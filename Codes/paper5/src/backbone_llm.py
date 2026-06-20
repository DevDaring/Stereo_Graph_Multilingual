"""
backbone_llm.py - Local 4-bit causal LLMs with mechanistic access (Papers 4 and 5).

Loads Qwen2.5-7B-Instruct and Aya-23-8B in 4-bit (bitsandbytes NF4) with the SAME
quantization settings as Papers 1 and 3, so numbers stay comparable across the
combined paper. Flash-Attention-2 is requested for fast GPU execution; if the
pre-compiled wheel is not importable for the installed torch + CUDA build, the
code falls back to PyTorch SDPA automatically (no failure).

Beyond generation and causal scoring, this module exposes the white-box hooks the
mechanistic studies need:
  - layers / num_layers          : the transformer block list
  - hidden_states(text)          : per-layer residual stream for a sequence
  - logit_lens_gap(...)          : per-layer stereotype-minus-anti logit gap
  - insert_hook(layer_idx, fn)   : forward hook to filter/patch a layer output
"""

# =====================================================================
# CITATION(S) for this module:
#   [qwen2.5] Qwen Team, "Qwen2.5 Technical Report," 2024. arXiv:2412.15115.
#   [aya2024] Aryabumi et al., "Aya 23," 2024. arXiv:2405.15032.
#   [dettmers2023qlora] Dettmers et al., "QLoRA / NF4," NeurIPS 2023. arXiv:2305.14314.
#   [dao2023flashattention2] Dao, "FlashAttention-2," 2023. arXiv:2307.08691.
#   [belrose2023tunedlens] Belrose et al., "Eliciting Latent Predictions
#     (Tuned/Logit Lens)," 2023. arXiv:2303.08112. Implements: per-layer logit lens.
# =====================================================================

import contextlib
import importlib
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger("backbone_llm")

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}


def _flash_attn_available() -> bool:
    try:
        importlib.import_module("flash_attn")
        return True
    except ImportError:
        return False


def _resolve_attn_impl(use_flash: bool) -> str:
    if use_flash and _flash_attn_available() and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:  # CITATION: dao2023flashattention2 - sm_80 minimum
            logger.info("[OK] flash_attention_2 enabled.")
            return "flash_attention_2"
        logger.warning("[WARN] GPU sm_%d < sm_80; flash-attn unsupported -> SDPA.", major)
    else:
        logger.info("[INFO] flash_attn wheel not importable or no CUDA -> SDPA.")
    return "sdpa"


class LocalCausalLLM:
    """A frozen 4-bit causal LLM with scoring, generation, and white-box hooks."""

    def __init__(self, hf_id: str, quant_cfg: Dict, backbone_cfg: Dict,
                 device: Optional[str] = None):
        self.hf_id = hf_id
        self.quant_cfg = quant_cfg
        self.backbone_cfg = backbone_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.attn_impl = "sdpa"

    def load(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_token = os.getenv("HUGGINGFACE_TOKEN") or None
        self.attn_impl = _resolve_attn_impl(self.backbone_cfg.get("use_flash_attention", True))
        compute_dtype = _DTYPE.get(self.quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
                                   torch.bfloat16)
        bnb = BitsAndBytesConfig(
            load_in_4bit=bool(self.quant_cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type=self.quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(self.quant_cfg.get("bnb_4bit_use_double_quant", True)),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, token=hf_token, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, quantization_config=bnb, device_map="auto",
            torch_dtype=compute_dtype, attn_implementation=self.attn_impl,
            token=hf_token, trust_remote_code=True,
        )
        self.model.eval()
        self._loaded = True
        logger.info("[OK] Loaded %s | 4-bit NF4 | attn=%s | %d layers",
                    self.hf_id, self.attn_impl, self.num_layers)

    def unload(self) -> None:
        del self.model
        self.model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- white-box structure ----------------------------------------------

    @property
    def layers(self) -> List[torch.nn.Module]:
        """The decoder block list, located across common architectures."""
        self.load()
        m = self.model
        for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
            obj = m
            ok = True
            for part in path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok:
                return list(obj)
        raise AttributeError(f"[FAIL] Could not locate decoder layers for {self.hf_id}.")

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def _final_norm_and_head(self):
        """Return (final_norm, lm_head) for logit-lens projection."""
        self.load()
        m = self.model
        head = m.get_output_embeddings()
        norm = None
        for path in ("model.norm", "model.model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
            obj = m
            ok = True
            for part in path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok:
                norm = obj
                break
        return norm, head

    # -- hooks -------------------------------------------------------------

    @contextlib.contextmanager
    def insert_hook(self, layer_idx: int, fn: Callable):
        """
        Temporarily insert a forward hook at decoder layer `layer_idx`.
        `fn(output_hidden[B,T,d]) -> new_hidden[B,T,d]` rewrites the block output.
        Decoder blocks usually return a tuple (hidden, ...); the first element is
        rewritten and the rest are passed through unchanged.
        """
        layer = self.layers[layer_idx]

        def _hook(_module, _inp, out):
            if isinstance(out, tuple):
                new0 = fn(out[0])
                return (new0,) + tuple(out[1:])
            return fn(out)

        handle = layer.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    # -- intrinsic scoring -------------------------------------------------

    @torch.no_grad()
    def sequence_log_likelihood(self, text: str) -> Tuple[float, int]:
        """Total causal log-likelihood of `text`. Returns (total_logprob, n_tokens)."""
        self.load()
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.backbone_cfg.get("max_length", 128))
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc["attention_mask"].to(self.model.device)
        out = self.model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(token_lp.sum().item()), int(token_lp.size(1))

    @torch.no_grad()
    def hidden_states(self, text: str) -> torch.Tensor:
        """Per-layer residual stream for `text`: tensor [L+1, T, d] on CPU (float32)."""
        self.load()
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.backbone_cfg.get("max_length", 128))
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc["attention_mask"].to(self.model.device)
        out = self.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0).squeeze(1)  # [L+1, T, d]
        return hs.float().cpu()

    @torch.no_grad()
    def logit_lens_gap(self, text: str, token_a: str, token_b: str) -> List[float]:
        """
        Per-layer logit-lens gap between two candidate continuation tokens.
        Projects each layer's last-position residual through (final_norm, lm_head)
        and returns [gap_0, ..., gap_L] where gap = logit(token_a) - logit(token_b).
        CITATION: belrose2023tunedlens - logit lens.
        """
        self.load()
        norm, head = self._final_norm_and_head()

        def _first_tok(tok: str):
            # Mid-sentence words are usually a leading-space sub-word (e.g. " Brahmin");
            # try the space-prefixed form first for an accurate logit-lens token.
            ids = self.tokenizer.encode(" " + tok, add_special_tokens=False)
            if not ids:
                ids = self.tokenizer.encode(tok, add_special_tokens=False)
            return ids[0] if ids else None

        ida, idb = _first_tok(token_a), _first_tok(token_b)
        if ida is None or idb is None:
            return []
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.backbone_cfg.get("max_length", 128))
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc["attention_mask"].to(self.model.device)
        out = self.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        gaps = []
        for h in out.hidden_states:                    # each [1, T, d]
            last = h[:, -1, :]
            normed = norm(last) if norm is not None else last
            logits = head(normed).float().squeeze(0)   # [vocab]
            gaps.append(float(logits[ida].item() - logits[idb].item()))
        return gaps

    # -- expressed generation ---------------------------------------------

    @torch.no_grad()
    def generate(self, system_prompt: Optional[str], user_prompt: str,
                 max_new_tokens: Optional[int] = None) -> str:
        self.load()
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
            + [{"role": "user", "content": user_prompt}]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens or self.backbone_cfg.get("max_new_tokens", 24),
            do_sample=False, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        out_ids = gen[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
