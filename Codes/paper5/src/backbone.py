"""
backbone.py - Frozen multilingual encoders for Paper 2.

Loads one of three frozen encoders and exposes two views of each:
  - load_encoder()    -> AutoModel (last_hidden_state) for representation
                         debiasing, SEAT, and the utility probe.
  - load_masked_lm()  -> AutoModelForMaskedLM for the pseudo-log-likelihood
                         bias score, with a hook to insert a debiasing
                         projection between the encoder and the MLM head.

All backbone weights are FROZEN. Flash-Attention-2 is requested when
dtype is fp16/bf16, the GPU compute capability is >= sm_80, and the
flash_attn package is importable; otherwise the code falls back to
PyTorch SDPA. On Windows flash-attn is normally unavailable, so SDPA is
used automatically (no failure).
"""

# =====================================================================
# CITATION(S) for this module:
#   [conneau2020xlmr] Conneau et al., "Unsupervised Cross-lingual
#     Representation Learning at Scale (XLM-R)," ACL 2020. arXiv:1911.02116.
#   [khanuja2021muril] Khanuja et al., "MuRIL: Multilingual Representations
#     for Indian Languages," 2021. arXiv:2103.10730.
#   [devlin2019bert] Devlin et al., "BERT," NAACL 2019.  (mBERT)
#   [dao2023flashattention2] Dao, "FlashAttention-2," 2023. arXiv:2307.08691.
# =====================================================================

import hashlib
import importlib
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger("backbone")

SUPPORTED_BACKBONES = [
    "xlm-roberta-base",
    "google/muril-base-cased",
    "bert-base-multilingual-cased",
]

_ATTN_IMPL_LOG: Dict[str, str] = {}


def _can_use_flash_attn(dtype: torch.dtype) -> bool:
    """Return True if flash-attn-2 can be used given dtype and GPU."""
    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("[INFO] flash-attn needs fp16/bf16; dtype=%s -> SDPA.", dtype)
        return False
    try:
        importlib.import_module("flash_attn")
    except ImportError:
        logger.info("[INFO] flash_attn not installed -> SDPA.")
        return False
    if not torch.cuda.is_available():
        logger.info("[INFO] No CUDA device -> SDPA.")
        return False
    major, _ = torch.cuda.get_device_capability(0)
    if major < 8:
        # CITATION: dao2023flashattention2 - sm_80 minimum for flash-attn 2.x
        logger.warning("[FAIL] GPU sm_%d < sm_80; flash-attn unsupported -> SDPA.", major)
        return False
    return True


def _resolve_attn_impl(backbone_name: str, dtype: torch.dtype, use_flash: bool) -> str:
    impl = "sdpa"  # encoders (XLM-R/mBERT/MuRIL) do NOT support FA2; SDPA is correct + tiny cost
    _ATTN_IMPL_LOG[backbone_name] = impl
    logger.info("[OK] %s attention implementation: %s", backbone_name, impl)
    return impl


def _dtype_from_str(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(name, torch.float16)


def _common_load_kwargs(dtype: torch.dtype, attn_impl: str) -> dict:
    token = os.getenv("HUGGINGFACE_TOKEN") or None
    return {"torch_dtype": dtype, "token": token, "attn_implementation": attn_impl}


def _freeze(model) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def load_encoder(
    backbone_name: str,
    dtype: torch.dtype = torch.float16,
    use_flash_attention: bool = True,
    device: Optional[str] = None,
) -> Tuple[object, object]:
    """Load a frozen AutoModel encoder and tokenizer. Returns (model, tokenizer)."""
    from transformers import AutoModel, AutoTokenizer

    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(f"[FAIL] Unsupported backbone '{backbone_name}'. Supported: {SUPPORTED_BACKBONES}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    attn_impl = _resolve_attn_impl(backbone_name, dtype, use_flash_attention)
    token = os.getenv("HUGGINGFACE_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=True, token=token)
    model = AutoModel.from_pretrained(backbone_name, **_common_load_kwargs(dtype, attn_impl))
    _freeze(model)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 0, "[FAIL] Encoder freeze failed (trainable params > 0)."
    logger.info("[OK] Encoder loaded: %s | device=%s | attn=%s", backbone_name, device, attn_impl)
    return model, tokenizer


def load_masked_lm(
    backbone_name: str,
    dtype: torch.dtype = torch.float16,
    use_flash_attention: bool = True,
    device: Optional[str] = None,
) -> Tuple[object, object]:
    """Load a frozen AutoModelForMaskedLM and tokenizer. Returns (model, tokenizer)."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(f"[FAIL] Unsupported backbone '{backbone_name}'. Supported: {SUPPORTED_BACKBONES}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    attn_impl = _resolve_attn_impl(backbone_name, dtype, use_flash_attention)
    token = os.getenv("HUGGINGFACE_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=True, token=token)
    model = AutoModelForMaskedLM.from_pretrained(backbone_name, **_common_load_kwargs(dtype, attn_impl))
    _freeze(model)
    model.to(device)
    logger.info("[OK] MaskedLM loaded: %s | device=%s | attn=%s", backbone_name, device, attn_impl)
    return model, tokenizer


def get_mlm_components(masked_lm_model):
    """
    Split a MaskedLM model into (encoder, head) so a debiasing projection can
    be inserted between them.

    encoder(input_ids, attention_mask).last_hidden_state -> [B, T, d]
    head(hidden_states) -> [B, T, vocab]

    Works for RoBERTa-family (lm_head) and BERT-family (cls) MaskedLM heads.
    """
    encoder = masked_lm_model.base_model  # RobertaModel / BertModel
    if hasattr(masked_lm_model, "lm_head"):
        head = masked_lm_model.lm_head              # RoBERTa / XLM-R
    elif hasattr(masked_lm_model, "cls"):
        head = masked_lm_model.cls                  # BERT / mBERT / MuRIL
    else:
        raise AttributeError("[FAIL] Could not locate the MLM head on this model.")
    return encoder, head


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
    batch_size: int = 32,
    pooling: str = "mean",
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Encode texts with a frozen encoder and return a pooled [N, d] float32 matrix.
    pooling: 'mean' (masked mean over tokens) or 'cls' (first token).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    out_vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        enc = tokenizer(batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        hidden = model(input_ids=input_ids, attention_mask=attn).last_hidden_state  # [B, T, d]
        if pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = attn.unsqueeze(-1).to(hidden.dtype)            # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
        out_vectors.append(pooled.float().cpu())

    return torch.cat(out_vectors, dim=0)


def get_attn_impl_log() -> Dict[str, str]:
    return dict(_ATTN_IMPL_LOG)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
