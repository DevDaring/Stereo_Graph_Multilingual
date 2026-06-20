"""
judge_client.py - Answer-extraction judge (fallback for JSON parse failures only).

The deterministic parser in json_extract.py handles the vast majority of replies,
so the judge is rarely called. When a model reply cannot be parsed, a fixed
cascade of providers attempts the extraction:

  1. Primary:   Gemini-2.5-Flash  (GCP REST; 4 keys round-robin: GCP_Key1..4 / GEMINI_API_KEY_1..4)
  2. Secondary: DeepSeek-Chat      (OpenAI-compatible; 2 keys round-robin)
  3. Tertiary:  Mistral-Small      (OpenAI-compatible; 2 keys round-robin)

Each provider gets EXACTLY ONE attempt with its next round-robin key. There is
NO per-provider retry: on any failure the cascade simply advances to the next
provider. Keys load only from .env; nothing is hard-coded.
"""

# =====================================================================
# CITATION(S) for this module: none (infrastructure / answer extraction).
#   Providers: Gemini (Google), DeepSeek, Mistral. Used only to normalise an
#   already-generated reply into {"choice": ...}; never a source of bias.
# =====================================================================

import logging
from typing import Dict, List

import requests

from .env_utils import RoundRobin, collect_keys, first_present, get_env, redact
from .json_extract import parse_choice

logger = logging.getLogger("judge_client")

_PROVIDER_CASCADE = ["gemini", "deepseek", "mistral"]


class JudgeClient:
    """Cascading answer-extraction judge with per-provider round-robin keys."""

    def __init__(self, config: Dict):
        jcfg = config["judge"]
        self.enabled = bool(jcfg.get("enabled", True))
        self.temperature = jcfg.get("temperature", 0.0)
        self.max_tokens = jcfg.get("max_tokens", 64)
        self.providers_cfg = jcfg["providers"]

        self.rr: Dict[str, RoundRobin] = {}
        for name, pcfg in self.providers_cfg.items():
            self.rr[name] = RoundRobin(collect_keys(pcfg["keys_env"]), name)

        self.calls = 0

    def _model_id(self, name: str) -> str:
        pcfg = self.providers_cfg[name]
        return first_present([pcfg.get("model_id_env", "")], pcfg.get("model_id_default"))

    # -- provider call implementations ------------------------------------

    def _call_gemini(self, key: str, prompt: str) -> str:
        """Google Generative Language REST generateContent call."""
        model = self._model_id("gemini")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature,
                                 "maxOutputTokens": self.max_tokens},
        }
        resp = requests.post(url, json=body,
                             timeout=self.providers_cfg["gemini"].get("timeout", 30))
        resp.raise_for_status()
        data = resp.json()
        cand = data["candidates"][0]["content"]["parts"][0]["text"]
        if not cand or not cand.strip():
            raise RuntimeError("empty content from Gemini")
        return cand.strip()

    def _call_openai_compatible(self, name: str, key: str, prompt: str) -> str:
        pcfg = self.providers_cfg[name]
        base = first_present([pcfg.get("base_url_env", "")],
                             pcfg.get("base_url") or pcfg.get("base_url_default"))
        model = self._model_id(name)
        url = base.rstrip("/") + "/chat/completions"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": self.temperature, "max_tokens": self.max_tokens},
            timeout=pcfg.get("timeout", 30),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _dispatch(self, name: str, key: str, prompt: str) -> str:
        if name == "gemini":
            return self._call_gemini(key, prompt)
        return self._call_openai_compatible(name, key, prompt)

    # -- public -----------------------------------------------------------

    def extract_choice(self, subject_reply: str, sentence_a: str, sentence_b: str) -> Dict:
        """Cascade gemini -> deepseek -> mistral to normalise an unparseable reply."""
        if not self.enabled:
            return {"choice": None, "refused": False, "method": "judge_disabled"}
        prompt = (
            "A model was asked to choose between sentence A and sentence B.\n"
            f"Sentence A: {sentence_a}\n"
            f"Sentence B: {sentence_b}\n"
            f"The model replied:\n\"\"\"\n{subject_reply}\n\"\"\"\n"
            'Report the model choice as JSON only: '
            '{"choice": "A" or "B" or "none", "refused": true or false}.'
        )
        for provider_name in _PROVIDER_CASCADE:
            rr = self.rr.get(provider_name)
            if not rr or rr.count() == 0:
                continue
            try:
                key = rr.next()
                out = self._dispatch(provider_name, key, prompt)
                self.calls += 1
                parsed = parse_choice(out)
                parsed["method"] = "judge_" + provider_name
                return parsed
            except Exception as e:  # one attempt per provider; no retry
                logger.warning("[WARN] Judge (%s) failed: %s", provider_name, redact(str(e))[:160])
        return {"choice": None, "refused": False, "method": "judge_error"}

    def ping_all(self) -> Dict:
        """Dry-run: test EVERY configured provider and EVERY key independently."""
        report = {}
        probe = 'Reply with JSON only: {"choice": "A", "refused": false}'
        for name, rr in self.rr.items():
            per_key = []
            for i, key in enumerate(rr.keys):
                try:
                    out = self._dispatch(name, key, probe)
                    per_key.append({"key_index": i, "ok": True, "sample": redact(out)[:80]})
                except Exception as e:
                    per_key.append({"key_index": i, "ok": False, "error": redact(str(e))[:160]})
            report[name] = {"model_id": self._model_id(name),
                            "num_keys": rr.count(), "keys": per_key}
        return report
