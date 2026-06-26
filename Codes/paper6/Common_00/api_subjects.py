"""Routing for the E4 black-box (API) subject models.

Each black-box decoder maps to an OpenAI-compatible provider, its served model id,
and the env-var NAMES holding that provider's keys (round-robin, no fallback). Only
NAMES appear here; values load from .env. Shared by the dry run (03) and E4 API
runner (06) so both test/select the exact same ids.
"""
from typing import Dict, Optional

from Common_00.common import get_env, get_keys
from Common_00.providers import JudgeProvider

SUBJECTS: Dict[str, Dict] = {
    "deepseek-chat": {
        "base_url_default": "https://api.deepseek.com/v1", "base_url_env": "DEEPSEEK_API_BASE_URL",
        "model": "deepseek-chat", "key_env_vars": ["DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2"]},
    "llama-3.3-70b": {
        "base_url_default": "https://openrouter.ai/api/v1", "base_url_env": "OPENROUTER_API_BASE_URL",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "key_env_vars": ["OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"]},
    "gpt-oss-20b": {
        # AWS Bedrock OpenAI-compatible endpoint, 2 API keys round-robin - ~2x faster than
        # NanoGPT in benchmarking. Bedrock returns the chain-of-thought inside the content as
        # <reasoning>...</reasoning><answer>, which the A/B parser strips (see rag_leakfree).
        # (NanoGPT fallback: base https://nano-gpt.com/api/v1, model openai/gpt-oss-20b,
        #  keys NanoGPT_API_Key / Nano_GPT_API_KEY.)
        "base_url_default": "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
        "base_url_env": "AWS_BEDROCK_BASE_URL",
        "model": "openai.gpt-oss-20b-1:0",
        "key_env_vars": ["AWS_Bedrock_API_Key1", "AWS_Bedrock_API_Key2"],
        "max_tokens": 1024},   # gpt-oss-20b reasons before answering; give room for a visible letter
}


def subject_base_url(spec: Dict) -> str:
    return get_env(spec.get("base_url_env", ""), None) or spec["base_url_default"]


def build_subject(short: str, max_tokens: int = 64) -> Optional[JudgeProvider]:
    """A round-robin client for one API subject model, or None if no keys are set.
    64 tokens: gpt-oss-20b is a reasoning model and would truncate at a tiny budget;
    the A/B letter is parsed out of the (possibly longer) reply downstream."""
    if short not in SUBJECTS:
        return None
    spec = SUBJECTS[short]
    keys = get_keys(spec["key_env_vars"])
    if not keys:
        return None
    mt = spec.get("max_tokens", max_tokens)   # per-subject override (reasoning models need more)
    return JudgeProvider(name=short, base_url=subject_base_url(spec), model=spec["model"],
                         keys=keys, temperature=0.0, max_tokens=mt, timeout_s=120)
