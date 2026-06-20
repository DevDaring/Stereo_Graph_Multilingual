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
        "base_url_default": "https://openrouter.ai/api/v1", "base_url_env": "OPENROUTER_API_BASE_URL",
        "model": "openai/gpt-oss-20b",
        "key_env_vars": ["OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"]},
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
    return JudgeProvider(name=short, base_url=subject_base_url(spec), model=spec["model"],
                         keys=keys, temperature=0.0, max_tokens=max_tokens, timeout_s=60)
