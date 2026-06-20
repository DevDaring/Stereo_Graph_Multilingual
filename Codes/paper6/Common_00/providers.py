"""Paper 6 judge / answer-extraction client.

PRIMARY provider is Gemini 2.5 Flash via 4 GCP keys (round-robin). DeepSeek,
Mistral Small, and OpenRouter are SELECTABLE alternatives. There is NO automatic
cross-provider fallback: the active provider is fixed by config.judge.provider, and
a failed call is reported as a failure (the caller records it), never silently
retried on a different provider.

All providers are reached through their OpenAI-compatible chat endpoint, so a single
client implementation covers them all. Keys load from .env only.
"""
from typing import Dict, List, Optional, Tuple

from Common_00.common import RoundRobin, get_env, get_keys


def _provider_keys(spec: Dict) -> List[str]:
    return get_keys(spec.get("key_env_vars", []))


def _provider_base_url(spec: Dict) -> Optional[str]:
    if spec.get("base_url"):
        return spec["base_url"]
    if spec.get("base_url_env"):
        v = get_env(spec["base_url_env"])
        if v:
            return v
    return spec.get("base_url_default")


class JudgeProvider:
    """One provider (e.g., gemini). Round-robin over its keys. No fallback."""

    def __init__(self, name: str, base_url: str, model: str, keys: List[str],
                 temperature: float = 0.0, max_tokens: int = 16, timeout_s: int = 60,
                 extra_params: Optional[Dict] = None):
        self.name = name
        self.base_url = base_url
        self.model = model
        self.keys = RoundRobin(keys)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        # extra create() kwargs, e.g. {"reasoning_effort": "none"} to disable
        # gemini-2.5-flash thinking so a small max_tokens still returns visible text.
        self.extra_params = dict(extra_params or {})
        self._clients: Dict[str, object] = {}

    @property
    def configured(self) -> bool:
        return bool(self.keys) and bool(self.base_url) and bool(self.model)

    def _client(self, key: str):
        if key not in self._clients:
            from openai import OpenAI  # imported lazily so dry-run can report cleanly
            self._clients[key] = OpenAI(api_key=key, base_url=self.base_url,
                                        timeout=self.timeout_s)
        return self._clients[key]

    def chat(self, system: Optional[str], user: str, max_attempts: int = 1) -> str:
        """Single round-robin call. No cross-provider fallback. `max_attempts`
        only rotates keys of THIS provider (default 1 = no retry)."""
        if not self.configured:
            raise RuntimeError(f"Judge provider '{self.name}' is not configured "
                               f"(missing keys, base_url, or model).")
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        last_err = None
        for _ in range(max(1, max_attempts)):
            key = self.keys.next()
            try:
                resp = self._client(key).chat.completions.create(
                    model=self.model, messages=msgs,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    **self.extra_params)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # same-provider only; never switch providers
                last_err = e
        raise RuntimeError(f"Judge provider '{self.name}' failed: {last_err}")


def build_provider(name: str, jcfg: Dict) -> JudgeProvider:
    spec = jcfg["providers"][name]
    return JudgeProvider(
        name=name,
        base_url=_provider_base_url(spec),
        model=spec["model"],
        keys=_provider_keys(spec),
        temperature=jcfg.get("temperature", 0.0),
        max_tokens=jcfg.get("max_tokens", 16),
        timeout_s=jcfg.get("timeout_s", 60),
        extra_params=spec.get("extra_params"),
    )


def get_judge(config: Dict) -> JudgeProvider:
    """The active judge provider (config.judge.provider). No fallback chain."""
    jcfg = config["judge"]
    return build_provider(jcfg["provider"], jcfg)


def build_all_providers(config: Dict) -> Dict[str, JudgeProvider]:
    """Every configured provider, for the dry-run key/model test."""
    jcfg = config["judge"]
    return {name: build_provider(name, jcfg) for name in jcfg["providers"]}


def provider_key_report(config: Dict) -> List[Tuple[str, int]]:
    """(provider, n_keys) for each provider; used by the dry run."""
    jcfg = config["judge"]
    out = []
    for name, spec in jcfg["providers"].items():
        out.append((name, len(get_keys(spec.get("key_env_vars", [])))))
    return out
