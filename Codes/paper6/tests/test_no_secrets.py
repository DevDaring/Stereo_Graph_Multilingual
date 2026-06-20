"""Guard test: no API key value is hard-coded anywhere in the Paper 6 tree.

Scans every .py / .md / .yaml / .csv / .json under paper6 (excluding .env and the
results cache) for credential-shaped strings and for assignment of any known key
env-var NAME to a non-empty literal. The only place a secret may live is .env.

Run:  python tests/test_no_secrets.py    (or: pytest tests/test_no_secrets.py)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_EXT = (".py", ".md", ".yaml", ".yml", ".csv", ".json", ".txt", ".cfg", ".ini")
SKIP_DIRS = {".git", "results", "__pycache__", ".pytest_cache"}
SKIP_FILES = {".env"}

# Credential-shaped literals that must never appear in tracked files.
PATTERNS = [
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "OpenAI/DeepSeek-style key"),
    (re.compile(r"\bsk-or-[A-Za-z0-9-]{20,}"), "OpenRouter key"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), "Google API key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "HuggingFace token"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "GitHub token"),
]

# env-var NAMES that may be referenced but never assigned a literal value in code.
KEY_NAMES = [
    "GCP_Key1", "GCP_Key2", "GCP_Key3", "GCP_key4",
    "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
    "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2",
    "MISTRAL_API_KEY1", "MISTRAL_API_KEY2",
    "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "HUGGINGFACE_TOKEN",
]
# `NAME = "..."`  or  `NAME: "..."` with a non-empty value (an actual hard-coded secret).
ASSIGN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in KEY_NAMES) + r")\b\s*[:=]\s*['\"]([^'\"]+)['\"]")


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn in SKIP_FILES or fn == ".env":
                continue
            if os.path.splitext(fn)[1].lower() in SCAN_EXT:
                yield os.path.join(dirpath, fn)


def scan():
    violations = []
    for path in iter_files():
        rel = os.path.relpath(path, ROOT)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        for rx, label in PATTERNS:
            for m in rx.findall(text):
                violations.append(f"{rel}: {label} -> {str(m)[:12]}...")
        for name, val in ASSIGN.findall(text):
            # an env-var-name placeholder (.env.example) has empty value; allow that.
            if val.strip() and not val.strip().startswith("$"):
                violations.append(f"{rel}: hard-coded value for {name}")
    return violations


def main():
    v = scan()
    if v:
        print("SECRET-LEAK CHECK FAILED:")
        for x in v:
            print("  -", x)
        sys.exit(1)
    print("SECRET-LEAK CHECK PASSED: no hard-coded credentials in the Paper 6 tree.")


def test_no_secrets():
    assert scan() == []


if __name__ == "__main__":
    main()
