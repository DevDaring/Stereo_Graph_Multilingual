"""Guard: no API key value is hard-coded anywhere in the Robust_Rerun tree.

Scans every tracked text file (excluding .env and results) for credential-shaped
strings and for assignment of any known key env-var NAME to a non-empty literal.
The only place a secret may live is the repo-root .env.

Run:  python tests/test_no_secrets.py    (or: pytest tests/test_no_secrets.py)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Robust_Rerun/
SCAN_EXT = (".py", ".md", ".yaml", ".yml", ".csv", ".json", ".txt", ".cfg", ".ini", ".html")
SKIP_DIRS = {".git", "results", "__pycache__", ".pytest_cache"}
SKIP_FILES = {".env"}

PATTERNS = [
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "OpenAI/DeepSeek/LinkAPI-style key"),
    (re.compile(r"\bsk-or-[A-Za-z0-9-]{20,}"), "OpenRouter key"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), "Google API key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "HuggingFace token"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "GitHub token"),
    (re.compile(r"\bfish_[A-Za-z0-9]{40,}"), "Sakana key"),
]

KEY_NAMES = [
    "Link_Gemini_Cheap_API_Key",
    "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2",
    "MISTRAL_API_KEY1", "MISTRAL_API_KEY2",
    "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "HUGGINGFACE_TOKEN",
]
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
            if val.strip() and not val.strip().startswith("$"):
                violations.append(f"{rel}: hard-coded value for {name}")
    return violations


def test_no_secrets():
    assert scan() == []


def main():
    v = scan()
    if v:
        print("SECRET-LEAK CHECK FAILED:")
        for x in v:
            print("  -", x)
        sys.exit(1)
    print("SECRET-LEAK CHECK PASSED: no hard-coded credentials in Robust_Rerun.")


if __name__ == "__main__":
    main()
