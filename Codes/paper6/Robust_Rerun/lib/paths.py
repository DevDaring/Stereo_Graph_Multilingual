"""Config access for Robust_Rerun. (sys.path is wired in lib/__init__.py.)"""
import json
import os

from Common_00.common import load_config, resolve

ROBUST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROBUST_ROOT, "config", "default.yaml")


def cfg():
    """Load the Robust_Rerun config (non-secret; keys load from .env separately)."""
    return load_config(CONFIG_PATH)


def results_path(*parts):
    return resolve(os.path.join(cfg()["paths"]["results"], *parts))


def load_split(config):
    path = resolve(config["paths"]["split_file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"concept split not built: {path}. Run step_02_split.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
