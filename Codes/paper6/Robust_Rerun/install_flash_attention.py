"""GPU_Only - Install a PRE-COMPILED flash-attention wheel (no compilation).

Flash-attention dramatically speeds up the local decoders. Building from source is
slow; this script detects the environment (torch, CUDA, Python, C++ ABI, platform)
and downloads the matching pre-built wheel from the Dao-AILab/flash-attention
GitHub releases, then pip-installs and verifies it.

Pre-built wheels exist only for Linux x86_64. On other OSes the script reports that
flash-attention must be skipped (the pipeline still runs, just slower, because
backbone loading falls back to eager/sdpa attention).

Usage:
  python Common_00/install_flash_attention.py            # auto-detect, default ver
  python Common_00/install_flash_attention.py --version 2.8.3
"""
import argparse
import platform
import subprocess
import sys
import urllib.request

REL = "https://github.com/Dao-AILab/flash-attention/releases/download"
DEFAULT_VER = "2.8.3"   # the version used previously on GCP (SM89 / Ada)


def detect():
    import torch
    info = {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch": torch.__version__.split("+")[0],
        "cuda": torch.version.cuda,
        "cxx11abi": bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)),
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        cap = torch.cuda.get_device_capability(0)
        info["sm"] = f"{cap[0]}{cap[1]}"
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def wheel_url(ver, info):
    torch_mm = ".".join(info["torch"].split(".")[:2])           # 2.7.1 -> 2.7
    cu = "cu" + (info["cuda"].split(".")[0] if info["cuda"] else "12")  # 12.1 -> cu12
    abi = "TRUE" if info["cxx11abi"] else "FALSE"
    py = info["python"]
    name = (f"flash_attn-{ver}+{cu}torch{torch_mm}cxx11abi{abi}"
            f"-{py}-{py}-linux_x86_64.whl")
    return f"{REL}/v{ver}/{name}", name


def url_exists(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=DEFAULT_VER)
    args = ap.parse_args()

    print("=== flash-attention pre-compiled install ===")
    try:
        info = detect()
    except Exception as e:
        print(f"  could not import torch: {e}")
        print("  install torch first (requirements.txt), then re-run.")
        sys.exit(1)
    for k, v in info.items():
        print(f"  {k}: {v}")

    if info["os"] != "Linux" or info["machine"] not in ("x86_64", "AMD64"):
        print("  Pre-built flash-attn wheels are Linux x86_64 only.")
        print("  SKIPPING flash-attn here; the pipeline will use eager/sdpa attention.")
        print("  Run this on the GPU (Linux) machine before the GPU steps.")
        return
    if not info["cuda_available"]:
        print("  No CUDA GPU visible; skipping flash-attn install.")
        return

    url, name = wheel_url(args.version, info)
    print(f"  target wheel: {name}")
    print(f"  url: {url}")
    if not url_exists(url):
        print("  That exact wheel was not found on the releases page.")
        print("  Open https://github.com/Dao-AILab/flash-attention/releases and pick the")
        print(f"  wheel matching: torch{'.'.join(info['torch'].split('.')[:2])}, "
              f"cu{(info['cuda'] or '').split('.')[0]}, {info['python']}, "
              f"cxx11abi{'TRUE' if info['cxx11abi'] else 'FALSE'}, then pip install it.")
        sys.exit(2)

    print("  downloading + installing (no build, no isolation)...")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", url,
                          "--no-build-isolation"])
    if rc != 0:
        sys.exit(rc)
    try:
        import flash_attn  # noqa: F401
        print(f"  flash_attn {flash_attn.__version__} installed and importable. OK.")
    except Exception as e:
        print(f"  installed but import failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
