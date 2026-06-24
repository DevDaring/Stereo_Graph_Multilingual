"""Importing anything from `lib` wires sys.path so the reused Paper 6 Common_00
library and this folder's modules both resolve, regardless of the caller's CWD.
"""
import os as _os
import sys as _sys

_LIB = _os.path.dirname(_os.path.abspath(__file__))
ROBUST_ROOT = _os.path.dirname(_LIB)            # Robust_Rerun/
PAPER6_ROOT = _os.path.dirname(ROBUST_ROOT)     # Codes/paper6/

for _p in (PAPER6_ROOT, ROBUST_ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
