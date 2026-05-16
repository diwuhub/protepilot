"""
optional_deps.py  ·  ProtePilot — Optional Dependency Registry
================================================================
Central registry for optional dependencies that may or may not be installed.
Provides clear error messages directing users to the correct requirements layer.

Usage:
    from src.optional_deps import check_layer, available

    # Check if a specific package is available
    if available("torch"):
        import torch

    # Raise a helpful error if a layer isn't installed
    check_layer("training")  # raises ImportError with install instructions

    # Get a summary of what's installed
    from src.optional_deps import layer_status
    status = layer_status()
    # {'core': True, 'analysis': True, 'training': False}
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict

log = logging.getLogger("ProtePilot.OptionalDeps")


# ═══════════════════════════════════════════════════════════════════════
#  Package → Layer mapping
# ═══════════════════════════════════════════════════════════════════════

# Layer 1: Core Runtime (requirements-core.txt)
_CORE_PACKAGES = [
    "numpy", "pandas", "Bio",       # Bio = biopython
    "plotly", "matplotlib", "kaleido",
    "docx",                          # docx = python-docx
    "h5py", "requests", "pytest",
]

# Layer 2: Analysis Runtime (requirements-analysis.txt)
_ANALYSIS_PACKAGES = [
    "streamlit", "py3Dmol", "stmol", "openai",
]

# Layer 3: Training Runtime (requirements-training.txt)
_TRAINING_PACKAGES = [
    "torch", "xgboost", "sklearn",  # sklearn = scikit-learn
    "joblib", "shap", "transformers", "sentencepiece",
]

_LAYER_NAMES = {
    "core":     ("Layer 1 — Core Runtime",     "requirements-core.txt"),
    "analysis": ("Layer 2 — Analysis Runtime",  "requirements-analysis.txt"),
    "training": ("Layer 3 — Training Runtime",  "requirements-training.txt"),
}

_PACKAGE_TO_LAYER = {}
for pkg in _CORE_PACKAGES:
    _PACKAGE_TO_LAYER[pkg] = "core"
for pkg in _ANALYSIS_PACKAGES:
    _PACKAGE_TO_LAYER[pkg] = "analysis"
for pkg in _TRAINING_PACKAGES:
    _PACKAGE_TO_LAYER[pkg] = "training"


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def available(package: str) -> bool:
    """Check if a package is importable without actually importing it.

    >>> available("torch")    # True if torch is installed
    >>> available("numpy")    # True (core dependency)
    """
    try:
        importlib.import_module(package)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def check_layer(layer: str) -> None:
    """Raise ImportError with install instructions if a layer isn't installed.

    Parameters
    ----------
    layer : str
        One of 'core', 'analysis', 'training'.

    Raises
    ------
    ImportError
        If any package in the specified layer is missing.
    """
    if layer not in _LAYER_NAMES:
        raise ValueError(f"Unknown layer: {layer!r}.  Use 'core', 'analysis', or 'training'.")

    layer_name, req_file = _LAYER_NAMES[layer]
    packages = {
        "core": _CORE_PACKAGES,
        "analysis": _ANALYSIS_PACKAGES,
        "training": _TRAINING_PACKAGES,
    }[layer]

    missing = [pkg for pkg in packages if not available(pkg)]
    if missing:
        raise ImportError(
            f"{layer_name} is not fully installed.  "
            f"Missing packages: {', '.join(missing)}.  "
            f"Install with:  pip install -r {req_file}"
        )


def require(package: str) -> None:
    """Raise ImportError with layer-specific install instructions if package is missing.

    >>> require("torch")  # raises ImportError pointing to requirements-training.txt
    """
    if available(package):
        return

    layer = _PACKAGE_TO_LAYER.get(package, "training")
    layer_name, req_file = _LAYER_NAMES[layer]
    raise ImportError(
        f"'{package}' is required but not installed.  "
        f"It belongs to {layer_name}.  "
        f"Install with:  pip install -r {req_file}"
    )


def layer_status() -> Dict[str, bool]:
    """Return which layers are fully installed.

    >>> layer_status()
    {'core': True, 'analysis': True, 'training': False}
    """
    return {
        "core":     all(available(p) for p in _CORE_PACKAGES),
        "analysis": all(available(p) for p in _ANALYSIS_PACKAGES),
        "training": all(available(p) for p in _TRAINING_PACKAGES),
    }


def print_status() -> None:
    """Print a human-readable dependency status report."""
    status = layer_status()
    print("ProtePilot — Dependency Status")
    print("=" * 50)
    for layer, ok in status.items():
        name, req_file = _LAYER_NAMES[layer]
        icon = "OK" if ok else "MISSING"
        print(f"  {name:40s}  [{icon}]")
        if not ok:
            packages = {
                "core": _CORE_PACKAGES,
                "analysis": _ANALYSIS_PACKAGES,
                "training": _TRAINING_PACKAGES,
            }[layer]
            missing = [p for p in packages if not available(p)]
            print(f"    Missing: {', '.join(missing)}")
            print(f"    Fix:     pip install -r {req_file}")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_status()
