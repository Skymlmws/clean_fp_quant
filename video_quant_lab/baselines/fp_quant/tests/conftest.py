"""Make the self-contained FP-Quant baseline importable during project tests."""

from pathlib import Path
import sys
import types

BASELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# CPU unit tests exercise the native PyTorch fallback and do not need the
# optional CUDA extension.  Keep collection working in lightweight dev envs.
try:
    import fast_hadamard_transform  # noqa: F401
except ModuleNotFoundError:
    fallback = types.ModuleType("fast_hadamard_transform")

    def _unavailable_hadamard_transform(*_args, **_kwargs):
        raise RuntimeError("fast_hadamard_transform is required for CUDA tests")

    fallback.hadamard_transform = _unavailable_hadamard_transform
    sys.modules["fast_hadamard_transform"] = fallback
