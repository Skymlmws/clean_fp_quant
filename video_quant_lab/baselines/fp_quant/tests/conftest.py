"""Make the self-contained FP-Quant baseline importable during project tests."""

from pathlib import Path
import sys


BASELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
