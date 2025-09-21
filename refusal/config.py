from __future__ import annotations

import os
from pathlib import Path
import torch
try:
    # Optional: load environment variables from a .env file if present
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "synthethic_data"
SAVED_OUTPUTS_DIR = PROJECT_ROOT / "saved_outputs"
SAVED_TENSORS_DIR = PROJECT_ROOT / "saved_tensors"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure important directories exist
for _p in (DATA_DIR, SAVED_OUTPUTS_DIR, SAVED_TENSORS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Devices and defaults
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DEFAULT_MODEL = os.environ.get("REFUSAL_DEFAULT_MODEL", "sshleifer/tiny-gpt2")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("REFUSAL_MAX_NEW_TOKENS", 40))
DEFAULT_BATCH_SIZE = int(os.environ.get("REFUSAL_BATCH_SIZE", 4))

# Random seed (optional)
DEFAULT_SEED = int(os.environ.get("REFUSAL_SEED", 42))
