"""Refusal analysis package.

Lightweight scaffolding to support migrating code out of notebooks into modules.
"""

from .config import (
    DEVICE as device,
    PROJECT_ROOT,
    SAVED_OUTPUTS_DIR,
    SAVED_TENSORS_DIR,
    FIGURES_DIR,
)

__all__ = [
    "device",
    "PROJECT_ROOT",
    "SAVED_OUTPUTS_DIR",
    "SAVED_TENSORS_DIR",
    "FIGURES_DIR",
]
