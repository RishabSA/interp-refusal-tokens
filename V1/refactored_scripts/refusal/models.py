from __future__ import annotations

from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf_model_and_tokenizer(model_name: str, device: str = None) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load a HF Causal LM and tokenizer in eval mode.

    - Ensures pad_token_id is set to eos_token_id if missing.
    - Returns model.to(device) and tokenizer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding often works better for generation with teacher forcing
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device).eval()
    return model, tokenizer


def load_hooked_transformer(hf_model: torch.nn.Module, model_name: str, device_map: str = "auto"):
    """Optionally wrap a HF model with TransformerLens HookedTransformer.

    This import is optional to keep a light dependency chain for quick smoke tests.
    Returns a HookedTransformer if possible, otherwise raises the underlying error.
    """
    try:
        from transformer_lens import HookedTransformer  # type: ignore
    except Exception as e:  # pragma: no cover - optional path
        raise RuntimeError(
            "transformer_lens is not installed or importable. Install it to use HookedTransformer."
        ) from e

    hooked = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        device_map=device_map,
    )
    return hooked
