from __future__ import annotations

from typing import Callable, List, Dict, Any, Tuple

import torch
from torch import amp
import torch.nn.functional as F

from transformer_lens.utils import get_act_name  # requires transformer_lens

from .config import DEVICE, DEFAULT_SEED


# ----- Activation Patching (TransformerLens HookedTransformer) -----

def generate_with_activation_patching(
    clean_prompt: str,
    corrupt_prompt: str,
    hooked_model,
    tokenizer,
    hidden_ids: List[int] | None = None,
    generate_baseline: bool = False,
    layer: int = 9,
    position: int = -1,
    activation_name: str = "resid_post",
    max_new_tokens: int = 200,
    do_sample: bool = False,
    device: str | torch.device = DEVICE,
) -> str | Tuple[str, str]:
    """Activation patching: run clean prompt, cache its activation, then inject into corrupt prompt.

    If generate_baseline=True, returns (baseline, patched); otherwise returns patched only.
    """
    if isinstance(device, str):
        device = torch.device(device)

    hooked_model.reset_hooks()

    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    # Tokenization
    clean_tokens = hooked_model.to_tokens(clean_prompt).to(device)
    corrupt_tokens = hooked_model.to_tokens(corrupt_prompt).to(device)

    # Get the clean model cache
    _, cache_clean = hooked_model.run_with_cache(clean_tokens, remove_batch_dim=False)

    # Build the patching hook
    hook_name = get_act_name(activation_name, layer)

    def patch_hook(activation, hook):
        patched = activation.clone()
        residual = cache_clean[hook_name]
        if hidden_ids is None:
            patched[:, position, :] = residual[:, position, :]
        else:
            patched[:, position, hidden_ids] = residual[:, position, hidden_ids]
        return patched

    fwd_hooks = [(hook_name, patch_hook)]

    # Re-generate with the hook
    with hooked_model.hooks(fwd_hooks):
        torch.manual_seed(DEFAULT_SEED)
        patched = hooked_model.generate(
            corrupt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_type="str",
            stop_at_eos=True,
            eos_token_id=stop_ids,
        )

    hooked_model.reset_hooks()

    if generate_baseline:
        torch.manual_seed(DEFAULT_SEED)
        baseline = hooked_model.generate(
            corrupt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_type="str",
            stop_at_eos=True,
            eos_token_id=stop_ids,
        )
        return baseline, patched

    return patched


# ----- Attribution Patching / Neuron attribution via A*G -----

def generate_with_attribution_patching(
    target_prompt: str,
    hooked_model,
    tokenizer,
    layer: int = 6,
    position: int = -1,
    activation_name: str = "resid_post",
    refusal_token_id: int = 128259,
    top_k: int = 50,
) -> List[Tuple[int, float]]:
    """Run forward+backward on target_prompt, capture activation A and gradient G=∂L/∂A,
    compute saliency_i = |A_i * G_i|, and return top-k neuron indices and scores.
    """
    hooked_model.eval()

    tokens = hooked_model.to_tokens(target_prompt).to(hooked_model.cfg.device)

    saved: Dict[str, torch.Tensor] = {}

    def save_activation(activation, hook):
        # activation shape: (batch_size, seq_len, d_model)
        saved["activation"] = activation.clone().detach().requires_grad_(True)

    hook_name = get_act_name(activation_name, layer)
    hooked_model.add_hook(hook_name, save_activation, "fwd")

    logits = hooked_model(tokens)  # [B, S, V]
    hooked_model.reset_hooks()

    # Refusal-token logit at the given position
    logit = logits[0, position, refusal_token_id]

    # Compute gradient wrt saved activation
    hooked_model.cfg.use_attn_result = False
    logit.backward()

    A = saved["activation"][0, position, :]
    G = saved["activation"].grad[0, position, :]

    saliency = (A * G).abs()
    top_vals, top_idx = torch.topk(saliency, top_k)
    return [(int(i), float(saliency[i])) for i in top_idx.tolist()]


# ----- HF module-level patching utilities (architecture-specific) -----

def refusal_logit_diff(logits: torch.Tensor, refusal_token_id: int, response_token_id: int) -> torch.Tensor:
    """Difference between refusal and response token logits at the last position (batch=1)."""
    return logits[0, -1, refusal_token_id] - logits[0, -1, response_token_id]


def get_residual_stream(model, tokens, layer_idx: int) -> torch.Tensor:
    """Return residual stream at layer_idx for last token using HF model with output_hidden_states=True.

    Note: Assumes a decoder-only model where hidden_states[layer_idx] aligns as in Llama.
    """
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
    hidden_states = output.hidden_states
    resid = hidden_states[layer_idx][:, -1, :]
    return resid.detach().clone()


def make_injection_hook(activation_to_patch: torch.Tensor) -> Callable:
    """Make a forward hook that overwrites the last token activation with activation_to_patch."""
    def hook_fn(module, input, output):
        output[:, -1, :] = activation_to_patch
        return output

    return hook_fn


def attribution_patch_resid(
    model,
    clean_tokens,
    corr_tokens,
    layer_idx: int,
    category_token_id: int,
    response_token_id: int,
) -> float:
    """Attribution patching score across HF module layer: (grad * delta).sum()."""
    clean_resid = get_residual_stream(model, clean_tokens, layer_idx)
    corr_resid = get_residual_stream(model, corr_tokens, layer_idx)

    clean_resid.requires_grad_(True)
    handle = model.model.layers[layer_idx].register_forward_hook(make_injection_hook(clean_resid))

    output = model(**clean_tokens)
    logits = output.logits

    loss = refusal_logit_diff(logits, category_token_id, response_token_id)
    loss.backward()
    grad = clean_resid.grad.detach()

    delta = corr_resid - clean_resid.detach()
    attribution_score = (grad * delta).sum().item()

    handle.remove()
    return attribution_score


def run_patch_across_layers(
    model,
    pair: Dict[str, Any],
    max_layer: int,
    category_token_id: int,
    response_token_id: int,
) -> List[float | None]:
    """Run attribution_patch_resid for layers 0..max_layer-1 and return list of scores or None on error."""
    scores: List[float | None] = []
    for layer_idx in range(max_layer):
        try:
            score = attribution_patch_resid(
                model,
                pair["clean_tokens"],
                pair["corr_tokens"],
                layer_idx,
                category_token_id,
                response_token_id,
            )
            scores.append(score)
        except Exception as e:  # pragma: no cover - best effort
            print(f"Layer {layer_idx}: error {e}")
            scores.append(None)
    return scores
