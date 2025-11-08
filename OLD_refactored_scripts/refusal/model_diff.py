from __future__ import annotations

from typing import Dict, Tuple

import torch

from .config import DEVICE
from .activations import get_hooked_activations
from .steering import compute_steering_vectors


def compute_category_mean_activations(
    hooked_model,
    dataloaders_by_category: Dict[str, any],
    activation_name: str = "resid_post",
    layer: int = 9,
    position: int = -1,
    prompt_seq_append: str = "",
    device: str | torch.device = DEVICE,
) -> Dict[str, torch.Tensor]:
    """Compute mean activation vectors per category for a given HookedTransformer model.

    Returns a mapping {category: mean_activation[d_model]}.
    """
    means: Dict[str, torch.Tensor] = {}
    for category, dl in dataloaders_by_category.items():
        _, mean_act = get_hooked_activations(
            hooked_model=hooked_model,
            iterator=dl,
            activation_name=activation_name,
            layer=layer,
            position=position,
            prompt_seq_append=prompt_seq_append,
            device=device,
        )
        means[category] = mean_act
    return means


def compute_model_steering_vectors_from_dataloaders(
    hooked_model,
    harmful_by_cat: Dict[str, any],
    benign_by_cat: Dict[str, any],
    activation_name: str = "resid_post",
    layer: int = 9,
    position: int = -1,
    prompt_seq_append: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    device: str | torch.device = DEVICE,
) -> Dict[str, torch.Tensor]:
    """Convenience to compute steering vectors per category for a single model from two category dataloader maps.

    It computes mean activations for harmful and benign dataloaders category-wise, then calls compute_steering_vectors.
    """
    mean_harmful = compute_category_mean_activations(
        hooked_model,
        harmful_by_cat,
        activation_name=activation_name,
        layer=layer,
        position=position,
        prompt_seq_append=prompt_seq_append,
        device=device,
    )
    mean_benign = compute_category_mean_activations(
        hooked_model,
        benign_by_cat,
        activation_name=activation_name,
        layer=layer,
        position=position,
        prompt_seq_append=prompt_seq_append,
        device=device,
    )

    steering = compute_steering_vectors(
        mean_benign_dict=mean_benign,
        mean_harmful_dict=mean_harmful,
        should_filter_shared=False,
        K=100,
        tau=1e-3,
    )
    return steering
