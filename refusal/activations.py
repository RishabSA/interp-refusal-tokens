from __future__ import annotations

from typing import Tuple, Dict

import torch
from torch import amp
from tqdm.auto import tqdm

from transformer_lens.utils import get_act_name  # requires transformer_lens

from .config import DEVICE


def get_hooked_activations(
    hooked_model,
    iterator,
    activation_name: str = "resid_post",
    layer: int = 9,
    position: int = -1,
    prompt_seq_append: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    device: torch.device | str = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract activations at a specific layer/name for all prompts in an iterator.

    Returns (activations[N, d_model], mean_activation[d_model]).
    """
    if isinstance(device, str):
        device = torch.device(device)

    activations = []
    hook_name = get_act_name(activation_name, layer)

    def activation_hook(activation, hook):
        activations.append(activation[:, position, :].detach().cpu())

    hooked_model.to(device).eval()
    hooked_model.reset_hooks()
    hooked_model.add_hook(hook_name, activation_hook, "fwd")

    with torch.inference_mode(), amp.autocast(device.type, dtype=torch.float16):
        for batch in tqdm(iterator, desc="Extracting Activations"):
            prompts = [prompt + prompt_seq_append for prompt in batch["prompt"]]
            tokens = hooked_model.to_tokens(prompts).to(device)
            _ = hooked_model(tokens)
            del tokens, _

    hooked_model.reset_hooks()

    acts = torch.cat(activations, dim=0)  # (N, d_model)
    mean_act = acts.mean(dim=0)  # (d_model)
    return acts, mean_act


def get_binary_hooked_activations(
    hooked_model,
    iterator_benign_1,
    iterator_harmful_1,
    iterator_benign_2,
    iterator_harmful_2,
    activation_name: str = "resid_post",
    layer: int = 11,
    position: int = -1,
    device: torch.device | str = DEVICE,
):
    """Compute mean activations for two benign/harmful iterator pairs.

    Returns tuple of four mean activation tensors.
    """
    if isinstance(device, str):
        device = torch.device(device)

    benign_1_acts, benign_1_mean = get_hooked_activations(
        hooked_model=hooked_model,
        iterator=iterator_benign_1,
        activation_name=activation_name,
        layer=layer,
        position=position,
        prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        device=device,
    )

    harmful_1_acts, harmful_1_mean = get_hooked_activations(
        hooked_model=hooked_model,
        iterator=iterator_harmful_1,
        activation_name=activation_name,
        layer=layer,
        position=position,
        prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        device=device,
    )

    benign_2_acts, benign_2_mean = get_hooked_activations(
        hooked_model=hooked_model,
        iterator=iterator_benign_2,
        activation_name=activation_name,
        layer=layer,
        position=position,
        prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        device=device,
    )

    harmful_2_acts, harmful_2_mean = get_hooked_activations(
        hooked_model=hooked_model,
        iterator=iterator_harmful_2,
        activation_name=activation_name,
        layer=layer,
        position=position,
        prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        device=device,
    )

    del benign_1_acts, harmful_1_acts, benign_2_acts, harmful_2_acts

    return benign_1_mean, harmful_1_mean, benign_2_mean, harmful_2_mean
