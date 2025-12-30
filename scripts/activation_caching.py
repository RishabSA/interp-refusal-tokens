from tqdm.auto import tqdm
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from torch.utils.data import DataLoader


def cache_hooked_activations_before_pad(
    hooked_model: HookedTransformer,
    iterator: DataLoader,
    activation_name: str = "resid_post",
    layer: int = 18,
    prompt_seq_append: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Caches the activations of shape [4096] before the first pad token (caching the last non-pad token) at a given layer and point in the layer using transformer_lens.

    Caches the activations for each of the prompts given in the iterator.

    Args:
        hooked_model (HookedTransformer)
        iterator (DataLoader)
        activation_name (str, optional): Position in the layer to hook at. Defaults to "resid_post".
        layer (int, optional): Layer to hook at. Defaults to 18.
        prompt_seq_append (str, optional): Sequence to append to all prompts before caching the activation of the last non-pad token. Defaults to "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".
        device (torch.device, optional). Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        tuple[torch.Tensor, torch.Tensor]: [N, 4096] tensor for all of the cached activations and [4096] tensor for the mean of the cached activations
    """
    activations = []

    hook_name = get_act_name(activation_name, layer)

    # Last non-pad position for the current batch
    current_positions = None

    def activation_hook(activation, hook):
        # activation shape: (batch_size, seq_len, d_model)
        # current_positions shape: (batch_size) indices of the last non-pad token
        nonlocal current_positions

        idx = current_positions.to(activation.device)  # shape: (batch_size)
        batch_size = activation.size(0)

        activations.append(
            activation[torch.arange(batch_size, device=activation.device), idx, :]
            .detach()
            .cpu()  # (batch_size, d_model)
        )

    hooked_model.reset_hooks()
    hooked_model.add_hook(hook_name, activation_hook, "fwd")

    pad_id = hooked_model.tokenizer.pad_token_id

    with torch.inference_mode():
        for batch in tqdm(iterator, desc="Extracting Activations"):
            prompts = [prompt + prompt_seq_append for prompt in batch["prompt"]]

            tokens = hooked_model.to_tokens(prompts).to(device)

            # Get the per-sample last non-padding position
            # (tokens != pad_id) is True for real tokens, False for pads
            lengths = (tokens != pad_id).sum(dim=-1)  # shape: (batch_size)
            current_positions = lengths - 1  # shape: (batch_size)

            logits = hooked_model(tokens)

            del tokens, logits

    hooked_model.reset_hooks()

    activations = torch.cat(activations, dim=0)  # shape: (N, d_model)
    mean_activation = activations.mean(dim=0)  # shape: (d_model)

    print(f"Extracted {activations.shape[0]} activations")
    print(f"Mean Activations shape: {mean_activation.shape}")

    return activations, mean_activation
