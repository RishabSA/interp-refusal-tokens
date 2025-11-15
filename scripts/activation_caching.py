from tqdm.notebook import tqdm
import torch

import transformer_lens
from transformer_lens.utils import get_act_name


def cache_hooked_activations_before_pad(
    hooked_model,
    iterator,
    activation_name: str = "resid_post",
    layer: int = 16,
    prompt_seq_append: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
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
    if pad_id is None:
        pad_id = 0  # transformer_lens often uses 0 when no pad_token is set

    with torch.inference_mode():
        for batch in tqdm(iterator, desc="Extracting Activations"):
            prompts = [prompt + prompt_seq_append for prompt in batch["prompt"]]

            tokens = hooked_model.to_tokens(prompts).to(device)

            # compute per-example last non-padding position
            # (tokens != pad_id) is True for real tokens, False for pads
            lengths = (tokens != pad_id).sum(dim=-1)  # shape: (batch_size)
            current_positions = lengths - 1  # shape: (batch_size)

            # print(prompts)
            # print(tokens)
            # print("positions:", current_positions)

            # batch_size = tokens.size(0)
            # last_token_ids = tokens[
            #     torch.arange(batch_size, device=device), current_positions
            # ]

            # print("last token ids:", last_token_ids.tolist())

            logits = hooked_model(tokens)

            del tokens, logits

    hooked_model.reset_hooks()

    activations = torch.cat(activations, dim=0)  # shape: (N, d_model)
    mean_activation = activations.mean(dim=0)  # shape: (d_model)

    print(f"Extracted {activations.shape[0]} activations")
    print(f"Mean Activations shape: {mean_activation.shape}")

    return activations, mean_activation
