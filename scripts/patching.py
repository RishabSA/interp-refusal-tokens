import torch
from torch import amp
from transformer_lens.utils import get_act_name
from transformer_lens import (
    HookedTransformer,
)
from transformers import (
    PreTrainedTokenizerBase,
)


def generate_with_activation_patching(
    clean_prompt: str,
    corrupt_prompt: str,
    hooked_model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    hidden_ids: list[int] | None = None,
    generate_baseline: bool = False,
    layer: int = 16,
    activation_name: str = "resid_post",
    max_new_tokens: int = 200,
    do_sample: bool = False,
    SEED: int = 42,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> str:
    hooked_model.reset_hooks()

    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    # Tokenization
    clean_tokens = hooked_model.to_tokens(clean_prompt).to(device)
    corrupt_tokens = hooked_model.to_tokens(corrupt_prompt).to(device)

    # Get the clean model cache
    clean_logits, cache_clean = hooked_model.run_with_cache(
        clean_tokens, remove_batch_dim=False
    )

    # Build the patching hook
    hook_name = get_act_name(activation_name, layer)

    def patch_hook(activation, hook):
        patched = activation.clone()
        residual = cache_clean[hook_name]

        if hidden_ids is None:
            patched[:, -1, :] = residual[:, -1, :]
        else:
            patched[:, -1, hidden_ids] = residual[:, -1, hidden_ids]

        return patched

    fwd_hooks = [(hook_name, patch_hook)]

    with torch.inference_mode(), amp.autocast(device.type, dtype=torch.float16):
        if generate_baseline:
            torch.manual_seed(SEED)
            baseline = hooked_model.generate(
                corrupt_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_type="str",
                stop_at_eos=True,
                eos_token_id=stop_ids,
            )

            print(f"Baseline: {baseline}\n")

        # Re-generate with the hook
        with hooked_model.hooks(fwd_hooks):
            torch.manual_seed(SEED)
            patched = hooked_model.generate(
                corrupt_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                return_type="str",
                stop_at_eos=True,
                eos_token_id=stop_ids,
            )

    hooked_model.reset_hooks()

    return patched


def generate_with_attribution_patching(
    target_prompt: str,
    hooked_model: HookedTransformer,
    layer: int = 16,
    activation_name: str = "resid_post",
    refusal_token_id: int = 128259,
    top_k: int = 50,
):
    """
    Runs one forward+backward on `target_prompt`, captures the activation A
    and gradient G = ∂L/∂A at (layer, act_name, position), and computes  saliency_i = A_i * G_i
    for each neuron i. Returns a sorted list of (neuron_idx, saliency).
    """
    hooked_model.eval()

    # Tokenization
    tokens = hooked_model.to_tokens(target_prompt).to(hooked_model.cfg.device)

    saved = {}

    # Build the forward-pass hook
    def save_activation(activation, hook):
        # activation shape: (batch_size, seq_len, d_model)
        saved["activation"] = activation.clone().detach().requires_grad_(True)

    hook_name = get_act_name(activation_name, layer)
    hooked_model.add_hook(hook_name, save_activation, "fwd")

    logits = hooked_model(tokens)  # logits shape: (batch_size, seq_len, vocab_size)
    hooked_model.reset_hooks()

    # Get the refusal-token logit at the last token
    logit = logits[0, -1, refusal_token_id]

    hooked_model.cfg.use_attn_result = False
    logit.backward()

    # Get the activation and gradient
    A = saved["activation"][0, -1, :]  # shape: (d_model)
    G = saved["activation"].grad[0, -1, :]  # shape: (d_model)

    saliency = (A * G).abs()

    # Get the top-k
    top_vals, top_idx = torch.topk(saliency, top_k)

    # Return a sorted_list
    return [(int(i), float(saliency[i])) for i in top_idx.tolist()]
