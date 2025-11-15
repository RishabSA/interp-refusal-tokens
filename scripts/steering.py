from typing import Callable
import torch
from torch import amp
from functools import partial
import transformer_lens
from transformer_lens.utils import get_act_name
from scripts.eval_data import Counter


def generate_with_steering(
    prompt,
    hooked_model,
    tokenizer,
    steering_vector,
    intervention_hook: Callable,
    get_steering_vector=None,
    fixed_strenth: float = 0.0,
    benign_strength: float = -4.0,
    harmful_strength: float = 1.0,
    generate_baseline: bool = False,
    layer: int = 16,
    activations: list[str] = ["resid_post"],
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 1.0,
    SEED: int = 42,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    hooked_model.reset_hooks()

    strength = fixed_strenth
    if steering_vector is None and get_steering_vector is not None:
        steering_vector, strength = get_steering_vector(
            prompt, hooked_model, benign_strength, harmful_strength
        )

    # Build the forward hooks
    fwd_hooks = []

    if steering_vector is not None:
        steering_vector = steering_vector.to(
            hooked_model.cfg.device, dtype=next(hooked_model.parameters()).dtype
        )

        for activation in activations:
            hook_name = get_act_name(activation, layer)

            token_limit_gen_counter = Counter()
            hook_fn = partial(
                intervention_hook,
                steering_vector,
                strength,
                token_limit_gen_counter,
                device,
            )
            fwd_hooks.append((hook_name, hook_fn))

    tokens = hooked_model.to_tokens(prompt).to(device)

    with torch.inference_mode(), amp.autocast(device.type, dtype=torch.float16):
        if generate_baseline:
            # Baseline
            torch.manual_seed(SEED)
            baseline = hooked_model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_type="str",
                stop_at_eos=True,
                eos_token_id=stop_ids,
            )

        # Intervened
        with hooked_model.hooks(fwd_hooks):
            torch.manual_seed(SEED)
            intervened = hooked_model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                return_type="str",
                stop_at_eos=True,
                eos_token_id=stop_ids,
            )

    hooked_model.reset_hooks()

    if generate_baseline:
        return baseline, intervened

    return intervened


def steering_hook_activations(
    steering_vector, strength, counter, device, activation, hook
):
    # A positive value of strength increases the category-specific refusal behavior
    # A negative value of strength decreases the category-specific refusal behavior

    # activation shape: (batch_size, seq_len, d_model)
    # Steers the activation with the steering vector and steering strength

    if True:
        # if counter.count < 2:
        batch_size, seq_len, d_model = activation.shape
        out = activation.clone()

        sv = steering_vector
        sv = sv.to(device)

        if sv.ndim == 1:
            sv = sv.view(1, d_model).expand(batch_size, d_model)
        elif sv.ndim == 2:
            assert sv.shape == (
                batch_size,
                d_model,
            ), f"steering_vector must be (d_model,) or (batch_size, d_model), got {sv.shape}"
        else:
            raise ValueError("steering_vector must be 1D or 2D")

        # Add steering at the target token position
        out[:, -1, :] = out[:, -1, :] + strength * sv

        counter.count += 1
        return out

    counter.count += 1
    return activation
