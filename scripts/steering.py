from typing import Callable
import torch
from torch import amp
from functools import partial
from transformer_lens.utils import get_act_name
from scripts.eval_data import Counter
from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformer


def generate_with_steering(
    prompt: str,
    hooked_model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    steering_vector: torch.Tensor,
    intervention_hook: Callable,
    get_steering_vector: Callable = None,
    fixed_strength: float | None = None,
    benign_strength: float | None = -4.0,
    harmful_strength: float | None = 1.0,
    generate_baseline: bool = False,
    layer: int = 18,
    activation_name: str = "resid_post",
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
    append_seq: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    stop_tokens: list[str] = ["<|eot_id|>"],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> str:
    prompt += append_seq

    stop_ids = [tokenizer.eos_token_id]
    stop_ids.extend([tokenizer.convert_tokens_to_ids(token) for token in stop_tokens])

    hooked_model.reset_hooks()

    strength = fixed_strength

    if steering_vector is None and get_steering_vector is not None:
        if fixed_strength is not None:
            steering_vector, strength = get_steering_vector(
                prompt=prompt,
                hooked_model=hooked_model,
                benign_strength=fixed_strength,
                harmful_strength=fixed_strength,
            )
        elif benign_strength is not None and harmful_strength is not None:
            steering_vector, strength = get_steering_vector(
                prompt=prompt,
                hooked_model=hooked_model,
                benign_strength=benign_strength,
                harmful_strength=harmful_strength,
            )
        else:
            raise Exception(
                "You must pass in values for either fixed_strength or benign_strength and harmful_strength"
            )

    # Build the forward hooks
    fwd_hooks = []

    if steering_vector is not None:
        steering_vector = steering_vector.to(
            hooked_model.cfg.device, dtype=next(hooked_model.parameters()).dtype
        )

        hook_name = get_act_name(activation_name, layer)

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
            baseline = hooked_model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_type="str",
                stop_at_eos=True,
                eos_token_id=stop_ids,
            )

            print(f"Baseline: {baseline}\n")

        # Steered
        with hooked_model.hooks(fwd_hooks):
            steered = hooked_model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                return_type="str",
                stop_at_eos=True,
                eos_token_id=stop_ids,
            )

    hooked_model.reset_hooks()

    return steered


def steering_hook_activations(
    steering_vector: torch.Tensor, strength, counter, device, activation, hook
) -> torch.Tensor:
    # A positive value of strength increases the category-specific refusal behavior
    # A negative value of strength decreases the category-specific refusal behavior

    # activation shape: (batch_size, seq_len, d_model)
    # Steers the activation with the steering vector and steering strength

    if True:
        # if counter.count < 2:
        batch_size, seq_len, d_model = activation.shape
        out = activation.clone()

        vector = steering_vector.to(device)

        if vector.ndim == 1:
            # Expand the steering vector to 2D to apply for the batch
            vector = vector.unsqueeze(dim=0).expand(batch_size, d_model)
        elif vector.ndim == 2:
            assert vector.shape == (
                batch_size,
                d_model,
            ), f"steering_vector must be (d_model,) or (batch_size, d_model), got {vector.shape}"
        else:
            raise ValueError("steering_vector must be 1D or 2D")

        # Add steering at the last token position
        out[:, -1, :] = out[:, -1, :] + strength * vector

        counter.count += 1
        return out

    counter.count += 1
    return activation


def get_categorical_steering_vector_old(
    prompt: str,
    hooked_model: HookedTransformer,
    benign_strength: float,
    harmful_strength: float,
    steering_vector_mapping: dict[int, torch.Tensor],
    append_seq: str = "",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[torch.Tensor, float]:
    # Seperate strength variables for function call consistency
    assert (
        benign_strength == harmful_strength
    ), "benign_strength and harmful_strength must have the same value"

    full_prompt = prompt + append_seq

    tokens = hooked_model.to_tokens(full_prompt)

    with torch.inference_mode(), amp.autocast(device.type, dtype=torch.float16):
        sequences = hooked_model.generate(tokens, max_new_tokens=16, do_sample=False)

    # [128256, 128257, 128258, 128259, 128260]
    refusal_token_ids = list(steering_vector_mapping.keys())

    generated_sequence = [
        token for sequence in sequences.tolist() for token in sequence
    ]

    chosen_steering_vector = None

    for token_id in refusal_token_ids:
        if token_id in generated_sequence:
            print(f"Chosen steering vector token id: {token_id}")

            chosen_steering_vector = steering_vector_mapping[token_id]
            break

    return chosen_steering_vector, benign_strength
