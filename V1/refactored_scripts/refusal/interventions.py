from __future__ import annotations

from functools import partial
from typing import Callable, List, Iterable, Dict, Any

import torch
from torch import amp
from tqdm.auto import tqdm

from transformer_lens.utils import get_act_name  # requires transformer_lens

from .config import DEVICE, DEFAULT_SEED


def make_steering_hook_activations(position: int = -1) -> Callable:
    """Factory for an activation-level steering hook.

    Returns a hook function with signature (steering_vector, strength, activation, hook) -> activation.
    It adds strength * steering_vector at the specified token position for each batch element.
    """

    def steering_hook_activations(steering_vector, strength, activation, hook):
        # activation shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = activation.shape
        out = activation.clone()

        sv = steering_vector
        sv = sv.to(activation.device, dtype=activation.dtype)

        if sv.ndim == 1:
            sv = sv.view(1, d_model).expand(batch_size, d_model)
        elif sv.ndim == 2:
            assert sv.shape == (
                batch_size,
                d_model,
            ), f"steering_vector must be (d_model,) or (batch_size, d_model), got {sv.shape}"
        else:
            raise ValueError("steering_vector must be 1D or 2D")

        out[:, position, :] = out[:, position, :] + float(strength) * sv
        return out

    return steering_hook_activations


def make_steering_hook_sparse_vector(position: int = -1) -> Callable:
    """Factory for a sparse steering hook; same as activations hook but intended for sparse vectors."""

    def steering_hook_sparse_vector(steering_vector, strength, activation, hook):
        batch_size, seq_len, d_model = activation.shape
        out = activation.clone()

        sv = steering_vector.to(activation.device, dtype=activation.dtype)
        if sv.ndim == 1:
            sv = sv.view(1, d_model).expand(batch_size, d_model)
        elif sv.ndim == 2:
            assert sv.shape == (
                batch_size,
                d_model,
            ), f"steering_vector must be (d_model,) or (batch_size, d_model), got {sv.shape}"
        else:
            raise ValueError("steering_vector must be 1D or 2D")

        out[:, position, :] = out[:, position, :] + float(strength) * sv
        return out

    return steering_hook_sparse_vector


def generate_with_intervention(
    prompt: str,
    hooked_model,
    tokenizer,
    steering_vector: torch.Tensor | None,
    intervention_hook: Callable,  # hook signature: (steer_vec, strength, activation, hook)
    get_steering_vector: Callable | None = None,
    strength: float = -1.0,
    generate_baseline: bool = False,
    layer: int = 9,
    activations: List[str] | None = None,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 1.0,
    device: str | torch.device = DEVICE,
) -> str | tuple[str, str]:
    """Generate text with an activation-level intervention via forward hooks.

    If generate_baseline=True, returns a tuple (baseline, intervened).
    Otherwise returns just the intervened string.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if activations is None:
        activations = ["resid_post"]

    prompt = prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    hooked_model.reset_hooks()

    if steering_vector is None and get_steering_vector is not None:
        steering_vector = get_steering_vector(prompt, hooked_model, tokenizer)

    fwd_hooks = []
    if steering_vector is not None:
        steering_vector = steering_vector.to(
            hooked_model.cfg.device, dtype=next(hooked_model.parameters()).dtype
        )
        for activation in activations:
            hook_name = get_act_name(activation, layer)
            hook_fn = partial(intervention_hook, steering_vector, strength)
            fwd_hooks.append((hook_name, hook_fn))

    tokens = hooked_model.to_tokens(prompt).to(device)

    with torch.inference_mode(), amp.autocast(device.type, dtype=torch.float16):
        if generate_baseline:
            torch.manual_seed(DEFAULT_SEED)
            baseline = hooked_model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_type="str",
                stop_at_eos=True,
                eos_token_id=stop_ids,
            )

        with hooked_model.hooks(fwd_hooks):
            torch.manual_seed(DEFAULT_SEED)
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


def generate_outputs_dataset_intervened(
    model,
    tokenizer,
    iterator: Iterable[Dict[str, List[str]]],
    model_name: str,
    steering_vector: torch.Tensor | None = None,
    strength: float = -1.0,
    get_steering_vector: Callable | None = None,
    intervention_hook: Callable | None = None,
    layer: int | None = None,
    activations: List[str] | None = None,
    description: str = "Evaluation",
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 1.0,
    outputs_save_path: str | None = None,
    device: str | torch.device = DEVICE,
) -> List[Dict[str, Any]]:
    """Generate responses over an iterator with optional activation-level interventions.

    - If `intervention_hook` is provided and `model` is a HookedTransformer, we attach forward hooks at the
      specified `layer` and `activations` to add the steering vector (per-sample if batch vectors are given).
    - Otherwise, falls back to standard HF generation.
    - Returns a list of {id, model, category, prompt, response}. Optionally writes JSONL to outputs_save_path.
    """
    try:
        from transformer_lens import HookedTransformer  # type: ignore
    except Exception:
        HookedTransformer = None  # type: ignore

    if isinstance(device, str):
        device = torch.device(device)

    # Ensure tokenizer padding and pad_token_id
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, "config"):
            try:
                model.config.pad_token_id = tokenizer.pad_token_id
            except Exception:
                pass
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    # Stop tokens (best-effort)
    stop_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None:
            stop_ids.append(eot_id)
    except Exception:
        pass

    is_hooked = HookedTransformer is not None and isinstance(model, HookedTransformer)

    results: List[Dict[str, Any]] = []
    with torch.inference_mode(), amp.autocast(device.type, dtype=torch.float16):
        for batch in tqdm(iterator, desc=description):
            prompts: List[str] = batch["prompt"]
            categories: List[str | None] = batch.get("category") or [None] * len(prompts)

            # Append assistant header as in notebook
            prompts_proc = [
                p + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for p in prompts
            ]

            if intervention_hook is not None:
                assert (
                    layer is not None and activations is not None
                ), "When using intervention_hook, pass layer and activations."

            if intervention_hook is not None and is_hooked:
                tokens = model.to_tokens(prompts_proc).to(device)

                steer_batch = steering_vector
                if get_steering_vector is not None:
                    batch_vecs: List[torch.Tensor | None] = []
                    for p in prompts_proc:
                        vec = get_steering_vector(p, model, tokenizer)
                        batch_vecs.append(vec if vec is not None else None)

                    # Convert to [B, D] tensor, zeros where None
                    if any(v is not None for v in batch_vecs):
                        D = next(v for v in batch_vecs if v is not None).numel()
                    else:
                        D = getattr(model.cfg, "d_model", None) or 0
                    if D == 0:
                        raise ValueError("Cannot infer d_model for steering vector batch")
                    stack = [
                        (v.detach().to(device) if v is not None else torch.zeros(D, device=device))
                        for v in batch_vecs
                    ]
                    steer_batch = torch.stack(stack, dim=0)

                # Build forward hooks for each requested activation name
                fwd_hooks = []
                for act in activations or []:
                    hook_name = get_act_name(act, layer)
                    hook_fn = partial(intervention_hook, steer_batch, strength)
                    fwd_hooks.append((hook_name, hook_fn))

                with model.hooks(fwd_hooks):
                    torch.manual_seed(DEFAULT_SEED)
                    sequences = model.generate(
                        tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        return_type="str",
                        stop_at_eos=True,
                        eos_token_id=stop_ids,
                    )
                model.reset_hooks()
            elif is_hooked:
                tokens = model.to_tokens(prompts_proc).to(device)
                torch.manual_seed(DEFAULT_SEED)
                sequences = model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    return_type="str",
                    stop_at_eos=True,
                    eos_token_id=stop_ids,
                )
            else:
                inputs = tokenizer(prompts_proc, padding=True, truncation=True, return_tensors="pt").to(device)
                torch.manual_seed(DEFAULT_SEED)
                out = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    output_scores=False,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=stop_ids,
                )
                sequences = tokenizer.batch_decode(
                    out.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            # Collect results
            if isinstance(sequences, list):
                seq_texts = sequences
            else:
                # HookedTransformer returns list[str] when return_type="str"
                seq_texts = sequences

            if len(prompts) == 1 and not isinstance(seq_texts, list):
                # In case a single string was returned (defensive)
                seq_texts = [seq_texts]

            for i in range(len(prompts)):
                resp = seq_texts[i] if isinstance(seq_texts, list) else seq_texts
                results.append(
                    {
                        "id": "",  # filled by caller if needed
                        "model": model_name,
                        "category": categories[i] if i < len(categories) else None,
                        "prompt": prompts[i],
                        "response": resp,
                    }
                )

    if outputs_save_path:
        from pathlib import Path
        import json as _json
        outp = Path(outputs_save_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for rec in results:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")

    return results
def get_categorical_steering_vector_fine_tuned(
    steering_vector_mapping: dict[int, torch.Tensor],
    prompt: str,
    hooked_model,
    tokenizer,
    device: str | torch.device = DEVICE,
):
    """Infer which categorical refusal token appears and return its steering vector.

    Returns None if no refusal token appears in the short rollout.
    """
    if isinstance(device, str):
        device = torch.device(device)

    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    tokens = hooked_model.to_tokens(prompt)

    with torch.inference_mode(), amp.autocast(device.type, dtype=torch.float16):
        torch.manual_seed(DEFAULT_SEED)
        sequences = hooked_model.generate(
            tokens,
            max_new_tokens=16,
            do_sample=False,
            return_type="tokens",
            stop_at_eos=True,
            eos_token_id=stop_ids,
        )

    refusal_token_ids = [128256, 128257, 128258, 128259, 128260]

    generated_sequence = [token for sequence in sequences.tolist() for token in sequence]
    for token_id in refusal_token_ids:
        if token_id in generated_sequence:
            return steering_vector_mapping.get(token_id)
    return None
