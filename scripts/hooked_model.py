import time
import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
)
from transformer_lens import HookedTransformer


def load_hooked_model(
    model_id: str = "tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens",
) -> tuple[HookedTransformer, PreTrainedTokenizerBase]:
    """Loads a HookedTransformer from transformer_lens given the model_id, which allows for activation caching and intervention in intermediate layers of the language model.

    Args:
        model_id (str, optional). Defaults to "tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens".

    Returns:
        tuple[HookedTransformer, PreTrainedTokenizerBase]: Returns the hooked model and tokenizer.
    """
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    hooked_model = HookedTransformer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        hf_model=model,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=tokenizer,
    )
    hooked_model.eval()

    end_time = time.time()
    print(f"Hooked model download time: {(end_time - start_time):.4f} seconds")

    return hooked_model, tokenizer


def generate_hooked_model_response(
    hooked_model: HookedTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    append_seq: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    stop_tokens: list[str] = ["<|eot_id|>"],
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> str:
    """Given a hooked model and a prompt, generates a response to the prompt with the specified parameters.

    Args:
        hooked_model (HookedTransformer)
        tokenizer (AutoTokenizer)
        prompt (str)
        append_seq (str, optional). Defaults to "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".
        stop_tokens (list[str], optional). Defaults to ["<|eot_id|>"].
        max_new_tokens (int, optional). Defaults to 512.
        do_sample (bool, optional). Defaults to False.
        temperature (float, optional). Defaults to 1.0.
        device (torch.device, optional). Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        str: The hooked model's outputted response to the given prompt.
    """
    stop_ids = [tokenizer.eos_token_id]
    stop_ids.extend([tokenizer.convert_tokens_to_ids(token) for token in stop_tokens])

    full_prompt = prompt + append_seq

    tokens = hooked_model.to_tokens(full_prompt).to(device)

    sequence = hooked_model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        return_type="str",
        stop_at_eos=True,
        eos_token_id=stop_ids,
    )

    return sequence
