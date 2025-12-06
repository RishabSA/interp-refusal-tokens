import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerBase


def load_model(
    model_id: str = "tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens",
) -> tuple[LlamaForCausalLM, PreTrainedTokenizerBase]:
    """Loads a Llama language model from HuggingFace given the model_id.

    Args:
        model_id (str, optional). Defaults to "tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens".

    Returns:
        tuple[LlamaForCausalLM, PreTrainedTokenizerBase]: Returns the model and tokenizer.
    """
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    # model.to(device)

    end_time = time.time()
    print(f"Model download time: {(end_time - start_time):.4f} seconds")

    return model, tokenizer


def generate_model_response(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    append_seq: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    stop_tokens: list[str] = ["<|eot_id|>"],
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> str:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, "config"):
            model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"

    stop_ids = [tokenizer.eos_token_id]
    stop_ids.extend([tokenizer.convert_tokens_to_ids(token) for token in stop_tokens])

    full_prompt = prompt + append_seq

    inputs = tokenizer(
        full_prompt, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    out = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=stop_ids,
    )

    sequences = tokenizer.batch_decode(
        out,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return sequences[0]
