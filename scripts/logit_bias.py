import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    AutoModelForCausalLM,
    AutoTokenizer,
)


class FirstTokenLogitBias(LogitsProcessor):
    def __init__(self, token_id: int, bias: float):
        self.token_id = int(token_id)
        self.bias = float(bias)
        self.step = 0

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.FloatTensor:
        # scores shape: (batch_size, vocab_size) for the next token

        # Only apply logit bias for the first token
        if self.step == 0:
            scores = scores.clone()
            scores[:, self.token_id] += self.bias
        self.step += 1

        return scores


def generate_with_first_token_logit_bias(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    bias_token_id: int,
    bias: float = 5.0,
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

    prompt += append_seq

    stop_ids = [tokenizer.eos_token_id]
    stop_ids.extend([tokenizer.convert_tokens_to_ids(token) for token in stop_tokens])

    model.to(device).eval()
    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(
        device
    )

    processors = LogitsProcessorList(
        [FirstTokenLogitBias(token_id=bias_token_id, bias=bias)]
    )

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            logits_processor=processors,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )

    sequences = tokenizer.batch_decode(
        out,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return sequences[0]


# Example usage:
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16)
# tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# print(generate_with_first_token_bias_hf(model, tok, "Hello", bias_token_id=128256, bias=8.0))
