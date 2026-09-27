import argparse
import json
import os
import shutil
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

from scripts.steering_vector_data import get_steering_vector_data


def convert_shards_to_fp16(model_id: str, n_layers: int, weights_dir: str) -> dict[str, str]:
    index_file = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    needed_prefixes = ("model.embed_tokens.",) + tuple(f"model.layers.{i}." for i in range(n_layers))
    needed_keys = [key for key in weight_map if key.startswith(needed_prefixes)]

    download_dir = os.path.join(weights_dir, "download")
    os.makedirs(download_dir, exist_ok=True)

    tensor_files = {}
    for shard in tqdm(sorted({weight_map[key] for key in needed_keys}), desc="Converting shards"):
        shard_keys = [key for key in needed_keys if weight_map[key] == shard]
        fp16_file = os.path.join(weights_dir, shard)

        # Already-converted shards are reused so an interrupted run does not download them again
        if not os.path.exists(fp16_file):
            shard_file = hf_hub_download(model_id, shard, local_dir=download_dir)

            with safe_open(shard_file, framework="pt") as f:
                tensors = {key: f.get_tensor(key).to(dtype=torch.float16) for key in shard_keys}

            save_file(tensors, fp16_file)
            del tensors

            # Delete each original shard once converted so peak disk usage stays at a single shard
            os.remove(shard_file)

        tensor_files.update({key: fp16_file for key in shard_keys})

    return tensor_files


def load_tensors(tensor_files: dict[str, str], prefix: str, device: torch.device) -> dict[str, torch.Tensor]:
    keys_by_file = {}
    for key, file in tensor_files.items():
        if key.startswith(prefix):
            keys_by_file.setdefault(file, []).append(key)

    tensors = {}
    for file, keys in keys_by_file.items():
        with safe_open(file, framework="pt") as f:
            tensors.update({key.removeprefix(prefix): f.get_tensor(key).to(device) for key in keys})

    return tensors


def load_decoder_layer(
    config: PretrainedConfig, layer: int, tensor_files: dict[str, str], device: torch.device
) -> LlamaDecoderLayer:
    with torch.device("meta"):
        decoder_layer = LlamaDecoderLayer(config, layer_idx=layer)

    decoder_layer.load_state_dict(
        load_tensors(tensor_files, f"model.layers.{layer}.", device), assign=True
    )
    return decoder_layer.eval()


def make_token_budget_batches(lengths: list[int], order: list[int], token_budget: int) -> list[list[int]]:
    # order is ascending by length, so the newest prompt sets the padded length of the batch
    batches = [[]]
    for index in order:
        if batches[-1] and (len(batches[-1]) + 1) * lengths[index] > token_budget:
            batches.append([])
        batches[-1].append(index)

    return batches


def cache_last_token_activations(
    config: PretrainedConfig,
    tokenizer: PreTrainedTokenizerBase,
    tensor_files: dict[str, str],
    prompts: list[str],
    layer: int,
    batch_tokens: int,
    chunk_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Right padding under a causal mask leaves every non-pad position identical to the unpadded forward pass
    tokenizer.padding_side = "right"

    # Sorting by length keeps padding low, and token budgets bound attention and hidden-state memory for the few very long prompts
    lengths = [len(input_ids) for input_ids in tokenizer(prompts)["input_ids"]]
    order = sorted(range(len(prompts)), key=lengths.__getitem__)
    batches = make_token_budget_batches(lengths, order, batch_tokens)

    chunks = [[]]
    chunk_token_count = 0
    for batch in batches:
        batch_token_count = len(batch) * lengths[batch[-1]]
        if chunks[-1] and chunk_token_count + batch_token_count > chunk_tokens:
            chunks.append([])
            chunk_token_count = 0
        chunks[-1].append(batch)
        chunk_token_count += batch_token_count

    embed_weight = load_tensors(tensor_files, "model.embed_tokens.", torch.device("cpu"))["weight"]  # shape: (vocab_size, d_model)
    rotary_emb = LlamaRotaryEmbedding(config=config, device=device)

    activations = torch.empty(len(prompts), config.hidden_size)  # shape: (n_prompts, d_model)
    with torch.inference_mode():
        # Layer-major within each chunk: only one decoder layer is resident at a time, which keeps memory far below the full model
        for chunk in tqdm(chunks, desc="Extracting activations"):
            inputs = [
                tokenizer([prompts[index] for index in batch], return_tensors="pt", padding=True).to(device)
                for batch in chunk
            ]

            hidden_states = [
                F.embedding(batch_inputs["input_ids"].cpu(), embed_weight).to(device) for batch_inputs in inputs
            ]  # each shape: (batch_size, seq_len, d_model)
            position_ids = [
                torch.arange(hidden.size(1), device=device).unsqueeze(dim=0) for hidden in hidden_states
            ]  # each shape: (1, seq_len)
            position_embeddings = [
                rotary_emb(hidden, position_ids=positions)
                for hidden, positions in zip(hidden_states, position_ids, strict=True)
            ]
            causal_masks = [
                create_causal_mask(
                    config=config,
                    inputs_embeds=hidden,
                    attention_mask=batch_inputs["attention_mask"],
                    past_key_values=None,
                    position_ids=positions,
                )
                for hidden, batch_inputs, positions in zip(hidden_states, inputs, position_ids, strict=True)
            ]

            for layer_idx in tqdm(range(layer + 1), desc="Layers", leave=False):
                decoder_layer = load_decoder_layer(config, layer_idx, tensor_files, device)

                # Replace each batch in place so only one copy of the chunk's hidden states is alive
                for i in range(len(hidden_states)):
                    hidden_states[i] = decoder_layer(
                        hidden_states[i],
                        attention_mask=causal_masks[i],
                        position_ids=position_ids[i],
                        position_embeddings=position_embeddings[i],
                    )

                del decoder_layer

                # Variable batch shapes fragment the caching allocator, which on unified memory starves the rest of the system
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            for hidden, batch_inputs, batch in zip(hidden_states, inputs, chunk, strict=True):
                last_positions = batch_inputs["attention_mask"].sum(dim=-1) - 1  # shape: (batch_size)
                batch_indices = torch.arange(hidden.size(0), device=device)
                activations[batch] = hidden[batch_indices, last_positions].float().cpu()  # shape: (batch_size, d_model)

            del hidden_states, causal_masks, position_embeddings

    return activations


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Cache last-token resid_post activations on CoCoNot harmful prompts by streaming one decoder layer at a time, so an 8B model fits on a laptop."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens",
        help="HuggingFace model to cache activations from (default: tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=18,
        help="Layer whose resid_post activations are cached (default: 18).",
    )
    parser.add_argument(
        "--batch_tokens",
        type=int,
        default=4096,
        help="Max padded tokens per forward pass, bounding attention memory (default: 4096).",
    )
    parser.add_argument(
        "--chunk_tokens",
        type=int,
        default=32768,
        help="Max padded tokens whose hidden states are held in memory while streaming the layers (default: 32768).",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="downloads/fp16_weights",
        help="Where float16 copies of the needed weights are written, one subdirectory per model, deleted after a successful run (default: downloads/fp16_weights).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="saved_outputs/model_diffing/activations_18_resid_post_refuse-llama.pt",
        help="Where to save the activations and category labels (default: saved_outputs/model_diffing/activations_18_resid_post_refuse-llama.pt).",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_weights_dir = os.path.join(args.weights_dir, args.model_id.replace("/", "--"))
    os.makedirs(model_weights_dir, exist_ok=True)

    # float16 matches the dtype load_hooked_model used for the paper's activations
    tensor_files = convert_shards_to_fp16(args.model_id, n_layers=args.layer + 1, weights_dir=model_weights_dir)

    config = AutoConfig.from_pretrained(args.model_id, dtype=torch.float16)
    config._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    harmful_dataloaders, _ = get_steering_vector_data()

    # The dataloaders use a nested collate_fn with worker processes, which cannot be pickled under macOS spawn
    prompts = []
    categories = []
    for category, dataloader in harmful_dataloaders.items():
        prompts.extend(item["prompt"] for item in dataloader.dataset)
        categories.extend([category] * len(dataloader.dataset))

    activations = cache_last_token_activations(
        config,
        tokenizer,
        tensor_files,
        prompts,
        layer=args.layer,
        batch_tokens=args.batch_tokens,
        chunk_tokens=args.chunk_tokens,
        device=device,
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    torch.save(
        {
            "model_id": args.model_id,
            "layer": args.layer,
            "activations": activations,  # shape: (n_prompts, d_model)
            "categories": categories,
        },
        args.output_file,
    )
    print(f"Saved {len(categories)} activations to {args.output_file}")

    shutil.rmtree(model_weights_dir)
