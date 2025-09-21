#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple

# Ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from refusal.config import DEVICE, FIGURES_DIR, SAVED_TENSORS_DIR
from refusal.models import load_hf_model_and_tokenizer, load_hooked_transformer
from refusal.data import make_dataloader, split_dataloader_by_category
from refusal.activations import get_hooked_activations
from refusal.steering import (
    compute_steering_vectors,
    compute_caa_steering_vectors,
    compute_steering_vector_cosine_similarities,
)
from refusal.plots import plot_steering_vector_cosine_sims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute steering vectors/plots from harmful and benign prompt JSONL files.")
    p.add_argument("--model", required=True, help="HF model id to load")
    p.add_argument("--harmful-jsonl", required=True, help="JSONL with {prompt, category}")
    p.add_argument("--benign-jsonl", required=True, help="JSONL with {prompt, category}")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--activation-name", default="resid_post")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--out-prefix", default="steering")
    return p.parse_args()


def load_jsonl_items(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise ValueError("Each line must contain a 'prompt' field")
            if "category" not in obj:
                obj["category"] = None
            items.append({"prompt": obj["prompt"], "category": obj["category"]})
    return items


def main() -> None:
    args = parse_args()

    print(f"Loading HF model + tokenizer: {args.model}")
    hf_model, tokenizer = load_hf_model_and_tokenizer(args.model, device=DEVICE)

    print("Wrapping with TransformerLens HookedTransformer...")
    hooked_model = load_hooked_transformer(hf_model, args.model, device_map="auto")

    print("Reading JSONL prompt files...")
    harmful_items = load_jsonl_items(args.harmful_jsonl)
    benign_items = load_jsonl_items(args.benign_jsonl)

    harmful_dl = make_dataloader(harmful_items, batch_size=args.batch_size, shuffle=False)
    benign_dl = make_dataloader(benign_items, batch_size=args.batch_size, shuffle=False)

    harmful_by_cat = split_dataloader_by_category(harmful_dl, category_field="category")
    benign_by_cat = split_dataloader_by_category(benign_dl, category_field="category")

    print("Computing activations per category...")
    harmful_means: Dict[str, torch.Tensor] = {}
    benign_means: Dict[str, torch.Tensor] = {}

    for cat, harm_iter in harmful_by_cat.items():
        if cat not in benign_by_cat:
            print(f"Skipping category '{cat}' not present in benign set")
            continue
        acts, mean_act = get_hooked_activations(
            hooked_model=hooked_model,
            iterator=harm_iter,
            activation_name=args.activation_name,
            layer=args.layer,
            position=args.position,
            prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            device=DEVICE,
        )
        harmful_means[cat] = mean_act
        # Optionally save acts/means
        torch.save(acts, SAVED_TENSORS_DIR / f"{args.out_prefix}_harmful_acts_{cat}.pt")
        torch.save(mean_act, SAVED_TENSORS_DIR / f"{args.out_prefix}_harmful_mean_{cat}.pt")
        del acts

    for cat, ben_iter in benign_by_cat.items():
        if cat not in harmful_by_cat:
            print(f"Skipping category '{cat}' not present in harmful set")
            continue
        acts, mean_act = get_hooked_activations(
            hooked_model=hooked_model,
            iterator=ben_iter,
            activation_name=args.activation_name,
            layer=args.layer,
            position=args.position,
            prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            device=DEVICE,
        )
        benign_means[cat] = mean_act
        torch.save(acts, SAVED_TENSORS_DIR / f"{args.out_prefix}_benign_acts_{cat}.pt")
        torch.save(mean_act, SAVED_TENSORS_DIR / f"{args.out_prefix}_benign_mean_{cat}.pt")
        del acts

    print("Computing steering vectors...")
    steering = compute_steering_vectors(
        mean_benign_dict=benign_means,
        mean_harmful_dict=harmful_means,
        should_filter_shared=False,
        K=100,
        tau=1e-3,
    )

    print("Computing cosine similarity matrix and plotting...")
    sims = compute_steering_vector_cosine_similarities(steering)
    plot_steering_vector_cosine_sims(
        sims,
        layer=args.layer,
        activation_name=args.activation_name,
        title=f"Steering Vector Cosine Similarities ({args.activation_name} L{args.layer})",
        out_dir=FIGURES_DIR,
    )

    print("Done.")


if __name__ == "__main__":
    main()
