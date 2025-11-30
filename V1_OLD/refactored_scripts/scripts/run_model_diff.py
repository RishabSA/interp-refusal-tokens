#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure project root import
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from refusal.config import DEVICE, FIGURES_DIR, SAVED_TENSORS_DIR
from refusal.models import load_hf_model_and_tokenizer, load_hooked_transformer
from refusal.data import load_synthetic_refusal_json
from refusal.model_diff import compute_model_steering_vectors_from_dataloaders
from refusal.steering import compute_steering_vector_cosine_similarities
from refusal.plots import plot_steering_vector_cosine_sims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute model steering vectors from category dataloaders and plot sims")
    p.add_argument("--model", required=True, help="HF model id to load")
    p.add_argument("--synthetic-json", default="", help="Path to synthetic JSON (defaults to synthethic_data/refusal_dataset.json)")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--activation-name", default="resid_post")
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--out-prefix", default="modeldiff")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading HF model + tokenizer: {args.model}")
    hf_model, tokenizer = load_hf_model_and_tokenizer(args.model, device=DEVICE)

    print("Wrapping with HookedTransformer...")
    hooked = load_hooked_transformer(hf_model, args.model, device_map="auto")

    print("Loading synthetic category dataloaders...")
    loaders = load_synthetic_refusal_json(args.synthetic_json or None, batch_size=args.batch_size)

    harmful_by_cat = loaders["harmful"]
    benign_by_cat = loaders["benign"]

    print("Computing steering vectors from dataloaders...")
    steering = compute_model_steering_vectors_from_dataloaders(
        hooked_model=hooked,
        harmful_by_cat=harmful_by_cat,
        benign_by_cat=benign_by_cat,
        activation_name=args.activation_name,
        layer=args.layer,
        position=args.position,
        prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        device=DEVICE,
    )

    # Save steering vectors
    torch.save(steering, SAVED_TENSORS_DIR / f"{args.out_prefix}_steering_vectors.pt")

    print("Plotting cosine similarities...")
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
