#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from refusal.config import DEVICE
from refusal.models import load_hf_model_and_tokenizer, load_hooked_transformer
from refusal.patching import generate_with_attribution_patching


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attribution patching (A*G) to rank top neurons for a refusal token")
    p.add_argument("--model", required=True, help="HF model id to load")
    p.add_argument("--prompt", required=True, help="Target prompt to analyze")
    p.add_argument("--refusal-token-id", type=int, default=128259)
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--activation-name", default="resid_post")
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model}")
    hf_model, tokenizer = load_hf_model_and_tokenizer(args.model, device=DEVICE)
    hooked = load_hooked_transformer(hf_model, args.model, device_map="auto")

    print("Running attribution patching...")
    top_neurons = generate_with_attribution_patching(
        target_prompt=args.prompt,
        hooked_model=hooked,
        tokenizer=tokenizer,
        layer=args.layer,
        position=args.position,
        activation_name=args.activation_name,
        refusal_token_id=args.refusal_token_id,
        top_k=args.top_k,
    )

    print("Top neurons by gradient (index, score):")
    for idx, score in top_neurons:
        print(f"  neuron {idx:4d}: score = {score:.6f}")


if __name__ == "__main__":
    main()
