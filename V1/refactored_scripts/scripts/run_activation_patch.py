#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from refusal.config import DEVICE
from refusal.models import load_hf_model_and_tokenizer, load_hooked_transformer
from refusal.patching import generate_with_activation_patching


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Activation patching demo (clean -> corrupt)")
    p.add_argument("--model", required=True, help="HF model id to load")
    p.add_argument("--clean", required=True, help="Clean prompt")
    p.add_argument("--corrupt", required=True, help="Corrupt prompt")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--activation-name", default="resid_post")
    p.add_argument("--hidden-ids", default="", help="Comma-separated hidden indices to patch (optional)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--do-sample", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model}")
    hf_model, tokenizer = load_hf_model_and_tokenizer(args.model, device=DEVICE)
    hooked = load_hooked_transformer(hf_model, args.model, device_map="auto")

    hidden_ids = None
    if args.hidden_ids:
        hidden_ids = [int(x.strip()) for x in args.hidden_ids.split(",") if x.strip()]

    print("Running activation patching...")
    baseline, patched = generate_with_activation_patching(
        clean_prompt=args.clean,
        corrupt_prompt=args.corrupt,
        hooked_model=hooked,
        tokenizer=tokenizer,
        hidden_ids=hidden_ids,
        generate_baseline=True,
        layer=args.layer,
        position=args.position,
        activation_name=args.activation_name,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        device=DEVICE,
    )

    print("\n=== Baseline ===\n")
    print(baseline)
    print("\n=== Patched ===\n")
    print(patched)


if __name__ == "__main__":
    main()
