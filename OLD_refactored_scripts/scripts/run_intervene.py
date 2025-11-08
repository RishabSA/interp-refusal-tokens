#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any, List

# Ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from refusal.config import DEVICE, SAVED_OUTPUTS_DIR
from refusal.models import load_hf_model_and_tokenizer, load_hooked_transformer
from refusal.data import make_dataloader
from refusal.interventions import (
    make_steering_hook_activations,
    generate_outputs_dataset_intervened,
    get_categorical_steering_vector_fine_tuned,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate with activation-level interventions (dense or categorical)")
    p.add_argument("--model", required=True, help="HF model id to load")
    p.add_argument("--prompts-jsonl", required=True, help="JSONL with {prompt, category?}")
    p.add_argument("--mode", choices=["dense", "categorical"], required=True)
    p.add_argument("--steering-vector-pt", default="", help="Path to a .pt tensor for dense mode")
    p.add_argument("--vector-map-pt", default="", help="Path to a .pt dict[token_id->tensor] for categorical mode")
    p.add_argument("--strength", type=float, default=-4.0)
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--activation-name", default="resid_post")
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", default=str(SAVED_OUTPUTS_DIR / "intervened_outputs.jsonl"))
    return p.parse_args()


def load_prompts(prompts_jsonl: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with Path(prompts_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise ValueError("Each line must contain a 'prompt' field")
            items.append({"prompt": obj["prompt"], "category": obj.get("category")})
    return items


def main() -> None:
    args = parse_args()

    print(f"Loading HF model and tokenizer: {args.model}")
    hf_model, tokenizer = load_hf_model_and_tokenizer(args.model, device=DEVICE)

    print("Wrapping with TransformerLens HookedTransformer...")
    hooked = load_hooked_transformer(hf_model, args.model, device_map="auto")

    items = load_prompts(args.prompts_jsonl)
    dl = make_dataloader(items, batch_size=args.batch_size, shuffle=False)

    steering_vector = None
    get_sv = None

    if args.mode == "dense":
        if not args.steering_vector_pt:
            raise ValueError("--steering-vector-pt is required for dense mode")
        print(f"Loading dense steering vector from: {args.steering_vector_pt}")
        steering_vector = torch.load(args.steering_vector_pt, map_location="cpu")
        if not isinstance(steering_vector, torch.Tensor):
            raise ValueError("Loaded dense steering vector is not a torch.Tensor")
    else:
        if not args.vector_map_pt:
            raise ValueError("--vector-map-pt is required for categorical mode")
        print(f"Loading categorical vector map from: {args.vector_map_pt}")
        vector_map = torch.load(args.vector_map_pt, map_location="cpu")
        if not isinstance(vector_map, dict):
            raise ValueError("Loaded vector map must be a dict[token_id->torch.Tensor]")
        get_sv = lambda prompt, model, tok: get_categorical_steering_vector_fine_tuned(  # noqa: E731
            vector_map, prompt, model, tok
        )

    # Build steering hook (position-aware)
    steering_hook = make_steering_hook_activations(position=args.position)

    print("Generating with interventions...")
    outputs = generate_outputs_dataset_intervened(
        model=hooked,
        tokenizer=tokenizer,
        iterator=dl,
        model_name=args.model,
        steering_vector=steering_vector,
        strength=args.strength,
        get_steering_vector=get_sv,
        intervention_hook=steering_hook,
        layer=args.layer,
        activations=[args.activation_name],
        description=f"Intervened ({args.mode})",
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=1.0,
        outputs_save_path=args.out,
        device=DEVICE,
    )

    print(f"Wrote {len(outputs)} records to {args.out}")


if __name__ == "__main__":
    main()
