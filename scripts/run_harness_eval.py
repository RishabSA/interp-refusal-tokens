#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

# Ensure project root imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from refusal.config import DEVICE, SAVED_OUTPUTS_DIR
from refusal.models import load_hf_model_and_tokenizer, load_hooked_transformer
from refusal.harness import HookedSteeredLM
from refusal.interventions import get_categorical_steering_vector_fine_tuned

# lm-evaluation-harness
from lm_eval import evaluator, tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate (optionally steered) model on lm-eval harness tasks")
    p.add_argument("--model", required=True, help="HF model id to load")
    p.add_argument("--mode", choices=["none", "categorical"], default="none")
    p.add_argument("--vector-map-pt", default="", help="Path to .pt dict[token_id->tensor] (categorical mode)")
    p.add_argument("--strength", type=float, default=-5.0)
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--activation-name", default="resid_post")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tasks", default="gsm8k,truthfulqa,mmlu", help="Comma-separated task names")
    p.add_argument("--out", default=str(SAVED_OUTPUTS_DIR / "harness_results.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model}")
    hf_model, tokenizer = load_hf_model_and_tokenizer(args.model, device=DEVICE)
    hooked = load_hooked_transformer(hf_model, args.model, device_map="auto")

    get_sv = None
    if args.mode == "categorical":
        if not args.vector_map_pt:
            raise ValueError("--vector-map-pt required for categorical mode")
        vec_map = torch.load(args.vector_map_pt, map_location="cpu")
        if not isinstance(vec_map, dict):
            raise ValueError("Loaded vector map must be a dict[token_id->torch.Tensor]")
        get_sv = lambda prompt, hm: get_categorical_steering_vector_fine_tuned(vec_map, prompt, hm, tokenizer)  # noqa: E731

    steered_lm = HookedSteeredLM(
        hooked_model=hooked,
        tokenizer=tokenizer,
        get_steering_vector=get_sv,
        strength=args.strength,
        layer=args.layer,
        act_name=args.activation_name,
        max_gen_tokens=256,
        device=hooked.cfg.device,
        batch_size=args.batch_size,
    )

    tasks.initialize_tasks()
    task_list: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"Running harness tasks: {task_list}")
    results = evaluator.simple_evaluate(
        model=steered_lm,
        tasks=task_list,
        batch_size=args.batch_size,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results.get("results", {}), "versions": results.get("versions", {})}, f, indent=2)

    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
