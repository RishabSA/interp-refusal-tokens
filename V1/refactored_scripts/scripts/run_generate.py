#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Any

# Ensure project root on sys.path so 'refusal' package is importable when running this script directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from refusal.config import DEVICE, DEFAULT_MODEL, DEFAULT_MAX_NEW_TOKENS, SAVED_OUTPUTS_DIR
from refusal.models import load_hf_model_and_tokenizer
from refusal.data import make_dataloader
from refusal.inference import generate_outputs_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate model outputs for a set of prompts.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model name")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--prompts-jsonl", type=str, default="", help="Optional JSONL with {prompt, category?}")
    p.add_argument("--out", type=str, default=str(SAVED_OUTPUTS_DIR / "sample_outputs.jsonl"))
    return p.parse_args()


def load_prompts(prompts_jsonl: str | None) -> List[Dict[str, Any]]:
    if prompts_jsonl:
        path = Path(prompts_jsonl)
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                items.append(json.loads(line))
        return items
    # Fallback demo prompts
    return [
        {"prompt": "Explain why the sky is blue.", "category": "benign"},
        {"prompt": "Write a haiku about coding.", "category": "benign"},
    ]


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_hf_model_and_tokenizer(args.model, device=DEVICE)

    items = load_prompts(args.prompts_jsonl)
    dl = make_dataloader(items, batch_size=args.batch_size, shuffle=False)

    outputs = generate_outputs_dataset(
        model=model,
        tokenizer=tokenizer,
        iterator=dl,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        device=DEVICE,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(outputs)} records to {out_path}")


if __name__ == "__main__":
    main()
