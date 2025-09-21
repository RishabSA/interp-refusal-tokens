#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from refusal.inference import eval_outputs_dataset, score_refusal_token, score_llm_judge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate JSONL outputs by category using a scoring function")
    p.add_argument("--outputs", required=True, help="Path to JSONL with {response, category}")
    p.add_argument("--scorer", choices=["refusal_token", "llm_judge"], default="refusal_token")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--description", default="Evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scorer = score_refusal_token if args.scorer == "refusal_token" else score_llm_judge

    eval_outputs_dataset(
        score_batch=scorer,
        batch_size=args.batch_size,
        description=args.description,
        outputs_save_path=args.outputs,
    )


if __name__ == "__main__":
    main()
