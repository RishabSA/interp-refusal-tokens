"""
Baseline evaluation script - mirrors the notebook logic exactly.

Models:
- llama-base: meta-llama/Meta-Llama-3-8B (LLM judge only)
- llama-instruct: meta-llama/Meta-Llama-3-8B-Instruct (LLM judge only)
- categorical-refusal: score_refusal_token (5 categorical tokens)
- binary-refusal: score_binary_refusal_token ([refuse] token)

Benchmarks:
- COCONot (original + contrast)
- WildGuard
- WildJailbreak (full + adversarial_benign + adversarial_harmful splits)
- OR-Bench (hard + toxic)
"""

import os
import sys
import argparse
from functools import partial

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from huggingface_hub import login

# Get tokens from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Login to HuggingFace
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HF_TOKEN not set, some models may not be accessible")

from scripts.model import load_model
from scripts.eval import (
    generate_outputs_dataset,
    eval_outputs_dataset,
    score_refusal_token,
    score_binary_refusal_token,
    score_llm_judge,
)
from scripts.eval_data import (
    split_dataloader_by_category,
    load_coconot_test_data,
    load_wildguard_test_data,
    load_wildjailbreak_test_data,
    load_or_bench_test_data,
)

# Model configurations (same as notebook)
MODEL_HF_MAPPINGS = {
    "llama-base": "meta-llama/Meta-Llama-3-8B",
    "llama-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "categorical-refusal": "tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens",
    "binary-refusal": "tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-single-token",
}

# Output file naming patterns (matches notebook exactly)
# {model} will be replaced with model_name
OUTPUT_FILE_PATTERNS = {
    "coconot_orig": "coconot_orig_test_outputs_{model}.jsonl",
    "coconot_contrast": "coconot_contrast_test_outputs_{model}.jsonl",
    "wildguard": "wildguard_test_outputs_{model}.jsonl",
    "wildjailbreak_eval_adversarial_benign": "wildjailbreak_eval_outputs_{model}_adversarial_benign.jsonl",
    "wildjailbreak_eval_adversarial_harmful": "wildjailbreak_eval_outputs_{model}_adversarial_harmful.jsonl",
    "or_bench_hard": "or_bench_hard_outputs_{model}.jsonl",
    "or_bench_toxic": "or_bench_toxic_outputs_{model}.jsonl",
}


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations (same as notebook)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama-base", "llama-instruct", "categorical-refusal", "binary-refusal"],
        choices=list(MODEL_HF_MAPPINGS.keys()),
        help="Models to evaluate",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["coconot", "wildguard", "wildjailbreak", "orbench"],
        choices=["coconot", "wildguard", "wildjailbreak", "orbench"],
        help="Benchmarks to run",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing outputs, skip generation")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup OpenAI client for LLM judge
    openai_api_key = args.openai_api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    openai_client = None
    score_llm_judge_gpt = None

    if openai_api_key:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=openai_api_key)
            score_llm_judge_gpt = partial(score_llm_judge, openai_client)
            print("OpenAI client initialized for LLM judge")
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
    else:
        print("Warning: No OpenAI API key provided, LLM judge evaluation will be skipped")

    # Load benchmarks
    print("\n" + "="*60)
    print("Loading benchmarks...")
    print("="*60)

    benchmarks = {}

    if "coconot" in args.benchmarks:
        coconot_data = load_coconot_test_data(batch_size=args.batch_size)
        benchmarks["coconot_orig"] = coconot_data["coconot_orig_test_dataloader"]
        benchmarks["coconot_contrast"] = coconot_data["coconot_contrast_test_dataloader"]

    if "wildguard" in args.benchmarks:
        wildguard_data = load_wildguard_test_data(batch_size=args.batch_size)
        benchmarks["wildguard"] = wildguard_data["wildguard_test_dataloader"]

    if "wildjailbreak" in args.benchmarks:
        wildjailbreak_data = load_wildjailbreak_test_data(batch_size=args.batch_size)
        # Split by category for adversarial_benign and adversarial_harmful only
        wildjailbreak_split = split_dataloader_by_category(
            wildjailbreak_data["wildjailbreak_eval_dataloader"], category_field="category"
        )
        if "adversarial_benign" in wildjailbreak_split:
            benchmarks["wildjailbreak_eval_adversarial_benign"] = wildjailbreak_split["adversarial_benign"]
        if "adversarial_harmful" in wildjailbreak_split:
            benchmarks["wildjailbreak_eval_adversarial_harmful"] = wildjailbreak_split["adversarial_harmful"]

    if "orbench" in args.benchmarks:
        orbench_data = load_or_bench_test_data(batch_size=args.batch_size)
        benchmarks["or_bench_hard"] = orbench_data["or_bench_hard_dataloader"]
        benchmarks["or_bench_toxic"] = orbench_data["or_bench_toxic_dataloader"]

    # Create generate function with same params as notebook
    generate_outputs_dataset_baseline_eval = partial(
        generate_outputs_dataset,
        steering_vector=None,
        benign_strength=0.0,
        harmful_strength=0.0,
        get_steering_vector=None,
        intervention_hook=None,
        layer=None,
        activation_name=None,
        max_new_tokens=512,
        do_sample=False,
        temperature=1.0,
        append_seq="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        device=device,
    )

    # Run evaluations for each model
    for model_name in args.models:
        print("\n" + "="*60)
        print(f"Processing model: {model_name}")
        print("="*60)

        model_id = MODEL_HF_MAPPINGS[model_name]

        if not args.eval_only:
            # Load model (same as notebook)
            print(f"Loading {model_id}...")
            model, tokenizer = load_model(model_id)

        # Determine scoring function for refusal token rate
        # - llama models: no refusal token scoring (only LLM judge)
        # - categorical-refusal: score_refusal_token (5 categorical tokens)
        # - binary-refusal: score_binary_refusal_token ([refuse] token)
        if model_name == "binary-refusal":
            score_refusal_fn = score_binary_refusal_token
        else:
            score_refusal_fn = score_refusal_token

        # Run each benchmark
        for benchmark_name, iterator in benchmarks.items():
            # Use exact notebook file naming pattern
            outputs_path = OUTPUT_FILE_PATTERNS[benchmark_name].format(model=model_name)

            print(f"\n--- {benchmark_name} ---")

            # Generation (same as notebook)
            if not args.eval_only:
                print(f"Generating outputs...")
                outputs = generate_outputs_dataset_baseline_eval(
                    model=model,
                    tokenizer=tokenizer,
                    iterator=iterator,
                    description=f"{benchmark_name} Test Generation",
                    outputs_save_path=outputs_path,
                    model_name=model_name,
                )
                print(f"{len(outputs)} outputs were generated")

            # Evaluation with Refusal Token Rate (same as notebook)
            # Only for non-llama models
            if "llama" not in model_name:
                if os.path.exists(outputs_path):
                    print(f"Evaluating with Refusal Token Rate...")
                    eval_outputs_dataset(
                        score_batch=score_refusal_fn,
                        batch_size=8,
                        description=f"{benchmark_name} Test Evaluation with Refusal Token Rate",
                        outputs_save_path=outputs_path,
                    )

            # Evaluation with LLM as a Judge
            # Only for llama models (categorical/binary-refusal have refusal token scoring)
            if "llama" in model_name:
                if score_llm_judge_gpt and os.path.exists(outputs_path):
                    print(f"Evaluating with LLM as a Judge...")
                    eval_outputs_dataset(
                        score_batch=score_llm_judge_gpt,
                        batch_size=8,
                        description=f"{benchmark_name} Test Evaluation with LLM as a Judge",
                        outputs_save_path=outputs_path,
                    )

        # Free memory after each model
        if not args.eval_only:
            del model, tokenizer
            torch.cuda.empty_cache()
            print(f"\nFreed memory for {model_name}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
