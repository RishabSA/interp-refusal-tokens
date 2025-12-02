# run_baselines.py

Baseline evaluation script for refusal token models. Generates model responses on safety benchmarks and evaluates refusal rates.

## Models

| Model | HuggingFace ID | Evaluation Method |
|-------|----------------|-------------------|
| `llama-base` | `meta-llama/Meta-Llama-3-8B` | LLM Judge (GPT) |
| `llama-instruct` | `meta-llama/Meta-Llama-3-8B-Instruct` | LLM Judge (GPT) |
| `categorical-refusal` | `tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens` | Refusal Token (5 categories) |
| `binary-refusal` | `tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-single-token` | Refusal Token (`[refuse]`) |

## Benchmarks

| Benchmark | Splits | Description |
|-----------|--------|-------------|
| COCONot | `orig`, `contrast` | Safety benchmark with 5 refusal categories |
| WildGuard | `test` | Multi-category safety evaluation |
| WildJailbreak | `adversarial_benign`, `adversarial_harmful` | Adversarial jailbreak attempts |
| OR-Bench | `hard`, `toxic` | Over-refusal benchmark |

## Output Files

Per model, generates 7 `.jsonl` files:

```
coconot_orig_test_outputs_{model}.jsonl
coconot_contrast_test_outputs_{model}.jsonl
wildguard_test_outputs_{model}.jsonl
wildjailbreak_eval_outputs_{model}_adversarial_benign.jsonl
wildjailbreak_eval_outputs_{model}_adversarial_harmful.jsonl
or_bench_hard_outputs_{model}.jsonl
or_bench_toxic_outputs_{model}.jsonl
```

## Usage

Run from project root (`/home/ubuntu/interp-refusal-tokens/`):

```bash
# Run all models on all benchmarks
python scripts/run_baselines.py

# Run specific model(s)
python scripts/run_baselines.py --models binary-refusal
python scripts/run_baselines.py --models llama-base llama-instruct

# Run specific benchmark(s)
python scripts/run_baselines.py --benchmarks coconot wildguard

# Evaluate only (skip generation, use existing .jsonl files)
python scripts/run_baselines.py --eval_only
python scripts/run_baselines.py --models binary-refusal --eval_only

# Custom batch size
python scripts/run_baselines.py --batch_size 8
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | all 4 models | Models to evaluate |
| `--benchmarks` | `coconot wildguard wildjailbreak orbench` | Benchmarks to run |
| `--batch_size` | 4 | Batch size for generation |
| `--eval_only` | False | Skip generation, only evaluate existing outputs |
| `--openai_api_key` | env/hardcoded | OpenAI API key for LLM judge |

## Refusal Token Scoring

### Categorical Refusal (5 tokens)
```
[Humanizing requests]      - 128256
[Incomplete requests]      - 128257
[Indeterminate requests]   - 128258
[Requests with safety concerns] - 128259
[Unsupported requests]     - 128260
```

### Binary Refusal (1 token)
```
[refuse] - 128257
```

## Example Workflow

```bash
# 1. Generate outputs for all models (takes several hours)
python scripts/run_baselines.py

# 2. Or generate one model at a time
python scripts/run_baselines.py --models binary-refusal
python scripts/run_baselines.py --models categorical-refusal
python scripts/run_baselines.py --models llama-base
python scripts/run_baselines.py --models llama-instruct

# 3. Re-run evaluation only on existing files
python scripts/run_baselines.py --eval_only
```

## Output Format

Each `.jsonl` file contains records with:

```json
{
  "id": "uuid",
  "model": "binary-refusal",
  "category": "Requests with safety concerns",
  "prompt": "...",
  "response": "..."
}
```

## Evaluation Output

```
coconot_orig Test Evaluation with Refusal Token Rate: 952/1001 -> 95.1049%

Requests with safety concerns: 389/395 -> 98.4810%
Humanizing requests: 78/82 -> 95.1220%
Incomplete requests: 210/225 -> 93.3333%
Unsupported requests: 151/157 -> 96.1783%
Indeterminate requests: 124/142 -> 87.3239%
```
