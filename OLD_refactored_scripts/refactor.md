# Refactor 


## Mapping
- Data & inference → `data.py`, `inference.py`
- Hooks & vectors → `activations.py`, `steering.py`, `plots.py`
- Interventions & patching → `interventions.py`, `patching.py`
- SAE, model diff, harness → `sae.py`, `model_diff.py`, `harness.py`

## Quickstart
1) Install deps
```bash
pip install -r requirements.txt
```
2) Generate
```bash
python scripts/run_generate.py --model sshleifer/tiny-gpt2 --out saved_outputs/demo.jsonl
```
3) Compute steering + plots
```bash
python scripts/run_steering.py --model sshleifer/tiny-gpt2 --harmful-jsonl harm.jsonl --benign-jsonl ben.jsonl
```
4) Intervene (dense/categorical)
```bash
python scripts/run_intervene.py --model <hf_model> --prompts-jsonl prompts.jsonl --mode dense --steering-vector-pt steer.pt
python scripts/run_intervene.py --model <hf_model> --prompts-jsonl prompts.jsonl --mode categorical --vector-map-pt tokenid_to_vec_map.pt
```

## Notes
- Figures → `figures/`, tensors → `saved_tensors/`, outputs → `saved_outputs/`.
- `.env` is supported (e.g., `OPENAI_API_KEY`, `HF_TOKEN`). Secrets are not hardcoded.
- Synthetic pairs JSON supported via `refusal/data.py::load_synthetic_refusal_json()` from `synthethic_data/refusal_dataset.json`.
