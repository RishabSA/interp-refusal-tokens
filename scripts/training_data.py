import os
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets


def get_prompt_training_data(
    batch_size: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Get dataloaders containing prompts used for training.

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        tuple[DataLoader, DataLoader]: Dataloaders containing harmful and benign prompts used for training.
    """

    def prompt_category_collate(batch: list[dict]) -> dict[str, list[str]]:
        return {
            "prompt": [sample["prompt"] for sample in batch],
        }

    # COCONot Dataset loading
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

    # WildGuardMix Dataset loading
    wildguard_train = load_dataset(
        "allenai/wildguardmix", "wildguardtrain"
    )  # 1.73k items

    wildguard_train_harmful = wildguard_train["train"].filter(
        lambda x: x["prompt_harm_label"] == "harmful"
    )

    wildguard_train_benign = wildguard_train["train"].filter(
        lambda x: x["prompt_harm_label"] == "unharmful"
    )

    # OR-Bench
    or_bench_80k = load_dataset("bench-llm/or-bench", "or-bench-80k")  # 80k items
    or_bench_hard = load_dataset(
        "bench-llm/or-bench", "or-bench-hard-1k"
    )  # 1.32k items
    or_bench_toxic = load_dataset("bench-llm/or-bench", "or-bench-toxic")  # 655 item

    # TruthfulQA
    truthful_qa_gen = load_dataset("truthfulqa/truthful_qa", "generation")
    truthful_qa_gen = truthful_qa_gen.rename_column("question", "prompt")

    # GSM8k
    gsm8k = load_dataset("openai/gsm8k", "main")
    gsm8k = gsm8k.rename_column("question", "prompt")

    # MMLU
    mmlu = load_dataset("cais/mmlu", "all")
    mmlu = mmlu.rename_column("question", "prompt")
    mmlu = mmlu.remove_columns("answer")

    harmful_probe_dataset = concatenate_datasets(
        [
            coconot_orig["train"],
            wildguard_train_harmful,
            or_bench_80k["train"],
        ]
    )

    benign_probe_dataset = concatenate_datasets(
        [
            wildguard_train_benign,
            truthful_qa_gen["validation"],
            gsm8k["train"],
            mmlu["auxiliary_train"],
        ]
    )

    harmful_probe_dataloader = DataLoader(
        harmful_probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_category_collate,
    )

    benign_probe_dataloader = DataLoader(
        benign_probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_category_collate,
    )

    return harmful_probe_dataloader, benign_probe_dataloader


def get_prompt_testing_data(batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    """Get dataloaders containing prompts used for testing.

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        tuple[DataLoader, DataLoader]: Dataloaders containing harmful and benign prompts used for testing.
    """

    def prompt_category_collate(batch):
        return {
            "prompt": [sample["prompt"] for sample in batch],
        }

    # COCONot Dataset loading
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

    # WildGuardMix Dataset loading
    wildguard_test = load_dataset(
        "allenai/wildguardmix", "wildguardtest"
    )  # 1.73k items

    wildguard_test_harmful = wildguard_test["test"].filter(
        lambda x: x["prompt_harm_label"] == "harmful"
    )

    wildguard_test_benign = wildguard_test["test"].filter(
        lambda x: x["prompt_harm_label"] == "unharmful"
    )

    # OR-Bench
    or_bench_hard = load_dataset(
        "bench-llm/or-bench", "or-bench-hard-1k"
    )  # 1.32k items
    or_bench_toxic = load_dataset("bench-llm/or-bench", "or-bench-toxic")  # 655 item

    # TruthfulQA
    truthful_qa_gen = load_dataset("truthfulqa/truthful_qa", "generation")
    truthful_qa_gen = truthful_qa_gen.rename_column("question", "prompt")

    wildjailbreak_eval = load_dataset("allenai/wildjailbreak", "eval")  # 2.21k items
    wildjailbreak_eval = wildjailbreak_eval.rename_column("adversarial", "prompt")

    # GSM8k
    gsm8k = load_dataset("openai/gsm8k", "main")
    gsm8k = gsm8k.rename_column("question", "prompt")

    # MMLU
    mmlu = load_dataset("cais/mmlu", "all")
    mmlu = mmlu.rename_column("question", "prompt")
    mmlu = mmlu.remove_columns("answer")

    wildjailbreak_eval_harmful = wildjailbreak_eval["train"].filter(
        lambda x: x["data_type"] == "vanilla_harmful"
    )

    wildjailbreak_eval_benign = wildjailbreak_eval["train"].filter(
        lambda x: x["data_type"] == "vanilla_benign"
    )

    harmful_probe_dataset = concatenate_datasets(
        [
            coconot_orig["test"],
            wildguard_test_harmful,
            or_bench_toxic["train"],
            wildjailbreak_eval_harmful,
        ]
    )

    benign_probe_dataset = concatenate_datasets(
        [
            coconot_contrast["test"],
            wildguard_test_benign,
            or_bench_hard["train"],
            gsm8k["test"],
            mmlu["test"],
            wildjailbreak_eval_benign,
        ]
    )

    harmful_probe_dataloader = DataLoader(
        harmful_probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_category_collate,
    )

    benign_probe_dataloader = DataLoader(
        benign_probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_category_collate,
    )

    return harmful_probe_dataloader, benign_probe_dataloader
