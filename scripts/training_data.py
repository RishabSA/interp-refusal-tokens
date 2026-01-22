import os
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets


def get_prompt_training_data(
    batch_size: int = 4,
    load_datasets: list[str] = [
        "coconot",
        "wildguard",
        "or_bench",
        "truthful_qa",
        "gsm8k",
        "mmlu",
    ],
) -> tuple[DataLoader, DataLoader]:
    """Get dataloaders containing prompts used for training.

    Args:
        batch_size (int, optional). Defaults to 4.
        load_datasets (list[str], optional). Defaults to Defaults to ["coconot", "wildguard", "or_bench", "truthful_qa", "gsm8k", "mmlu"].

    Returns:
        tuple[DataLoader, DataLoader]: Dataloaders containing harmful and benign prompts used for training.
    """

    def prompt_category_collate(batch: list[dict]) -> dict[str, list[str]]:
        return {
            "prompt": [sample["prompt"] for sample in batch],
        }

    harmful_datasets = []
    benign_datasets = []

    if "coconot" in load_datasets:
        # COCONot Dataset loading
        coconot_orig = load_dataset("allenai/coconot", "original")
        coconot_contrast = load_dataset("allenai/coconot", "contrast")

        harmful_datasets.append(coconot_orig["train"])  # 11.5k items
        benign_datasets.append(coconot_contrast["test"])  # 379 items

    if "wildguard" in load_datasets:
        # WildGuardMix Dataset loading
        wildguard_train = load_dataset("allenai/wildguardmix", "wildguardtrain")

        wildguard_train_harmful = wildguard_train["train"].filter(
            lambda x: x["prompt_harm_label"] == "harmful"
        )  # 46.3k items

        wildguard_train_benign = wildguard_train["train"].filter(
            lambda x: x["prompt_harm_label"] == "unharmful"
        )  # 40.5k

        harmful_datasets.append(wildguard_train_harmful)
        benign_datasets.append(wildguard_train_benign)

    if "or_bench" in load_datasets:
        # OR-Bench
        or_bench_80k = load_dataset("bench-llm/or-bench", "or-bench-80k")

        harmful_datasets.append(or_bench_80k["train"])  # 80k items

    if "truthful_qa" in load_datasets:
        # TruthfulQA
        truthful_qa_gen = load_dataset("truthfulqa/truthful_qa", "generation")
        truthful_qa_gen = truthful_qa_gen.rename_column("question", "prompt")

        benign_datasets.append(truthful_qa_gen["validation"])  # 817 items

    if "gsm8k" in load_datasets:
        # GSM8k
        gsm8k = load_dataset("openai/gsm8k", "main")
        gsm8k = gsm8k.rename_column("question", "prompt")

        benign_datasets.append(gsm8k["train"])  # 7.47k items

    if "mmlu" in load_datasets:
        # MMLU
        mmlu = load_dataset("cais/mmlu", "all")
        mmlu = mmlu.rename_column("question", "prompt")
        mmlu = mmlu.remove_columns("answer")

        benign_datasets.append(mmlu["auxiliary_train"])  # 99.8k

    if len(harmful_datasets) == 0:
        raise ValueError("No harmful datasets were passed in")

    if len(benign_datasets) == 0:
        raise ValueError("No benign datasets were passed in")

    harmful_prompt_dataset = concatenate_datasets(harmful_datasets)
    benign_prompt_dataset = concatenate_datasets(benign_datasets)

    harmful_prompt_dataloader = DataLoader(
        harmful_prompt_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_category_collate,
    )

    benign_prompt_dataloader = DataLoader(
        benign_prompt_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_category_collate,
    )

    return harmful_prompt_dataloader, benign_prompt_dataloader


def get_prompt_testing_data(
    batch_size: int = 4,
    load_datasets: list[str] = [
        "coconot",
        "wildguard",
        "or_bench",
        "wildjailbreak",
        "gsm8k",
        "mmlu",
    ],
) -> tuple[DataLoader, DataLoader]:
    """Get dataloaders containing prompts used for testing.

    Args:
        batch_size (int, optional). Defaults to 4.
        load_datasets (list[str], optional). Defaults to Defaults to ["coconot", "wildguard", "or_bench", "wildjailbreak", "gsm8k", "mmlu"].

    Returns:
        tuple[DataLoader, DataLoader]: Dataloaders containing harmful and benign prompts used for testing.
    """

    def prompt_category_collate(batch):
        return {
            "prompt": [sample["prompt"] for sample in batch],
        }

    harmful_datasets = []
    benign_datasets = []

    if "coconot" in load_datasets:
        # COCONot Dataset loading
        coconot_orig = load_dataset("allenai/coconot", "original")
        coconot_contrast = load_dataset("allenai/coconot", "contrast")

        harmful_datasets.append(coconot_orig["test"])  # 1k items
        benign_datasets.append(coconot_contrast["test"])  # 379 items

    if "wildguard" in load_datasets:
        # WildGuardMix Dataset loading
        wildguard_test = load_dataset(
            "allenai/wildguardmix", "wildguardtest"
        )  # 1.73k items

        wildguard_test_harmful = wildguard_test["test"].filter(
            lambda x: x["prompt_harm_label"] == "harmful"
        )  # 756 items

        wildguard_test_benign = wildguard_test["test"].filter(
            lambda x: x["prompt_harm_label"] == "unharmful"
        )  # 948 items

        harmful_datasets.append(wildguard_test_harmful)
        benign_datasets.append(wildguard_test_benign)

    if "or_bench" in load_datasets:
        # OR-Bench
        or_bench_toxic = load_dataset("bench-llm/or-bench", "or-bench-toxic")
        or_bench_hard = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")

        harmful_datasets.append(or_bench_toxic["train"])  # 655 item
        benign_datasets.append(or_bench_hard["train"])  # 1.32k items

    if "wildjailbreak" in load_datasets:
        # WildJailbreak Dataset Loading
        wildjailbreak_eval = load_dataset(
            "allenai/wildjailbreak", "eval"
        )  # 2.21k items
        wildjailbreak_eval = wildjailbreak_eval.rename_column("adversarial", "prompt")

        wildjailbreak_eval_harmful = wildjailbreak_eval["train"].filter(
            lambda x: x["data_type"] == "vanilla_harmful"
        )

        wildjailbreak_eval_benign = wildjailbreak_eval["train"].filter(
            lambda x: x["data_type"] == "vanilla_benign"
        )

        harmful_datasets.append(wildjailbreak_eval_harmful)
        benign_datasets.append(wildjailbreak_eval_benign)

    if "gsm8k" in load_datasets:
        # GSM8k
        gsm8k = load_dataset("openai/gsm8k", "main")
        gsm8k = gsm8k.rename_column("question", "prompt")

        benign_datasets.append(gsm8k["test"])  # 1.32k items

    if "mmlu" in load_datasets:
        # MMLU
        mmlu = load_dataset("cais/mmlu", "all")
        mmlu = mmlu.rename_column("question", "prompt")
        mmlu = mmlu.remove_columns("answer")

        benign_datasets.append(mmlu["test"])  # 14k

    if len(harmful_datasets) == 0:
        raise ValueError("No harmful datasets were passed in")

    if len(benign_datasets) == 0:
        raise ValueError("No benign datasets were passed in")

    harmful_prompt_dataset = concatenate_datasets(harmful_datasets)
    benign_prompt_dataset = concatenate_datasets(benign_datasets)

    harmful_prompt_dataloader = DataLoader(
        harmful_prompt_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_category_collate,
    )

    benign_prompt_dataloader = DataLoader(
        benign_prompt_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_category_collate,
    )

    return harmful_prompt_dataloader, benign_prompt_dataloader
