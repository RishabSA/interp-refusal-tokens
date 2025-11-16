import os
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from transformer_lens import (
    HookedTransformer,
)

from scripts.activation_caching import cache_hooked_activations_before_pad


class ActivationDataset(Dataset):
    def __init__(self, X, y):
        # X shape: (N, d_model)
        # y shape: (N) (0 - benign / 1 - harmful)

        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def get_probe_prompt_training_data(
    batch_size: int = 4,
) -> tuple[DataLoader, DataLoader]:
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


def get_probe_prompt_testing_data(batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
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


def get_probe_training_activations(
    hooked_model: HookedTransformer,
    harmful_probe_dataloader: DataLoader,
    benign_probe_dataloader: DataLoader,
    layer: int = 18,
    activation_name: str = "resid_post",
    batch_size: int = 512,
    val_split: float = 0.2,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[DataLoader, DataLoader]:
    hooked_model.to(device).eval()

    harmful_probe_activations, _ = cache_hooked_activations_before_pad(
        hooked_model=hooked_model,
        iterator=harmful_probe_dataloader,
        activation_name=activation_name,
        layer=layer,
        # prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        prompt_seq_append="",
        device=device,
    )  # (N_h, d_model)

    benign_probe_activations, _ = cache_hooked_activations_before_pad(
        hooked_model=hooked_model,
        iterator=benign_probe_dataloader,
        activation_name=activation_name,
        layer=layer,
        # prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        prompt_seq_append="",
        device=device,
    )  # (N_b, d_model)

    print(
        f"Harmful probe activations data has a shape of {harmful_probe_activations.shape}"
    )
    print(
        f"Benign probe activations data has a shape of {benign_probe_activations.shape}"
    )

    harmful_probe_activations = harmful_probe_activations.to(torch.float32)
    benign_probe_activations = benign_probe_activations.to(torch.float32)

    # Get labels (1 = harmful, 0 = benign)
    N_h = harmful_probe_activations.shape[0]
    N_b = benign_probe_activations.shape[0]

    X = torch.cat(
        [harmful_probe_activations, benign_probe_activations], dim=0
    )  # (N_h + N_b, d_model)
    y = torch.cat([torch.ones(N_h), torch.zeros(N_b)], dim=0)  # (N_h + N_b)

    probe_dataset = ActivationDataset(X, y)

    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_split, random_state=0, stratify=y.numpy()
    )

    train_probe_dataset = torch.utils.data.Subset(probe_dataset, train_idx)
    val_probe_dataset = torch.utils.data.Subset(probe_dataset, val_idx)

    train_probe_dataloader = DataLoader(
        train_probe_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    val_probe_dataloader = DataLoader(
        val_probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    return train_probe_dataloader, val_probe_dataloader


def get_probe_testing_activations(
    hooked_model: HookedTransformer,
    harmful_probe_dataloader: DataLoader,
    benign_probe_dataloader: DataLoader,
    layer: int = 18,
    activation_name: str = "resid_post",
    batch_size: int = 512,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> DataLoader:
    hooked_model.to(device).eval()

    harmful_probe_activations, _ = cache_hooked_activations_before_pad(
        hooked_model=hooked_model,
        iterator=harmful_probe_dataloader,
        activation_name=activation_name,
        layer=layer,
        # prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        prompt_seq_append="",
        device=device,
    )  # (N_h, d_model)

    benign_probe_activations, _ = cache_hooked_activations_before_pad(
        hooked_model=hooked_model,
        iterator=benign_probe_dataloader,
        activation_name=activation_name,
        layer=layer,
        # prompt_seq_append="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        prompt_seq_append="",
        device=device,
    )  # (N_b, d_model)

    print(
        f"Harmful probe activations data has a shape of {harmful_probe_activations.shape}"
    )
    print(
        f"Benign probe activations data has a shape of {benign_probe_activations.shape}"
    )

    harmful_probe_activations = harmful_probe_activations.to(torch.float32)
    benign_probe_activations = benign_probe_activations.to(torch.float32)

    # Get labels (1 = harmful, 0 = benign)
    N_h = harmful_probe_activations.shape[0]
    N_b = benign_probe_activations.shape[0]

    X = torch.cat(
        [harmful_probe_activations, benign_probe_activations], dim=0
    )  # (N_h + N_b, d_model)
    y = torch.cat([torch.ones(N_h), torch.zeros(N_b)], dim=0)  # (N_h + N_b)

    probe_dataset = ActivationDataset(X, y)

    test_probe_dataloader = DataLoader(
        probe_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    return test_probe_dataloader
