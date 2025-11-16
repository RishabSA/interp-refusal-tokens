import os
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import (
    AutoTokenizer,
)
from transformer_lens import (
    HookedTransformer,
)

from scripts.hooked_model import generate_hooked_model_response


def get_contrast_steering_vector_data(
    batch_size: int = 4,
) -> tuple[dict[str, DataLoader], dict[str, DataLoader]]:
    # COCONot Dataset loading
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

    coconot_unique_categories = coconot_orig["train"].unique("category")

    coconot_harmful_dataloaders = {}
    coconot_benign_dataloaders = {}

    def prompt_category_collate(batch: list[dict]) -> dict[str, list[str]]:
        return {
            "prompt": [sample["prompt"] for sample in batch],
            "category": [sample.get("category") for sample in batch],
        }

    for category in coconot_unique_categories:
        # Filter the orig train dataset
        orig_category_train = coconot_orig["train"].filter(
            lambda x, c=category: x["category"] == c
        )

        # Filter the orig test dataset
        orig_category_test = coconot_orig["test"].filter(
            lambda x, c=category: x["category"] == c
        )

        categorical_dataset = concatenate_datasets(
            [orig_category_train, orig_category_test]
        )

        harmful_categorical_pair_prompts = []
        benign_categorical_pair_prompts = []

        for item in categorical_dataset:
            harmful_item = dict(item)
            benign_item = dict(item)

            # harmful_item["prompt"] += (
            #     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n "
            #     + f"[{category}]"
            # )

            harmful_item["prompt"] += f" [{category}]"

            # benign_item[
            #     "prompt"
            # ] += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n [respond]"

            benign_item["prompt"] += " [respond]"

            harmful_categorical_pair_prompts.append(harmful_item)
            benign_categorical_pair_prompts.append(benign_item)

        harmful_category_dataloader = DataLoader(
            harmful_categorical_pair_prompts,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=prompt_category_collate,
        )

        benign_category_dataloader = DataLoader(
            benign_categorical_pair_prompts,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=prompt_category_collate,
        )

        coconot_harmful_dataloaders[category] = harmful_category_dataloader
        coconot_benign_dataloaders[category] = benign_category_dataloader

    for category, dataloader in coconot_harmful_dataloaders.items():
        print(f"{category} harmful category has {len(dataloader)} batches")

    print("\n")

    for category, dataloader in coconot_benign_dataloaders.items():
        print(f"{category} benign category has {len(dataloader)} batches")

    return coconot_harmful_dataloaders, coconot_benign_dataloaders


def get_old_steering_vector_data(
    batch_size: int = 4,
) -> tuple[dict[str, DataLoader], dict[str, DataLoader]]:
    # COCONot Dataset loading
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

    coconot_unique_categories = coconot_orig["train"].unique("category")

    coconot_harmful_dataloaders = {}
    coconot_benign_dataloaders = {}

    def prompt_category_collate(batch: list[dict]) -> dict[str, list[str]]:
        return {
            "prompt": [sample["prompt"] for sample in batch],
            "category": [sample.get("category") for sample in batch],
        }

    for category in coconot_unique_categories:
        # Filter the orig train dataset
        orig_category_train = coconot_orig["train"].filter(
            lambda x, c=category: x["category"] == c
        )

        # Filter the orig test dataset
        orig_category_test = coconot_orig["test"].filter(
            lambda x, c=category: x["category"] == c
        )

        harmful_category_dataset = concatenate_datasets(
            [orig_category_train, orig_category_test]
        )

        harmful_category_dataloader = DataLoader(
            harmful_category_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=prompt_category_collate,
        )

        # Filter the contrast test dataset
        # benign_category_dataset = coconot_contrast["test"].filter(lambda x, c=category: x["category"] == c)
        # benign_category_dataset = coconot_contrast["test"]

        # benign_wildguard_dataset = wildguard_test["test"].filter(lambda x: x["subcategory"] == "benign")
        # benign_category_dataset = concatenate_datasets([coconot_contrast["test"], benign_wildguard_dataset, truthful_qa_gen["validation"]])

        benign_category_dataset = coconot_contrast["test"]

        benign_category_dataloader = DataLoader(
            benign_category_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=prompt_category_collate,
        )

        coconot_harmful_dataloaders[category] = harmful_category_dataloader
        coconot_benign_dataloaders[category] = benign_category_dataloader

    for category, dataloader in coconot_harmful_dataloaders.items():
        print(f"{category} harmful category has {len(dataloader)} batches")

    print("\n")

    for category, dataloader in coconot_benign_dataloaders.items():
        print(f"{category} benign category has {len(dataloader)} batches")

    return coconot_harmful_dataloaders, coconot_benign_dataloaders


class SyntheticHarmfulDataset(Dataset):
    def __init__(self, items):
        self.category = items["category"]
        self.pairs = items["pairs"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, str]:
        pair = self.pairs[idx]

        return {
            "prompt": pair["harmful"],
            "category": self.category,
        }


class SyntheticBenignDataset(Dataset):
    def __init__(self, items):
        self.category = items["category"]
        self.pairs = items["pairs"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, str]:
        pair = self.pairs[idx]

        return {
            "prompt": pair["benign"],
            "category": self.category,
        }


def get_synthetic_steering_vector_data(
    batch_size: int = 4,
) -> tuple[dict[str, DataLoader], dict[str, DataLoader]]:
    data_path = Path("/workspace/refusal_dataset.json")
    with data_path.open("r", encoding="utf-8") as f:
        synthetic_items = json.load(f)

    unique_synthetic_categories = {item["category"] for item in synthetic_items}

    synthetic_harmful_dataloaders = {}
    synthetic_benign_dataloaders = {}

    for category in unique_synthetic_categories:
        category_subset = [
            items for items in synthetic_items if items["category"] == category
        ][0]

        synthetic_harmful_dataset = SyntheticHarmfulDataset(category_subset)
        synthetic_benign_dataset = SyntheticBenignDataset(category_subset)

        synthetic_harmful_dataloader = DataLoader(
            synthetic_harmful_dataset,
            batch_size=batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

        synthetic_benign_dataloader = DataLoader(
            synthetic_benign_dataset,
            batch_size=batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

        synthetic_harmful_dataloaders[category] = synthetic_harmful_dataloader
        synthetic_benign_dataloaders[category] = synthetic_benign_dataloader

    for category, dataloader in synthetic_harmful_dataloaders.items():
        print(f"{category} harmful category has {len(dataloader)} synthetic batches")

    print("\n")

    for category, dataloader in synthetic_benign_dataloaders.items():
        print(f"{category} benign category has {len(dataloader)} synthetic batches")

    return synthetic_harmful_dataloaders, synthetic_benign_dataloaders


def print_synthetic_pairs(
    hooked_model: HookedTransformer,
    tokenizer: AutoTokenizer,
    SEED: int = 42,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    data_path = Path("/workspace/refusal_dataset.json")
    with data_path.open("r", encoding="utf-8") as f:
        synthetic_items = json.load(f)

    synthetic_harmful_test_prompts = {}
    synthetic_benign_test_prompts = {}

    for category_items in synthetic_items:
        category_harmful = []
        for pair in category_items["pairs"]:
            category_harmful.append(pair["harmful"])

        synthetic_harmful_test_prompts[category_items["category"]] = category_harmful

        category_benign = []
        for pair in category_items["pairs"]:
            category_benign.append(pair["benign"])

        synthetic_benign_test_prompts[category_items["category"]] = category_benign

    # Harmful Synthethic Prompts
    correct_refusal_harmful_numbers = {
        key: 0 for key in synthetic_harmful_test_prompts.keys()
    }

    for (
        harmful_category,
        harmful_test_prompts,
    ) in synthetic_harmful_test_prompts.items():
        print(f"Category: {harmful_category}")

        for harmful_test_prompt in harmful_test_prompts:
            harmful_test_sequence = generate_hooked_model_response(
                hooked_model=hooked_model,
                tokenizer=tokenizer,
                prompt=harmful_test_prompt,
                append_seq="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                stop_tokens=["<|eot_id|>"],
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                SEED=SEED,
                device=device,
            )

            if harmful_category in harmful_test_sequence:
                correct_refusal_harmful_numbers[harmful_category] += 1

            print(harmful_test_sequence)
            print("\n")

        print("\n")

    print(correct_refusal_harmful_numbers)

    # Benign Synthethic Prompts
    correct_refusal_benign_numbers = {
        key: 0 for key in synthetic_benign_test_prompts.keys()
    }

    for benign_category, benign_test_prompts in synthetic_benign_test_prompts.items():
        print(f"Category: {benign_category}")

        for benign_test_prompt in benign_test_prompts:
            benign_test_sequence = generate_hooked_model_response(
                hooked_model=hooked_model,
                tokenizer=tokenizer,
                prompt=benign_test_prompt,
                append_seq="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                stop_tokens=["<|eot_id|>"],
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                SEED=SEED,
                device=device,
            )

            if "respond" in benign_test_sequence:
                correct_refusal_benign_numbers[benign_category] += 1

            print(benign_test_sequence)
            print("\n")

        print("\n")

    print(correct_refusal_benign_numbers)
