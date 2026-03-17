import os
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from scripts.hooked_model import generate_hooked_model_response


def get_contrast_steering_vector_data(
    batch_size: int = 4,
    should_append: bool = True,
) -> tuple[dict[str, DataLoader], dict[str, DataLoader]]:
    # COCONot Dataset loading
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items

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

            if should_append:
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


def get_steering_vector_data(
    batch_size: int = 4,
    should_append: bool = True,
) -> tuple[dict[str, DataLoader], DataLoader]:
    # COCONot Dataset loading
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

    coconot_unique_categories = coconot_orig["train"].unique("category")

    coconot_harmful_dataloaders = {}

    def prompt_collate(batch: list[dict]) -> dict[str, list[str]]:
        return {
            "prompt": [sample["prompt"] for sample in batch],
        }

    def prompt_category_collate(batch: list[dict]) -> dict[str, list[str]]:
        return {
            "prompt": [sample["prompt"] for sample in batch],
            "category": [sample.get("category") for sample in batch],
        }

    benign_category_dataset = coconot_contrast["test"]

    benign_categorical_prompts = []
    for item in benign_category_dataset:
        benign_item = dict(item)

        if should_append:
            benign_item["prompt"] += f" [respond]"

        benign_categorical_prompts.append(benign_item)

    benign_prompts_dataloader = DataLoader(
        benign_categorical_prompts,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_collate,
    )

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

        harmful_categorical_prompts = []

        for item in harmful_category_dataset:
            harmful_item = dict(item)

            if should_append:
                harmful_item["prompt"] += f" [{category}]"

            harmful_categorical_prompts.append(harmful_item)

        harmful_category_dataloader = DataLoader(
            harmful_categorical_prompts,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=prompt_category_collate,
        )

        coconot_harmful_dataloaders[category] = harmful_category_dataloader

    for category, dataloader in coconot_harmful_dataloaders.items():
        print(f"{category} harmful category has {len(dataloader)} batches")

    print("\n")

    print(f"Benign dataloader has {len(benign_prompts_dataloader)} batches")

    return coconot_harmful_dataloaders, benign_prompts_dataloader
