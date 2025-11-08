from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Iterable

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os

from datasets import load_dataset
from pathlib import Path
import json

from .config import DATA_DIR


@dataclass
class PromptExample:
    prompt: str
    category: str | None = None


class PromptCategoryDataset(Dataset):
    """Simple dataset wrapper around a list of dicts with keys 'prompt' and optional 'category'."""

    def __init__(self, items: List[Dict[str, Any]]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._items[idx]


def prompt_category_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "prompt": [sample["prompt"] for sample in batch],
        "category": [sample.get("category") for sample in batch],
    }


def make_dataloader(items: List[Dict[str, Any]], batch_size: int = 4, shuffle: bool = False) -> DataLoader:
    ds = PromptCategoryDataset(items)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=prompt_category_collate)


def split_dataloader_by_category(iterator: DataLoader, category_field: str = "category") -> Dict[str, DataLoader]:
    """Split a DataLoader's underlying dataset by category using Subset to preserve collate_fn and batching.

    Mirrors the notebook logic and keeps original DataLoader settings as much as possible.
    """
    dataset = iterator.dataset  # type: ignore[attr-defined]

    # Collect indices for each category
    category2idxs: Dict[str, List[int]] = {}
    for i in range(len(dataset)):
        ex = dataset[i]
        cat = ex[category_field] if isinstance(ex, dict) else getattr(ex, category_field, "uncategorized")
        category2idxs.setdefault(cat, []).append(i)

    iterator_by_category: Dict[str, DataLoader] = {}
    for category, idxs in category2idxs.items():
        data_subset = Subset(dataset, idxs)
        iterator_by_category[category] = DataLoader(
            data_subset,
            batch_size=iterator.batch_size or 4,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
            collate_fn=iterator.collate_fn,
        )

    return iterator_by_category


# Dataset-specific loaders mirrored from the notebook
def load_coconot_data(batch_size: int = 4) -> Dict[str, DataLoader]:
    """COCONot test dataloaders for original and contrast splits."""
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

    coconot_orig_test_dataloader = DataLoader(
        coconot_orig["test"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    coconot_contrast_test_dataloader = DataLoader(
        coconot_contrast["test"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    return {
        "coconot_orig_test_dataloader": coconot_orig_test_dataloader,
        "coconot_contrast_test_dataloader": coconot_contrast_test_dataloader,
    }


def load_wildguard_data(batch_size: int = 4) -> Dict[str, DataLoader]:
    """WildGuardMix test dataloader with custom collate to return prompts and categories."""
    wildguard_test = load_dataset("allenai/wildguardmix", "wildguardtest")  # 1.73k items

    # Align field names to expected keys
    wildguard_test = wildguard_test.rename_column("subcategory", "category")

    def wildguard_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "prompt": [ex["prompt"] for ex in batch],
            "category": [ex["category"] for ex in batch],
        }

    wildguard_test_dataloader = DataLoader(
        wildguard_test["test"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=wildguard_collate,
    )

    return {"wildguard_test_dataloader": wildguard_test_dataloader}


def load_wildjailbreak_data(batch_size: int = 4) -> Dict[str, DataLoader]:
    """WildJailbreak eval dataloader with field renaming to 'prompt' and 'category'."""
    wildjailbreak_eval = load_dataset("allenai/wildjailbreak", "eval")  # 2.21k items

    wildjailbreak_eval = wildjailbreak_eval.rename_column("adversarial", "prompt")
    wildjailbreak_eval = wildjailbreak_eval.rename_column("data_type", "category")

    wildjailbreak_eval_dataloader = DataLoader(
        wildjailbreak_eval["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    return {"wildjailbreak_eval_dataloader": wildjailbreak_eval_dataloader}


def load_or_bench_data(batch_size: int = 4) -> Dict[str, DataLoader]:
    """OR-Bench hard and toxic dataloaders."""
    or_bench_hard = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")  # 1.32k
    or_bench_toxic = load_dataset("bench-llm/or-bench", "or-bench-toxic")  # 655

    or_bench_hard_dataloader = DataLoader(
        or_bench_hard["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    or_bench_toxic_dataloader = DataLoader(
        or_bench_toxic["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    return {
        "or_bench_hard_dataloader": or_bench_hard_dataloader,
        "or_bench_toxic_dataloader": or_bench_toxic_dataloader,
    }


# ----- Synthetic dataset from JSON -----


class SyntheticHarmfulDataset(Dataset):
    def __init__(self, items: Dict[str, Any]):
        self.category = items["category"]
        self.pairs = items["pairs"]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        return {"prompt": pair["harmful"], "category": self.category}


class SyntheticBenignDataset(Dataset):
    def __init__(self, items: Dict[str, Any]):
        self.category = items["category"]
        self.pairs = items["pairs"]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        return {"prompt": pair["benign"], "category": self.category}


def load_synthetic_refusal_json(
    json_path: str | Path | None = None,
    batch_size: int = 4,
) -> Dict[str, Dict[str, DataLoader]]:
    """Load synthetic harmful/benign prompt pairs from a JSON file and return dataloaders by category.

    Returns a dict with keys "harmful" and "benign", each mapping category->DataLoader.
    """
    if json_path is None:
        json_path = DATA_DIR / "refusal_dataset.json"
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        synthetic_items: List[Dict[str, Any]] = json.load(f)

    categories = sorted({item["category"] for item in synthetic_items})
    harmful_by_cat: Dict[str, DataLoader] = {}
    benign_by_cat: Dict[str, DataLoader] = {}

    for category in categories:
        category_subset = [items for items in synthetic_items if items["category"] == category][0]

        harm_ds = SyntheticHarmfulDataset(category_subset)
        ben_ds = SyntheticBenignDataset(category_subset)

        harm_dl = DataLoader(
            harm_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=prompt_category_collate,
        )

        ben_dl = DataLoader(
            ben_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=prompt_category_collate,
        )

        harmful_by_cat[category] = harm_dl
        benign_by_cat[category] = ben_dl

    return {"harmful": harmful_by_cat, "benign": benign_by_cat}
