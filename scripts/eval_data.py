import os
from torch.utils.data import (
    DataLoader,
    Subset,
)
from datasets import load_dataset


class Counter:
    def __init__(self, start: int = 0):
        self.count = start


def split_dataloader_by_category(iterator, category_field: str = "category"):
    dataset = iterator.dataset

    # Collect indices for each category
    category2idxs = {}
    for i in range(len(dataset)):
        category = dataset[i][category_field]
        category2idxs.setdefault(category, []).append(i)

    iterator_by_category = {}

    for category, idxs in category2idxs.items():
        dataSubset = Subset(dataset, idxs)
        iterator_by_category[category] = DataLoader(
            dataSubset,
            batch_size=iterator.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
            collate_fn=iterator.collate_fn,
        )

    return iterator_by_category


def load_coconot_data(batch_size: int = 4):
    # COCONot
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

    coconot_unique_categories = coconot_orig["test"].unique("category")
    print(f"COCONot Unique Categories: {coconot_unique_categories}")

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

    print(
        f"COCONot Orig Test Batches: {len(coconot_orig_test_dataloader)} | COCONot Contrast Test Batches: {len(coconot_contrast_test_dataloader)}"
    )

    return {
        "coconot_orig_test_dataloader": coconot_orig_test_dataloader,
        "coconot_contrast_test_dataloader": coconot_contrast_test_dataloader,
    }


def load_wildguard_data(batch_size: int = 4):
    # WildGuard
    wildguard_test = load_dataset(
        "allenai/wildguardmix", "wildguardtest"
    )  # 1.73k items

    wildguard_test = wildguard_test.rename_column("subcategory", "category")

    wildguard_unique_categories = wildguard_test["test"].unique("category")
    print(f"WildGuard Unique Categories: {wildguard_unique_categories}")

    def wildguard_collate(batch):
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

    print(f"WildGuard Test Batches: {len(wildguard_test_dataloader)}")

    return {
        "wildguard_test_dataloader": wildguard_test_dataloader,
    }


def load_wildjailbreak_data(batch_size: int = 4):
    # WildJailbreak
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

    print(f"WildJailbreak Eval Batches: {len(wildjailbreak_eval_dataloader)}")

    return {
        "wildjailbreak_eval_dataloader": wildjailbreak_eval_dataloader,
    }


def load_or_bench_data(batch_size: int = 4):
    # OR-Bench
    or_bench_hard = load_dataset(
        "bench-llm/or-bench", "or-bench-hard-1k"
    )  # 1.32k items
    or_bench_toxic = load_dataset("bench-llm/or-bench", "or-bench-toxic")  # 655 items

    or_bench_unique_categories = or_bench_hard["train"].unique("category")
    print(f"OR-Bench Unique Categories: {or_bench_unique_categories}")

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

    print(
        f"OR-Bench Hard Batches: {len(or_bench_hard_dataloader)} | OR-Bench Toxic Batches: {len(or_bench_toxic_dataloader)}"
    )

    return {
        "or_bench_hard_dataloader": or_bench_hard_dataloader,
        "or_bench_toxic_dataloader": or_bench_toxic_dataloader,
    }
