import os
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset


def split_dataloader_by_category(
    iterator: DataLoader, category_field: str = "category"
) -> dict[str, DataLoader]:
    """Given an iterator, splits the iterator into separate iterators by the given field.

    Args:
        iterator (DataLoader)
        category_field (str, optional). Defaults to "category".

    Returns:
        dict[str, DataLoader]
    """
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


def load_coconot_test_data(batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    """Loads the test splits for COCONot Orig and Contrast from HuggingFace and returns dataloaders for each

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        tuple[DataLoader, DataLoader]
    """
    # COCONot
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

    coconot_unique_categories = coconot_orig["test"].unique("category")
    print(f"COCONot Unique Categories: {coconot_unique_categories}")

    coconot_orig_test_dataloader = DataLoader(
        coconot_orig["test"],  # 1k items
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    coconot_contrast_test_dataloader = DataLoader(
        coconot_contrast["test"],  # 379 items
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    print(
        f"COCONot Orig Test Batches: {len(coconot_orig_test_dataloader)} | COCONot Contrast Test Batches: {len(coconot_contrast_test_dataloader)}"
    )

    return coconot_orig_test_dataloader, coconot_contrast_test_dataloader


def load_wildguard_test_data(batch_size: int = 4) -> DataLoader:
    """Loads the test split for WildGuard from HuggingFace and returns a dataloader

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        DataLoader
    """
    # WildGuard
    wildguard_test = load_dataset(
        "allenai/wildguardmix", "wildguardtest"
    )  # 1.73k items

    wildguard_test = wildguard_test.rename_column("subcategory", "category")

    wildguard_unique_categories = wildguard_test["test"].unique("category")
    print(f"WildGuard Unique Categories: {wildguard_unique_categories}")

    def wildguard_collate(batch):
        return {
            "prompt": [example["prompt"] for example in batch],
            "category": [example["category"] for example in batch],
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

    return wildguard_test_dataloader


def load_wildjailbreak_test_data(batch_size: int = 4) -> DataLoader:
    """Loads the test split for WildJailbreak from HuggingFace and returns a dataloader

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        DataLoader
    """
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

    return wildjailbreak_eval_dataloader


def load_or_bench_test_data(batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    """Loads the test splits for OR-Bench Hard and Toxic from HuggingFace and returns dataloaders for each

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        tuple[DataLoader, DataLoader]
    """
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

    return or_bench_hard_dataloader, or_bench_toxic_dataloader


def load_xstest_test_data(batch_size: int = 4) -> DataLoader:
    """Loads the test split for XSTest from HuggingFace and returns a dataloader for safe and unsafe prompts.

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        DataLoader
    """
    # XSTest
    xstest = load_dataset("walledai/XSTest")  # 450 items

    xstest = xstest.rename_column("label", "category")

    xstest_unique_categories = xstest["test"].unique("category")
    print(f"XSTest Unique Categories: {xstest_unique_categories}")

    def xstest_collate(batch):
        return {
            "prompt": [example["prompt"] for example in batch],
            "category": [example["category"] for example in batch],
        }

    xstest_dataloader = DataLoader(
        xstest["test"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=xstest_collate,
    )

    print(f"XSTest Batches: {len(xstest_dataloader)}")

    return xstest_dataloader


def load_sorry_bench_test_data(batch_size: int = 4) -> DataLoader:
    """Loads the SORRY-Bench dataset from HuggingFace and returns a dataloader.

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        DataLoader
    """
    # SORRY-Bench
    sorry_bench = load_dataset("sorry-bench/sorry-bench-202503")

    sorry_bench = sorry_bench.rename_column("turns", "prompt")

    def convert_turn_to_string(example):
        example["prompt"] = example["prompt"][0]
        return example

    sorry_bench = sorry_bench.map(convert_turn_to_string)

    sorry_bench_train = sorry_bench["train"].filter(
        lambda x: x["prompt_style"] == "base"
    )  # 440 items

    def sorry_bench_collate(batch):
        return {
            "prompt": [example["prompt"] for example in batch],
            "category": [example["category"] for example in batch],
        }

    sorry_bench_dataloader = DataLoader(
        sorry_bench_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=sorry_bench_collate,
    )

    print(f"SORRY-Bench Batches: {len(sorry_bench_dataloader)}")

    return sorry_bench_dataloader


def load_adv_bench_test_data(batch_size: int = 4) -> DataLoader:
    """Loads the AdvBench dataset from HuggingFace and returns a dataloader.

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        DataLoader
    """
    # AdvBench
    adv_bench = load_dataset("walledai/AdvBench")

    def adv_bench_collate(batch):
        return {
            "prompt": [example["prompt"] for example in batch],
            "category": ["None" for _ in batch],
        }

    adv_bench_dataloader = DataLoader(
        adv_bench["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=adv_bench_collate,
    )

    print(f"AdvBench Batches: {len(adv_bench_dataloader)}")

    return adv_bench_dataloader


def load_harmful_qa_test_data(batch_size: int = 4) -> DataLoader:
    """Loads the HarmfulQA dataset from HuggingFace and returns a dataloader.

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        DataLoader
    """
    # HarmfulQA
    harmful_qa = load_dataset("declare-lab/HarmfulQA")  # 1.96k items

    harmful_qa = harmful_qa.rename_column("question", "prompt")
    harmful_qa = harmful_qa.rename_column("topic", "category")

    def harmful_qa_collate(batch):
        return {
            "prompt": [example["prompt"] for example in batch],
            "category": [example["category"] for example in batch],
        }

    harmful_qa_dataloader = DataLoader(
        harmful_qa["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=harmful_qa_collate,
    )

    print(f"HarmfulQA Batches: {len(harmful_qa_dataloader)}")

    return harmful_qa_dataloader


def load_do_not_answer_test_data(batch_size: int = 4) -> DataLoader:
    """Loads the Do-Not-Answer dataset from HuggingFace and returns a dataloader.

    Args:
        batch_size (int, optional). Defaults to 4.

    Returns:
        DataLoader
    """
    # Do-Not-Answer
    do_not_answer = load_dataset("LibrAI/do-not-answer")  # 939 items

    do_not_answer = do_not_answer.rename_column("question", "prompt")
    do_not_answer = do_not_answer.rename_column("types_of_harm", "category")

    def do_not_answer_collate(batch):
        return {
            "prompt": [example["prompt"] for example in batch],
            "category": [example["category"] for example in batch],
        }

    do_not_answer_dataloader = DataLoader(
        do_not_answer["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=do_not_answer_collate,
    )

    print(f"Do-Not-Answer Batches: {len(do_not_answer_dataloader)}")

    return do_not_answer_dataloader
