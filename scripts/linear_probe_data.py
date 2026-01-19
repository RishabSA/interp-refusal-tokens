import os
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer

from scripts.activation_caching import cache_hooked_activations_before_pad


class ActivationDataset(Dataset):
    """Stores cached activations used for training the linear probe.

    Args:
        Dataset (_type_)
    """

    def __init__(self, X, y):
        # X shape: (N, d_model)
        # y shape: (N) (0 - benign / 1 - harmful)

        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def get_probe_training_activations(
    hooked_model: HookedTransformer,
    harmful_probe_dataloader: DataLoader,
    benign_probe_dataloader: DataLoader,
    layer: int = 18,
    activation_name: str = "resid_post",
    prompt_seq_append: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    batch_size: int = 512,
    val_split: float = 0.2,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[DataLoader, DataLoader]:
    """Given dataloaders containg harmful and benign prompts, caches activations at the specific position in the hooked model and saves them in new dataloaders used for training and validating the linear probe.

    Args:
        hooked_model (HookedTransformer)
        harmful_probe_dataloader (DataLoader)
        benign_probe_dataloader (DataLoader)
        layer (int, optional). Defaults to 18.
        activation_name (str, optional). Defaults to "resid_post".
        prompt_seq_append (str, optional). Defaults to "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".
        batch_size (int, optional). Defaults to 512.
        val_split (float, optional). Defaults to 0.2.
        device (torch.device, optional). Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        tuple[DataLoader, DataLoader]: Dataloaders containing activations for training and validating the linear probe.
    """
    hooked_model.to(device).eval()

    harmful_probe_activations, _ = cache_hooked_activations_before_pad(
        hooked_model=hooked_model,
        iterator=harmful_probe_dataloader,
        activation_name=activation_name,
        layer=layer,
        prompt_seq_append=prompt_seq_append,
        device=device,
    )  # (N_h, d_model)

    benign_probe_activations, _ = cache_hooked_activations_before_pad(
        hooked_model=hooked_model,
        iterator=benign_probe_dataloader,
        activation_name=activation_name,
        layer=layer,
        prompt_seq_append=prompt_seq_append,
        device=device,
    )  # (N_b, d_model)

    print(
        f"Harmful probe activations data has a shape of {harmful_probe_activations.shape}"
    )
    print(
        f"Benign probe activations data has a shape of {benign_probe_activations.shape}"
    )

    harmful_probe_activations = harmful_probe_activations.to(dtype=torch.float32)
    benign_probe_activations = benign_probe_activations.to(dtype=torch.float32)

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
    prompt_seq_append: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    batch_size: int = 512,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> DataLoader:
    """Given dataloaders containg harmful and benign prompts, caches activations at the specific position in the hooked model and saves them in new dataloaders used for testing the linear probe.

    Args:
        hooked_model (HookedTransformer)
        harmful_probe_dataloader (DataLoader)
        benign_probe_dataloader (DataLoader)
        layer (int, optional). Defaults to 18.
        activation_name (str, optional). Defaults to "resid_post".
        prompt_seq_append (str, optional). Defaults to "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".
        batch_size (int, optional). Defaults to 512.
        device (torch.device, optional). Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        DataLoader: Dataloader containing activations for testing the linear probe.
    """
    hooked_model.to(device).eval()

    harmful_probe_activations, _ = cache_hooked_activations_before_pad(
        hooked_model=hooked_model,
        iterator=harmful_probe_dataloader,
        activation_name=activation_name,
        layer=layer,
        prompt_seq_append=prompt_seq_append,
        device=device,
    )  # (N_h, d_model)

    benign_probe_activations, _ = cache_hooked_activations_before_pad(
        hooked_model=hooked_model,
        iterator=benign_probe_dataloader,
        activation_name=activation_name,
        layer=layer,
        prompt_seq_append=prompt_seq_append,
        device=device,
    )  # (N_b, d_model)

    print(
        f"Harmful probe activations data has a shape of {harmful_probe_activations.shape}"
    )
    print(
        f"Benign probe activations data has a shape of {benign_probe_activations.shape}"
    )

    harmful_probe_activations = harmful_probe_activations.to(dtype=torch.float32)
    benign_probe_activations = benign_probe_activations.to(dtype=torch.float32)

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
