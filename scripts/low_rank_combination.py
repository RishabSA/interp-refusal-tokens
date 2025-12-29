import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


def get_low_rank_combination_data(
    batch_size: int = 4,
) -> DataLoader:
    # COCONot Dataset loading
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

    def prompt_category_collate(batch: list[dict]) -> dict[str, list[str]]:
        return {
            "prompt": [sample["prompt"] for sample in batch],
            "category": [sample.get("category") for sample in batch],
        }

    coconot_benign_dataloader = DataLoader(
        coconot_contrast["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=prompt_category_collate,
    )

    print(f"Benign prompts dataloader has {len(coconot_benign_dataloader)} batches")

    return coconot_benign_dataloader


def compute_covariance_sigma(activations: torch.Tensor) -> torch.Tensor:
    activations_centered = activations - activations.mean(dim=0, keepdim=True)

    covariance_sigma = (
        activations_centered.T @ activations_centered
    ) / activations_centered.shape[0]

    return covariance_sigma  # covariance_sigma shape: (d_model, d_model)


def compute_steering_basis(
    steering_vectors: dict[str, torch.Tensor],
    covariance_sigma: torch.Tensor,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # covariance_sigma shape: (d_model, d_model)

    # W = U \Sigma^{-1/2} U^\top

    U, S, V = torch.linalg.svd(
        covariance_sigma
        + eps * torch.eye(covariance_sigma.shape[0], device=covariance_sigma.device)
    )

    whitening_matrix = torch.matmul(
        (U * (S.clamp_min(eps).rsqrt())), U.T
    )  # shape: (d_model, d_model)

    # Stack steering vectors
    categories = list(steering_vectors.keys())
    steering_vector_stacked = torch.stack(
        [steering_vectors[category] for category in categories], dim=1
    )  # shape: (d_model, 5)

    whitened_vectors = torch.matmul(
        whitening_matrix, steering_vector_stacked
    )  # shape: (d_model, 5)

    # Orthonormalize with QR Decomposition (whitening_matrix = QR)
    Q, R = torch.linalg.qr(
        whitened_vectors, mode="reduced"
    )  # Q shape: (d_model, 5}, R shape: (5, 5)

    # Q is the orthonormal steering basis

    return Q, R, whitening_matrix
