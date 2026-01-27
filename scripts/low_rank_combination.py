import torch


def compute_covariance_sigma(
    activations: torch.Tensor,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor:
    activations = activations.to(device, dtype=torch.float32)

    activations_centered = activations - activations.mean(dim=0, keepdim=True)

    covariance_sigma = (
        activations_centered.T @ activations_centered
    ) / activations_centered.shape[0]

    return covariance_sigma  # covariance_sigma shape: (d_model, d_model)


def compute_steering_basis(
    steering_vectors: dict[str, torch.Tensor],
    covariance_sigma: torch.Tensor,
    eps: float = 1e-4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # covariance_sigma shape: (d_model, d_model)

    # W = U \Sigma^{-1/2} U^\top

    U, S, V = torch.linalg.svd(
        covariance_sigma
        + eps * torch.eye(covariance_sigma.shape[0], device=covariance_sigma.device)
    )

    # Used to remove activation-space aniostropy, so that all steering directions are equal
    whitening_matrix = torch.matmul(
        (U * (S.clamp_min(eps).rsqrt())), U.T
    )  # shape: (d_model, d_model)

    # Stack steering vectors
    categories = list(steering_vectors.keys())
    steering_vector_stacked = torch.stack(
        [steering_vectors[category] for category in categories], dim=1
    ).to(
        device, dtype=torch.float32
    )  # shape: (d_model, 5)

    whitened_vectors = torch.matmul(
        whitening_matrix, steering_vector_stacked
    )  # shape: (d_model, 5)

    # Orthonormalize with QR Decomposition (whitened_vectors = QR)
    Q, R = torch.linalg.qr(
        whitened_vectors, mode="reduced"
    )  # Q shape: (d_model, 5), R shape: (5, 5)

    # Q is the orthonormal steering basis where each category direction is uncorrelated

    return Q, R, whitening_matrix
