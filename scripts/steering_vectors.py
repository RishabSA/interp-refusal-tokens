import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def compute_contrastive_steering_vectors(
    benign_activations: dict[str, torch.Tensor],
    harmful_activations: dict[str, torch.Tensor],
    K: int | None = 100,
    tau: float | None = 1e-3,
) -> dict[str, torch.Tensor]:
    steering_vectors = {}

    # Enforce sparsity by only keeping the top-K values and setting the others to 0
    def get_topk_sparse_vector(vector, K):
        vals, idxs = torch.topk(vector.abs(), K)
        mask = torch.zeros_like(vector)
        mask[idxs] = 1.0

        return vector * mask

    # L2 Normalization
    def l2_norm(vector, eps=1e-8):
        return vector / (vector.norm(dim=-1, keepdim=True) + eps)

    for (
        (harmful_category, harmful),
        (benign_category, benign),
    ) in zip(
        harmful_activations.items(),
        benign_activations.items(),
    ):
        if harmful_category != benign_category:
            print("Error: harmful and benign are not the same category")
            break

        steering_harmful = (harmful - benign).mean(dim=0)

        if tau is not None:
            # Filter out inactive features with values < tau
            # boolean mask of shape (d_model)

            tau_mask = steering_harmful.abs() >= tau

            # Convert the bool masks to float masks to multiply
            tau_mask = tau_mask.float()

            # Apply the masks to each of the mean features
            steering_harmful = steering_harmful * tau_mask

        if K is not None:
            steering_harmful = get_topk_sparse_vector(steering_harmful, K)

        steering_harmful = l2_norm(steering_harmful)

        steering_vectors[harmful_category] = steering_harmful

    return steering_vectors


def whiten_steering_vectors(
    steering_vectors: dict[str, torch.Tensor],
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    categories = list(steering_vectors.keys())
    vectors = torch.stack(
        [steering_vectors[category].to(torch.float32) for category in categories],
        dim=0,
    )  # (5, d_model)

    # Mean-center across categories (so each dimension has mean 0 across vectors)
    vectors_centered = vectors - vectors.mean(dim=0, keepdim=True)  # (5, d)

    # Covariance over dimensions: Cov[i, j] = cov(v_i, v_j) across d_model components
    # This gives an 5 x 5 matrix that captures how similar the vectors are.
    covariance_matrix = (vectors_centered @ vectors_centered.T) / max(
        vectors_centered.shape[1] - 1, 1
    )  # (5, 5)

    # Add epsilon to the diagonal (identity matrix) for stability
    covariance_matrix = covariance_matrix + eps * torch.eye(
        vectors_centered.shape[0],
        device=covariance_matrix.device,
        dtype=covariance_matrix.dtype,
    )

    # Eigendecompose the covariance matrix
    # Cov = U diag(w) U^T
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Cov^{-1/2} = U diag(1/sqrt(w)) U^T
    eigenvalues_clamped = eigenvalues.clamp_min(eps)
    covariance_matrix_inv_sqrt = (
        eigenvectors @ torch.diag(eigenvalues_clamped.rsqrt()) @ eigenvectors.T
    )  # (5, 5)

    # Apply whitening transform in "vector space"
    # Vw[i] is a linear combo of original vectors but now decorrelated
    vectors_whitened = covariance_matrix_inv_sqrt @ vectors_centered  # (5, d)

    # L2-normalization
    vectors_whitened = vectors_whitened / (
        vectors_whitened.norm(dim=1, keepdim=True) + eps
    )

    return {category: vectors_whitened[i] for i, category in enumerate(categories)}


def compute_old_steering_vectors(
    mean_benign_activations: dict[str, torch.Tensor],
    mean_harmful_activations: dict[str, torch.Tensor],
    K: int | None = None,
    tau: float | None = None,
    should_filter_shared: bool = False,
) -> dict[str, torch.Tensor]:
    steering_vectors = {}

    # Enforce sparsity by only keeping the top-K values and setting the others to 0
    def get_topk_sparse_vector(vector, K):
        vals, idxs = torch.topk(vector.abs(), K)
        mask = torch.zeros_like(vector)
        mask[idxs] = 1.0

        return vector * mask

    # Normalize the steering vectors to have magnitude = 1
    def normalize_steering_vector(vector):
        norm = vector.norm()

        # Prevent division by 0 error
        return vector / norm if norm > 0 else vector

    for (
        (harmful_category, mean_harmful),
        (benign_category, mean_benign),
    ) in zip(
        mean_harmful_activations.items(),
        mean_benign_activations.items(),
    ):
        if harmful_category != benign_category:
            print("Error: harmful and benign are not the same category")
            break

        if tau is not None:
            # Filter out inactive features with values < tau
            # boolean mask of shape (d_model)

            benign_mask = mean_benign.abs() >= tau
            harmful_mask = mean_harmful.abs() >= tau

        if should_filter_shared:
            # Filter out features that are shared between the mean category-specific harmful activations and the benign activations to isolate behavior-specific components
            harmful_mask = harmful_mask & (~benign_mask)

        if tau is not None or should_filter_shared:
            # Convert the bool masks to float masks to multiply
            benign_mask = benign_mask.float()
            harmful_mask = harmful_mask.float()

            # Apply the masks to each of the mean features
            mean_benign = mean_benign * benign_mask
            mean_harmful = mean_harmful * harmful_mask

        # Subtract the mean benign activations from the mean category-specific harmful activations to get the steering vector for the specific category

        steering_harmful = mean_harmful - mean_benign

        if K is not None:
            steering_harmful = get_topk_sparse_vector(steering_harmful, K)

        steering_harmful = normalize_steering_vector(steering_harmful)

        steering_harmful_cosine_sim = F.cosine_similarity(
            mean_harmful, mean_benign, dim=-1, eps=1e-8
        )
        print(
            f"Harmful category {harmful_category} has cosine similarity of {steering_harmful_cosine_sim} with benign"
        )

        steering_vectors[harmful_category] = steering_harmful

    return steering_vectors
