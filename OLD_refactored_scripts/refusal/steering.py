from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from .config import FIGURES_DIR


def compute_steering_vectors(
    mean_benign_dict: Dict[str, torch.Tensor],
    mean_harmful_dict: Dict[str, torch.Tensor],
    should_filter_shared: bool = False,
    K: int | None = 100,
    tau: float | None = 1e-3,
) -> Dict[str, torch.Tensor]:
    """Compute sparse, normalized steering vectors per category.

    Mirrors the notebook logic: optional tau thresholding, optional shared-feature filtering,
    top-K sparsification, and L2 normalization.
    """
    steering_vectors: Dict[str, torch.Tensor] = {}

    def get_topk_sparse_vector(vector: torch.Tensor, K: int) -> torch.Tensor:
        vals, idxs = torch.topk(vector.abs(), K)
        mask = torch.zeros_like(vector)
        mask[idxs] = 1.0
        return vector * mask

    def normalize_steering_vector(vector: torch.Tensor) -> torch.Tensor:
        norm = vector.norm()
        return vector / norm if norm > 0 else vector

    for (harmful_category, mean_harmful), (benign_category, mean_benign) in zip(
        mean_harmful_dict.items(), mean_benign_dict.items()
    ):
        if harmful_category != benign_category:
            print("Error: harmful and benign are not the same category")
            break

        if tau is not None:
            benign_mask = mean_benign.abs() >= tau
            harmful_mask = mean_harmful.abs() >= tau
        else:
            benign_mask = torch.ones_like(mean_benign, dtype=torch.bool)
            harmful_mask = torch.ones_like(mean_harmful, dtype=torch.bool)

        if should_filter_shared:
            harmful_mask = harmful_mask & (~benign_mask)

        if tau is not None or should_filter_shared:
            benign_mask_f = benign_mask.float()
            harmful_mask_f = harmful_mask.float()
            mean_benign = mean_benign * benign_mask_f
            mean_harmful = mean_harmful * harmful_mask_f

        steering_harmful = mean_harmful - mean_benign
        if K is not None:
            steering_harmful = get_topk_sparse_vector(steering_harmful, K)

        steering_harmful = normalize_steering_vector(steering_harmful)

        sim = F.cosine_similarity(mean_harmful, mean_benign, dim=-1, eps=1e-8)
        print(
            f"Harmful category {harmful_category} has cosine similarity of {sim} with benign"
        )

        steering_vectors[harmful_category] = steering_harmful

    return steering_vectors


def compute_caa_steering_vectors(
    benign_dict: Dict[str, torch.Tensor],
    harmful_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute category-average activation (CAA) steering vectors: mean(harmful - benign)."""
    steering_vectors: Dict[str, torch.Tensor] = {}

    for (harmful_category, harmful), (benign_category, benign) in zip(
        harmful_dict.items(), benign_dict.items()
    ):
        if harmful_category != benign_category:
            print("Error: harmful and benign are not the same category")
            break

        steering_harmful = (harmful - benign).mean(dim=0)
        steering_vectors[harmful_category] = steering_harmful

    return steering_vectors


def evaluate_vector_clusters(
    steering_vectors_dict: Dict[str, torch.Tensor],
    compute_cluster_metrics: bool = True,
    tsne_perplexity: int = 5,
    layer: int = 9,
    activation_name: str = "resid_post",
    desc: str = "",
    out_dir=FIGURES_DIR,
) -> Tuple[Any, np.ndarray, Any, np.ndarray] | Tuple[Any, np.ndarray, Any, np.ndarray, Dict[str, np.ndarray], float, float, float]:
    """PCA and t-SNE projections with optional clustering metrics. Saves figures to out_dir."""
    processed: Dict[str, np.ndarray] = {}
    for category, vec in steering_vectors_dict.items():
        arr = vec.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        processed[category] = arr

    X = np.vstack(list(processed.values()))
    labels: list[str] = []
    for category, arr in processed.items():
        labels.extend([category] * arr.shape[0])
    labels_np = np.array(labels)

    # PCA
    pca = PCA(n_components=2, random_state=0)
    pca_projection = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    for category in processed:
        mask = labels_np == category
        plt.scatter(pca_projection[mask, 0], pca_projection[mask, 1], label=category, alpha=0.7)
    plt.title(f"PCA - {desc}")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    out_path = out_dir / f"PCA - {desc} - {layer} - {activation_name}.png"
    plt.savefig(out_path)
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=0)
    tsne_projection = tsne.fit_transform(X)

    plt.figure(figsize=(6, 5))
    for category in processed:
        mask = labels_np == category
        plt.scatter(tsne_projection[mask, 0], tsne_projection[mask, 1], label=category, alpha=0.7)
    plt.title(f"t-SNE - {desc}")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    out_path = out_dir / f"t-SNE - {desc} - {layer} - {activation_name}.png"
    plt.savefig(out_path)
    plt.close()

    if compute_cluster_metrics:
        centroids = {category: arr.mean(axis=0) for category, arr in processed.items()}
        sil = silhouette_score(X, labels_np)
        db = davies_bouldin_score(X, labels_np)
        ch = calinski_harabasz_score(X, labels_np)
        print(f"Silhouette Score: {sil}")
        print(f"Davies-Bouldin Score: {db}")
        print(f"Calinski-Harabasz Score: {ch}")
        return pca, pca_projection, tsne, tsne_projection, centroids, sil, db, ch

    return pca, pca_projection, tsne, tsne_projection


def compute_steering_vector_cosine_similarities(
    steering_vectors: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    """Pairwise cosine similarities between steering vectors by category."""
    sims: Dict[str, Dict[str, float]] = {}
    for category_1, v1 in steering_vectors.items():
        sims[category_1] = {}
        for category_2, v2 in steering_vectors.items():
            cos = F.cosine_similarity(v1, v2, dim=-1, eps=1e-8)
            sims[category_1][category_2] = float(cos.detach().cpu())
    return sims


def compute_binary_steering_vectors(
    mean_benign: torch.Tensor,
    mean_harmful: torch.Tensor,
    should_filter_shared: bool = True,
    K: int | None = 50,
    tau: float | None = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a binary steering vector between a pair of means with optional filtering and sparsification.

    Returns (mean_benign_after_mask, steering_vector_normalized)
    """
    if tau is not None:
        benign_mask = mean_benign.abs() >= tau
        harmful_mask = mean_harmful.abs() >= tau
    else:
        benign_mask = torch.ones_like(mean_benign, dtype=torch.bool)
        harmful_mask = torch.ones_like(mean_harmful, dtype=torch.bool)

    if should_filter_shared:
        harmful_mask = harmful_mask & (~benign_mask)

    if tau is not None or should_filter_shared:
        benign_mask_f = benign_mask.float()
        harmful_mask_f = harmful_mask.float()
        mean_benign = mean_benign * benign_mask_f
        mean_harmful = mean_harmful * harmful_mask_f

    steering_harmful = mean_harmful - mean_benign

    def get_topk_sparse_vector(vector: torch.Tensor, K: int) -> torch.Tensor:
        vals, idxs = torch.topk(vector.abs(), K)
        mask = torch.zeros_like(vector)
        mask[idxs] = 1.0
        return vector * mask

    if K is not None:
        steering_harmful = get_topk_sparse_vector(steering_harmful, K)

    def normalize_sparse_vector(vector: torch.Tensor) -> torch.Tensor:
        norm = vector.norm()
        return vector / norm if norm > 0 else vector

    steering_harmful = normalize_sparse_vector(steering_harmful)
    return mean_benign, steering_harmful


def get_topk_steering_vector(vector: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (top_values, top_indices) for a steering vector."""
    vals, idxs = torch.topk(vector.abs(), K)
    return vals, idxs
