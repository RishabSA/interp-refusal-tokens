import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)


def project_activations_and_evaluate_clusters(
    activations_dict: dict[str, torch.Tensor],
    should_compute_cluster_metrics: bool = True,
    tsne_perplexity: int = 5,
    layer: int = 16,
    activation_name: str = "resid_post",
    desc: str = "2D Projection of Clustered Residual-Stream Activations",
) -> tuple:
    """Projects activations in the 4096-dimensional space into 2-dimensions with PCA and t-SNE.

    Also, computes Silhouette Score and Davies-Bouldin Score clustering metrics.

    Args:
        activations_dict (dict[str, torch.Tensor]): High-dimensional activations separated by category
        should_compute_cluster_metrics (bool, optional). Defaults to True.
        tsne_perplexity (int, optional). Defaults to 5.
        layer (int, optional): Layer to hook at. Defaults to 16.
        activation_name (str, optional): Position in the layer to hook at. Defaults to "resid_post".
        desc (str, optional): Decription displayed as part of the title on the PCA and t-SNE plots. Defaults to "2D Projection of Clustered Residual-Stream Activations".

    Returns:
        tuple: Returns the PCA model, PCA projections, t-SNE model, and t-SNE projections. Also returns the cluster centroids, Silhouette Score, and Davies-Bouldin Score clustering metrics if should_compute_cluster_metrics is True.
    """
    numpy_activations_dict = {}
    for category, activations in activations_dict.items():
        numpy_activations = activations.detach().cpu().numpy()

        if numpy_activations.ndim == 1:
            numpy_activations = numpy_activations[None, :]

        numpy_activations_dict[category] = numpy_activations

    # Stack all activations and build labels
    X = np.vstack(list(numpy_activations_dict.values()))  # (N * 5, 4096)
    labels = []

    for category, activations in numpy_activations_dict.items():
        labels.extend([category] * activations.shape[0])

    labels = np.array(labels)

    # PCA
    pca = PCA(n_components=2, random_state=0)
    pca_projection = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    for category in numpy_activations_dict:
        mask = labels == category
        plt.scatter(
            pca_projection[mask, 0], pca_projection[mask, 1], label=category, alpha=0.7
        )

    plt.title(f"PCA - {desc}")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    plt.savefig(f"PCA_{layer}_{activation_name}.png")
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=0)
    tsne_projection = tsne.fit_transform(X)

    plt.figure(figsize=(6, 5))
    for category in numpy_activations_dict:
        mask = labels == category
        plt.scatter(
            tsne_projection[mask, 0],
            tsne_projection[mask, 1],
            label=category,
            alpha=0.7,
        )

    plt.title(f"t-SNE - {desc}")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    plt.savefig(f"t-SNE_{layer}_{activation_name}.png")
    plt.show()

    if should_compute_cluster_metrics:
        # Compute cluster centroids
        centroids = {
            category: steering.mean(axis=0)
            for category, steering in numpy_activations_dict.items()
        }

        # Clustering Metrics
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)

        print("Silhouette Score Interpretation - Range: (-1, 1)")
        print("Value ~1: clusters are dense and well-separated")
        print("Value ~0: overlapping clusters")
        print("Value <0: some data points may have been assigned to the wrong cluster")

        print("\n")

        print("Davies-Bouldin Score Interpretation - Range: (0, infinity)")
        print("Lower score: better clustering")
        print("Value 0: ideal minimum")

        print("\n\n")

        print(f"Silhouette Score: {sil_score}")
        print(f"Davies-Bouldin Score: {db_score}")

        return (
            pca,
            pca_projection,
            tsne,
            tsne_projection,
            centroids,
            sil_score,
            db_score,
        )

    return pca, pca_projection, tsne, tsne_projection


def compute_inter_steering_vector_cosine_sims(
    steering_vectors: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Computes cosine similarities between categorical steering vectors.

    Args:
        steering_vectors (dict[str, torch.Tensor])

    Returns:
        dict[str, dict[str, float]]: Computed cosine similarities
    """
    steering_vector_cosine_similarities = {}

    for category_1, steering_vector_1 in steering_vectors.items():
        steering_vector_cosine_similarities[category_1] = {}

        for category_2, steering_vector_2 in steering_vectors.items():
            steering_cosine_sim = F.cosine_similarity(
                steering_vector_1, steering_vector_2, dim=-1, eps=1e-8
            )

            steering_vector_cosine_similarities[category_1][category_2] = float(
                steering_cosine_sim.detach().cpu()
            )

    return steering_vector_cosine_similarities


def plot_inter_steering_vector_cosine_sims(
    steering_vector_cosine_similarities: dict[str, dict[str, float]],
    title: str = "Inter-Steering Vector Cosine Similarities",
) -> None:
    """Plots the cosine similarities between categorical steering vectors.

    Args:
        steering_vector_cosine_similarities (dict[str, dict[str, float]])
        title (str, optional). Defaults to "Inter-Steering Vector Cosine Similarities".
    """
    row_labels = list(steering_vector_cosine_similarities.keys())
    col_labels = list(next(iter(steering_vector_cosine_similarities.values())).keys())

    N = len(row_labels)

    M = np.zeros((N, N), dtype=np.float32)
    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            cosine_sim = steering_vector_cosine_similarities[row][col]

            M[i, j] = cosine_sim

    fig, ax = plt.subplots(figsize=(max(6, N * 1.05), max(6, N * 1.05)))

    im = ax.imshow(M, cmap="RdBu_r", interpolation="nearest", vmin=0.0, vmax=1.0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add the cosine similarity values to each cell
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cm = plt.get_cmap("RdBu_r")

    for i in range(N):
        for j in range(N):
            ax.text(
                j,
                i,
                f"{M[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    # Ticks & labels
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    # Gridlines at cell boundaries
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title)
    fig.tight_layout()

    plt.savefig("steering_vector_cos_sim.png")
    plt.show()


def get_topk_steering_vector(
    vector: torch.Tensor, K: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the top-K features' values and indices given a vector

    Args:
        vector (torch.Tensor)
        K (int, optional). Defaults to 10.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: top-K values and top-K indices
    """
    vals, idxs = torch.topk(vector.abs(), K)
    return vals, idxs


def print_topk_refusal_features(steering_vectors: dict[str, torch.Tensor]) -> None:
    """Prints the top-K features' values and indices given steering vectors

    Args:
        steering_vectors (dict[str, torch.Tensor])
    """
    for category, steering_vector in steering_vectors.items():
        steering_vector_vals, steering_vector_idxs = get_topk_steering_vector(
            steering_vector, K=10
        )

        print(f"{category} categorical steering vector has top-K:")
        print(steering_vector_vals)
        print(steering_vector_idxs)


def plot_grouped_steering_vector_features(
    steering_vectors: dict[str, torch.Tensor], feature_ids: list[int]
) -> None:
    """Plots the values of the given features for all of the steering vectors

    Args:
        steering_vectors (dict[str, torch.Tensor])
        feature_ids (list[int])
    """
    items = list(steering_vectors.items())
    categories = [category for category, steering_vector in items]

    len_categories = len(categories)
    len_features = len(feature_ids)

    vals = np.array(
        [
            [float(steering_vector[f].abs().detach().cpu()) for f in feature_ids]
            for category, steering_vector in items
        ]
    )

    x = np.arange(len_categories)
    width = min(0.22, 0.8 / len_features)
    offsets = (np.arange(len_features) - (len_features - 1) / 2.0) * width

    cmap = plt.colormaps.get("tab10")
    colors = [cmap(i) for i in range(len_features)]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for j in range(len_features):
        label = f"Feature {feature_ids[j]}"
        ax.bar(
            x + offsets[j],
            vals[:, j],
            width=width,
            label=label,
            color=colors[j],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x, categories, rotation=30, ha="right")

    ax.set_ylabel("Feature Absolute Value")
    ax.set_title("Top Steering Vector Values")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(ncol=len_features, frameon=False)

    ax.set_ylim(0, 0.5)

    plt.savefig(f"steering_vectors_grouped.png")
    plt.show()


def plot_steering_vector_feature(
    steering_vectors: dict[str, torch.Tensor], feature_id: int
) -> None:
    """Plots the values of the given feature for all of the steering vectors

    Args:
        steering_vectors (dict[str, torch.Tensor])
        feature_id (int)
    """
    items = list(steering_vectors.items())

    categories = [category for category, vector in items]
    values = [
        float(vector[feature_id].abs().detach().cpu()) for category, vector in items
    ]

    cmap = plt.colormaps.get("tab20")
    bar_colors = [cmap(i % cmap.N) for i in range(len(categories))]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.bar(
        categories,
        values,
        width=0.25,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Steering Vectors")
    ax.set_ylabel("Absolute Values")
    ax.set_title(f"Steering Vector Values for Feature #{feature_id}")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.savefig(f"steering_vectors_feature_{str(feature_id)}.png")
    plt.show()
