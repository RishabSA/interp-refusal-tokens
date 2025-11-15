import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def evaluate_vector_clusters(
    steering_vectors_dict,
    compute_cluster_metrics: bool = True,
    tsne_perplexity: int = 5,
    layer: int = 16,
    activation_name: str = "resid_post",
    desc: str = "",
):
    processed = {}
    for category, vec in steering_vectors_dict.items():
        arr = vec.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        processed[category] = arr

    # Stack all steering vectors and build labels
    X = np.vstack(list(processed.values()))
    labels = []

    for category, steering in processed.items():
        labels.extend([category] * steering.shape[0])

    labels = np.array(labels)

    # PCA
    pca = PCA(n_components=2, random_state=0)
    pca_projection = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    for category in processed:
        mask = labels == category
        plt.scatter(
            pca_projection[mask, 0], pca_projection[mask, 1], label=category, alpha=0.7
        )

    plt.title(f"PCA - {desc}")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    plt.savefig(f"PCA - {desc} - {layer} - {activation_name}.png")
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=0)
    tsne_projection = tsne.fit_transform(X)

    plt.figure(figsize=(6, 5))
    for category in processed:
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

    plt.savefig(f"t-SNE - {desc} - {layer} - {activation_name}.png")
    plt.show()

    if compute_cluster_metrics:
        # Compute cluster centroids
        centroids = {
            category: steering.mean(axis=0) for category, steering in processed.items()
        }

        # Clustering Metrics
        sil_score = silhouette_score(X, labels)  # Higher score is better (-1 - +1)
        db_score = davies_bouldin_score(X, labels)  # Lower score is better (>= 0)
        ch_score = calinski_harabasz_score(X, labels)  # Higher score is better

        print(f"Silhouette Score: {sil_score}")
        print(f"Davies-Bouldin Score: {db_score}")
        print(f"Calinski-Harabasz Score: {ch_score}")

        return (
            pca,
            pca_projection,
            tsne,
            tsne_projection,
            centroids,
            sil_score,
            db_score,
            ch_score,
        )

    return pca, pca_projection, tsne, tsne_projection


def compute_steering_vector_cosine_similarities(steering_vectors):
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


def plot_steering_vector_cosine_sims(
    steering_vector_cosine_similarities,
    layer: int = 16,
    activation_name: str = "resid_post",
    title: str = "Steering Vector Cosine Similarities",
):
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

    plt.savefig(f"steering_vector_cos_sim - {layer} - {activation_name}.png")
    plt.show()


def get_topk_steering_vector(vector, K: int = 10):
    vals, idxs = torch.topk(vector.abs(), K)
    return vals, idxs


def print_topk_steering_features(steering_vectors_activations):
    for category, steering_vector in steering_vectors_activations.items():
        steering_vector_vals, steering_vector_idxs = get_topk_steering_vector(
            steering_vector, K=10
        )

        print(f"{category} categorical steering vector has top-K:")
        print(steering_vector_vals)
        print(steering_vector_idxs)


def plot_steering_vectors_grouped(steering_vectors, feature_ids):
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


def plot_steering_vector_feature(steering_vectors, feature_id: int):
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
