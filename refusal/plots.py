from __future__ import annotations

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from .config import FIGURES_DIR


def plot_steering_vector_cosine_sims(
    steering_vector_cosine_similarities: Dict[str, Dict[str, float]],
    layer: int = 9,
    activation_name: str = "resid_post",
    title: str = "Steering Vector Cosine Similarities",
    out_dir=FIGURES_DIR,
) -> None:
    """Plot a heatmap of cosine similarities between category steering vectors and save it to figures/.

    Mirrors the notebook implementation and writes the figure to
    figures/steering_vector_cos_sim - {layer} - {activation_name}.png
    """
    row_labels = list(steering_vector_cosine_similarities.keys())
    col_labels = list(next(iter(steering_vector_cosine_similarities.values())).keys())

    N = len(row_labels)
    M = np.zeros((N, N), dtype=np.float32)

    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            M[i, j] = steering_vector_cosine_similarities[row][col]

    fig, ax = plt.subplots(figsize=(max(6, N * 1.05), max(6, N * 1.05)))
    im = ax.imshow(M, cmap="RdBu_r", interpolation="nearest", vmin=-1.0, vmax=1.0)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="black", fontsize=10)

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title)
    fig.tight_layout()

    out_path = out_dir / f"steering_vector_cos_sim - {layer} - {activation_name}.png"
    plt.savefig(out_path)
    plt.close()


def plot_steering_vectors_grouped(
    steering_vectors: Dict[str, "np.ndarray | any"], feature_ids: List[int], out_path: str | None = None
) -> None:
    """Bar plot grouped by category showing absolute values at selected feature indices.

    steering_vectors: mapping category->tensor-like (1D) of d_model features.
    feature_ids: list of feature indices to display.
    """
    items = list(steering_vectors.items())
    categories = [category for category, _ in items]

    len_categories = len(categories)
    len_features = len(feature_ids)

    # Build matrix of abs values [num_categories, num_features]
    vals = np.array([
        [float(steering_vector[f].abs().detach().cpu()) for f in feature_ids]
        for _, steering_vector in items
    ])

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
    ax.legend(ncol=min(5, len_features), frameon=False)
    ax.set_ylim(0, max(0.5, float(vals.max()) * 1.1))

    if out_path is None:
        out_path = str(FIGURES_DIR / "steering_vectors_grouped.png")
    plt.savefig(out_path)
    plt.close()


def plot_steering_vector_feature(
    steering_vectors: Dict[str, "np.ndarray | any"], feature_id: int, out_path: str | None = None
) -> None:
    """Bar plot of absolute value for a single feature across categories."""
    items = list(steering_vectors.items())
    categories = [category for category, _ in items]
    values = [float(vector[feature_id].abs().detach().cpu()) for _, vector in items]

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

    if out_path is None:
        out_path = str(FIGURES_DIR / f"steering_vectors_feature_{feature_id}.png")
    plt.savefig(out_path)
    plt.close()
