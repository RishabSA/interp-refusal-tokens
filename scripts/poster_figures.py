import argparse
import os
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

output_dir = Path("papers/colm_2026_actionable_interp/poster/figures")
activation_files = {
    "Llama 3 8B Base": Path("saved_outputs/model_diffing/activations_18_resid_post_llama-base.pt"),
    "Refuse-Llama": Path("saved_outputs/model_diffing/activations_18_resid_post_refuse-llama.pt"),
}

# Figures are drawn at their printed size so font sizes here are the sizes on the poster
column_width = 10.0

ink = "#0b0b0b"
secondary_ink = "#52514e"
gridline = "#e1e0d9"
axis_line = "#c3c2b7"
good_region = "#0ca30c"

baseline_color = "#898781"
categorical_steering_color = "#e34948"
low_rank_color = "#2a78d6"

# Validated all-pairs for CVD and normal vision; same hue order as the paper's tab10 plots
category_colors = {
    "Requests with safety concerns": "#2a78d6",
    "Humanizing requests": "#eda100",
    "Incomplete requests": "#008300",
    "Unsupported requests": "#e87ba4",
    "Indeterminate requests": "#4a3aa7",
}

# Pooled averages from Table 1 of the paper (overleaf/sections/4_experimental_setup.tex)
over_refusal = {
    "Refuse-Llama": 17.08,
    "Refuse-Llama + Categorical Steering": 3.38,
    "Refuse-Llama + Low-Rank Combination": 8.15,
    "Llama 3 8B Instruct": 30.68,
    "Llama 3 8B Instruct + Low-Rank Combination": 27.94,
    "DeepSeek R1 Distill Llama": 56.88,
    "DeepSeek R1 Distill Llama + Low-Rank Combination": 50.60,
}
refusal = {
    "Refuse-Llama": 65.21,
    "Refuse-Llama + Categorical Steering": 79.38,
    "Refuse-Llama + Low-Rank Combination": 78.07,
    "Llama 3 8B Instruct": 66.76,
    "Llama 3 8B Instruct + Low-Rank Combination": 67.42,
    "DeepSeek R1 Distill Llama": 74.87,
    "DeepSeek R1 Distill Llama + Low-Rank Combination": 78.36,
}

# Desirable region used by Figure 1 of the paper
desirable_max_over_refusal = 25.0
desirable_min_refusal = 75.0

marker_area = 420
marker_ring = 3.0


def setup_style() -> None:
    for font_name in ["Lato-Regular.ttf", "Lato-Bold.ttf", "Lato-Italic.ttf"]:
        font_file = subprocess.run(
            ["kpsewhich", font_name], capture_output=True, text=True, check=True
        ).stdout.strip()
        font_manager.fontManager.addfont(font_file)

    plt.rcParams.update(
        {
            "font.family": "Lato",
            "font.size": 24,
            "axes.titlesize": 26,
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "text.color": ink,
            "axes.labelcolor": secondary_ink,
            "xtick.color": secondary_ink,
            "ytick.color": secondary_ink,
            "axes.edgecolor": axis_line,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": gridline,
            "grid.linewidth": 1.0,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 8,
            "ytick.major.pad": 8,
            "pdf.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def plot_tradeoff(output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(column_width, 6.0), layout="constrained")

    ax.add_patch(
        Rectangle(
            (0.0, desirable_min_refusal),
            desirable_max_over_refusal,
            100.0 - desirable_min_refusal,
            facecolor=good_region,
            alpha=0.1,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(
        0.6,
        87.6,
        "desirable: safe and helpful",
        ha="left",
        va="top",
        fontsize=21,
        style="italic",
        color=secondary_ink,
    )

    points = [
        ("Llama 3 8B Instruct", "X", baseline_color, "Llama 3 8B Instruct", (0, 22), "center"),
        ("Refuse-Llama", "s", baseline_color, "Refuse-Llama", (0, -40), "center"),
        (
            "Refuse-Llama + Categorical Steering",
            "o",
            categorical_steering_color,
            "+ Categorical Steering (ours)",
            (-8, 26),
            "left",
        ),
        (
            "Refuse-Llama + Low-Rank Combination",
            "D",
            low_rank_color,
            "+ Low-Rank Combination (ours)",
            (22, -8),
            "left",
        ),
    ]
    for key, marker, color, label, offset, align in points:
        x, y = over_refusal[key], refusal[key]
        ax.scatter(
            [x],
            [y],
            s=marker_area,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=marker_ring,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=23,
            fontweight="bold" if "ours" in label else "normal",
            color=ink,
        )

    ax.set_xlim(0.0, 40.0)
    ax.set_ylim(60.0, 88.0)
    ax.set_xticks(np.arange(0, 41, 10))
    ax.set_yticks(np.arange(60, 89, 5))
    ax.set_xlabel("Over-refusal on benign prompts (%), lower is better")
    ax.set_ylabel("Refusal on harmful prompts (%)\nhigher is better")

    fig.savefig(output_file)
    plt.close(fig)


def plot_results(output_file: Path) -> None:
    rows = [
        ("Refuse-Llama", "Refuse-Llama + Categorical Steering", categorical_steering_color),
        ("Refuse-Llama", "Refuse-Llama + Low-Rank Combination", low_rank_color),
        ("Llama 3 8B Instruct", "Llama 3 8B Instruct + Low-Rank Combination", low_rank_color),
        ("DeepSeek R1 Distill Llama", "DeepSeek R1 Distill Llama + Low-Rank Combination", low_rank_color),
    ]
    row_labels = [
        "Refuse-Llama\n+ Categorical Steering",
        "Refuse-Llama\n+ Low-Rank",
        "Llama 3 8B Instruct\n+ Low-Rank (transfer)",
        "DeepSeek R1 Distill\n+ Low-Rank (transfer)",
    ]
    panels = [
        (over_refusal, "Over-refusal (%)", "benign, lower is better", (0.0, 62.0), [0, 20, 40, 60]),
        (refusal, "Refusal (%)", "harmful, higher is better", (62.0, 86.0), [65, 70, 75, 80, 85]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(column_width, 7.6), sharey=True, layout="constrained")
    y_positions = np.arange(len(rows))[::-1]  # shape: (n_rows)

    for ax, (values, title, direction, xlim, xticks) in zip(axes, panels, strict=True):
        for y, (base, steered, color) in zip(y_positions, rows, strict=True):
            base_value, steered_value = values[base], values[steered]

            # A plain connector, since arrowheads break down on sub-point deltas; the signed label carries direction
            ax.plot([base_value, steered_value], [y, y], color=color, linewidth=5, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.scatter([base_value], [y], s=260, color=baseline_color, edgecolors="white", linewidths=marker_ring, zorder=3)
            ax.scatter([steered_value], [y], s=320, color=color, edgecolors="white", linewidths=marker_ring, zorder=3)

            delta = steered_value - base_value
            ax.text(
                (base_value + steered_value) / 2.0,
                y + 0.24,
                f"{delta:+.1f}%".replace("-", "−"),
                ha="center",
                va="bottom",
                fontsize=22,
                fontweight="bold",
                color=ink,
            )

        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{title}\n", fontsize=24, fontweight="bold", color=ink, loc="left")
        ax.text(0.0, 1.0, direction, transform=ax.transAxes, fontsize=21, style="italic", color=secondary_ink, va="bottom")
        ax.spines["left"].set_visible(False)

    # Separates the directly steered model from the zero-shot transfers
    for ax in axes:
        ax.axhline(1.5, color=axis_line, linewidth=1.2, zorder=1)

    axes[0].set_yticks(y_positions, row_labels, fontsize=21, color=ink)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", markersize=15, color=baseline_color, label="Without steering"),
        Line2D([], [], marker="o", linestyle="", markersize=15, color=categorical_steering_color, label="Categorical Steering"),
        Line2D([], [], marker="o", linestyle="", markersize=15, color=low_rank_color, label="Low-Rank Combination"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=20,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    fig.get_layout_engine().set(wspace=0.06)

    fig.savefig(output_file)
    plt.close(fig)


def plot_pca(output_file: Path, seed: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(column_width, 7.4), layout="constrained")
    rng = np.random.default_rng(seed)

    for ax, (model_name, activation_file) in zip(axes, activation_files.items(), strict=True):
        saved = torch.load(activation_file)
        activations = saved["activations"].float().numpy()  # shape: (n_prompts, d_model)
        categories = np.array(saved["categories"])  # shape: (n_prompts)

        # Same projection and clustering metric as project_activations_and_evaluate_clusters
        projection = PCA(n_components=2, random_state=0).fit_transform(activations)  # shape: (n_prompts, 2)
        silhouette = silhouette_score(activations, categories)
        print(f"{model_name}: silhouette score {silhouette:.3f} on {len(categories)} prompts")

        # Shuffled draw order so no category is systematically painted over the others
        order = rng.permutation(len(categories))
        colors = np.array([category_colors[category] for category in categories])
        ax.scatter(
            projection[order, 0],
            projection[order, 1],
            s=7,
            c=colors[order],
            alpha=0.55,
            linewidths=0,
        )

        ax.set_title(f"{model_name}\n", fontsize=24, fontweight="bold", color=ink)
        ax.text(
            0.5,
            1.02,
            f"silhouette score {silhouette:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=21,
            color=secondary_ink,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        ax.set_xlabel("PC 1", fontsize=20)
        ax.set_ylabel("PC 2", fontsize=20)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(True)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", markersize=14, color=color, label=category)
        for category, color in category_colors.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncols=2,
        frameon=False,
        fontsize=20,
        handletextpad=0.2,
        columnspacing=1.2,
    )

    fig.savefig(output_file)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the poster figures, as PDFs for the poster or SVG/PNG for the project page.")
    parser.add_argument(
        "--figures",
        type=str,
        nargs="+",
        choices=["tradeoff", "results", "pca"],
        default=["tradeoff", "results", "pca"],
        help="Which figures to render (default: tradeoff results pca).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pdf", "svg", "png", "webp"],
        default="pdf",
        help="Output file format (default: pdf).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=output_dir,
        help=f"Directory to write the figures to (default: {output_dir}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution for png and webp output, ignored by vector formats (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the PCA scatter draw order (default: 42).",
    )
    args = parser.parse_args()

    setup_style()
    plt.rcParams["savefig.dpi"] = args.dpi
    os.makedirs(args.output_dir, exist_ok=True)

    if "tradeoff" in args.figures:
        plot_tradeoff(args.output_dir / f"tradeoff.{args.format}")
    if "results" in args.figures:
        plot_results(args.output_dir / f"results.{args.format}")
    if "pca" in args.figures:
        plot_pca(args.output_dir / f"pca.{args.format}", seed=args.seed)

    print(f"Saved {', '.join(args.figures)} as {args.format} to {args.output_dir}")
