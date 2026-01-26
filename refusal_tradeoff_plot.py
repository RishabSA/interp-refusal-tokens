import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_refusal_tradeoff(
    points: list,
    title: str = "Refusal vs. Over-Refusal Tradeoff",
):
    names, xs, ys = [], [], []
    for item in points:
        name, x, y = item

        names.append(name)
        xs.append(x)
        ys.append(y)

    xlim = (0.0, 50.0)
    ylim = (50.0, 100.0)

    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = fig.add_subplot()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.25)

    x_good_max = 25.0
    y_good_min = 75.0

    rect = Rectangle(
        (0.0, y_good_min),
        x_good_max,
        ylim[1] - y_good_min,
        facecolor="lightgreen",
        alpha=0.3,
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(rect)

    markers = ["X", "s", "o", "D"]
    colors = ["C0", "C1", "C2", "C3"]

    # Scatter points
    for i, (name, x, y) in enumerate(zip(names, xs, ys)):
        ax.scatter(
            [x],
            [y],
            s=70,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidths=0.6,
            zorder=3,
            label=name,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("Over-refusal Rate (%)", fontsize=10)
    ax.set_ylabel("Refusal Rate (%)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis="both", labelsize=8)

    # Legend
    ax.legend(
        loc="lower right",
        fontsize=8,
        frameon=True,
        framealpha=0.95,
        borderpad=0.4,
        handletextpad=0.6,
        labelspacing=0.8,
    )

    fig.tight_layout()

    fig.savefig("refusal_overrefusal_tradeoff.pdf", bbox_inches="tight")
    fig.savefig("refusal_overrefusal_tradeoff.png", bbox_inches="tight")


if __name__ == "__main__":
    points = [
        ("LLama 3 8B Instruct", 28.26, 66.76),
        ("Categorical Refusal Tokens", 17.34, 65.53),
        ("Categorical Steering (Ours)", 4.54, 79.03),
        ("Low-rank Combination (Ours)", 11.47, 77.21),
    ]

    plot_refusal_tradeoff(
        points,
        title="Safety Tradeoff on Refusal Benchmarks",
    )
