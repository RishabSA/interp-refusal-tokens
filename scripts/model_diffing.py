import matplotlib.pyplot as plt
import torch.nn.functional as F


def compute_model_diffing_cosine_sims(
    steering_vectors_activations, steering_vectors_activations_llama
):
    model_diffing_cosine_sims = {}

    for (category, steering_vector), (category_llama, steering_vector_llama) in zip(
        steering_vectors_activations.items(), steering_vectors_activations_llama.items()
    ):
        if category != category_llama:
            print("Error: categories do not match")
            break

        steering_cosine_sim = F.cosine_similarity(
            steering_vector, steering_vector_llama, dim=-1, eps=1e-8
        )
        print(f"{category} has a cosine similarity of {steering_cosine_sim}")
        model_diffing_cosine_sims[category] = float(steering_cosine_sim.detach().cpu())

    return model_diffing_cosine_sims


def plot_model_diffing_cosine_sims(model_diffing_cosine_sims):
    items = list(model_diffing_cosine_sims.items())

    categories = [category for category, value in items]
    values = [value for category, value in items]

    cmap = plt.colormaps.get("tab20")
    bar_colors = [cmap(i % cmap.N) for i in range(len(categories))]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.bar(categories, values, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Steering Vectors")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(
        "Cosine Similarity of Steering Vectors (Llama-3-8b vs Categorical Refusal Token Model)"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, 1)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.savefig("steering_vector_cos_sim_model_diffing.png")
    plt.show()
