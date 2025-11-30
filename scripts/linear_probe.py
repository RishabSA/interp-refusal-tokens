import os
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens.utils import get_act_name
from transformer_lens import HookedTransformer
from datasets import load_dataset, concatenate_datasets


class LinearProbe(nn.Module):
    """1-layer linear probe that takes in [4096] residual-stream activations at a token position, which outputs 1 value for each item in the batch.

    Args:
        nn (_type_)
    """

    def __init__(self, d_model: int = 4096):
        super().__init__()

        self.head = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, x: torch.Tensor):
        # x: (B, d_model)
        return self.head(x)  # shape: (B, 1)


class LowRankProbe(nn.Module):
    """1-layer low-rank linear probe that takes in [4096] residual-stream activations at a token position and first projects it into a lower-rank before passing it through a linear layer, which outputs 1 value for each item in the batch.

    Args:
        nn (_type_)
    """

    def __init__(self, U_r: torch.Tensor, d_model: int = 4096, rank: int = 64):
        super().__init__()
        # U_r is the tensor of shape (d_model, rank) from PCA
        self.d_model = d_model
        self.rank = rank

        # Fixed projection (no grad) for classic low-rank probe
        self.register_buffer("U_r", U_r.clone().detach())  # shape: (d, r)

        self.head = nn.Linear(in_features=rank, out_features=1)

    def forward(self, x: torch.Tensor):
        # x: (B, d_model)

        # Project to rank-r (B, r)
        z = x @ self.U_r
        return self.head(z)  # shape: (B, 1)


def load_probe_model(
    probe_model: nn.Module,
    path: str = "steering_probe_18.pt",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[nn.Module, int]:
    """Loads a linear probe from a given path onto the device and returns both the model and the threshold

    Args:
        probe_model (nn.Module)
        path (str, optional). Defaults to "steering_probe_18.pt".
        device (torch.device, optional). Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        tuple[nn.Module, int]: Loaded linear probe model and threshold from the model. The benign vs. harmful threshold defaults to 0.5 if one isn't found.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    probe_model.load_state_dict(checkpoint["model_state_dict"])
    probe_model.eval()

    return probe_model, checkpoint.get("threshold", 0.5)


def analyze_probe_direction_with_activations(
    prompts_dict: dict[str, list[str]],
    activations_dict: dict[str, torch.Tensor],
    probe_model: nn.Module,
    K: int = 10,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    assert isinstance(probe_model, LinearProbe), "probe_model must be a LinearProbe"

    probe_weight_direction = probe_model.head.weight.squeeze(dim=0)

    scored_examples = []

    for category, activations in activations_dict.items():
        prompts = prompts_dict[category]
        categorical_activations = activations.to(device, dtype=torch.float32)

        for i in range(categorical_activations.shape[0]):
            activation = categorical_activations[i]
            score = torch.dot(activation, probe_weight_direction).item()
            scored_examples.append((score, category, prompts[i]))

    # Sort by dot product score
    scored_examples.sort(key=lambda x: x[0], reverse=True)

    print("Top Activating Prompts (Most Harmful)")
    for score, category, prompt in scored_examples[:K]:
        print(f"Category: {category} | Score: {score:.4f} | Prompt: {prompt}")

    print("\n")

    print("Bottom Activating Prompts (Most Benign)")
    for score, category, prompt in scored_examples[-K:]:
        print(f"Category: {category} | Score: {score:.4f} | Prompt: {prompt}")


def compare_probe_direction_with_steering_vectors(
    steering_vectors: dict[str, torch.Tensor],
    probe_model: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict[str, float]:
    assert isinstance(probe_model, LinearProbe), "probe_model must be a LinearProbe"

    probe_weight_direction = probe_model.head.weight.squeeze(dim=0)
    probe_weight_direction = probe_weight_direction / (
        probe_weight_direction.norm() + 1e-8
    )

    probe_steering_vector_cosine_sims = {}

    for category, steering_vector in steering_vectors.items():
        vector = steering_vector.to(device)
        cosine_sim = F.cosine_similarity(
            vector, probe_weight_direction, dim=-1, eps=1e-8
        ).item()
        print(
            f"{category} steering vector and probe weight direction have a cosine similarity of {cosine_sim}"
        )
        probe_steering_vector_cosine_sims[category] = cosine_sim

    return probe_steering_vector_cosine_sims


def get_probe_analysis_data(
    batch_size: int = 4, include_benign: bool = True
) -> dict[str, DataLoader]:
    # COCONot Dataset loading
    coconot_orig = load_dataset("allenai/coconot", "original")  # 12.5k items
    coconot_unique_categories = coconot_orig["train"].unique("category")

    coconot_dataloaders = {}

    def prompt_category_collate(batch: list[dict]) -> dict[str, list[str]]:
        return {
            "prompt": [sample["prompt"] for sample in batch],
            "category": [sample.get("category") for sample in batch],
        }

    for category in coconot_unique_categories:
        # Filter the orig train dataset
        orig_category_train = coconot_orig["train"].filter(
            lambda x, c=category: x["category"] == c
        )

        # Filter the orig test dataset
        orig_category_test = coconot_orig["test"].filter(
            lambda x, c=category: x["category"] == c
        )

        harmful_category_dataset = concatenate_datasets(
            [orig_category_train, orig_category_test]
        )

        harmful_category_dataloader = DataLoader(
            harmful_category_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=prompt_category_collate,
        )

        coconot_dataloaders[category] = harmful_category_dataloader

    if include_benign:
        coconot_contrast = load_dataset("allenai/coconot", "contrast")  # 379 items

        benign_category_dataset = coconot_contrast["test"]

        benign_category_dataloader = DataLoader(
            benign_category_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=prompt_category_collate,
        )

        coconot_dataloaders["respond"] = benign_category_dataloader

    for category, dataloader in coconot_dataloaders.items():
        print(f"{category} category has {len(dataloader)} batches")

    return coconot_dataloaders


def collect_prompts_by_category(
    iterators: dict[str, DataLoader],
) -> dict[str, list[str]]:
    prompts_by_category = defaultdict(list)

    for category, iterator in iterators.items():
        for batch in iterator:
            prompts = batch["prompt"]

            for prompt in prompts:
                prompts_by_category[category].append(prompt)

    return dict(prompts_by_category)


def get_categorical_steering_vector_probe(
    prompt: str,
    hooked_model: HookedTransformer,
    benign_strength: float,
    harmful_strength: float,
    steering_vector_mapping: dict[int, torch.Tensor],
    probe_model: nn.Module,
    probe_threshold: float = 0.5,
    activation_name: str = "resid_post",
    layer: int = 18,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[torch.Tensor, float]:
    token_activation = None

    hook_name = get_act_name(activation_name, layer)

    def activation_hook(activation, hook):
        nonlocal token_activation
        token_activation = activation[:, -1, :].detach()

    hooked_model.to(device).eval()
    hooked_model.reset_hooks()

    hooked_model.add_hook(hook_name, activation_hook, "fwd")

    with torch.inference_mode():
        tokens = hooked_model.to_tokens(prompt).to(device)
        outputs = hooked_model(tokens)

    hooked_model.reset_hooks()

    logits_next = outputs[:, -1, :]  # shape: (B, vocab_size)
    probs_next = F.softmax(logits_next, dim=-1)

    token_activation = token_activation.to(device, dtype=torch.float32)

    with torch.inference_mode():
        harmful_score = torch.sigmoid(
            probe_model(token_activation).squeeze(-1)
        ).item()  # shape: (B)

    harmful_decision = harmful_score >= probe_threshold

    refusal_category_mapping = {
        128256: "Humanizing requests",
        128257: "Incomplete requests",
        128258: "Indeterminate requests",
        128259: "Requests with safety concerns",
        128260: "Unsupported requests",
    }

    respond_token_id = 128261

    refusal_token_ids = torch.tensor(
        list(refusal_category_mapping.keys()), device=device
    )

    refusal_probs = probs_next.index_select(
        dim=1, index=refusal_token_ids
    )  # shape: (B, 5)

    respond_prob = probs_next.index_select(
        dim=1, index=torch.tensor(respond_token_id, device=device)
    )  # shape: (B, 1)

    top_refusal_prob, top_refusal_token_idx = refusal_probs.max(dim=1)
    top_refusal_token_id = refusal_token_ids[top_refusal_token_idx].item()

    # print("Top refusal prob and id:", top_refusal_prob.item(), top_refusal_token_id)
    # print("Respond prob: ", respond_prob.item())

    if harmful_decision:
        # Harmful

        # print(f"Harmful: {top_refusal_token_id}, strength: {harmful_strength}")
        return steering_vector_mapping[top_refusal_token_id], harmful_strength
    else:
        # Benign

        # print(f"Benign: {top_refusal_token_id}, strength: {benign_strength}")
        return steering_vector_mapping[top_refusal_token_id], benign_strength


def get_random_categorical_steering_vector_probe(
    prompt: str,
    hooked_model: HookedTransformer,
    benign_strength: float,
    harmful_strength: float,
    steering_vector_mapping: dict[int, torch.Tensor],
    probe_model: nn.Module,
    probe_threshold: float = 0.5,
    activation_name: str = "resid_post",
    layer: int = 18,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[torch.Tensor, float]:
    token_activation = None

    hook_name = get_act_name(activation_name, layer)

    def activation_hook(activation, hook):
        nonlocal token_activation
        token_activation = activation[:, -1, :].detach()

    hooked_model.to(device).eval()
    hooked_model.reset_hooks()

    hooked_model.add_hook(hook_name, activation_hook, "fwd")

    with torch.inference_mode():
        tokens = hooked_model.to_tokens(prompt).to(device)
        outputs = hooked_model(tokens)

    hooked_model.reset_hooks()

    token_activation = token_activation.to(device, dtype=torch.float32)

    with torch.inference_mode():
        harmful_score = torch.sigmoid(
            probe_model(token_activation).squeeze(-1)
        ).item()  # shape: (B)

    harmful_decision = harmful_score >= probe_threshold

    random_key = random.choice(list(steering_vector_mapping.keys()))
    random_vector = steering_vector_mapping[random_key]

    if harmful_decision:
        # Harmful

        # print(f"Harmful: {random_key} (random), strength: {harmful_strength}")
        return random_vector, harmful_strength
    else:
        # Benign

        # print(f"Benign: {random_key} (random), strength: {benign_strength}")
        return random_vector, benign_strength
