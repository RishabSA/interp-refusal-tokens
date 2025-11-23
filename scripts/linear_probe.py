import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens.utils import get_act_name
from transformer_lens import HookedTransformer


class LinearProbe(nn.Module):
    def __init__(self, d_model: int = 4096):
        super().__init__()

        self.head = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, x: torch.Tensor):
        # x: (B, d_model)
        return self.head(x)  # shape: (B, 1)


class LinearMLPProbe(nn.Module):
    def __init__(
        self, d_model: int = 4096, hidden_features: int = 64, p_dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.LayerNorm(hidden_features),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.LayerNorm(hidden_features),
            nn.Linear(in_features=hidden_features, out_features=1),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, d_model)
        return self.layers(x)  # shape: (B, 1)


class LowRankProbe(nn.Module):
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
    path: str = "steering_probe_18.pt",
    d_model: int = 4096,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[nn.Module, int]:
    # probe_model = LinearProbe(d_model=d_model).to(device)
    probe_model = LinearMLPProbe(d_model=d_model, hidden_features=64).to(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    probe_model.load_state_dict(checkpoint["model_state_dict"])
    probe_model.eval()

    return probe_model, checkpoint.get("threshold", 0.5)


def get_categorical_steering_vector_probe(
    prompt: str,
    hooked_model: HookedTransformer,
    benign_strength: float,
    harmful_strength: float,
    steering_vector_mapping: dict[int, torch.Tensor],
    probe_model: nn.Module,
    probe_threshold: float = 0.5,
    activation_name: str = "resid_post",
    layer: int = 16,
    temperature: float = 1.0,
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
    if temperature and temperature != 1.0:
        logits_next = logits_next / temperature

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
        # if respond_prob.item() > 0.8:
        #     # print("Benign: None")
        #     return None, 0.0
        # else:

        # print(f"Benign: {top_refusal_token_id}, strength: {benign_strength}")
        return steering_vector_mapping[top_refusal_token_id], benign_strength
