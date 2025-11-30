from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import amp
from tqdm.auto import tqdm

from .config import DEVICE


def load_sae_from_hub(repo_id: str, hookpoint: str, device: str | torch.device = DEVICE):
    """Load a Sparse Autoencoder (SAE) from the EleutherAI hub via `eai-sparsify`.

    Requires `eai-sparsify` to be installed. Returns the SAE module on device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    try:
        from sparsify import Sae  # type: ignore
    except Exception as e:  # pragma: no cover - optional path
        raise RuntimeError(
            "eai-sparsify is not installed. Please `pip install eai-sparsify`."
        ) from e

    sae = Sae.load_from_hub(repo_id, hookpoint=hookpoint).to(device)
    sae.train()
    return sae


def batch_SAE_encode(
    activations: torch.Tensor,
    sae,
    batch_size: int = 4,
    device: str | torch.device = DEVICE,
) -> torch.Tensor:
    """Encode a set of activation vectors into SAE latent space, returning dense sparse vectors.

    Returns a tensor of shape (N, sae.num_latents), constructed by scattering the top-K indices/acts.
    Mirrors the notebook logic.
    """
    if isinstance(device, str):
        device = torch.device(device)

    ds = TensorDataset(activations)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_z: List[torch.Tensor] = []
    sae.eval()
    with torch.inference_mode(), amp.autocast(device.type, dtype=torch.float16):
        for (batch_x,) in tqdm(dl, desc="Extracting Sparse Vectors"):
            batch_x = batch_x.to(device)
            enc_out = sae.encode(batch_x)
            top_vals = enc_out.top_acts  # (B, k)
            top_idxs = enc_out.top_indices  # (B, k)

            z = batch_x.new_zeros((top_vals.shape[0], sae.num_latents))
            z.scatter_(1, top_idxs, top_vals)
            all_z.append(z.cpu())

    return torch.cat(all_z, dim=0)


def make_sae_steering_hook(sae, position: int = -1):
    """Factory for a steering hook that operates in SAE latent space.

    Encodes activations at `position`, adds strength * steering_vector in latent space, selects top-K,
    decodes back to residual space, and replaces the token's activation.
    """

    def sae_steering_hook(steering_vector, strength, activation, hook):
        batch_size, seq_len, d_model = activation.shape
        target_token = activation[:, position, :]
        enc_out = sae.encode(target_token)
        z = enc_out.pre_acts  # (B, num_latents)
        sae_hidden_dim = z.shape[-1]

        sv = steering_vector.to(activation.device, dtype=activation.dtype)
        if sv.ndim == 1:
            sv = sv.view(1, sae_hidden_dim).expand(batch_size, sae_hidden_dim)
        elif sv.ndim == 2:
            assert sv.shape == (
                batch_size,
                sae_hidden_dim,
            ), f"steering_vector must be (sae_hidden_dim,) or (batch_size, sae_hidden_dim), got {sv.shape}"
        else:
            raise ValueError("steering_vector must be 1D or 2D")

        steered_z = z + float(strength) * sv
        top_acts, top_idx = steered_z.topk(sae.cfg.k, dim=-1)
        recon = sae.decode(top_acts, top_idx)  # (B, d_model)

        out = activation.clone()
        out[:, position, :] = recon
        return out

    return sae_steering_hook


def compute_sae_steering_vectors(
    benign_sparse_vectors: Dict[str, torch.Tensor],
    harmful_sparse_vectors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute SAE-space steering vectors per category as mean(harmful - benign)."""
    out: Dict[str, torch.Tensor] = {}
    for (h_cat, h_vecs), (b_cat, b_vecs) in zip(
        harmful_sparse_vectors.items(), benign_sparse_vectors.items()
    ):
        if h_cat != b_cat:
            print("Error: harmful and benign categories do not match")
            break
        out[h_cat] = (h_vecs - b_vecs).mean(dim=0)
    return out
