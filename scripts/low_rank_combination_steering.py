import torch
import torch.nn as nn


class LowRankSteeringCombination(nn.Module):
    def __init__(self, steering_basis: torch.Tensor, d_model: int = 4096):
        super().__init__()

        self.U = nn.Parameter(steering_basis.clone())  # shape: (d_model, rank)
        self.V = nn.Parameter(steering_basis.clone())  # shape: (d_model, rank)
        self.z = nn.Parameter(0.01 * torch.randn(d_model))  # shape: (d_model)

    def forward(self) -> torch.Tensor:
        return self.U @ (self.V.T @ self.z)  # shape: (d_model)
