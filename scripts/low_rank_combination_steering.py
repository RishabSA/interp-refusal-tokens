import torch
import torch.nn as nn


class LowRankSteeringShift(nn.Module):
    def __init__(self, steering_basis: torch.Tensor, d_model: int = 4096):
        super().__init__()

        self.U = nn.Parameter(steering_basis.clone())  # shape: (d_model, rank)
        self.V = nn.Parameter(steering_basis.clone())  # shape: (d_model, rank)
        self.z = nn.Parameter(torch.zeros(d_model))  # shape: (d_model)

    def delta(self) -> torch.Tensor:
        return self.U @ (self.V.T @ self.z)  # shape: (d_model)


class LowRankSteeringMap(nn.Module):
    def __init__(self, steering_basis: torch.Tensor):
        super().__init__()

        self.U = nn.Parameter(steering_basis.clone())  # shape: (d_model, rank)
        self.V = nn.Parameter(steering_basis.clone())  # shape: (d_model, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, d_model)
        x = x.to(dtype=self.U.dtype)

        return (x @ self.V) @ self.U.T  # shape: (batch_size, d_model)


class LowRankSteeringMap(nn.Module):
    def __init__(self, steering_basis: torch.Tensor):
        super().__init__()
        Q = steering_basis.detach()  # shape: (d_model, rank)
        self.register_buffer("Q", Q)

        rank = Q.shape[1]
        self.M = nn.Parameter(torch.zeros(rank, rank))  # shape: (rank, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, d_model)
        x = x.to(dtype=self.Q.dtype)

        return (x @ self.Q) @ self.M @ self.Q.T  # shape: (batch_size, d_model)
