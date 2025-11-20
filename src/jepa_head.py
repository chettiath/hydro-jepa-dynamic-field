"""
jepa_head.py

JEPA-style predictor that maps (u_t, I_t) → u_{t+Δ} in the latent field space.

We use a small Conv1d network:
    - Input channels: 2 (field + input)
    - Hidden channels: configurable
    - Output channels: 1 (predicted field)
"""

from typing import Tuple

import torch
import torch.nn as nn


class JEPAHead(nn.Module):
    """
    JEPA-style prediction head.

    Args:
        hidden_channels: Number of hidden channels in the conv network.
        kernel_size: Size of the 1D convolution kernel.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same' padding."

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
        )

    def forward(
        self,
        u_t: torch.Tensor,
        I_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            u_t: (batch, 1, field_size)
            I_t: (batch, 1, field_size)

        Returns:
            u_pred: (batch, 1, field_size)
        """
        x = torch.cat([u_t, I_t], dim=1)  # (B, 2, N)
        u_pred = self.net(x)             # (B, 1, N)
        return u_pred