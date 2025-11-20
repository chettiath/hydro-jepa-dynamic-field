"""
dynamic_field.py

1D dynamic neural field implementing PDE-like dynamics:

    u_{t+1} = tanh( conv_local(u_t) + D * Laplacian(u_t) + I_t )

- conv_local: learned local interaction / reaction term
- Laplacian: fixed discrete ∇² operator for diffusion
- I_t: external input (stimulus)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicField(nn.Module):
    """
    1D dynamic neural field.

    Args:
        field_size: Length of the 1D field (number of spatial positions).
        kernel_size: Size of the local convolution kernel.
        diffusion: Diffusion coefficient for Laplacian term.
        use_diffusion: Whether to include diffusion term in updates.
    """

    def __init__(
        self,
        field_size: int,
        kernel_size: int = 7,
        diffusion: float = 0.1,
        use_diffusion: bool = True,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same' padding."

        self.field_size = field_size
        self.kernel_size = kernel_size
        self.diffusion = diffusion
        self.use_diffusion = use_diffusion

        # Local interaction / reaction kernel
        self.local_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

        # Fixed Laplacian kernel [1, -2, 1] ≈ ∇² operator in 1D
        lap_kernel = torch.tensor([1.0, -2.0, 1.0]).view(1, 1, 3)
        self.register_buffer("lap_kernel", lap_kernel)

    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """
        Discrete 1D Laplacian via conv1d with fixed kernel [1, -2, 1].

        Args:
            u: (batch, 1, field_size)

        Returns:
            (batch, 1, field_size)
        """
        return F.conv1d(u, self.lap_kernel, padding=1)

    def forward(
        self,
        u_t: torch.Tensor,
        I_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one time-step update of the field.

        Args:
            u_t: (batch, 1, field_size)
            I_t: (batch, 1, field_size)

        Returns:
            u_next: (batch, 1, field_size)
        """
        # Local interaction / reaction term
        local = self.local_conv(u_t)  # ≈ -R_θ(u_t)

        # Diffusion term: D ∇²u_t
        if self.use_diffusion and self.diffusion > 0.0:
            diff_term = self.laplacian(u_t)
            local = local + self.diffusion * diff_term

        # Add external input (forcing) and apply tanh nonlinearity
        u_next = torch.tanh(local + I_t)
        return u_next