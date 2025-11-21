"""
dynamic_field.py

1D dynamic neural field with Mexican-hat reaction + diffusion + optional global leak:

    u_{t+1} = tanh( ConvLocal(u_t) - leak * u_t + D * Laplacian(u_t) + I_t )

- ConvLocal: Mexican-hat-like kernel (center-excite / surround-inhibit, net slightly negative)
- leak: extra global negative self-coupling (uniform damping)
- Laplacian: fixed [1, -2, 1] stencil for diffusion
- I_t: external input (Gaussian bump at t=0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicField(nn.Module):
    """
    1D dynamic neural field.

    Args:
        field_size: Length of the 1D field (number of spatial positions).
        kernel_size: Size of the local convolution kernel (must be odd).
        diffusion: Diffusion coefficient for Laplacian term.
        use_diffusion: Whether to include diffusion term in updates.
        leak: Extra global negative self-coupling factor. If > 0, the field
              decays faster toward baseline even in regions without spatial
              structure.
        use_mexican_hat_init: If True and kernel_size == 7, initialize the
              conv kernel to a concrete Mexican-hat-like kernel whose total
              sum is slightly negative. Otherwise use PyTorch's default init.
        kernel_scale: Scalar multiplier for the default Mexican-hat kernel.
    """

    def __init__(
        self,
        field_size: int,
        kernel_size: int = 7,
        diffusion: float = 0.1,
        use_diffusion: bool = True,
        leak: float = 0.0,
        use_mexican_hat_init: bool = True,
        kernel_scale: float = 0.2,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same' padding."

        self.field_size = field_size
        self.kernel_size = kernel_size
        self.diffusion = diffusion
        self.use_diffusion = use_diffusion
        self.leak = leak

        # Local interaction / reaction kernel
        self.local_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

        # Optional: initialize as a tiny Mexican-hat kernel when k=7
        # Shape: [-0.05, 0.10, 0.30, -0.80, 0.30, 0.10, -0.05]
        # Sum = -0.10 → net slightly damping.
        if use_mexican_hat_init and kernel_size == 7:
            with torch.no_grad():
                base_k = torch.tensor(
                    [-0.05, 0.10, 0.30, -0.80, 0.30, 0.10, -0.05],
                    dtype=torch.float32,
                )
                k = kernel_scale * base_k
                self.local_conv.weight.zero_()
                self.local_conv.weight[0, 0, :] = k
                self.local_conv.bias.zero_()
        # else: keep PyTorch’s default initialization

        # Fixed Laplacian kernel [1, -2, 1] ≈ ∇² in 1D
        lap_kernel = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32).view(1, 1, 3)
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

        Dynamics:

            R_theta(u_t) = ConvLocal(u_t) - leak * u_t
            u_{t+1}     = tanh( R_theta(u_t)
                                + diffusion * Laplacian(u_t)
                                + I_t )

        Args:
            u_t: (batch, 1, field_size)  – current field
            I_t: (batch, 1, field_size)  – external input at this step

        Returns:
            u_next: (batch, 1, field_size)
        """
        # Local reaction term (pattern shaping)
        local = self.local_conv(u_t)

        # Extra global damping: -leak * u_t
        if self.leak != 0.0:
            local = local - self.leak * u_t

        # Diffusion term: D ∇²u_t
        if self.use_diffusion and self.diffusion > 0.0:
            diff_term = self.laplacian(u_t)
            local = local + self.diffusion * diff_term

        # Add external input (forcing) and squash to keep field bounded
        u_next = torch.tanh(local + I_t)
        return u_next
