"""
plotting.py

Plotting helpers for visualizing 1D field trajectories.

We provide:
    - plot_field_trajectory: line plots + heatmap of u(x, t)
"""

from typing import Optional

import matplotlib.pyplot as plt
import torch


def plot_field_trajectory(
    u_seq: torch.Tensor,
    title: str = "Field trajectory",
) -> plt.Figure:
    """
    Plot a representation of the field over time.

    Args:
        u_seq: (T, 1, N) tensor of field states over time.
        title: Title for the plot.

    Returns:
        fig: Matplotlib Figure object.
    """
    # Ensure on CPU and detached
    u_seq = u_seq.detach().cpu()
    T, C, N = u_seq.shape
    assert C == 1, "Expected single-channel field for plotting."

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    data = u_seq.squeeze(1)  # (T, N)
    vmax = float(data.abs().max().item() + 1e-6)

    # Left: heatmap of u(x, t) with symmetric color scale
    im = axes[0].imshow(
        data,
        aspect="auto",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        cmap="viridis",
    )
    axes[0].set_title(f"{title} (heatmap)")
    axes[0].set_xlabel("Position (x)")
    axes[0].set_ylabel("Time (t)")
    fig.colorbar(im, ax=axes[0])

    # Right: line plots at a few time steps
    times_to_plot = [0, T // 2, T - 1] if T >= 3 else list(range(T))
    for t in times_to_plot:
        axes[1].plot(u_seq[t, 0].numpy(), label=f"t={t}")
    axes[1].set_ylim(-vmax, vmax)
    axes[1].set_title(f"{title} (slices)")
    axes[1].set_xlabel("Position (x)")
    axes[1].set_ylabel("u(x, t)")
    axes[1].legend()

    fig.tight_layout()
    return fig
