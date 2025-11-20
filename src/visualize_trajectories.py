"""
visualize_trajectories.py

Script for visualizing dynamic field trajectories over time.

Behavior:
    - Builds a DynamicField and StimulusDataset.
    - Draws one sample trajectory.
    - Plots u(x, t) as heatmap + line slices.
    - Saves figure to figures/ directory.
"""

import os
from typing import Optional

import torch

from .configs import TRAIN_CONFIG
from .dynamic_field import DynamicField
from .stimulus_dataset import StimulusDataset
from .utils.plotting import plot_field_trajectory


def visualize_single_trajectory(output_path: Optional[str] = None) -> None:
    """
    Generate a single trajectory and plot it.
    """
    cfg = TRAIN_CONFIG

    # For visualization, everything can live on CPU
    device = torch.device("cpu")

    field = DynamicField(
        field_size=cfg.field_size,
        kernel_size=7,
        diffusion=cfg.diffusion,
        use_diffusion=cfg.use_diffusion,
    )

    dataset = StimulusDataset(
        field=field,
        num_sequences=1,
        seq_len=cfg.seq_len,
        field_size=cfg.field_size,
        device=device,
    )

    u_seq, I_seq = dataset[0]  # (T, 1, N), (T, 1, N)
    _ = I_seq  # unused for now

    fig = plot_field_trajectory(u_seq, title="Dynamic field trajectory")

    if output_path is None:
        output_path = os.path.join("figures", "trajectory_baseline.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path)
    print(f"Saved trajectory figure to: {output_path}")


if __name__ == "__main__":
    visualize_single_trajectory()