"""
visualize_trajectories.py

Script for visualizing dynamic field trajectories over time.

Behavior:
    - Builds a DynamicField and StimulusDataset.
    - Draws one sample trajectory.
    - Plots u(x, t) as heatmap + line slices.
    - Saves figure to figures/ directory.
"""

import argparse
import os
from typing import Optional, Tuple

import torch

from .configs import TRAIN_CONFIG
from .dynamic_field import DynamicField
from .stimulus_dataset import StimulusDataset
from .utils.plotting import plot_field_trajectory
from .utils.seed import set_seed


def _build_sample(seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a single (u_seq, I_seq) sample deterministically if seed provided.
    """
    cfg = TRAIN_CONFIG
    if cfg.seq_len <= cfg.delta:
        raise ValueError(
            f"seq_len ({cfg.seq_len}) must be greater than delta ({cfg.delta}) "
            "for visualization."
        )
    if seed is not None:
        set_seed(seed)

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
    return dataset[0]


def visualize_single_trajectory(
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Generate a single trajectory and plot it.
    """
    u_seq, I_seq = _build_sample(seed=seed)
    _ = I_seq  # unused for now

    fig = plot_field_trajectory(u_seq, title="Dynamic field trajectory")

    if output_path is None:
        output_path = os.path.join("figures", "trajectory_baseline.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path)
    print(f"Saved trajectory figure to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a single dynamic field trajectory."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="figures/trajectory_baseline.png",
        help="Where to save the trajectory figure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible stimulus.",
    )
    args = parser.parse_args()
    visualize_single_trajectory(output_path=args.output_path, seed=args.seed)


if __name__ == "__main__":
    main()
