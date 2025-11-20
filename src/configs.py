"""
configs.py

Centralized configuration / hyperparameters for the dynamic field + JEPA project.
Modify these values or override them in experiment-specific configs.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Field / dynamics
    field_size: int = 64
    seq_len: int = 10
    diffusion: float = 0.1
    use_diffusion: bool = True

    # Training
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-3
    delta: int = 1  # prediction horizon (u_{t+delta})

    # Misc
    device: str = "cuda"  # will fall back to CPU if cuda not available
    seed: int = 42
    num_sequences: int = 10_000  # dataset size


# Default training config
TRAIN_CONFIG = TrainConfig()