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
    kernel_size: int = 7
    seq_len: int = 20
    diffusion: float = 0.05
    use_diffusion: bool = True
    leak: float = 0.03
    use_mexican_hat_init: bool = True

    # Training
    batch_size: int = 64
    num_epochs: int = 8
    learning_rate: float = 1e-3
    delta: int = 1  # prediction horizon (u_{t+delta})
    jepa_hidden_channels: int = 32
    jepa_kernel_size: int = 5

    # Misc
    device: str = "auto"  # auto-selects cuda → mps → cpu
    seed: int = 42
    num_sequences: int = 1_000  # dataset size


# Default training config
TRAIN_CONFIG = TrainConfig()
