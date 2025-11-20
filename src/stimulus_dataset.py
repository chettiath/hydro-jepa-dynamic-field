"""
stimulus_dataset.py

Dataset that generates sequences of dynamic field states (u_seq) and
input patterns (I_seq) by simulating the DynamicField forward in time
from an initial condition and randomly sampled stimulus.

Behavior:
    - u_0 = 0
    - I_0 = stimulus (1–2 Gaussian bumps)
    - I_t = 0 for t > 0
    - u_{t+1} = DynamicField(u_t, I_t)
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset

from .dynamic_field import DynamicField


class StimulusDataset(Dataset):
    """
    Dataset for dynamic field sequences.

    Args:
        field: DynamicField instance used for simulation (on CPU).
        num_sequences: Number of sequences in the dataset.
        seq_len: Length of each sequence (time steps).
        field_size: Size of the 1D field.
        device: torch.device for tensors (usually CPU for dataset).
    """

    def __init__(
        self,
        field: DynamicField,
        num_sequences: int = 10_000,
        seq_len: int = 10,
        field_size: int = 64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.field = field.to(device)
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.field_size = field_size
        self.device = device

    def __len__(self) -> int:
        return self.num_sequences

    def _sample_stimulus(self) -> torch.Tensor:
        """
        Create a single (1, field_size) input pattern with 1–2 Gaussian bumps.

        Returns:
            I: (1, field_size)
        """
        N = self.field_size
        x = torch.linspace(-1.0, 1.0, steps=N, device=self.device)
        I = torch.zeros(1, N, device=self.device)

        # Number of bumps (concepts) in this sequence
        num_bumps = torch.randint(low=1, high=3, size=(1,), device=self.device).item()

        for _ in range(num_bumps):
            center = float(torch.empty(1, device=self.device).uniform_(-0.7, 0.7))
            width = float(torch.empty(1, device=self.device).uniform_(0.05, 0.2))

            bump = torch.exp(-0.5 * ((x - center) ** 2) / (width ** 2))
            I += bump.unsqueeze(0)

        # Normalize amplitude to keep things bounded
        max_val = I.max()
        if max_val > 0:
            I = I / max_val

        return I  # shape: (1, N)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate one sequence.

        Returns:
            u_seq: (T, 1, field_size)  – field states over time
            I_seq: (T, 1, field_size)  – input over time
        """
        _ = idx  # unused

        with torch.no_grad():
            # Initial condition: u_0 = 0
            u_t = torch.zeros(1, 1, self.field_size, device=self.device)

            # Stimulus at t=0
            I_0 = self._sample_stimulus().unsqueeze(0)  # (1, 1, N)

            u_seq = []
            I_seq = []

            for t in range(self.seq_len):
                if t == 0:
                    I_t = I_0
                else:
                    I_t = torch.zeros_like(I_0)

                # store current state and input (strip batch=1 for T,1,N)
                u_seq.append(u_t[0].clone())  # (1, N)
                I_seq.append(I_t[0].clone())  # (1, N)

                # evolve the field
                u_t = self.field(u_t, I_t)

        u_seq_tensor = torch.stack(u_seq, dim=0)  # (T, 1, N)
        I_seq_tensor = torch.stack(I_seq, dim=0)  # (T, 1, N)
        return u_seq_tensor, I_seq_tensor