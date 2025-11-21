import torch

from src.dynamic_field import DynamicField
from src.jepa_head import JEPAHead
from src.stimulus_dataset import StimulusDataset


def test_stimulus_dataset_shapes() -> None:
    field = DynamicField(field_size=8, kernel_size=3, diffusion=0.0, use_diffusion=False)
    ds = StimulusDataset(
        field=field,
        num_sequences=2,
        seq_len=4,
        field_size=8,
        device=torch.device("cpu"),
    )

    u_seq, I_seq = ds[0]
    assert u_seq.shape == (4, 1, 8)
    assert I_seq.shape == (4, 1, 8)


def test_jepa_head_forward_shape() -> None:
    u = torch.zeros(2, 1, 8)
    I = torch.zeros(2, 1, 8)

    model = JEPAHead(hidden_channels=4, kernel_size=3)
    out = model(u, I)

    assert out.shape == u.shape
