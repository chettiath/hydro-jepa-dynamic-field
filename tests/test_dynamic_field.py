import torch

from src.dynamic_field import DynamicField


def test_mexican_hat_init_and_leak() -> None:
    field = DynamicField(
        field_size=8,
        kernel_size=7,
        diffusion=0.0,
        use_diffusion=False,
        leak=0.3,
        use_mexican_hat_init=True,
    )
    u0 = torch.zeros(2, 1, 8)
    I = torch.zeros(2, 1, 8)
    out = field(u0, I)
    # Leak on zeros should keep zeros; tanh(0) = 0
    assert torch.allclose(out, torch.zeros_like(out))


def test_device_selection_fallback_cpu(monkeypatch) -> None:
    # Simulate no cuda/mps availability by patching torch flags
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    from src.train_field_jepa import select_device

    device = select_device("auto")
    assert device.type == "cpu"


def test_device_selection_prefers_cuda_then_mps(monkeypatch) -> None:
    # CUDA available → pick cuda
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    from src.train_field_jepa import select_device

    device = select_device("auto")
    assert device.type == "cuda"

def test_device_selection_explicit_unavailable(monkeypatch) -> None:
    # Explicit bad device falls back to cpu with warning pathway
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    from src.train_field_jepa import select_device

    device = select_device("cuda")  # request unavailable cuda
    assert device.type == "cpu"
