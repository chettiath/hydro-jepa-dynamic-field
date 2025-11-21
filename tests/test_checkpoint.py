import torch

from src.dynamic_field import DynamicField
from src.train_field_jepa import TRAIN_CONFIG, DynamicField as DF, JEPAHead, torch as train_torch


def test_checkpoint_save_load(tmp_path) -> None:
    # Build tiny model/field and fake checkpoint
    cfg = TRAIN_CONFIG
    field = DynamicField(
        field_size=cfg.field_size,
        kernel_size=cfg.kernel_size,
        diffusion=cfg.diffusion,
        use_diffusion=cfg.use_diffusion,
        leak=cfg.leak,
        use_mexican_hat_init=cfg.use_mexican_hat_init,
    )
    model_kwargs = {"hidden_channels": 32, "kernel_size": 5}
    model = JEPAHead(**model_kwargs)
    # Save
    ckpt_path = tmp_path / "ckpt.pth"
    data_gen_kwargs = {
        "field_size": cfg.field_size,
        "kernel_size": cfg.kernel_size,
        "diffusion": cfg.diffusion,
        "use_diffusion": cfg.use_diffusion,
        "leak": cfg.leak,
        "use_mexican_hat_init": cfg.use_mexican_hat_init,
        "seq_len": cfg.seq_len,
        "delta": cfg.delta,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "model_kwargs": model_kwargs,
            "data_gen_kwargs": data_gen_kwargs,
        },
        ckpt_path,
    )

    # Load via visualize helper to ensure data_gen_kwargs and model_kwargs are picked
    from src.visualize_predictions import _load_model

    loaded_model, loaded_cfg, loaded_dg = _load_model(str(ckpt_path), torch.device("cpu"))
    assert loaded_dg == data_gen_kwargs
    # ensure model kwargs match what was saved
    assert model_kwargs["hidden_channels"] == loaded_model.net[0].out_channels
    assert model_kwargs["kernel_size"] == loaded_model.net[0].kernel_size[0]
    assert "model_state" not in loaded_cfg  # config is a dict of scalars
