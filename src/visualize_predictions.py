"""
visualize_predictions.py

Compare the ground-truth field trajectory with the JEPAHead's predicted
trajectory using a saved checkpoint.
"""

import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .configs import TRAIN_CONFIG
from .dynamic_field import DynamicField
from .jepa_head import JEPAHead
from .stimulus_dataset import StimulusDataset
from .utils.seed import set_seed


def _select_device(device_pref: str = "auto") -> torch.device:
    if device_pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(device_pref)
    except Exception:
        return torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available; using CPU.")
        return torch.device("cpu")

    return device


def _load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[JEPAHead, Dict, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_kwargs = ckpt.get(
        "model_kwargs",
        {
            "hidden_channels": 32,
            "kernel_size": 5,
        },
    )
    model = JEPAHead(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("config", {}), ckpt.get("data_gen_kwargs", {})


def _predict_sequence(
    model: JEPAHead,
    u_seq: torch.Tensor,
    I_seq: torch.Tensor,
    delta: int,
    device: torch.device,
    rollout: bool = False,
) -> torch.Tensor:
    """
    Generate predictions aligned with the ground-truth sequence.

    If rollout is False, predicts only one step ahead for each context slice.
    If rollout is True, performs iterative rollout using available predictions
    as the source state for subsequent horizons.
    """
    T, _, _ = u_seq.shape
    pred_seq = torch.full_like(u_seq, float("nan"))

    if T <= delta:
        return pred_seq

    with torch.no_grad():
        pred_seq[:delta] = u_seq[:delta]

        for t in range(T - delta):
            # source state for predicting u_{t+delta}
            if rollout and not torch.isnan(pred_seq[t]).any():
                u_src = pred_seq[t]
            else:
                u_src = u_seq[t]

            I_src = I_seq[t]

            u_in = u_src.unsqueeze(0).to(device)  # (1,1,N)
            I_in = I_src.unsqueeze(0).to(device)  # (1,1,N)
            pred = model(u_in, I_in).cpu()        # (1,1,N)
            pred_seq[t + delta] = pred
    return pred_seq


def _plot_truth_vs_pred(
    u_true: torch.Tensor,
    u_pred: torch.Tensor,
    title: str,
    output_path: str,
) -> None:
    u_true = u_true.detach().cpu()
    u_pred = u_pred.detach().cpu()
    u_true_np = u_true.squeeze(1).numpy()
    u_pred_np = u_pred.squeeze(1).numpy()
    error_np = u_pred_np - u_true_np

    vmax_true = float(np.nanmax(np.abs(u_true_np)) + 1e-6)
    vmax_pred = float(np.nanmax(np.abs(u_pred_np)) + 1e-6)
    vmax = max(vmax_true, vmax_pred)

    cmap_sym = plt.cm.get_cmap("viridis").copy()
    cmap_sym.set_bad(color="lightgray")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(
        u_true_np,
        aspect="auto",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        cmap=cmap_sym,
        interpolation="nearest",
    )
    axes[0].set_title("Ground truth (heatmap)")
    axes[0].set_xlabel("Position (x)")
    axes[0].set_ylabel("Time (t)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        np.ma.masked_invalid(u_pred_np),
        aspect="auto",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        cmap=cmap_sym,
        interpolation="nearest",
    )
    axes[1].set_title("JEPA prediction (heatmap)")
    axes[1].set_xlabel("Position (x)")
    axes[1].set_ylabel("Time (t)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        np.ma.masked_invalid(error_np),
        aspect="auto",
        origin="lower",
        cmap=cmap_sym,
        interpolation="nearest",
    )
    axes[2].set_title("Prediction error (pred - truth)")
    axes[2].set_xlabel("Position (x)")
    axes[2].set_ylabel("Time (t)")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(title)
    fig.tight_layout()

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def visualize_predictions(
    checkpoint_path: str = "checkpoints/jepa_head.pth",
    output_path: str = "figures/trajectory_pred_vs_truth.png",
    rollout: bool = False,
) -> None:
    cfg = TRAIN_CONFIG

    set_seed(cfg.seed)

    device = _select_device(cfg.device)
    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train first to create it."
        )

    model, saved_cfg, data_gen_kwargs = _load_model(checkpoint_path, device)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # Data generation on CPU for reproducibility
    # Prefer data-gen params from checkpoint; fall back to current cfg with warning
    if data_gen_kwargs:
        dg = data_gen_kwargs
    else:
        dg = {
            "field_size": cfg.field_size,
            "kernel_size": cfg.kernel_size,
            "diffusion": cfg.diffusion,
            "use_diffusion": cfg.use_diffusion,
            "leak": cfg.leak if hasattr(cfg, "leak") else 0.0,
            "use_mexican_hat_init": getattr(cfg, "use_mexican_hat_init", False),
            "seq_len": cfg.seq_len,
            "delta": cfg.delta,
        }

    delta_ckpt = dg.get("delta") if isinstance(dg, dict) else None
    delta_cfg = saved_cfg.get("delta") if isinstance(saved_cfg, dict) else None
    delta_effective = delta_ckpt or delta_cfg or cfg.delta
    if delta_effective != cfg.delta:
        print(
            f"Warning: using checkpoint delta={delta_effective} (current cfg.delta={cfg.delta})."
        )

    seq_len_effective = dg.get("seq_len", cfg.seq_len)
    if seq_len_effective <= delta_effective:
        raise ValueError(
            f"seq_len ({seq_len_effective}) must be greater than delta ({delta_effective}) "
            "for visualization."
        )

    field = DynamicField(
        field_size=dg["field_size"],
        kernel_size=dg.get("kernel_size", 7),
        diffusion=dg.get("diffusion", 0.1),
        use_diffusion=dg.get("use_diffusion", True),
        leak=dg.get("leak", 0.0),
        use_mexican_hat_init=dg.get("use_mexican_hat_init", True),
    )
    dataset = StimulusDataset(
        field=field,
        num_sequences=1,
        seq_len=dg.get("seq_len", cfg.seq_len),
        field_size=dg["field_size"],
        device=torch.device("cpu"),
    )

    u_seq, I_seq = dataset[0]
    pred_seq = _predict_sequence(
        model=model,
        u_seq=u_seq,
        I_seq=I_seq,
        delta=delta_effective,
        device=device,
        rollout=rollout,
    )

    _plot_truth_vs_pred(
        u_true=u_seq,
        u_pred=pred_seq,
        title="Ground truth vs JEPA prediction",
        output_path=output_path,
    )
    print(f"Saved comparison figure to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ground truth vs JEPAHead predictions."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/jepa_head.pth",
        help="Path to saved JEPAHead checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="figures/trajectory_pred_vs_truth.png",
        help="Where to save the comparison figure.",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Use iterative rollout beyond delta (closed-loop predictions).",
    )
    args = parser.parse_args()
    visualize_predictions(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        rollout=args.rollout,
    )


if __name__ == "__main__":
    main()
