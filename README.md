# Hydro-JEPA Dynamic Field (Project 1)

This repo is a **toy implementation** of a 1D dynamic neural field with **PDE-like dynamics**, optional **Mexican-hat coupling + global leak**, and a small **JEPA-style predictor** trained to forecast the field’s future state.

It’s the first concrete step in a larger research agenda about **fluid-dynamics-like latent world models**, where internal cognitive state is modeled as a **conceptual fluid** that ripples, flows, and forms “hydrogen bonds” between ideas.

---

## Conceptual overview

We model a 1D field \(u(x, t)\) that evolves in discrete time:

\[
u_{t+1} = \tanh\big( \text{ConvLocal}(u_t) - \text{leak} \cdot u_t + D \cdot \text{Laplacian}(u_t) + I_t \big)
\]

- **ConvLocal**: learned local interaction / reaction term; can be initialized as a 7-tap Mexican-hat kernel (center-excite / surround-inhibit, net slightly negative).
- **leak**: optional global damping term.
- **Laplacian**: fixed discrete \(\nabla^2\) operator (diffusion).
- **\(D\)**: diffusion coefficient.
- **\(I_t\)**: external input (a “drop” into the fluid, Gaussian bump at t=0).
- **tanh**: smooth, bounded nonlinearity (keeps the field stable).

We generate synthetic sequences by:
- starting from \(u_0 = 0\),
- injecting a stimulus \(I_0\) made of 1–2 Gaussian bumps,
- evolving the field forward for \(T\) steps.

Then we train a small **JEPAHead** network to predict \(u_{t+\Delta}\) from \((u_t, I_t)\) using an MSE loss in latent space.

---

## Directory structure

Core layout (under `src/`):

- `configs.py` – central hyperparameters (`TrainConfig`)
- `dynamic_field.py` – `DynamicField`: PDE-like update rule with optional Mexican-hat init and global leak
- `stimulus_dataset.py` – `StimulusDataset`: generates `(u_seq, I_seq)` via simulation
- `jepa_head.py` – `JEPAHead`: conv-based predictor `(u_t, I_t) -> u_{t+Δ}`
- `train_field_jepa.py` – training loop
- `visualize_trajectories.py` – script to visualize field trajectories
- `utils/`
  - `seed.py` – reproducible seeding
  - `plotting.py` – helper to plot field trajectories as heatmaps + line plots

---

## Installation

Create a virtual environment and install dependencies:

```bash```
```
python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt
```

## Quick checks

- Smoke tests: `pytest tests/test_smoke.py`
- Visualization: `python -m src.visualize_trajectories`
- Model vs ground truth (requires trained checkpoint):\
  `python -m src.visualize_predictions --checkpoint-path checkpoints/jepa_head.pth`
- Reproducible baseline trajectory example:\
  `python -m src.visualize_trajectories --seed 42 --output-path figures/trajectory_seed42.png`
