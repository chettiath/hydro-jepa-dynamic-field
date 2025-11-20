# Hydro-JEPA Dynamic Field (Project 1)

This repo is a **toy implementation** of a 1D dynamic neural field with **PDE-like dynamics** and a small **JEPA-style predictor** trained to forecast the field’s future state.

It’s the first concrete step in a larger research agenda about **fluid-dynamics-like latent world models**, where internal cognitive state is modeled as a **conceptual fluid** that ripples, flows, and forms “hydrogen bonds” between ideas.

---

## Conceptual overview

We model a 1D field \(u(x, t)\) that evolves in discrete time:

\[
u_{t+1} = \tanh\big( \text{conv\_local}(u_t) + D \cdot \text{Laplacian}(u_t) + I_t \big)
\]

- **conv_local**: learned local interaction / reaction term
- **Laplacian**: fixed discrete \(\nabla^2\) operator (diffusion)
- **\(D\)**: diffusion coefficient
- **\(I_t\)**: external input (a “drop” into the fluid)
- **tanh**: smooth, bounded nonlinearity (keeps the field stable and symmetric)

We generate synthetic sequences by:
- starting from \(u_0 = 0\),
- injecting a stimulus \(I_0\) made of 1–2 Gaussian bumps,
- evolving the field forward for \(T\) steps.

Then we train a small **JEPAHead** network to predict \(u_{t+\Delta}\) from \((u_t, I_t)\) using an MSE loss in latent space.

---

## Directory structure

Core layout (under `src/`):

- `configs.py` – central hyperparameters (`TrainConfig`)
- `dynamic_field.py` – `DynamicField`: PDE-like update rule
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
python -m venv .venv

source .venv/bin/activate        # On Windows: .venv\Scripts\activate

pip install --upgrade pip

pip install -r requirements.txt
