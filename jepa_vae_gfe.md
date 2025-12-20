Here is the comprehensive technical specification and mathematical assembly of the **JEPA-VAE-GFE World Model**.

This document represents the finalized, mathematically sound version of your agent. It integrates the **Deep Active Inference** architecture (Ensemble Variance, Reward Predictor, Frozen JEPA) with the **Tri-Partite Intrinsic Drive** (Mass, Guidance, Constraint) into a single rigorous framework.

---

# The JEPA-VAE-GFE World Model

### *A Deep Active Inference Agent with Tri-Partite Intrinsic Motivation*

## 1. Executive Summary

This system defines an autonomous agent that minimizes **Generalized Expected Free Energy (GFE)** within a learned latent space. Unlike standard RL, it does not merely maximize scalar rewards; it acts to minimize the divergence between its internal world model and external reality (Self-Evidencing) while actively pursuing structured novelty.

The system solves the stability problems of previous iterations (e.g., "Noisy TV," "Manifold Collapse") by decoupling the **Physics Engine** (JEPA) from the **Observer** (VAE) and driving exploration through a **Tri-Partite Intrinsic Drive**:

1. **The Map (Epistemic Mass):** Drives the agent to resolve uncertainty about dynamics ().
2. **The Compass (Epistemic Guidance):** specific directional guidance toward the steepest increase in learnable novelty ().
3. **The Tether (Epistemic Constraint):** Prevents topological collapse by penalizing impossible states ().

---

## 2. Component I: The Physics Engine (JEPA)

*The Ground Truth State Space*

The foundation of the agent is the **Joint Embedding Predictive Architecture**. Its role is to learn a latent representation  that encodes the dynamics of the environment while discarding unpredictable noise. To ensure the "Map" (VAE) remains valid during planning, the JEPA representation is **frozen** (or updated via stop-gradient) relative to the planner.

### 2.1 The Equations

* **Observation Encoder:** 
* **Physics Dynamics (Ensemble):** To validly measure Information Gain, we use a Deep Ensemble of  independent predictors.
$$ \hat{z}*{t+1}^{(k)} = P*{\phi_k}(z_t, a_t) \quad \text{for } k=1 \dots K $$
* **Trajectory Mean:**
$$ \bar{z}*{t+1} = \frac{1}{K} \sum*{k=1}^K \hat{z}_{t+1}^{(k)} $$

**Training Objective:**
The JEPA minimizes InfoNCE loss to structure the manifold:
$$ \mathcal{L}*{JEPA} = -\mathbb{E} \left[ \log \frac{\exp(sim(\hat{z}*{t+1}, z'*{t+1}) / \tau)}{\sum*{j} \exp(sim(\hat{z}*{t+1}, z*{j}) / \tau)} \right] $$

---

## 3. Component II: The Observer (VAE Density)

*The Epistemic Potential Field*

A Variational Autoencoder (VAE) is trained on the **frozen** JEPA latents. It provides a differentiable scalar field representing the "Familiarity" or "Density" of the state space. It does not reconstruct pixels; it reconstructs .

### 3.1 The Equations

* **Input:** 
* **Surprisal Field ():** We define the potential energy  as the VAE's Negative Log-Likelihood (NLL).
$$ U(z) = \mathcal{L}*{VAE}(z) \approx || z - D*\psi(E_\psi(z)) ||^2 + \beta D_{KL} $$

*Note: This creates a "smooth surface" over the JEPA latents. Low  means the state is familiar; high  means it is novel.*

---

## 4. Component III: The Pragmatic Link (Reward Predictor)

*The Extrinsic Drive*

Since the JEPA discards pixel-level information, the agent cannot "decode"  to check for rewards. We substitute a differentiable Reward Predictor to link the latent manifold to extrinsic goals.

### 4.1 The Equations

* **Predictor:** 
* **Extrinsic Risk ():**
$$ R_{ext}(z_t) = - \hat{r}_t $$

---

## 5. Component IV: The Tri-Partite Intrinsic Drive

*The Core Innovation*

This section defines the mathematically valid replacement for heuristic creativity terms. It balances three specific forces: **Mass** (Uncertainty), **Guidance** (Gradient), and **Constraint** (Density).

$$ E_{epi}(z) = \lambda_{1} \underbrace{V_{ens}(z)}*{\text{Mass}} + \lambda*{2} \underbrace{\mathcal{G}*{ent}(z)}*{\text{Guidance}} - \lambda_{3} \underbrace{U(z)}_{\text{Constraint}} $$

### 5.1 The Map: Epistemic Mass

*Replaces  with Ensemble Variance.*
This term attracts the agent to states where the physics models disagree. It guarantees **Information Gain** (resolving ignorance).
$$ V_{ens}(z_t) = \frac{1}{K-1} \sum_{k=1}^K || \hat{z}*{t+1}^{(k)} - \bar{z}*{t+1} ||^2 $$

### 5.2 The Compass: Epistemic Guidance

*Replaces  with the Gradient of Entropy.*
This term provides a vector direction. Instead of just asking "Is this novel?", the agent calculates the slope of the VAE's surprisal surface to find the *direction* where novelty increases most rapidly.

* **Gradient Vector:** 
* **The Term:**
$$ \mathcal{G}_{ent}(z) = \tilde{I}(z) \cdot || g_H(z) ||_2 $$

### 5.3 The Safety: Learning Progress Gate

*The Noise Filter.*
To prevent the Compass from guiding the agent into unlearnable noise (the "Noisy TV" problem), we gate the gradient term with **Learning Progress** ().

* **Equation:**
$$ \tilde{I}(z_t) = \sigma \left( \gamma \cdot (\mathcal{L}*{JEPA}^{t-k} - \mathcal{L}*{JEPA}^{t}) - \beta \right) $$
* **Logic:** If  (Not Learning), , effectively disabling the Compass.

---

## 6. The Master Equation: Generalized Expected Free Energy

The final output of the system assembly is the unified control functional. The planner optimizes a policy  (sequence of actions ) to minimize the GFE .

$$ G(\pi) = \sum_{t=1}^{T} \mathbb{E}*{q*\pi} \bigg[ \underbrace{- M_\omega(\bar{z}*t)}*{\text{Pragmatic Value}} - \underbrace{E_{epi}(\bar{z}*t)}*{\text{Intrinsic Creativity}} \bigg] $$

Expanding the terms:

$$ G(\pi) = \sum_{t=1}^{T} \bigg( \underbrace{- M_\omega(\bar{z}*t)}*{\text{Reward}} - \lambda_1 \underbrace{V_{ens}(\bar{z}*t)}*{\text{Mass}} - \lambda_2 \underbrace{\left( \tilde{I}*t \cdot || \nabla*{\bar{z}} U(\bar{z}*t) || \right)}*{\text{Guidance}} + \lambda_3 \underbrace{U(\bar{z}*t)}*{\text{Constraint}} \bigg) $$

### Summary of Forces:

1. **:** "Go to high reward."
2. **:** "Go to where the physics is unknown."
3. **:** "Climb the slope of novelty (if learnable)."
4. **:** "Do not violate the laws of the latent manifold."

---

## 7. Implementation: The "Probe" Cycle

To implement the **Entropy Gradient** inside a differentiable planner, you must use a "Probe" step to avoid breaking the computational graph.

**Step 1: Simulation**
Run the ensemble forward to get the predicted future state .
$$ \bar{z}_{t+1} = \text{Ensemble}(z_t, a_t) $$

**Step 2: The Probe (Gradient Calculation)**
Detach the predicted state from the graph, enable gradients, and pass it through the VAE.
$$ z_{probe} = \bar{z}*{t+1}.\text{detach}().\text{requires_grad_}(True) $$
$$ \text{Surprisal} = U(z*{probe}) $$
$$ \text{GradVector} = \nabla_{z_{probe}} (\text{Surprisal}) $$
$$ \text{CompassVal} = || \text{GradVector} || $$

**Step 3: Evaluation**
Plug `CompassVal` back into the GFE equation (multiplied by the Gating term ) as a scalar reward for the original .