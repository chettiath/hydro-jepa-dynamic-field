## 1. The Tri-Partite Intrinsic Drive (Map-Compass-Gate)

*The Core Innovation*

This section defines the three specific balancing forces of the epistemic intrinsic reward: **The Map** (Epistemic Topography), **The Compass** (Gradient of Uncertainty), and **Learning Progress Gate** (Epistemic Constraint).

$$ r_{t}^{\text{int}} = \lambda_{1} \cdot \underbrace{V_{\text{ens}}(\bar{z}_t, a_t)}_{\text{Map}} + \lambda_{2} \cdot \underbrace{G_{\text{ent}}(\bar{z}_t)}_{\text{Compass}} - \lambda_{3} \cdot \underbrace{U(\bar{z}_t)}_{\text{Learning Progress Gate}} $$

## 1.1 The Map: Epistemic Topography (Ensemble Disagreement)

This term attracts the agent toward regions where an ensemble of predictive dynamics models disagree. Ensemble disagreement is a proxy for epistemic uncertainty; when gated by local learning progress (Section 5.3), it preferentially targets reducible uncertainty (i.e., uncertainty that can actually be learned away), rather than irreducible noise (the "Noisy TV" failure mode).

Ensemble variance (scalar epistemic topographical map) defined as:

$$ \mathcal{v}_{\text{ens}}(\bar{z}_t, a_t) = \frac{1}{K-1}\sum_{k=1}^{K}\left\|\hat{z}_{t+1}^{(k)}-\bar{z}_{t+1}\right\|_2^2 $$
Finally, apply the local learning-progress gate (defined in 5.3):

$$ V_{\text{ens}}(\bar{z}_t, a_t) = \tilde{I}\left(r_t\right)\cdot\mathcal{v}_{\text{ens}}(\bar{z}_t, a_t) $$

where $(r_t)$ is the region/context index associated with the current state (defined below).


## 1.2 The Compass: Gradient of Uncertainty (VAE Surprisal Field)

This term provides a directional intrinsic drive by following the gradient of an uncertainty / surprisal field over the JEPA latent space. Let the VAE define an energy-like surprisal proxy $(U(z))$ derived from the negative ELBO:

$$ U(\bar{z}_t)\approx-\mathrm{ELBO}(\bar{z}_t) $$
Define a density-like proxy:

$$ P_{\text{density}}(\bar{z}_t)=\exp\left(-\frac{U(\bar{z}_t)}{T}\right) $$
Compute the gradient of uncertainty (the "compass direction"):

$$ g_H(\bar{z}_t) = \nabla_{\bar{z}_t}U(\bar{z}_t) $$
Define the compass magnitude:

$$ \mathrm{CompassVal}(\bar{z}_t) = \left\|g_H(\bar{z}_t)\right\|_2^2 $$
Define the learning progress gate guidance term (same local gate as the Map):

$$ G_{\text{ent}}(\bar{z}_t) = \tilde{I}\left(r_t\right)\cdot\mathrm{CompassVal}(\bar{z}_t)\cdot\exp\left(\lambda_{p} \cdot P_{\text{density}}(\bar{z}_t)\right)$$
- $\exp(\lambda_p \cdot P_{\text{density}})$ factor biases guidance toward gradients that lie on/near high-density (structured) regions, suppressing "pure noise" basins.
- Gate $\tilde{I}(r_t)$ ensures guidance is active only where local learning progess is actually happening (5.3).

## 1.3 The Learning Progress Gate: Epistemic Constraint  (Noise Filter)

To prevent both the Map and Compass from driving behavior into irreducible stochasticity (e.g., "Noisy TV" pixel noise), we gate them using a local learning-progress signal computed per region / per context anchor, not globally.

### 1.3.1 Region / context assignment (locality)

Let $\rho(\cdot)$ map a latent (or its associated context anchor) to a discrete region index:

$$ r_t = \rho(\bar{z}_t) $$
Examples of $\rho(\cdot)$:
- learned clustering over latents,
- discretized grid bin in latent space,
- "anchor state" identity in toy domains,
- k-means codebook index,
- nearest prototype,
- any stable partition used to assign per-context JEPA losses to local buckets.    

### 1.3.2 Local JEPA loss tracking (per-region)

For each JEPA training example $i$ at update step $t$, you have a per-example JEPA loss $\ell_{\text{JEPA}}^{(i)}(t)$ and an anchor region $r^{(i)}=\rho(z^{(i)}_{\text{anchor}})$.

Maintain a per-region exponentially weighted moving average (EWMA) of JEPA loss:

$$ \bar{\mathcal{L}}_{\text{JEPA}}(r,t) = (1-\alpha)\cdot\bar{\mathcal{L}}_{\text{JEPA}}(r,t-1) + \alpha\cdot\mathbb{E}\left[\ell_{\text{JEPA}}^{(i)}(t)\mid r^{(i)}=r\right]$$
Define the local learning progress over a lag of $k$ updates:

$$ \Delta\bar{\mathcal{L}}_{\text{JEPA}}(r,t) = \bar{\mathcal{L}}_{\text{JEPA}}(r,t-k) - \bar{\mathcal{L}}_{\text{JEPA}}(r,t)$$
- If $\Delta\bar{\mathcal{L}}_{\text{JEPA}}(r,t)$ is positive and large, JEPA is improving locally in region $r$.
- If it is near zero or negative, JEPA is not learning locally in that region.

### 1.3.3 Gate

Use a clipped rational gate that is near-zero unless progress exceeds a margin $\beta$:

Let:

$$ x(r_t,t)=\max\left(0,\Delta\bar{\mathcal{L}}_{\text{JEPA}}(r_t,t)-\beta\right) $$
Define the gate as:

$$ \tilde{I}(r_t) = \mathrm{clip}\left(\frac{x(r_t,t)}{c+x(r_t,t)},0,1\right) $$
- $\beta$ > 0 sets the "meaningful progress" threshold.
- $c$ > 0 controls saturation speed (smaller $c$ makes the gate turn on harder).

### 1.3.4 Effect: Disabling Map and Compass in unlearnable noise

If local progress is absent $(\Delta\bar{\mathcal{L}}\le \beta)$, then $x$ = 0 and:

$$ \tilde{I}(r_t)=0 $$
So both intrinsic drivers shut off locally:

$$ V_{\text{ens}}(\bar{z}_t,a_t)=0, \qquad G_{\text{ent}}(\bar{z}_t)=0 $$
If local progress is strong:

$$ \tilde I \to 1 $$
This is the intended "Noisy TV" safety behavior: irreducible noise yields no local learning progress, so intrinsic attraction disappears.

### 1.3.5 Alternatives

A lot of curiosity work moved from raw prediction error to things like:

#### A) Learning progress (Schmidhuber-style “compression progress” / learning-curve slope)

**Core idea:** don’t reward *being wrong*; reward *getting better* at modeling/compressing what you’ve seen. In Schmidhuber’s framework, the agent maintains (i) a **predictor/compressor** whose performance on the growing history can improve, and (ii) a **controller** (RL policy) that chooses actions to maximize *expected future improvement* of that predictor/compressor.

**Common methodology (as implemented in practice):**

1. **Maintain a learnable world model / compressor** $p(t)$ that predicts observations (or compresses the history) given past interaction data $h(\le t)$. This can be a dynamics model, predictive RNN, transformer, etc. Schmidhuber explicitly treats prediction as a route to compression: better prediction ⇒ fewer bits needed to encode the history.
    
2. **Choose a performance functional $C(p, h)$** that measures “how well the model explains/compresses the same data.” In the paper, $C$ is framed in description-length terms (bits to encode), optionally trading off runtime as well. In modern instantiations, $C$ is often a proxy like negative log-likelihood, cross-entropy, MDL-style code length, or reconstruction loss.
    
3. **Compute intrinsic reward as improvement on a matched evaluation set.** The canonical definition is: evaluate an “old” model $p(t)$ and an updated model $p(t+1)$ **on the same reference data**, then reward the reduction in cost. Schmidhuber writes this as a discrete-time progress reward of the form:

$$ r_{\text{int}}(t+1)=f\!\left(C(p_{\text{old}},h),\,C(p_{\text{new}},h)\right) $$
* with the most obvious choice being $f(a,b)=a-b$ (saved bits / reduced cost). The key point is _progress_, not raw error.
    
4. **Handle timing/credit assignment.** Schmidhuber notes an **asynchronous** setup: the compressor may train/evaluate over many steps, and the resulting progress reward can be delayed relative to the actions that produced the data. Practically, many systems approximate this by using (a) sliding windows, (b) periodic evaluation checkpoints, or (c) per-batch “before/after” loss deltas to reduce delay.
    
5. **Use an RL controller to seek learnable novelty.** The policy is trained to select actions expected to produce data that is *currently not well-modeled but is learnable*, because that maximizes the “steepness of the learning curve.” This is the mechanism that avoids getting stuck on (i) fully predictable regions (no progress) and (ii) pure noise (no progress).

#### B) Information gain about parameters (a.k.a. Bayesian surprise / parameter-uncertainty reduction)

**Core idea:** prefer experiences that *meaningfully update* the agent’s beliefs about its model parameters (or latent hypotheses), not merely those with high instantaneous error. High error caused by irreducible noise is not rewarded unless it changes what the model can learn.

**Common methodology:**

1. **Represent epistemic uncertainty over model parameters (or hypotheses).** Maintain a belief distribution $p(\theta \mid D)$ over parameters $\theta$ given data $D$ (interaction history). In modern deep RL this is usually approximated via:
    - ensembles (bootstrapped models),
    - Bayesian neural nets / variational posteriors,
    - particles (particle filters) or Laplace approximations,
    - or implicit uncertainty proxies (e.g., ensemble disagreement).
    
2. **Define “information gain” as a KL between posterior and prior.** For a candidate action $a$, the agent considers possible next observations $o$ and measures how much the belief would change:

$$ \text{IG}(a)=\mathbb{E}_{o \sim p(o \mid a, D)}\left[\mathrm{KL}\!\left(p(\theta \mid D,o,a)\,|\,p(\theta \mid D)\right)\right] $$
* Equivalent views: expected entropy reduction $H[p(\theta\mid D)]- \mathbb{E}_{o}[H[p(\theta\mid D,o,a)]]$, or expected reduction in posterior variance.
    
3. **Plan/act to maximize expected IG (or use it as intrinsic reward).**
    - **Planning form:** choose $a$ that maximizes $\text{IG}(a)$ directly (active learning / optimal experiment design flavor).
    - **RL form:** set $r_{\text{int}} = \text{IG}$ (or an approximation) and learn a policy that maximizes long-run cumulative IG.
    
4. **Approximate IG efficiently (what people actually do):**
    - **Ensemble disagreement:** treat variance across ensemble predictions as a proxy for epistemic uncertainty; large disagreement implies an observation will likely move parameters (high $\text{IG}$).
    - **Bayesian surprise proxy:** compute KL between “pre-update” and “post-update” predictive distributions or parameter distributions.
    - **Gradient/Fisher proxies:** approximate “how much parameters would move” if trained on that transition (useful when full Bayesian updates are infeasible).
    
5. **Why this differs from raw error:** the signal is explicitly about *learnable structure*. If an input is unpredictable due to noise, the posterior won’t concentrate and the KL/entropy reduction stays small—so the agent won’t chase it.

#### C) Novelty distance in learned latent spaces (representation-based novelty, not “noisy surprise”)

**Core idea:** compute novelty **in a representation space** where semantically meaningful differences are preserved and nuisance noise is collapsed. Then reward states that are “far” from what has been seen *in that latent geometry*, rather than states that simply yield high pixel prediction error.

**Common methodology:**

1. **Learn an embedding $z = f_\phi(o)$** from observation $o$ using self-supervised or auxiliary objectives (contrastive learning, autoencoding, predictive coding, inverse/forward models, etc.). The goal is that $z$ captures *regularities and invariances* (so sensor noise doesn’t dominate).
    
2. **Maintain a novelty memory in latent space.**
    - A replay buffer of past embeddings {$z_i$},
    - or a compact episodic memory (e.g., per-episode dictionary),
    - or a density model over $z$.
    
3. **Compute novelty as distance or density.** Typical scoring functions:
* **kNN distance:**

$$ r_{\text{int}}(z)=\text{mean}_{k}\|z - \text{kNN}_k(z)\| \quad \text{(or min distance)} $$
* **Kernel / density:**

$$ r_{\text{int}}(z)= -\log \hat{p}(z) $$
* where $\hat{p}$ is a learned density model (KDE, normalizing flow, Gaussian mixture, etc.).
* **Count-style in latent space:** discretize/hash $z$ and use pseudo-counts (rare hash buckets → higher reward).
    
4. **Stabilize the metric so “novelty” doesn’t become noise chasing.**
    - Normalize embeddings and distances; use cosine distance in high-dim.
    - Update representations slowly (target networks) so the novelty measure doesn’t drift too fast.
    - Use learned representations that are explicitly invariant to stochastic textures / observation noise.
    
5. **Common practical patterns:**
    - **Episodic novelty + long-term novelty:** reward being new _this episode_ (exploration) but decay across episodes to avoid revisiting the same novelty endlessly.
    - **Combine with learning-progress gating:** novelty gets rewarded most when it also yields learning progress (or reduces uncertainty), which further suppresses “random but far” artifacts.