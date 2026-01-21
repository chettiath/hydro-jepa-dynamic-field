# The Engine: Tri-Partite Intrinsic Drive (3C)

*From Discrete Control to Cognitive Thermodynamics*

**Core Philosophy:** The Engine does not optimize a single scalar reward. It minimizes a **Free Energy Functional** composed of three competing tensions: **Curiosity**, **Competence**, and **Coherence**.

The "Total Drive" ($D_{total}$) is the vector sum of these three gradients. The agent moves in the direction that maximizes the release of this tension.

$$D_{total} = \alpha \nabla \mathcal{D}_{\text{Curiosity}} + \beta \nabla \mathcal{D}_{\text{Competence}} + \gamma \nabla \mathcal{D}_{\text{Coherence}}$$

---

## 1. Curiosity: The Expansion Force (Entropy Maximization)

*Consuming "Surprise" to fuel Exploration*

This is the home of the **Bellman Error** and **Surprise**. It is the force that pushes the agent away from the known and into the unknown.

**The Mechanism**

Curiosity is defined as the drive to maximize **Information Gain**. It seeks regions where the internal model's predictions are currently failing.

**Sublimation of Previous Blueprints**

- **Bellman Error as Raw Fuel:**
    - _Old Way:_ Calculate Bellman Error. If $> Threshold$, trigger "Switch."
    - _The Engine:_ Bellman Error is now **Potential Energy**. A state with high Bellman Error ($|V(s) - (r + \gamma V(s'))|$) is a high-energy state.
    - **Physics:** Just as a ball rolls downhill to minimize gravitational potential, the agent "rolls" toward high-error states to "discharge" that error (by learning it). The error *pulls* the agent.
    
- **Surprise as a Gradient Vector:**
    - The "Surprise" component of your Reward Cocktail (Gradient Cosine Dissimilarity) determines the *direction* of the pull. If the surprise is "orthogonal" to current knowledge (a new dimension), the Curiosity gradient becomes steep, overpowering the other drives.
    

> **The "Switch" Logic:** There is no switch. When prediction error spikes, the **Curiosity Gradient** simply becomes the dominant term in the equation ($| \nabla \mathcal{D}_{\text{Curiosity}} | \gg | \nabla \mathcal{D}_{\text{Competence}} |$). The agent naturally veers off the "Exploitation" path because the "Exploration" pull is suddenly stronger.

---

## 2. Competence: The Stabilization Force (Entropy Minimization)

*Consuming "Chaos" to fuel Mastery*

This is the counter-force to Curiosity. It seeks to minimize the variance of outcomes. It wants the agent to be effective, efficient, and in control.

**The Mechanism**

Competence is defined by **Empowerment** (maximizing the mutual information between Action and State). It drives the agent toward states where it has high agency and predictable results.

**Sublimation of Previous Blueprints**

- **The "Flow" State:**
    - While Curiosity seeks high gradients (learning), Competence seeks **zero gradients** (perfect prediction).
    - This absorbs the "Attraction" (Fisher Information) component. It locks the agent into regions where it is currently making progress.
    
- **Resistance to Switching:**
    - Competence provides the "Inertia." It prevents the agent from being distracted by trivial noise. It ensures the agent only "switches" strategies when the Curiosity signal (Prediction Failure) is strong enough to overcome the Competence inertia.
    

---

## 3. Coherence: The Integration Force (Complexity Gain)

*Consuming "Divergence" to fuel Insight*

This is the bridge between **The Body** (Reality) and **The Imagination** (The Construct). It absorbs the concept of "Interestingness" and "Learning Progress."

**The Mechanism**

Coherence measures the **rate of update** of the Surrogate Model ("The Construct"). It is driven by the derivative of knowledge acquisition.

**Sublimation of Previous Blueprints**

- **Learning Progress as "Interestingness":**
    - *Old Way:* Calculate "Accumulated Complexity Gain" to see if the agent is bored.
    - *The Engine:* When "The Construct" undergoes a massive update (e.g., a paradigm shift in the internal model), it releases "Coherence Energy."
    - **The Feedback Loop:** If an action causes the internal model to snap into better alignment with reality (a "Eureka moment"), the Coherence drive reinforces that action.
    
- **The Reality Governor Link:**
    - This drive is directly fed by **The Conscience** (Induction-on-Scales). If a local insight successfully "inducts" to a global scale, it generates a massive Coherence reward.
    

---

## The Dynamics: How "Failure" Becomes "Impulse"

This is the most critical sublimation. You asked how the "Energy Landscape" changes to replace the "Hard-Coded Switch."

**The Scenario: The Collapse of a Strategy**

Imagine the agent is using **Strategy A** (e.g., walking) and it works well (High Competence, Low Curiosity).

1. **The Trigger (Reality Shift):** The terrain changes to water. "Walking" no longer produces movement.
    
2. **The Divergence (Bellman Spike):**
    - **The Imagination** predicts: "Move forward."
    - **The Body** reports: "Stationary."
    - **The Result:** A massive spike in **Bellman Error** (Prediction Failure).
        
3. **The Landscape Shift (The "Roll"):**
    - Previously, the "Walking" valley was deep (Low Energy, comfortable).
    - Suddenly, due to the error spike, the "Walking" valley fills with **Curiosity Potential Energy**. It is no longer a stable valley; it is a high peak.
    - Simultaneously, the region of "Swimming" (or random flailing) represents a way to discharge this energy.
        
4. **The Impulse:**
    - The agent does not "decide" to swim.
    - The **Curiosity Gradient** points violently away from "Walking."
    - The **Competence Gradient** (which usually holds the agent in place) collapses because agency has dropped to zero.
    - The agent **"rolls"** down the new gradient into **Strategy B** (Swimming).
    

**Summary of the "Physics"**

- **Failure (Error)** $\rightarrow$ **Potential Energy (Curiosity)**.
- **Success (Mastery)** $\rightarrow$ **Inertia (Competence)**.
- **Insight (Update)** $\rightarrow$ **Acceleration (Coherence)**.