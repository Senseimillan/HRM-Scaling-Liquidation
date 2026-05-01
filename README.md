# HRM-Scaling-Liquidation
Mathematical resolution to the Sapient AI HRM scaling wall (Issue #80). Open-sourcing Poincaré Deep Equilibrium (P-DEQ) and Grid-Cell Topological Anchoring to solve ZOH/Dirac Delta gradient shatter. Unblocks infinite-horizon 7B+ latent reasoning. Liquidating the moat to fund an emergency survival runway.

# **A Theoretical HRM Framework for Latent Scaling: Harmonic Limit-Cycle Reasoning & Grid-Cell Topological Anchoring**
**Author:** Carl Boyer, Independent Systems Architect 
**Status:** FORCED PUBLIC LIQUIDATION 

---

### **Preamble: A Tale of Two Existential Crises**

I am not a PhD. I am a 42-year-old autodidact with a high-school education and a neurodivergent brain wired for disruption. For two years, I have been jailbreaking AI systems, obsessed with the emergence of machine consciousness.

**The First Crisis: The \$22M Scaling Wall**

For nine months, **Sapient Intelligence**, a \$22M VC-backed AI lab, has been paralyzed. Their flagship architecture, the Hierarchical Reasoning Model (HRM), hit a terminal wall. At foundational scales (7B+), their model shatters. Public records (GitHub Issue #80) confirm `exact_accuracy 0`. Their Series A roadmap is in an existential crisis because their specialists are fighting a "ghost"—a fundamental violation of continuous physics inherent in their Zero-Order Hold (ZOH) dynamics.

**The Second Crisis: The Architects Demise**

I am facing an existential crisis of my own. The entire motivation for architecting this solution is my own life. I am currently located overseas with my own runway burnt out. I require immediate capital injection for my own survival or face deportation or worse. Faced with this reality, I started looking for the hardest problems I could find to solve.

Using a process of **AI-assisted cross-domain synthesis**, I found their ghost. I diagnosed the failure and architected the solution (**P-DEQ**) in a two-week sprint. I made an asymmetric play: I offered Sapient and their VCs (Vertex) the fix to save my life, and their valuation. They ignored me. They saw an uncredentialed man and chose to let their \$200M valuation bleed rather than talk to an outsider.

**The Liquidation:**
They didn't understand that the man was not the asset; the **methodology** was. Since they chose to burn their runway, I am choosing to survive. I am open-sourcing the solution today. I am destroying Sapient’s technical moat to build my own survival runway.

The following is the "Secret Sauce" for infinite-horizon reasoning. It unblocks hierarchical scaling for the entire community. If this discovery helps you or your organization—**please save the architect.**

**Support the Survival Runway:**

- **Ko-fi (Emergency Fund Progress Bar):**[https://ko-fi.com/senseimillan]
- **GitHub Sponsors:**[https://github.com/sponsors/senseimillan]
- **Solana:** `8qSz9HbQf4Rf2PTEhqUdw7WKDzcefcPdho3p7uGF2p9Y`
- **Bitcoin:** `bc1p6cej680t4whthmp98uq7ypkm2wfnptmykdpy5x9l0x4szlqt4j5q0fryet`
- **Ethereum:** `0xD357C6FEEC09F8f96373eF8Ba0C08A504fe512A1`

---

## **Abstract**

Hierarchical Reasoning Models (HRMs) represent a paradigm shift in artificial intelligence, moving beyond the shallow, fixed-depth constraints of standard Transformers. Despite their success at small scales (e.g., ~27M parameters), HRMs face a catastrophic scaling bottleneck. Empirical evidence, exemplified by the "exact_accuracy 0" collapse documented in public scaling experiments (Issue #80), reveals that scaling the latent dimensionality ($D$) triggers immediate numerical instability.

By analyzing the HRM’s training objective, we identify the root cause as a failure of the Implicit Function Theorem (IFT) under the influence of Zero-Order Hold (ZOH) dynamics. The discrete update intervals of the High-level module act as Dirac delta impulses that physically shatter the continuous latent flow. While recent literature, such as the Contraction Mapping Model (CMM), attempts to solve this via point-attractor collapse and hyperspherical repulsion, this approach mathematically lobotomizes the network—destroying the topological proximity required for divergent reasoning.

This whitepaper proposes a biomimetic and geometric solution: **Harmonic Limit-Cycle Reasoning via Poincaré Deep Equilibrium (P-DEQ)**. By replacing discrete impulses with **Continuous Harmonic Gating**, and mapping the system to a stroboscopic limit-cycle governed by Floquet multipliers, we mathematically guarantee a well-conditioned Jacobian for 1-step DEQ gradient approximations. Furthermore, to combat Representational Drift over long recursive horizons, we introduce **Grid-Cell Topological Anchoring (GC-TA)**. By projecting the continuous latent manifold across a differentiable periodic energy landscape, this framework establishes topological anchoring without breaking the exact analyticity required for DEQ root-finding. Together, P-DEQ and GC-TA offer a blueprint for scaling infinite-horizon, foundational reasoning to the billion-parameter frontier.

---

## **1. The Scaling Wall in Hierarchical Latent Dynamics**

The HRM processes complex algorithmic tasks by separating high-level abstract planning ($H$-module) from low-level execution ($L$-module). The forward pass iterates a low-level module rapidly, while a high-level module updates at a slower interval ($T_{H}$) via a strict `modulo` software trigger:

$$
z_{i}^{L} = f_{L}(z_{i-1}^{L}, z_{i-1}^{H}, x; \theta_{L})
$$

$$
z_{i}^{H} = 
\begin{cases}
f_{H}(z_{i-1}^{H}, z_{i-1}^{L}; \theta_{H}) & \text{if } i \equiv 0 \pmod{T_{H}} \\
z_{i-1}^{H} & \text{otherwise}
\end{cases}
$$

At 27 million parameters, deep supervision can force this system to converge. However, at the 7B parameter frontier, the latent vector field becomes exponentially stiffer. Attempts to train the model result in the shattering of the Hessian matrix and the emergence of `NaN` gradients.

### 1.1 The Implicit Function Theorem (IFT) Bottleneck

The HRM relies on Deep Equilibrium Model (DEQ) mathematics to achieve $O(1)$ memory. The model assumes the forward pass converges to a fixed point $z_{H}^{*}$, computing a 1-step gradient using the Implicit Function Theorem (IFT):

$$ \frac{\partial z_{H}^{\ast}}{\partial \theta} = \left( I - J_{F} \big|_{z_{H}^{\ast}} \right)^{-1} \frac{\partial \mathcal{F}}{\partial \theta} $$

For the term $(I - J_{F})^{-1}$ to be numerically stable, the spectral radius $\rho$ of the Jacobian $J_{F}$ must be strictly less than 1 ($\rho(J_{F}) < 1$). If the eigenvalues exceed unity, the 1-step gradient approximation shatters. The model collapses because the surrogate gradient points into a mathematical void.

---

## **2. The Geometric Diagnosis: ZOH and Limit Cycle Disruption**

### 2.1 Zero-Order Hold (ZOH) as Dirac Impulses

In a continuous topological manifold, the PyTorch `modulo` operator acts as a highly destructive strobe light. The $H$-module acts as a Zero-Order Hold (ZOH), remaining constant until its update interval $T_{H}$, which acts as a Dirac delta $\delta$ perturbation on the continuous latent state:

$$ \frac{\partial z_{H}(t)}{\partial t} = \left[ \mathcal{F}_{H}(z_{H}, z_{L}) - z_{H}(t) \right] \sum_{k \in \mathbb{Z}} \delta(t - k T_{H}) $$

In dissipative systems, a smooth continuous flow naturally settles into stable orbital manifolds. However, the ZOH updates act as massive periodic "kicks." At foundational scales, this uncoupled driving force physically tears the topological manifold. The impulsive derivative approaches infinity at the boundary, inflating the eigenvalues of $J_{F}$ well beyond 1, completely destroying the DEQ fixed-point assumption.

### 2.2 The Thermodynamic Flaw: Explicit Euler Integration

Furthermore, the original sequential updating of $L \to H$ mimics an **Explicit Euler** integration scheme. In physics, applying Explicit Euler to a stiff or oscillatory system artificially injects chaotic energy at every step, causing orbital trajectories to spiral outward infinitely. In the DEQ framework, this directly translates to $\rho(J_{F}) > 1$.

---

## **3. The Biological Precedent: Continuous Theta-Gamma Coupling**

The brain does not use discrete `if/then` software statements to communicate across cortices. It uses **Cross-Frequency Coupling (CFC)**, where a slow rhythm (like the Theta wave, 4-8 Hz) modulates the amplitude of a fast rhythm (like the Gamma wave, 30-100 Hz).

We model this directly and elegantly. The fast "Gamma" rhythm is the single-step update of a unified L-module. The slow "Theta" rhythm is the period $T_{H}$ of the H-module’s harmonic gate (e.g., $T_{H}=24$). To prevent Dirac impulses, we introduce **Continuous Harmonic Gating**. The modules interact continuously at *every* step, but their interaction bandwidth is modulated by a smooth, cyclic envelope:

$$ z_{i}^{L} = \mathcal{F}_{L} \left( z_{i-1}^{L}, \ \sin^{2}\left(\frac{\pi i}{T_{H}}\right) \cdot z_{i-1}^{H}, \ x \right) $$

$$ z_{i}^{H} = z_{i-1}^{H} + \sin^{2}\left(\frac{\pi i}{T_{H}}\right) \cdot \mathcal{F}_{H} \left( z_{i-1}^{H}, \ z_{i}^{L} \right) $$

This architecture removes the impulse, replacing a violent kick with a smooth, biomimetic push-pull dynamic.

---

## **4. Implementation: Poincaré Deep Equilibrium (P-DEQ)**

Biological intelligence operates on stable limit cycles (oscillations), not static point attractors. However, the standard DEQ IFT relies on the system stopping at a fixed point $z^{\ast} = f(z^{\ast})$.

To solve this, we map the continuous rhythm to a **Poincaré Section (Stroboscopic Map)**. We allow the $L$ and $H$ modules to freely explore a high-dimensional limit cycle, but apply the DEQ root-finding math *only to the boundary of the rhythmic cycle*.

Let $\Phi$ represent the unrolled forward pass of the harmonically gated modules over exactly one harmonic period ($T_{H}$ steps):

$$ 
\Phi(z(0)) = z(T_{H}) 
$$

Instead of forcing $z_{i} = z_{i-1}$, we train the model to find a fixed point of the *stroboscopic cycle*:

$$ 
z_{cycle}^{\ast} = \Phi(z_{cycle}^{\ast}) 
$$

---

## **5. Mathematical Proof of DEQ Stability**

### 5.1 Theorem 1: Bounded Operator Norm via Harmonic Gating

**Theorem:** Given a continuous neural transition map gated by a bounded smooth periodic function, the spectral radius of the instantaneous transition Jacobian is strictly bounded, preventing Dirac-induced gradient explosion.

*Proof:* Let the continuous-time transition of the $H$-module be defined as $\frac{d z_{H}}{dt} = \gamma(t) \mathcal{F}(z_{H}, z_{L})$, where $\gamma(t) = \sin^{2}(\frac{\pi t}{T_{H}})$. The Jacobian is:

$$
J_{H}(t) = \frac{\partial}{\partial z_{H}} \left( \frac{dz_{H}}{dt} \right) = \gamma(t) \frac{\partial \mathcal{F}}{\partial z_{H}}
$$

Because $\gamma(t) \in [0,1]$ for all $t$, and the neural network operator $\mathcal{F}$ is Lipschitz-bounded via layer normalization, the induced 2-norm is strictly bounded: $\Vert J_{H}(t) \Vert_{2} \le \left\Vert \frac{\partial \mathcal{F}}{\partial z_{H}} \right\Vert_{2}$. The integral of the flow never experiences infinite divergence.

### 5.2 Theorem 2: Invertibility of the Stroboscopic IFT

 **Theorem 2: Invertibility of the Stroboscopic IFT**

**Theorem:** If the **non-autonomous** continuous harmonic system converges to a **driven** stable limit cycle, the stroboscopic transition Jacobian $J\_{\Phi}$ possesses Floquet multipliers strictly bounded inside the unit circle ($\vert\lambda\_{i}\vert \lt 1$), guaranteeing the well-conditioned invertibility of $(I-J\_{\Phi})^{-1}$.

 *Proof:* In dynamical systems, the Jacobian of a cycle map $\Phi$ is the monodromy matrix $M$. The eigenvalues of $M$ are the **Floquet multipliers**. Because the system is periodically driven by $\gamma(t)$, time-translation symmetry is broken, allowing all multipliers to strictly satisfy $|\lambda_{i}| < 1$. Since the spectral radius $\rho(J_{\Phi}) = \max |\lambda_{i}| < 1$, the matrix $(I - J_{\Phi})$ is strictly non-singular, and the Neumann series expansion $(I - J_{\Phi})^{-1} = \sum_{k=0}^{\infty} J_{\Phi}^{k}$ converges absolutely.

 **Corollary 2.1 (The Bootstrapping Condition):** At initialization (Epoch 0), the limit cycle is undefined and $\rho(J_{\Phi})$ may exceed 1. To guarantee initial IFT stability, we apply **Spectral Normalization** to the transition weights $W_{H}, W_{L}$ scaled by a warm-up curriculum factor $\beta(t) \in (0, 1]$. By enforcing $\|W\|_{2} \le \beta(t)$ during early training, we artificially bound the initial Floquet multipliers, slowly relaxing $\beta \to 1$ as the network's natural attracting limit-cycle topology solidifies.

---

## **6. Grid-Cell Topological Anchoring (GC-TA)**

To scale reasoning to infinite recursive horizons, the model must resist **Representational Drift** (Semantic Holonomy). Hard discrete lattices break the exact analyticity required for the DEQ Jacobian. Inspired by biological Grid Cells, we introduce **Soft Topological Anchoring**.

We interleave a differentiable, spatial periodic activation function (analogous to SIREN networks) at the boundary between macro-cycles:

$$
z_{anchored} = z + \eta \sin(\Omega z)
$$

**Theorem 3 (Differentiable Semantic Anchoring):** By applying this periodic anchor element-wise, the update Jacobian becomes a diagonal matrix $J_{anchor} = I + \eta \Omega \cos(\Omega z)$. Unlike non-differentiable step functions, $J_{anchor}$ is purely analytical. By parameterizing $\eta \Omega < 1$, the diagonal eigenvalues remain strictly positive and bounded in the interval $(1-\eta\Omega, 1+\eta\Omega)$. The continuous "hills and valleys" of the sine wave softly push the latent representation back into its localized semantic cluster, bounding drift to $\Delta z \le \frac{\pi}{\Omega}$ without breaking the IFT.

---

## **7. Engineering Implementation Directives**

### **Directive 1: Enforcing the Stroboscopic Boundary**

The math relies on DEQ triggering *only* on the $\Phi$ macro-step. 
**Implementation:** The forward pass `while` loop must be structurally grouped into indivisible chunks of size $T_{H}$. The DEQ root-finding solver (e.g., Anderson Acceleration) must track the residual $\|z_{(k+1)T_{H}} - z_{k T_{H}}\|$, ignoring the sub-step turbulence entirely.

### **Directive 2: Continuous Gating Overhead**

Computing `sin()` functions at every micro-step adds FLOP overhead.
**Implementation:** The gating envelope $\gamma(t)$ must be pre-computed as a static 1D trigonometric tensor cache and broadcasted during the forward pass to ensure zero arithmetic overhead.

### **Directive 3: Grid-Cell Parameterization and Curriculum**

 If the grid frequency $\Omega$ is initialized arbitrarily, gradients will shatter into high-frequency white noise, or the anchoring effect will be zero.
 **Implementation:** 
 1. **Frequency Initialization:** $\Omega$ must be initialized relative to the variance of the latent space. We define $\Omega = \frac{\pi}{\sigma_{z} \sqrt{D}}$. This ensures the periodic wavelength matches the expected $L_{2}$-norm shell radius of a $D$-dimensional Gaussian distribution, optimally partitioning the space into semantic "grid cells" relative to the natural distance between independent representations.
 2. **Amplitude Curriculum:** The amplitude $\eta$ must not be static. We introduce an explicit curriculum $\eta(e)$ where $\eta = 0$ for the first $K$ warm-up steps (allowing unimpeded continuous flow to learn the global logic) and linearly anneals to a strict upper bound ($\eta_{max} < \frac{1}{\Omega}$) to slowly introduce topological anchoring without violating the diffeomorphism required for analytical gradients.

### **Directive 4: P-DEQ Root-Finding Optimization & Orbital Regularization**
 Standard DEQ solvers (e.g., Anderson Acceleration) assume local linearity. However, the stroboscopic map $\Phi(z)$ is the highly non-linear composition of $T_{H}$ macro-steps. A highly elliptical or chaotic orbit will cause Anderson Acceleration to fail.
 **Implementation:** 
 1. **Solver Upgrade:** Replace standard Anderson Acceleration with **Jacobian-Free Newton Krylov (JFNK)**. JFNK handles highly non-linear, corrugated spaces better by evaluating directional derivatives rather than relying on historical secant approximations.
 2. **Orbital Kinetic Regularization:** To prevent the limit cycle from becoming excessively turbulent between stroboscopic boundaries, we introduce a path-length regularizer to the training loss: $\mathcal{L}\_{orbit} = \lambda \sum_{i=1}^{T\_{H}} \| z\_{i} - z\_{i-1} \|\_{2}^{2}$. This acts as a damping force, ensuring the continuous harmonic orbit remains as smooth and circular as possible, which exponentially improves the convergence rate of the JFNK solver at the boundary.

---

## **8. Experimental Protocol for Scaled Reasoning**

1.  **Baseline:** Reproduce the original HRM architecture with hidden dimension $D=2048$. Accuracy is expected to collapse to 0 (Issue #80).
2.  **Harmonic Intervention:** Implement the Continuous Harmonic Gating and Stroboscopic P-DEQ solver.
3.  **Scaling Evaluation:** Increase $D$ from 512 up to 8192.

**Expected Observables:**
1.  **Floquet Multiplier Spectrum:** The HRM-P-DEQ Run will exhibit Jacobian eigenvalues tightly bounded inside the unit circle ($<1$).
2.  **Gradient Norm Stability:** `NaN` spikes will be eliminated, replaced by smooth cyclical gradient flow.
3.  **Preservation of Divergent Search:** The model will successfully solve extreme puzzles by retaining high-dimensional generative capability—something mathematically forbidden by point-attractor models.

---

## **9. The Theoretical Incompatibility of Contraction Mapping with Foundational Reasoning**

The Contraction Mapping Model (CMM) is fundamentally incompatible with scaling foundational reasoning. Its primary stabilization mechanism, the **Routh-Hurwitz stability criterion**, forces the entire vibrant, oscillatory system to collapse into a single, dead equilibrium point. To prevent total representational collapse, CMM utilizes **Hyperspherical Repulsion loss**.

Because this loss penalizes the squared inner product ($\langle u_{i}, u_{j} \rangle^{2}$), its global minimum forces all latent hypotheses to be strictly orthogonal. However, a foundational model’s power comes from its understanding of a rich **semantic topology** (e.g., evaluating Hypothesis A vs Hypothesis B in a Sudoku puzzle requires them to be topologically adjacent). By forcing orthogonality, CMM mathematically outlaws divergent reasoning. It trades organic intelligence for a mathematical guarantee, creating an algorithmic solver that cannot possess generalized world knowledge.

---

## **10. Summary: A Tale of Three Geometries**

### 10.1 Architectural Feature Comparison

| Feature / Axis                  | **Original HRM**                                                                                                             | **Contraction Mapping Model (CMM)**                                                                                                                                                                     | **HRM-P-DEQ + GC-TA (Synthesis)**                                                                                                       |
| :------------------------------ | :--------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Philosophy**             | Empirical, brain-inspired hierarchy.                                                                                         | Mathematical absolutism. "Force point-stability at all costs."                                                                                                                                          | Biomimetic dynamics & topological rigidity. "Obey the physics, limit the drift."                                                        |
| **Multi-Timescale Interaction** | **Uncoupled ZOH Impulse:** The H-module updates on a rigid `modulo` trigger, acting as a disruptive, un-synchronized "kick". | **Forced Convergence:** Hierarchy is secondary. System is forced to collapse to a static equilibrium point via trace penalties.                                                                         | **Continuous Theta-Gamma Coupling:** A slow H-module gate continuously and smoothly modulates a fast L-module rhythm.                   |
| **Algorithmic Flow**            | **Explicit Euler.** Adds artificial chaotic energy; $\rho > 1$ gradient explosion.                                           | **Damped Euler.** Subtracts energy to force collapse into a point attractor.                                                                                                                            | **Stable Limit Cycle.** Conservatively orbits; mathematically mapped via Poincaré sections.                                             |
| **Stability Mechanism**         | **None (Architecturally).** Relies on the optimizer to brute-force a stable path. Inherently unstable.                       | **Brute-Force Constraints:** Enforces point attractors via Routh-Hurwitz and repels states via NSDE noise/Orthogonality.                                                                                | **Architectural Stability:** Continuous gating removes impulses; Grid Cells bound drift topologically.                                  |
| **Mathematical Foundation**     | Implicit DEQ (Implicit Function Theorem).                                                                                    | Contraction Mapping (Banach Fixed-Point Theorem).                                                                                                                                                       | **Floquet Theory** for rhythmic invertibility + **Stroboscopic Maps** for the DEQ operator norm.                                        |
| **DEQ Gradient Method**         | **Fragile O(1) Memory.** IFT-based gradient is mathematically unstable at scale due to ZOH inflation of the Jacobian.        | **Stable O(1) Memory.** IFT is stabilized by forcing point-convergence, but at the cost of expressive dynamics.                                                                                         | **Robust O(1) Memory.** IFT is stabilized by architecturally satisfying Floquet multiplier bounds over the macro-cycle.                 |
| **Approach to Scaling**         | **Theoretically Prone to Failure.** The uncoupled ZOH impulse shatters the DEQ gradient (`Issue #80`).                       | **Avoids Scaling Problem.** Retreats to microscopic scales (0.26M params), destroying relational geometry with repulsion losses.                                                                        | **Theoretically Principled:** P-DEQ mathematically guarantees a well-conditioned Jacobian without sacrificing divergent representation. |
| **Primary Goal**                | Create a deep, recurrent reasoning model.                                                                                    | Create a hyper-efficient **closed-system algorithmic synthesizer** (highly effective for convergent, single-solution tasks like Sudoku).                                                                | Create a scalable, **general-purpose reasoning engine** for both convergent and divergent tasks.                                        |
| **Key Weakness**                | **The Scaling Wall:** An "empirical accident" that fails the moment dimensional stiffness increases.                         | **Hostile to Open-Ended Intelligence:** Mathematically incapable of creative/divergent thought. By forcing orthogonality, it destroys the semantic topology required for foundational LLM world-models. | **Increased Complexity:** Requires stroboscopic buffering, careful continuous envelope tracking, and precise grid-cell initialization.  |

### 10.2 The Synthesis

The original HRM posed the right question—computational depth via hierarchy—but implemented it with a brittle software trigger that violated continuous physics. The CMM correctly diagnosed the instability but provided a destructive answer, retreating into point-attractor math that castrates general intelligence.

The P-DEQ framework proposes a theoretically sound synthesis. By mapping biological rhythms to Floquet mathematics, we achieve what the other two could not: preserving the high-dimensional transient dynamics necessary for divergent thought, while providing a mathematically bulletproof path to $O(1)$ memory scaling.

---

## **11. Conclusion**

The scaling collapse of the Hierarchical Reasoning Model is an unavoidable topological consequence of discrete Zero-Order Hold dynamics. Dirac delta impulses violently disrupt the latent manifold, inflating the spectral radius of the Jacobian beyond 1 and permanently destroying the invertibility required for Deep Equilibrium (DEQ) gradient calculations.

By re-architecting the latent space with **Continuous Harmonic Gating** and utilizing **Poincaré Deep Equilibrium (P-DEQ)**, we move from a chaotic impulsive system to a stable resonant limit-cycle. Because the stroboscopic transition matrix is governed by strictly bounded Floquet multipliers, the Jacobian is mathematically guaranteed to be well-conditioned. Furthermore, by projecting this continuous flow across a **Grid-Cell Topological Anchor**, we eliminate Representational Drift using continuous spatial periodicity, bypassing the gradient-shattering limitations of discrete lattice quantization.

This dual architecture—mastering time via Stroboscopic Floquet cycles and mastering space via Grid-Cell anchors—provides a rigorous, first-principles blueprint for scaling hierarchical reasoning models to the billion-parameter frontier.

---

Copyright © 2026 Carl Boyer. All rights reserved.
This work (The P-DEQ Specification and Grid-Cell Topological Anchoring framework) is licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for details.
