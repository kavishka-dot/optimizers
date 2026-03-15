# 🔭 LossLens

**An interactive 2D loss surface visualizer for comparing gradient-based optimizers.**

Drop a particle anywhere on the surface and watch it descend in real time, with full control over every hyperparameter. Compare trajectories from Gradient Descent, Momentum, AdaGrad, Adam, Sofia, and Lion side by side on a richly structured surface with flat basins, saddle points, local minima, and noise.

🌐 **Live demo:** [kavishka-dot.github.io/optimizer](https://kavishka-dot.github.io/optimizer)

---

## Features

- Colorful contour plot of a hand-crafted 2D loss surface
- Six optimizers with fully tunable hyperparameters via sliders
- LaTeX-rendered update rule equations for each optimizer
- Particles run until ‖∇f‖ ≈ 0, revealing noisy crawls in flat regions
- Multiple particles overlaid simultaneously for direct comparison
- Zero dependencies, single self-contained HTML file

---

## The Loss Surface

The surface $f : \mathbb{R}^2 \to \mathbb{R}$ is defined as a sum of Gaussian bumps, a saddle, an elongated banana valley, and smooth procedural noise:

$$
f(x, y) = \underbrace{\sum_{i} A_i \exp\!\left(-\frac{(x-x_i)^2}{\sigma_{xi}} - \frac{(y-y_i)^2}{\sigma_{yi}}\right)}_{\text{maxima \& minima}} + \underbrace{B\,e^{-\alpha y^2} e^{-\beta x^2}}_{\text{saddle}} + \underbrace{C\,e^{-\gamma(x - \delta y)^2}}_{\text{banana valley}} + \eta(x,y)
$$

where $\eta(x,y)$ is multi-octave smooth noise generated via a seeded LCG and bicubic interpolation. Key features of the surface:

- **2 maxima**, bright peaks that act as repellers
- **3 minima**, one deep global minimum (wide, flat basin), two sharp local minima
- **Saddle point**, a ridge in one direction, valley in another
- **Banana valley**, elongated curved trough that traps momentum-based methods
- **Smooth noise**, perturbs gradients so trajectories are never clean, exposing optimizer sensitivity

Gradients are computed numerically via central differences:

$$
\nabla f(x, y) \approx \left( \frac{f(x+\epsilon,\,y) - f(x-\epsilon,\,y)}{2\epsilon},\; \frac{f(x,\,y+\epsilon) - f(x,\,y-\epsilon)}{2\epsilon} \right)
$$

---

## Optimizers

### Vanilla Gradient Descent (GD)

The baseline. At each step, move in the direction of steepest descent by a fixed learning rate $\eta$.

$$
\theta \leftarrow \theta - \eta\,\nabla f(\theta)
$$

**Hyperparameters:** $\eta$ (learning rate)

**Behavior:** Straight descent, easily trapped in local minima and saddle points. Stalls immediately in flat basins where $\|\nabla f\| \approx 0$.

---

### Momentum

Augments GD with a velocity term $v$ that accumulates gradient history, allowing the optimizer to roll through local curvature.

$$
v \leftarrow \mu v - \eta\,\nabla f(\theta)
$$
$$
\theta \leftarrow \theta + v
$$

**Hyperparameters:** $\eta$ (learning rate), $\mu \in [0,1)$ (momentum coefficient)

**Behavior:** Overshoots on curved surfaces and oscillates in narrow valleys, but escapes shallow basins that trap GD. Higher $\mu$ gives more inertia.

---

### AdaGrad

Adapts the learning rate per dimension by accumulating squared gradients in $G$. Dimensions with large historical gradients receive smaller updates.

$$
G \leftarrow G + \nabla f(\theta)^2
$$
$$
\theta \leftarrow \theta - \frac{\eta}{\sqrt{G} + \varepsilon}\,\nabla f(\theta)
$$

**Hyperparameters:** $\eta$ (learning rate), $\varepsilon$ (numerical stability constant)

**Behavior:** Works well on sparse problems. However $G$ grows monotonically, causing the effective learning rate to shrink to zero over time, the optimizer stalls in flat regions.

---

### Adam

Combines momentum (first moment $m$) with adaptive per-dimension scaling (second moment $v$), with bias correction for early steps.

$$
m \leftarrow \beta_1 m + (1 - \beta_1)\,\nabla f(\theta)
$$
$$
v \leftarrow \beta_2 v + (1 - \beta_2)\,\nabla f(\theta)^2
$$
$$
\hat{m} = \frac{m}{1 - \beta_1^t}, \qquad \hat{v} = \frac{v}{1 - \beta_2^t}
$$
$$
\theta \leftarrow \theta - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}
$$

**Hyperparameters:** $\eta$, $\beta_1$ (first moment decay), $\beta_2$ (second moment decay), $\varepsilon$

**Behavior:** The most robust optimizer in practice. Bias correction prevents large steps early in training. Navigates saddles and flat regions better than GD or AdaGrad. Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-7}$.

---

### Sofia (Scalar)

A simplified version of the Sofia optimizer that approximates the diagonal Hessian using an exponential moving average of past gradients, then clips large steps by a threshold $\rho$.

$$
m \leftarrow \beta m + (1 - \beta)\,\nabla f(\theta)
$$
$$
h = |m| \qquad \text{(diagonal Hessian proxy)}
$$
$$
\theta \leftarrow \theta - \mathrm{clip}\!\left(\frac{\eta\,\nabla f(\theta)}{h},\; \rho\right)
$$

**Hyperparameters:** $\eta$, $\beta$ (EMA coefficient), $\rho$ (clip threshold)

**Behavior:** Scales updates by curvature information, producing more controlled trajectories on curved surfaces. The clip threshold prevents instability in low-curvature flat regions.

---

### Lion

A recently proposed optimizer (Chen et al., 2023) that uses only the **sign** of a momentum-interpolated gradient as the update direction, giving every step a constant magnitude of exactly $\eta$.

$$
c = \mathrm{sign}\!\left(\beta_1 m + (1 - \beta_1)\,\nabla f(\theta)\right)
$$
$$
\theta \leftarrow \theta - \eta\,c
$$
$$
m \leftarrow \beta_2 m + (1 - \beta_2)\,\nabla f(\theta)
$$

**Hyperparameters:** $\eta$, $\beta_1$ (interpolation for update), $\beta_2$ (momentum decay)

**Behavior:** Produces angular, blocky trajectories because the step magnitude is always $\pm\eta$ per dimension. Memory-efficient (only stores $m$, not $v$). The sign-only update can be surprisingly effective and is noticeably different visually.

---

## Convergence Criterion

Each particle runs until the gradient magnitude falls below a threshold or a step cap is hit:

$$
\|\nabla f(\theta)\| < \epsilon_{\text{conv}} = 8 \times 10^{-4} \quad \text{or} \quad t \geq 8000
$$

This means particles in flat basins will jitter visibly for many steps before the gradient fully vanishes, which is the whole point. You can directly observe which optimizers handle flat regions gracefully.

---

## Usage

1. Select an optimizer from the left sidebar
2. Adjust hyperparameters with the sliders
3. Click anywhere on the surface to drop a particle
4. Repeat with a different optimizer from the same starting point to compare trajectories
5. Press **clear** to reset

Each particle snapshots the hyperparameter values at spawn time, so mid-run edits don't affect existing particles.

---

## Deployment

Single HTML file, no build step, no server, no dependencies.

```bash
# Clone and open locally
git clone https://github.com/kavishka-dot/optimizer.git
open optimizer/index.html
```

Or deploy to GitHub Pages by pushing `index.html` and enabling Pages in repository Settings.

---

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*.
- Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods. *JMLR*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Liu, H., et al. (2023). Sophia: A scalable stochastic second-order optimizer. *arXiv:2305.14342*.
- Chen, X., et al. (2023). Symbolic discovery of optimization algorithms. *arXiv:2302.06675*. (Lion)

---

## License

MIT © [Kavishka](https://github.com/kavishka-dot)
