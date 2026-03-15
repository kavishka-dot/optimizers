# 🔭 LossLens

**An interactive 2D loss surface visualizer for comparing gradient-based optimizers.**

Drop a particle anywhere on the surface and watch it descend in real time , with full control over every hyperparameter. Compare trajectories from Gradient Descent, Momentum, AdaGrad, Adam, Sofia, and Lion side by side on a richly structured surface with flat basins, saddle points, local minima, and noise.

🌐 **Live demo:** [kavishka-dot.github.io/optimizer](https://kavishka-dot.github.io/optimizer)

---

## Features

- Colorful contour plot of a hand-crafted 2D loss surface
- Six optimizers with fully tunable hyperparameters via sliders
- LaTeX-rendered update rule equations for each optimizer (rendered in-app via MathJax)
- Particles run until `‖∇f‖ ≈ 0` , revealing noisy crawls in flat regions
- Multiple particles overlaid simultaneously for direct comparison
- Zero dependencies , single self-contained HTML file

---

## The Loss Surface

The surface is a sum of Gaussian bumps, a saddle, an elongated banana valley, and smooth procedural noise. In compact form:

```
f(x, y) = Σ Aᵢ · exp(-(x-xᵢ)²/σxᵢ - (y-yᵢ)²/σyᵢ)   ← maxima & minima
         + B · exp(-α·y²) · exp(-β·x²)                  ← saddle
         + C · exp(-γ·(x - δy)²)                         ← banana valley
         + η(x, y)                                        ← smooth noise
```

Gradients are computed numerically via **central differences**:

```
∂f/∂x ≈ (f(x+ε, y) - f(x-ε, y)) / 2ε
∂f/∂y ≈ (f(x, y+ε) - f(x, y-ε)) / 2ε
```

Key surface features:

| Feature | Description |
|---|---|
| 2 maxima | Bright peaks acting as repellers |
| 1 global minimum | Deep, wide, flat basin , hard for GD to escape once inside |
| 2 local minima | Sharp wells that trap naive optimizers |
| Saddle point | Ridge in x, valley in y , reveals optimizer sensitivity to curvature |
| Banana valley | Curved elongated trough , causes momentum oscillation |
| Smooth noise | Multi-octave LCG noise perturbs gradients throughout |

---

## Optimizers

### 1. Vanilla Gradient Descent (GD)

The baseline. Move in the direction of steepest descent by a fixed learning rate `η`.

```
θ ← θ - η · ∇f(θ)
```

| Parameter | Symbol | Default |
|---|---|---|
| Learning rate | η | 9.0 |

**Behavior:** Straight cautious descent. Easily trapped in local minima and saddle points. Stalls in flat basins where `‖∇f‖ ≈ 0`.

---

### 2. Momentum

Augments GD with a velocity term `v` that accumulates gradient history.

```
v ← μ·v - η·∇f(θ)
θ ← θ + v
```

| Parameter | Symbol | Default |
|---|---|---|
| Learning rate | η | 9.0 |
| Momentum coefficient | μ | 0.88 |

**Behavior:** Overshoots on curved surfaces and oscillates in narrow valleys, but escapes shallow basins that trap GD. Higher `μ` gives more inertia.

---

### 3. AdaGrad

Adapts the learning rate per dimension by accumulating squared gradients in `G`.

```
G ← G + ∇f(θ)²
θ ← θ - (η / (√G + ε)) · ∇f(θ)
```

| Parameter | Symbol | Default |
|---|---|---|
| Learning rate | η | 9.0 |
| Stability constant | ε | 1e-6 |

**Behavior:** Dimensions with historically large gradients receive smaller updates. `G` grows monotonically though, causing the effective learning rate to decay to zero , stalls in flat regions.

---

### 4. Adam

Combines momentum (first moment `m`) with adaptive per-dimension scaling (second moment `v`), with bias correction for early steps.

```
m ← β₁·m + (1 - β₁)·∇f(θ)
v ← β₂·v + (1 - β₂)·∇f(θ)²

m̂ = m / (1 - β₁ᵗ)
v̂ = v / (1 - β₂ᵗ)

θ ← θ - η · m̂ / (√v̂ + ε)
```

| Parameter | Symbol | Default |
|---|---|---|
| Learning rate | η | 9.0 |
| First moment decay | β₁ | 0.900 |
| Second moment decay | β₂ | 0.999 |
| Stability constant | ε | 1e-7 |

**Behavior:** Most robust optimizer in practice. Bias correction prevents overshooting early in training. Navigates saddles and flat regions better than GD or AdaGrad.

---

### 5. Sofia (Scalar)

Approximates the diagonal Hessian using an EMA of past gradients, then clips large steps.

```
m ← β·m + (1 - β)·∇f(θ)
h = |m|                              ← diagonal Hessian proxy
θ ← θ - clip(η·∇f(θ) / h, ρ)
```

| Parameter | Symbol | Default |
|---|---|---|
| Learning rate | η | 9.0 |
| EMA coefficient | β | 0.900 |
| Clip threshold | ρ | 18 |

**Behavior:** Scales updates by curvature, producing tighter controlled trajectories on curved surfaces. Clipping prevents instability in low-curvature flat regions.

---

### 6. Lion

Uses only the **sign** of a momentum-interpolated gradient as the update direction. Every step has constant magnitude `η`.

```
c = sign(β₁·m + (1 - β₁)·∇f(θ))
θ ← θ - η·c
m ← β₂·m + (1 - β₂)·∇f(θ)
```

| Parameter | Symbol | Default |
|---|---|---|
| Learning rate | η | 9.0 |
| Interpolation coeff | β₁ | 0.900 |
| Momentum decay | β₂ | 0.990 |

**Behavior:** Produces angular, blocky trajectories since step magnitude is always exactly `±η` per dimension. Memory-efficient , stores only `m`, not `v`. The sign-only update is visually distinctive and surprisingly effective.

---

## Convergence Criterion

Each particle runs until the gradient magnitude falls below a threshold or a step cap is hit:

```
stop when:  ‖∇f(θ)‖ < 8×10⁻⁴   OR   steps ≥ 8000
```

Particles in flat basins jitter visibly for many steps before the gradient fully vanishes , directly exposing which optimizers handle flat regions gracefully.

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

Single HTML file , no build step, no server, no dependencies.

```bash
git clone https://github.com/kavishka-dot/optimizer.git
open optimizer/index.html
```

Or push `index.html` to a GitHub repo and enable GitHub Pages under **Settings → Pages → Branch: main**.

Live at: `https://<your-username>.github.io/<repo-name>`

---

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*.
- Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning. *JMLR*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Liu, H., et al. (2023). Sophia: A scalable stochastic second-order optimizer. [arXiv:2305.14342](https://arxiv.org/abs/2305.14342)
- Chen, X., et al. (2023). Symbolic discovery of optimization algorithms (Lion). [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)

---

## License

MIT © [Kavishka](https://github.com/kavishka-dot)
