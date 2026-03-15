# SDE Diffusion Simulator

An interactive, browser-based simulator for understanding stochastic differential equations (SDEs) and diffusion probabilistic models. Built as a single self-contained HTML file with no backend or build step required.

**Live demo:** [kavishka-dot.github.io/sde_simulations](https://kavishka-dot.github.io/sde_simulations)

---

## Overview

This tool bridges the gap between the abstract mathematics of Ito calculus and the practical mechanics of modern diffusion models (DDPM, DDIM, score-based generative models). All three tabs are interactive and equation-driven, with live MathJax rendering throughout.

---

## Features

### Tab 1: Diffusion Paths

Simulates 2D Ito stochastic differential equations in real time.

$$dX_t = \mu \, dt + \sigma \, dW_t$$

- Click anywhere on the canvas to place an origin point
- 100 Brownian motion paths are simulated forward from each origin
- Adjust drift ($\mu$), diffusion coefficient ($\sigma$), number of steps ($T$), and time step ($dt$) via sliders
- Multiple origins can be placed simultaneously; resimulate reruns all with updated parameters
- Live statistics: origin count, total paths, endpoint spread, total time

### Tab 2: Image Diffusion (Forward Process)

Visualises the DDPM forward noising process on a user-drawn image.

$$dX_t = -\tfrac{1}{2}\beta(t) X_t \, dt + \sqrt{\beta(t)} \, dW_t \qquad \Rightarrow \qquad X_t | X_0 \sim \mathcal{N}\!\left(\sqrt{\bar\alpha_t} X_0,\ (1 - \bar\alpha_t)I\right)$$

- Draw directly on the canvas with a colour palette, pen, eraser, and adjustable brush size
- Click **Generate Diffusion** to produce 9 independent noisy trajectories at 10 evenly-spaced timesteps
- A 10th cell shows the **Monte Carlo mean** image, averaged over $N$ independent noise draws (configurable from 10 to 500)
- As $N \to \infty$, noise cancels and only $\sqrt{\bar\alpha_t} X_0$ survives, demonstrating the marginal mean analytically
- Scrub through timesteps with the slider; all 10 cells update simultaneously
- Configurable noise schedule: $\beta_0$, $\beta_T$, $T$, and $N_\text{avg}$

### Tab 3: Oracle Denoising (Reverse Process)

Demonstrates the closed-form oracle denoiser, the theoretical upper bound on reconstruction quality.

$$\hat{X}_0 = \frac{X_t - \sqrt{1 - \bar\alpha_t}\, \varepsilon}{\sqrt{\bar\alpha_t}}$$

- Full drawing canvas identical to Tab 2, with the same tools and colour palette
- Slide the noise level $t$ to corrupt $X_0$ into $X_t$ and immediately recover $\hat{X}_0$
- Three-panel display: **Original** $X_0$ | **Noisy** $X_t$ | **Denoised** $\hat{X}_0$
- Since the true noise $\varepsilon$ is known exactly, the oracle achieves perfect inversion (PSNR $\to \infty$), illustrating why a real model must learn to estimate $\varepsilon$ from $X_t$ alone
- Timestep strip showing $X_t$ at 11 evenly-spaced noise levels, from clean to fully corrupted
- MSE curve plotting $1 - \bar\alpha_t$ vs $t$ with a live marker tracking the current slider position
- Sidebar statistics: $\bar\alpha(t)$, SNR (dB), PSNR, MSE
- **"Use current drawing"** button copies the drawing from Tab 2 directly into Tab 3

---

## Mathematical Background

The forward process of a diffusion model is an Ito SDE that gradually corrupts data into Gaussian noise. The DDPM forward SDE has a closed-form marginal:

$$X_t | X_0 \sim \mathcal{N}\!\left(\sqrt{\bar\alpha_t} X_0,\ (1 - \bar\alpha_t)I\right), \qquad \bar\alpha_t = \prod_{s=1}^{t}(1 - \beta_s)$$

The oracle denoiser uses knowledge of the true noise $\varepsilon \sim \mathcal{N}(0, I)$ to exactly invert this corruption. A trained diffusion model approximates this oracle by learning $\varepsilon_\theta(X_t, t) \approx \varepsilon$, which is what enables generation from pure noise.

---

## Tech Stack

| Concern | Solution |
|---|---|
| Rendering | Vanilla HTML5 Canvas API |
| Mathematics | MathJax 3 (`tex-svg.js`) |
| Typography | Inter + JetBrains Mono (Google Fonts) |
| Random sampling | Box-Muller transform for $\mathcal{N}(0,1)$ draws |
| Deployment | Single `.html` file, GitHub Pages |

No frameworks, no build tools, no dependencies beyond CDN-loaded MathJax and Google Fonts.

---

## Usage

### Run locally

```bash
git clone https://github.com/kavishka-dot/sde_simulations.git
cd sde_simulations
# Open in browser — no server needed
open index.html
```

### Deploy

The repository is configured for GitHub Pages. Any push to `main` automatically updates the live demo.

---

## Project Structure

```
sde_simulations/
└── index.html      # Entire application — self-contained
└── README.md
```

---

## License

MIT
