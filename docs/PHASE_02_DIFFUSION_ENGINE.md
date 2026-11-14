# TabSynth AI: Phase 02 – Tabular Diffusion Engine Specification & Documentation

## 1. Diffusion Overview

Tabular datasets are inherently heterogeneous, consisting of continuous numerical columns alongside discrete categorical variables with arbitrary cardinalities. Standard generative diffusion models (such as image DDPMs) rely on continuous Gaussian noise perturbations $\mathcal{N}(0, \sigma^2 \mathbf{I})$ that corrupt discrete category identifiers into invalid out-of-simplex floating point vectors.

**TabSynth AI** addresses this challenge through a mathematically decoupled diffusion formulation:
1. **Continuous Numerical Diffusion**: Standard Gaussian DDPM with closed-form posteriors and $\epsilon$-prediction score matching.
2. **Discrete Categorical Diffusion**: Discrete Multinomial Markov diffusion inspired by D3PM (Austin et al., 2021) operating directly on categorical probability simplexes.
3. **Mixed-Type Coordination**: The `MixedTypeDiffusion` interface independently manages forward corruption, joint composite losses, and ancestral reverse sampling without concatenating mixed data into a single continuous representation.

---

## 2. Gaussian Diffusion Mathematical Formulation

Continuous features $\mathbf{x}_0^{\text{num}} \in \mathbb{R}^{B \times D_{\text{num}}}$ are normalized to zero-mean and unit-variance (or normal quantiles in $[-5, 5]$).

### Forward Process
The forward noising kernel adds Gaussian noise over discrete timesteps $t \in \{0, 1, \dots, T-1\}$:
$$q(\mathbf{x}_t^{\text{num}} \mid \mathbf{x}_0^{\text{num}}) = \mathcal{N}\left(\mathbf{x}_t^{\text{num}}; \sqrt{\bar{\alpha}_t} \mathbf{x}_0^{\text{num}}, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
Reparameterized as:
$$\mathbf{x}_t^{\text{num}} = \sqrt{\bar{\alpha}_t} \mathbf{x}_0^{\text{num}} + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=0}^t \alpha_s$.

### Objective Function
The neural network predicts the injected noise vector $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$. The numerical loss is the Mean Squared Error:
$$\mathcal{L}_{\text{num}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t^{\text{num}}, t)\|^2 \right]$$

---

## 3. Categorical Diffusion Mathematical Formulation

Categorical attributes $\mathbf{x}_0^{\text{cat}, (j)} \in \{e_1, \dots, e_{K_j}\}$ with $K_j \ge 2$ classes are represented as one-hot vectors on the probability simplex:
$$\Delta^{K_j - 1} = \left\{ \mathbf{p} \in \mathbb{R}^{K_j} \;\middle|\; p_k \ge 0, \; \sum_{k=1}^{K_j} p_k = 1 \right\}$$

Rather than corrupting integers with continuous Gaussian noise, discrete transitions are defined by a Markov chain with transition probability matrices $\mathbf{Q}_t \in \mathbb{R}^{K_j \times K_j}$.

---

## 4. Transition Matrix Derivation

Adhering to the **row-vector probability convention**:
$$\mathbf{x}_t = \mathbf{x}_{t-1} \mathbf{Q}_t$$
where the $i$-th row of $\mathbf{Q}_t$ defines the conditional transition probability from state $i$:
$$[\mathbf{Q}_t]_{ij} = q(x_t = j \mid x_{t-1} = i)$$

Every row of $\mathbf{Q}_t$ sums to 1.0 ($\sum_j [\mathbf{Q}_t]_{ij} = 1.0$). For a uniform discrete corruption kernel with retention parameter $\alpha_t = 1 - \beta_t$:
$$\mathbf{Q}_t = \alpha_t \mathbf{I} + (1 - \alpha_t) \mathbf{U}$$
where:
- $\mathbf{I} \in \mathbb{R}^{K \times K}$ is the identity matrix.
- $\mathbf{U} = \frac{1}{K} \mathbf{1}\mathbf{1}^T \in \mathbb{R}^{K \times K}$ is the uniform matrix where every element is $\frac{1}{K}$.

Element-wise:
$$[\mathbf{Q}_t]_{ij} = \begin{cases} \alpha_t + \frac{1 - \alpha_t}{K}, & \text{if } i = j \\ \frac{1 - \alpha_t}{K}, & \text{if } i \ne j \end{cases}$$

---

## 5. Cumulative Transition Derivation

Under row-vector multiplication:
$$\mathbf{x}_t = \mathbf{x}_0 \bar{\mathbf{Q}}_t, \quad \text{where } \bar{\mathbf{Q}}_t = \mathbf{Q}_1 \mathbf{Q}_2 \dots \mathbf{Q}_t$$

Notice that for the uniform matrix:
$$\mathbf{U}^2 = \left(\frac{1}{K} \mathbf{1}\mathbf{1}^T\right) \left(\frac{1}{K} \mathbf{1}\mathbf{1}^T\right) = \frac{1}{K^2} \mathbf{1} (\mathbf{1}^T \mathbf{1}) \mathbf{1}^T = \frac{K}{K^2} \mathbf{1}\mathbf{1}^T = \mathbf{U}$$
and $\mathbf{I}\mathbf{U} = \mathbf{U}\mathbf{I} = \mathbf{U}$.

Therefore, multiplying two consecutive transition matrices:
$$\begin{aligned}
\mathbf{Q}_1 \mathbf{Q}_2 &= \left[\alpha_1 \mathbf{I} + (1 - \alpha_1) \mathbf{U}\right] \left[\alpha_2 \mathbf{I} + (1 - \alpha_2) \mathbf{U}\right] \\
&= \alpha_1 \alpha_2 \mathbf{I} + \left[\alpha_1 (1 - \alpha_2) + \alpha_2 (1 - \alpha_1) + (1 - \alpha_1)(1 - \alpha_2)\right] \mathbf{U} \\
&= (\alpha_1 \alpha_2) \mathbf{I} + (1 - \alpha_1 \alpha_2) \mathbf{U}
\end{aligned}$$

By mathematical induction over $t$ steps:
$$\bar{\mathbf{Q}}_t = \bar{\alpha}_t \mathbf{I} + (1 - \bar{\alpha}_t) \mathbf{U}, \quad \text{where } \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

Consequently, the marginal forward distribution for a one-hot vector $\mathbf{x}_0$ is computed in closed form without matrix powers:
$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathbf{x}_0 \bar{\mathbf{Q}}_t = \bar{\alpha}_t \mathbf{x}_0 + \frac{1 - \bar{\alpha}_t}{K} \mathbf{1}$$

---

## 6. Reverse Posterior Derivation

### Exact Posterior Given Clean State $x_0$
By Bayes' rule:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \, q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)} \propto q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \, q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)$$

In row-vector convention, for state $x_t = e_i$ and candidate prior state $x_{t-1} = e_j$:
- $q(x_t = i \mid x_{t-1} = j) = [\mathbf{Q}_t]_{ji} = [\mathbf{Q}_t^T]_{ij} = [\mathbf{x}_t \mathbf{Q}_t^T]_j = \alpha_t [\mathbf{x}_t]_j + \frac{1 - \alpha_t}{K}$
- $q(x_{t-1} = j \mid \mathbf{x}_0) = [\mathbf{x}_0 \bar{\mathbf{Q}}_{t-1}]_j = \bar{\alpha}_{t-1} [\mathbf{x}_0]_j + \frac{1 - \bar{\alpha}_{t-1}}{K}$

Therefore, the unnormalized posterior probability vector over $x_{t-1} \in \{1, \dots, K\}$ is given by the Hadamard (element-wise) product:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \propto \left(\alpha_t \mathbf{x}_t + \frac{1 - \alpha_t}{K} \mathbf{1}\right) \odot \left(\bar{\alpha}_{t-1} \mathbf{x}_0 + \frac{1 - \bar{\alpha}_{t-1}}{K} \mathbf{1}\right)$$
Normalized by dividing by the sum across the $K$ categories.

### Learned Reverse Distribution via Probability Marginalization
At test time or during training, clean state $\mathbf{x}_0$ is unknown. The neural network predicts clean class logits $\hat{\mathbf{z}} \in \mathbb{R}^K$, producing predicted probabilities:
$$p_\theta(\mathbf{x}_0 = e_k \mid \mathbf{x}_t) = [\text{softmax}(\hat{\mathbf{z}})]_k$$

The learned reverse step is obtained through exact probability marginalization over all $K$ basis states:
$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \sum_{k=1}^K q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0 = e_k) \, p_\theta(\mathbf{x}_0 = e_k \mid \mathbf{x}_t)$$

> [!IMPORTANT]
> **No Argmax Discretization**: The model NEVER applies $\text{argmax}$ over predicted categories during reverse posterior calculation. The full probability distribution is marginalized, guaranteeing differentiability, smoothness, and avoiding collapse to mode estimates.

---

## 7. Categorical KL Objective

The discrete training loss optimizes the Variational Lower Bound (VLB) on data log-likelihood:
$$\mathcal{L}_{\text{cat}}(\theta) = \begin{cases}
D_{\text{KL}}\left( q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \;\parallel\; p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \right), & \text{for } t > 0 \\
-\sum_{k=1}^K [\mathbf{x}_0]_k \log p_\theta(\mathbf{x}_0 = e_k \mid \mathbf{x}_1), & \text{for } t = 0
\end{cases}$$

### Auxiliary Cross-Entropy Regularizer
To accelerate representation learning in early training phases, an auxiliary cross-entropy loss against predicted clean logits is optionally included:
$$\mathcal{L}_{\text{aux}} = \text{CrossEntropy}(\hat{\mathbf{z}}, \text{argmax}(\mathbf{x}_0))$$
This auxiliary objective is kept strictly separate from the formal variational KL objective.

---

## 8. Numerical Reverse Process

Using predicted noise $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t^{\text{num}}, t)$, the model reconstructs the clean state estimate:
$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t^{\text{num}} - \sqrt{1 - \bar{\alpha}_t} \hat{\boldsymbol{\epsilon}}}{\sqrt{\bar{\alpha}_t}}$$

The reverse posterior distribution $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \hat{\mathbf{x}}_0) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t \mathbf{I})$ has parameters:
$$\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \hat{\mathbf{x}}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t^{\text{num}}$$
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

Sampling step:
$$\mathbf{x}_{t-1}^{\text{num}} = \tilde{\boldsymbol{\mu}}_t + \mathbf{1}(t > 0) \cdot \sqrt{\tilde{\beta}_t} \, \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$
At $t = 0$, stochastic noise injection is suppressed ($\mathbf{1}(t=0) = 0$).

---

## 9. Mixed-Type Diffusion Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                  MixedTypeDiffusion                    │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
     ┌─────────────────────────────┐                   ┌─────────────────────────────┐
     │      GaussianDiffusion      │                   │    CategoricalDiffusion     │
     │    (Continuous Features)    │                   │     (Discrete Features)     │
     └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                    │                                                 │
          q_sample(x_0, t)                                q_sample_categorical(x_0, t)
                    │                                                 │
                    ▼                                                 ▼
             Noisy x_t^num                                     Noisy x_t^cat
                    │                                                 │
                    └────────────────────────┬────────────────────────┘
                                             ▼
                                  ┌────────────────────┐
                                  │   Denoising Net    │
                                  │  (Hybrid / MLP)    │
                                  └──────────┬─────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
             Predicted eps                                     Predicted logits
                    │                                                 │
                    ▼                                                 ▼
           MSE Loss: L_num                                   D3PM KL Loss: L_cat
                    │                                                 │
                    └────────────────────────┬────────────────────────┘
                                             ▼
                             L_total = lambda_num * L_num
                                     + lambda_cat * L_cat
                                     + lambda_aux * L_aux
```

---

## 10. Tensor Shapes and Data Conventions

| Tensor | Symbol | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| Clean numerical features | $\mathbf{x}_0^{\text{num}}$ | `(B, D_num)` | `torch.float32` | Standardized continuous attributes |
| Clean categorical feature $j$ | $\mathbf{x}_0^{\text{cat}, (j)}$ | `(B, K_j)` | `torch.float32` | One-hot simplex vector for column $j$ |
| Numerical timesteps | $t_{\text{num}}$ | `(B,)` | `torch.int64` | Discrete timesteps in $[0, T_{\text{num}}-1]$ |
| Categorical timesteps | $t_{\text{cat}}$ | `(B,)` | `torch.int64` | Discrete timesteps in $[0, T_{\text{cat}}-1]$ |
| Predicted continuous noise | $\hat{\boldsymbol{\epsilon}}$ | `(B, D_num)` | `torch.float32` | Model noise prediction |
| Predicted categorical logits $j$ | $\hat{\mathbf{z}}^{(j)}$ | `(B, K_j)` | `torch.float32` | Model clean logit prediction for column $j$ |
| Transition matrix | $\mathbf{Q}_t$ | `(K_j, K_j)` | `torch.float32` | Row-stochastic Markov kernel |

---

## 11. Device Support & Hardware Optimization

The engine natively detects and leverages Apple Silicon Metal Performance Shaders (`mps`) with automated fallback to CPU:
- **Zero Incompatible Ops**: All operations (matrix multiplications `bmm`, multinomial sampling `torch.multinomial`, gather operations, elementwise products) are MPS-compatible in PyTorch 2.15.0+.
- **Strict float32 Precision**: Enforces `float32` everywhere to prevent underflow in cumulative product chains $\prod \alpha_s$.
- **Zero Inter-Device Copies**: Diffusion schedule constants are cached directly on the target device upon initialization.

---

## 12. Configuration Reference (`configs/diffusion.yaml`)

```yaml
diffusion:
  numerical_timesteps: 100
  categorical_timesteps: 100
  numerical_schedule: "cosine"
  categorical_schedule: "cosine"
  categorical_transition: "uniform"
  linear:
    beta_start: 0.0001
    beta_end: 0.02
  cosine:
    s: 0.008
    max_beta: 0.999
  loss:
    numerical_loss_type: "mse"
    categorical_loss_type: "kl"
    lambda_numerical: 1.0
    lambda_categorical: 1.0
    lambda_auxiliary: 0.2
```

---

## 13. Test Instructions

Execute the diffusion engine test suite:
```bash
# Run all dedicated Phase 02 diffusion tests
.venv/bin/pytest tests/diffusion/ -v

# Run full project regression suite (44 tests)
.venv/bin/pytest -v

# Run synthetic smoke test on CPU
.venv/bin/python scripts/smoke_test_diffusion.py cpu

# Run synthetic smoke test on Apple Silicon MPS
.venv/bin/python scripts/smoke_test_diffusion.py mps
```

---

## 14. Known Limitations & Architectural Notes

1. **Uniform Transition Kernel**: The current implementation uses uniform categorical transitions $\mathbf{Q}_t = \alpha_t \mathbf{I} + (1 - \alpha_t) \mathbf{U}$. Absorbing-state kernels or frequency-weighted transitions can be added in future iterations if rare categories require non-uniform corruptions.
2. **Computational Complexity of Marginalization**: The probability marginalization over $K$ basis states scales as $\mathcal{O}(B \cdot K^2)$ per categorical column. For high-cardinality columns ($K > 500$), hierarchical or clustering approximations can be applied.
3. **Decoupled Timesteps**: While $T_{\text{num}}$ and $T_{\text{cat}}$ can be configured independently, setting $T_{\text{num}} = T_{\text{cat}}$ simplifies synchronized reverse trajectories.
