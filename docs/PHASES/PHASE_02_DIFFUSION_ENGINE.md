# Phase 02: Mixed-Type Tabular Diffusion Engine

## 1. Objective
Implement a mathematically sound continuous-discrete joint diffusion engine capable of perturbing and reconstructing continuous variables (via Gaussian diffusion) and discrete categories (via D3PM transition matrices) synchronously across arbitrary diffusion timesteps $t \in [1, T]$.

## 2. Methodology & Mathematics
### 2.1 Continuous Numerical Diffusion (Gaussian DDPM)
- **Forward Process:**
  $$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
- **Analytical Reverse Posterior:**
  $$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}\right)$$
  where $\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$, and $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$.
- **Noise Schedules:** Linear, Cosine (Nichol & Dhariwal, 2021), and Sigmoid schedules.

### 2.2 Discrete Categorical Diffusion (D3PM)
- **Transition Matrices:** For a categorical variable with $K$ states, $\mathbf{Q}_t = (1 - \beta_t) \mathbf{I} + \beta_t \frac{1}{K} \mathbf{1}\mathbf{1}^\top$.
- **Cumulative Transition:** $\bar{\mathbf{Q}}_t = \prod_{s=1}^t \mathbf{Q}_s$.
- **Exact Bayes Posterior:**
  $$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{\mathbf{x}_t \mathbf{Q}_t \mathbf{x}_{t-1}^\top \cdot \mathbf{x}_0 \bar{\mathbf{Q}}_{t-1} \mathbf{x}_{t-1}^\top}{\mathbf{x}_0 \bar{\mathbf{Q}}_t \mathbf{x}_t^\top}$$
- **Learned Reverse Distribution:** Probability marginalization over model-predicted logits $\hat{\mathbf{p}}_0$:
  $$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \sum_{k=1}^K q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{e}_k) \cdot \hat{p}_{0,k}$$

## 3. Architecture & Implementation
- **`src/tabsynth/diffusion/numerical/gaussian_diffusion.py`:** Gaussian forward process, posterior calculations, and $\epsilon$-parameterized reverse step.
- **`src/tabsynth/diffusion/categorical/categorical_diffusion.py`:** D3PM multinomial transition matrices, forward one-hot corruption, and exact posterior evaluation.
- **`src/tabsynth/diffusion/mixed/mixed_diffusion.py`:** `MixedTypeDiffusion` synchronizing continuous and discrete diffusion processes into a unified training and sampling interface.
- **`src/tabsynth/diffusion/schedules.py`:** Beta schedule generators and SNR accumulators.

## 4. Key Design Decisions
- **Out-of-Simplex Elimination:** Continuous embedding diffusion forces categorical variables onto unconstrained continuous vectors, leading to invalid discrete decodings. D3PM maintains strict categorical validity on the discrete simplex.
- **Numerical Stability:** Epsilon terms ($10^{-12}$) in categorical log probabilities and float32 precision for cumulative product schedules prevent underflow.

## 5. Relevant Source Files
- `src/tabsynth/diffusion/schedules.py`
- `src/tabsynth/diffusion/numerical/gaussian_diffusion.py`
- `src/tabsynth/diffusion/categorical/categorical_diffusion.py`
- `src/tabsynth/diffusion/categorical/transition_matrices.py`
- `src/tabsynth/diffusion/mixed/mixed_diffusion.py`
- `src/tabsynth/diffusion/utils.py`

## 6. Tests Performed
- `tests/diffusion/test_numerical_diffusion.py` (5 tests)
- `tests/diffusion/test_categorical_diffusion.py` (4 tests)
- `tests/diffusion/test_categorical_transition.py` (4 tests)
- `tests/diffusion/test_mixed_diffusion.py` (5 tests)
- `tests/diffusion/test_schedules.py` (5 tests)
- `tests/diffusion/test_reverse_sampling.py` (3 tests)
- `tests/unit/test_diffusion_math.py` (3 tests)

## 7. Actual Results
- Analytical posterior variance and mean reconstruction match theoretical derivations ($L_1 \text{ error} < 10^{-4}$).
- 25 diffusion math and step tests pass in 3.12s.

## 8. Known Limitations
- Categorical transitions scale quadratically $\mathcal{O}(K^2)$ in state space size without uniform/absorbing matrix simplifications.

## 9. Reproduction Instructions
```bash
pytest tests/diffusion/ -v
python scripts/smoke_test_diffusion.py
```
