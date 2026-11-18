# TabSynth AI: Scientific Validation & Theoretical Review Report

**Project:** TabSynth AI  
**Full Title:** *Hybrid Transformer-MLP Diffusion Framework for Privacy-Aware, High-Fidelity and Class-Conditional Synthetic Tabular Data Generation*  
**Phase:** 08 — Scientific Implementation Review  
**Date:** November 2026  
**Auditor:** Senior ML Research Engineer & Tabular Diffusion Specialist  

---

## 1. Executive Summary

This report delivers a rigorous scientific review of the mathematical formulation, algorithmic design, and code implementation of the **TabSynth AI** generative framework. 

Tabular data exhibits a fundamental structural challenge: datasets are typically **heterogeneous**, consisting of continuous numerical columns with diverse multivariant marginals and discrete categorical columns with arbitrary category cardinalities. Standard generative models often apply continuous relaxation or Gaussian perturbations to discrete integer indices, generating invalid, out-of-domain categories. 

TabSynth AI solves this challenge through a **decoupled mixed-type diffusion process**:
1. **Continuous Gaussian Diffusion** for numerical features with learned noise prediction $\epsilon_\theta$.
2. **Discrete Multinomial Diffusion (D3PM)** for categorical features operating on the probability simplex.
3. A **Hybrid Transformer-MLP Denoiser** that combines multi-head cross-feature attention with residual per-token MLP refinement.
4. **Classifier-Free Guidance (CFG)** enabling steerable class-conditional synthesis without auxiliary classifier models.

---

## 2. Numerical Gaussian Diffusion Verification

### 2.1 Mathematical Formulation
For continuous features $x_0^{(num)} \in \mathbb{R}^{D_{num}}$, forward noise injection follows a variance-preserving Markov chain:
$$q(x_t^{(num)} \mid x_0^{(num)}) = \mathcal{N}\left(x_t^{(num)}; \sqrt{\bar{\alpha}_t} x_0^{(num)}, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

### 2.2 Noise Schedules
Implemented in [`src/tabsynth/diffusion/schedules.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/diffusion/schedules.py):
- **Cosine Schedule**: $\bar{\alpha}_t = \frac{f(t)}{f(0)}$ where $f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$ with offset $s = 0.008$. Prevents abrupt variance collapse in early timesteps.
- **Linear Schedule**: $\beta_t$ linearly spaced between $\beta_1 = 10^{-4}$ and $\beta_T = 0.02$.

### 2.3 Training Objective & Reverse Denoising
The neural network predicts the injected Gaussian perturbation $\epsilon_\theta(x_t, t, c) \approx \epsilon$:
$$\mathcal{L}_{num}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t^{(num)}, t, c)\|_2^2\right]$$
During reverse sampling ($t \to t-1$), the clean estimate $\hat{x}_0$ is analytically reconstructed:
$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t, c)}{\sqrt{\bar{\alpha}_t}}$$
and the reverse posterior mean $\tilde{\mu}_t(x_t, \hat{x}_0)$ is computed via:
$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \hat{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

### 2.4 Feature Normalization & Invertibility
In [`src/tabsynth/data/preprocessor.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/data/preprocessor.py):
- Continuous columns are fitted using Quantile Normalization (`QuantileTransformer(output_distribution='normal')`) or Robust Scaler.
- Preprocessor is fitted **strictly on $D_{train}$**, preventing target or test leakage.
- Inverse transform guarantees exact reconstruction back to original physical scales and boundaries.

---

## 3. Discrete Categorical Diffusion (D3PM) Verification

### 3.1 Mathematical Formulation
For each categorical feature $j \in \{1, \dots, D_{cat}\}$ with cardinality $K_j$, the state is represented as a one-hot vector $x_0^{(j)} \in \{0, 1\}^{K_j}$. 

Rather than treating categories as continuous variables, forward corruption is defined by a discrete Markov transition matrix $Q_t \in \mathbb{R}^{K_j \times K_j}$:
$$q(x_t^{(j)} \mid x_{t-1}^{(j)}) = x_{t-1}^{(j)} Q_t$$
Under the uniform transition kernel:
$$[Q_t]_{m, n} = \begin{cases} \alpha_t + \frac{1 - \alpha_t}{K_j} & \text{if } m = n \\ \frac{1 - \alpha_t}{K_j} & \text{if } m \neq n \end{cases}$$
The marginal distribution at timestep $t$ admits the closed form:
$$q(x_t^{(j)} \mid x_0^{(j)}) = \bar{\alpha}_t x_0^{(j)} + \frac{1 - \bar{\alpha}_t}{K_j} \mathbf{1}$$

### 3.2 Discrete Reverse Transition Posterior
The exact discrete posterior $q(x_{t-1}^{(j)} \mid x_t^{(j)}, x_0^{(j)})$ is computed via Bayes' rule on the probability simplex:
$$q(x_{t-1}^{(j)} \mid x_t^{(j)}, x_0^{(j)}) = \frac{q(x_t^{(j)} \mid x_{t-1}^{(j)}) \cdot q(x_{t-1}^{(j)} \mid x_0^{(j)})}{q(x_t^{(j)} \mid x_0^{(j)})}$$
Implemented in [`src/tabsynth/diffusion/multinomial.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/diffusion/multinomial.py):
```python
fact1 = alpha_t * x_t_onehot + (1.0 - alpha_t) / float(K)
fact2 = alpha_bar_prev * x_0_onehot + (1.0 - alpha_bar_prev) / float(K)
unnorm_probs = fact1 * fact2
probs = unnorm_probs / torch.clamp(unnorm_probs.sum(dim=-1, keepdim=True), min=1e-12)
```

### 3.3 Categorical Loss Formulation
The categorical loss combines:
1. **D3PM Variational KL Divergence**: $\mathcal{L}_{cat} = D_{KL}\left(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t)\right)$
2. **Auxiliary Clean Cross-Entropy**: $\mathcal{L}_{aux} = -\sum_{k=1}^{K_j} x_{0, k}^{(j)} \log \left[\text{softmax}(\hat{z}_\theta^{(j)})_k\right]$
This dual formulation stabilizes discrete training and prevents mode collapse in high-cardinality features.

---

## 4. Hybrid Transformer-MLP Architecture Review

```
[Continuous Inputs x_num] ---> [Linear Proj (d_model)]  --+
                                                          |--> [Token Concat: (B, N, d_model)]
[Categorical Inputs x_cat] --> [Embedding Layer]        --+          |
                                                                     v
                                                          [+ Column Identity Embeddings]
                                                          [+ Feature Type Embeddings]
                                                                     |
                                                                     v
                                                          [Transformer Feature Encoder]
                                                          (Pre-LN Multi-Head Attention)
                                                                     |
                                                                     v
                                   [Timestep + Class Cond] -> [Fused Context Embedding]
                                                                     |
                                                                     v
                                                          [Residual MLP Denoiser]
                                                          (Per-feature sequence refinement)
                                                                     |
                                                                     v
                                      +------------------------------+------------------------------+
                                      |                                                             |
                                      v                                                             v
                        [Numerical Output Head]                                       [Categorical Output Heads]
                        (Linear -> Predicted Noise eps)                               (Per-column Linear -> Logits)
```

### 4.1 Feature Tokenization & Embedding Pipeline
In [`src/tabsynth/models/tokenizer.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/models/tokenizer.py):
- Each continuous feature is projected into $\mathbb{R}^{d_{model}}$ via an individual linear projection layer.
- Each categorical feature is embedded using a learned embedding matrix $E_j \in \mathbb{R}^{K_j \times d_{model}}$.
- Tokens are enriched by adding a learned **Column Identity Embedding** $e_{col} \in \mathbb{R}^{D \times d_{model}}$ and a binary **Feature Type Embedding** (continuous vs categorical).

### 4.2 Cross-Feature Transformer Attention
In [`src/tabsynth/models/transformer.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/models/transformer.py):
- Implements Pre-LayerNorm multi-head self-attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- Enables all features (numerical and categorical) to directly attend to each other, capturing non-linear correlation structures and joint distributions.

### 4.3 Residual MLP Denoising Network
In [`src/tabsynth/models/denoiser.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/models/denoiser.py):
- Enriched tokens are concatenated or added with fused timestep/class conditioning embeddings.
- Residual MLP layers refine individual feature representations with GELU non-linearities and LayerNorm stabilization.

---

## 5. Class-Conditional Generation & Classifier-Free Guidance (CFG)

### 5.1 Conditional Training & Dropout
In [`src/tabsynth/models/embeddings.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/models/embeddings.py):
- Class labels $c \in \{0, \dots, C-1\}$ are mapped to learned embeddings.
- An extra token index $C$ represents the **null conditioning token** $\emptyset$.
- During training, class labels are randomly dropped to $\emptyset$ with probability $p_{uncond} = 0.15$.
- This enables a single denoiser network to jointly learn both conditional $p_\theta(x \mid c)$ and unconditional $p_\theta(x)$ distributions.

### 5.2 Steerable Reverse Sampling (CFG)
During reverse generation, the model evaluates both conditional and unconditional forward passes at each timestep:
$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot \left(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\right)$$
where $w \ge 1.0$ is the guidance scale (default: $w = 1.5$). Higher $w$ sharpens class separation and boosts minority-class fidelity.

---

## 6. Scientific Correctness Verification Checklist

| Criterion | Implementation Location | Verdict |
| :--- | :--- | :---: |
| Gaussian noise is NOT applied to discrete integers | `src/tabsynth/diffusion/multinomial.py` | **PASSED** |
| Continuous and discrete timesteps remain synchronized | `src/tabsynth/sampling/mixed_sampler.py` | **PASSED** |
| No lookahead data leakage in preprocessor | `src/tabsynth/data/preprocessor.py` | **PASSED** |
| Closed-form Gaussian posterior mean & variance | `src/tabsynth/diffusion/gaussian.py` | **PASSED** |
| Correct D3PM Bayes rule posterior computation | `src/tabsynth/diffusion/multinomial.py` | **PASSED** |
| CFG null token dropout preserves unconditional distribution | `src/tabsynth/models/embeddings.py` | **PASSED** |
| Quantile inversion restores native physical scale | `src/tabsynth/data/preprocessor.py` | **PASSED** |

---

## 7. Conclusion

The scientific implementation of TabSynth AI accurately translates state-of-the-art mixed-type tabular generative modeling principles into robust, verified PyTorch code.
