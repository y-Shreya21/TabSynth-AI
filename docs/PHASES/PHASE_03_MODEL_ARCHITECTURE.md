# Phase 03: Hybrid Transformer-MLP Denoising Architecture

## 1. Objective
Design and implement the complete denoising neural network for TabSynth AI. The architecture combines Pre-LayerNorm Transformer feature-attention tokenization with a residual MLP denoising backbone to capture multi-column interdependencies while remaining computationally efficient on local Apple Silicon M4 hardware.

## 2. Methodology & Mathematics
### 2.1 Feature-Wise Tokenization
Each tabular feature $j \in \{1, \dots, D\}$ is projected into a unified $d_{\text{model}}$-dimensional embedding:
- Continuous features: $\mathbf{e}_j^{\text{num}} = \mathbf{W}_j x_j + \mathbf{b}_j \in \mathbb{R}^{d_{\text{model}}}$.
- Categorical features: $\mathbf{e}_j^{\text{cat}} = \mathbf{E}_j[k] \in \mathbb{R}^{d_{\text{model}}}$, where $\mathbf{E}_j \in \mathbb{R}^{K_j \times d_{\text{model}}}$.
- Column Identity Embeddings: A learnable column token $\mathbf{c}_j \in \mathbb{R}^{d_{\text{model}}}$ is added to preserve feature ordering:
  $$\mathbf{z}_j = \mathbf{e}_j + \mathbf{c}_j$$

### 2.2 Pre-LN Transformer Cross-Feature Encoder
- Pre-LayerNorm multi-head self-attention operating across the $D$ column tokens:
  $$\mathbf{H}^{(0)} = [\mathbf{z}_1, \dots, \mathbf{z}_D] \in \mathbb{R}^{B \times D \times d_{\text{model}}}$$
  $$\mathbf{H}^{(\ell)\prime} = \mathbf{H}^{(\ell-1)} + \text{MHA}(\text{LN}(\mathbf{H}^{(\ell-1)}))$$
  $$\mathbf{H}^{(\ell)} = \mathbf{H}^{(\ell)\prime} + \text{FFN}(\text{LN}(\mathbf{H}^{(\ell)\prime}))$$

### 2.3 Residual MLP Denoiser with Conditioning Fusion
- The contextualized column representations are flattened to $\mathbf{h} \in \mathbb{R}^{B \times (D \cdot d_{\text{model}})}$.
- Fused with sinusoidal time step embedding $\mathbf{e}_t$ and class-conditioning embedding $\mathbf{e}_c$:
  $$\mathbf{e}_{\text{cond}} = \text{MLP}_t(\mathbf{e}_t) + \text{Embedding}_c(c)$$
  $$\mathbf{h}^{(0)} = \mathbf{W}_{\text{in}} \mathbf{h} + \mathbf{e}_{\text{cond}}$$
- Residual dense blocks with SwiGLU / SiLU activations, LayerNorm, and dropout.

### 2.4 Dual Output Heads
- Numerical Head: Linear projection predicting continuous Gaussian noise $\hat{\boldsymbol{\epsilon}} \in \mathbb{R}^{B \times D_{\text{cont}}}$.
- Categorical Heads: Individual linear projection heads for each categorical column $j$ predicting logits $\hat{\mathbf{z}}_j \in \mathbb{R}^{B \times K_j}$.

## 3. Architecture & Implementation
- **`src/tabsynth/models/embeddings.py`:** `NumericalFeatureTokenizer`, `CategoricalFeatureTokenizer`, `SinusoidalTimeEmbedding`, and `ClassConditioningEmbedding` (with CFG dropout $p_{\text{uncond}} = 0.15$).
- **`src/tabsynth/models/transformer.py`:** `TransformerFeatureEncoder` with Pre-LN layers.
- **`src/tabsynth/models/denoiser.py`:** `ResidualMLPDenoiser` with conditioning projection.
- **`src/tabsynth/models/heads.py`:** `NumericalNoiseHead` and `CategoricalLogitsHead`.
- **`src/tabsynth/models/hybrid.py`:** `TabSynthHybridDenoiser` assembling all components.
- **`src/tabsynth/models/mlp_baseline.py`:** `TabSynthMLPDenoiser` flat baseline without Transformer attention.

## 4. Key Design Decisions
- **Pre-LayerNorm Stabilization:** Pre-LN provides stable gradient flow during diffusion training across varied noise scales $t \in [1, 100]$.
- **Architectural Ablation Baseline:** Implemented `TabSynthMLPDenoiser` to empirically isolate the exact value of cross-feature self-attention (RQ1 / E1).

## 5. Relevant Source Files
- `src/tabsynth/models/embeddings.py`
- `src/tabsynth/models/transformer.py`
- `src/tabsynth/models/denoiser.py`
- `src/tabsynth/models/heads.py`
- `src/tabsynth/models/hybrid.py`
- `src/tabsynth/models/mlp_baseline.py`
- `src/tabsynth/models/diagnostics.py`

## 6. Tests Performed
- `tests/models/test_tokenizer.py` (6 tests)
- `tests/models/test_time_embedding.py` (2 tests)
- `tests/models/test_conditioning.py` (4 tests)
- `tests/models/test_transformer_denoiser.py` (3 tests)
- `tests/models/test_mlp_denoiser.py` (3 tests)
- `tests/models/test_output_heads.py` (2 tests)
- `tests/models/test_model_integration.py` (3 tests)

## 7. Actual Results
- TabSynth Hybrid model on Adult Income: **240,198 parameters** (all trainable).
- TabDDPM MLP model on Adult Income: **177,478 parameters**.
- Forward pass executes in < 5ms per mini-batch on Apple Silicon MPS.

## 8. Known Limitations
- Transformer attention scales quadratically $\mathcal{O}(D^2)$ with the number of columns. For tables with > 200 columns, linear attention or sparse feature clustering is recommended.

## 9. Reproduction Instructions
```bash
pytest tests/models/ -v
python scripts/smoke_test_model.py
```
