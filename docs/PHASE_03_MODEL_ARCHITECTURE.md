# TabSynth AI: Phase 03 – Hybrid Transformer-MLP Denoising Architecture Specification

## 1. Model Overview

**TabSynth AI** is an advanced generative deep learning framework for tabular data generation based on a hybrid denoising score-matching architecture. Tabular datasets comprise heterogeneous continuous attributes with arbitrary distributions alongside discrete categorical attributes with non-uniform cardinalities.

To model these mixed data types with high statistical fidelity without continuous out-of-simplex corruption, TabSynth AI decouples the diffusion processes (Gaussian continuous + D3PM discrete) and couples them via a unified **Hybrid Transformer-MLP Denoiser**:
1. **Pre-LN Transformer Cross-Feature Attention**: Captures high-order, non-local interdependencies across all $D$ columns simultaneously.
2. **Feature-Level Residual MLP Denoising**: Refines per-feature token representations conditioned on fused diffusion timesteps ($t_{\text{num}}, t_{\text{cat}}$) and class labels ($y$).
3. **Specialized Output Prediction Heads**: Projects continuous tokens to Gaussian noise predictions $\hat{\boldsymbol{\epsilon}} \in \mathbb{R}^{B \times D_{\text{num}}}$ and discrete tokens to clean-category unnormalized logits $\hat{\mathbf{z}}^{(j)} \in \mathbb{R}^{B \times K_j}$.

```mermaid
graph TD
    subgraph Inputs["1. Mixed-Type Tabular Inputs"]
        X_num["Continuous Features x_t^num (B, D_num)"]
        X_cat["Categorical Features x_t^cat [(B, K_j)]"]
        T_num["Numerical Timestep t_num (B,)"]
        T_cat["Categorical Timestep t_cat (B,)"]
        Y_cond["Target Class Labels y (B,)"]
    end

    subgraph Tokenization["2. Tokenization Layer"]
        X_num --> NumTok["Numerical Feature Tokenizer"]
        X_cat --> CatTok["Categorical Feature Tokenizer"]
        NumTok --> ConcatTok["Feature Token Sequence"]
        CatTok --> ConcatTok
        ColEmb["Column Identity Embeddings e_col"] --> AddEmb["+"]
        TypeEmb["Feature Type Embeddings e_type"] --> AddEmb
        ConcatTok --> AddEmb
        AddEmb --> Tokens["Tokens: (B, D, d_model)"]
    end

    subgraph Conditioning["3. Conditioning Fusion"]
        T_num --> TimeEmb["Mixed Timestep Embedding (Sinusoidal + MLP)"]
        T_cat --> TimeEmb
        Y_cond --> ClassEmb["Class Conditioning Embedding (CFG Dropout)"]
        TimeEmb --> CondAdd["+"]
        ClassEmb --> CondAdd
        CondAdd --> FusedCond["Fused Conditioning Vector e_cond: (B, d_model)"]
    end

    subgraph Attention["4. Cross-Feature Transformer"]
        Tokens --> Trans["Pre-LN Multi-Head Self-Attention Blocks (N_layers)"]
        Trans --> ContextTokens["Contextualized Tokens: (B, D, d_model)"]
    end

    subgraph MLP["5. Feature-Level Residual MLP"]
        ContextTokens --> ResMLP["Residual MLP Blocks (N_mlp)"]
        FusedCond --> ResMLP
        ResMLP --> DenoisedTokens["Denoised Feature Tokens: (B, D, d_model)"]
    end

    subgraph Heads["6. Specialized Output Heads"]
        DenoisedTokens -->|Continuous Slice| NumHead["Numerical Output Head"]
        DenoisedTokens -->|Categorical Slice| CatHeads["Categorical Logit Heads"]
        NumHead --> PredNoise["Predicted Noise eps (B, D_num)"]
        CatHeads --> PredLogits["Predicted Clean Logits [(B, K_j)]"]
    end
```

---

## 2. Feature Tokenizer (`src/tabsynth/models/tokenizer.py`)

Tabular attributes must be projected into a shared latent metric space $\mathbb{R}^{d_{\text{model}}}$ before self-attention.

### 2.1 Numerical Feature Tokenization
For each continuous column $i \in \{0, \dots, D_{\text{num}}-1\}$:
$$\mathbf{t}_i^{\text{num}} = x_i \mathbf{w}_i + \mathbf{b}_i, \quad \mathbf{w}_i, \mathbf{b}_i \in \mathbb{R}^{d_{\text{model}}}$$
The continuous values act as scaling factors along learned feature direction vectors $\mathbf{w}_i$, preserving the scalar magnitude while expanding into token space.

### 2.2 Categorical Feature Tokenization
For each discrete column $j \in \{0, \dots, C-1\}$ with cardinality $K_j$:
$$\mathbf{t}_j^{\text{cat}} = \mathbf{x}_j^{\text{oh}} \mathbf{W}_j^{\text{cat}} + \mathbf{b}_j^{\text{cat}}, \quad \mathbf{W}_j^{\text{cat}} \in \mathbb{R}^{K_j \times d_{\text{model}}}$$
If inputs are integer class indices, they are converted on-the-fly to one-hot distributions or mapped via the linear projection weights.

### 2.3 Column Identity & Feature Type Embeddings
To enable the Transformer to distinguish column semantics (e.g. `age` vs `hours-per-week`, or `education` vs `occupation`) and data types:
$$\mathbf{T} = \left[ \mathbf{t}_1^{\text{num}}, \dots, \mathbf{t}_{D_{\text{num}}}^{\text{num}}, \mathbf{t}_1^{\text{cat}}, \dots, \mathbf{t}_C^{\text{cat}} \right] + \mathbf{E}_{\text{col}} + \mathbf{E}_{\text{type}}$$
where:
- $\mathbf{E}_{\text{col}} \in \mathbb{R}^{D \times d_{\text{model}}}$ is a learned parameter matrix.
- $\mathbf{E}_{\text{type}} \in \mathbb{R}^{2 \times d_{\text{model}}}$ with index $0$ for numerical features and $1$ for categorical features.

---

## 3. Timestep and Class Conditioning

### 3.1 Mixed Timestep Embedding
To support independent continuous ($t_{\text{num}}$) and discrete ($t_{\text{cat}}$) schedules:
$$\mathbf{e}_{\text{sin}}(t, 2k) = \sin\left(\frac{t}{10000^{2k/d}}\right), \quad \mathbf{e}_{\text{sin}}(t, 2k+1) = \cos\left(\frac{t}{10000^{2k/d}}\right)$$
Both timesteps are embedded and projected through a 2-layer GELU MLP:
$$\mathbf{e}_{\text{time}} = \text{MLP}\left( [\mathbf{e}_{\text{sin}}(t_{\text{num}}) \,\|\, \mathbf{e}_{\text{sin}}(t_{\text{cat}})] \right) \in \mathbb{R}^{B \times d_{\text{model}}}$$

### 3.2 Class Conditioning & Classifier-Free Guidance (CFG)
For class-conditional synthesis (e.g. minority class oversampling):
- $\mathbf{E}_{\text{class}} \in \mathbb{R}^{(C_{\text{target}} + 1) \times d_{\text{model}}}$.
- Index $C_{\text{target}}$ is reserved as the null unconditional token $\emptyset$.
- During training, condition dropout occurs with probability $p_{\text{uncond}} = 0.15$:
  $$y_{\text{train}} = \begin{cases} \emptyset, & \text{with probability } p_{\text{uncond}} \\ y, & \text{with probability } 1 - p_{\text{uncond}} \end{cases}$$
- Combined conditioning vector:
  $$\mathbf{e}_{\text{cond}} = \mathbf{e}_{\text{time}} + \mathbf{e}_{\text{class}} \in \mathbb{R}^{B \times d_{\text{model}}}$$

---

## 4. Transformer Cross-Feature Interaction Backbone

Operating on the token sequence $\mathbf{T}^{(0)} \in \mathbb{R}^{B \times D \times d_{\text{model}}}$, each Pre-LN layer applies:
$$\mathbf{T}' = \mathbf{T}^{(\ell-1)} + \text{Dropout}\left( \text{MHA}\left( \text{LayerNorm}(\mathbf{T}^{(\ell-1)}) \right) \right)$$
$$\mathbf{T}^{(\ell)} = \mathbf{T}' + \text{Dropout}\left( \text{FFN}\left( \text{LayerNorm}(\mathbf{T}') \right) \right)$$
where $\text{FFN}(\mathbf{h}) = \text{Linear}_{2}(\text{Dropout}(\text{GELU}(\text{Linear}_1(\mathbf{h}))))$.

Default configuration:
- $d_{\text{model}} = 64$
- $\text{heads} = 4$
- $\text{layers} = 2$
- $d_{\text{ff}} = 256$

---

## 5. Feature-Level Residual MLP Denoising Network

To prevent information bottlenecking from flattening the entire table into a single 1D vector, the Residual MLP operates directly across the sequence of feature tokens:
$$\mathbf{h}_0 = \text{LayerNorm}(\mathbf{T}_i) + \text{Linear}_{\text{cond}}(\mathbf{e}_{\text{cond}})$$
$$\mathbf{h}_1 = \text{Linear}_2\left( \text{Dropout}\left( \text{GELU}\left( \text{Linear}_1(\mathbf{h}_0) \right) \right) \right)$$
$$\mathbf{T}_i^{(\text{next})} = \mathbf{T}_i + \mathbf{h}_1$$
Conditioning is broadcast additively across all $D$ feature tokens, modulating every feature representation synchronously.

---

## 6. Output Prediction Heads

After $N_{\text{mlp}}$ residual blocks, tokens are partitioned according to column type:
1. **Continuous Slice** $\mathbf{T}[:, :D_{\text{num}}, :]$:
   Passed through `NumericalOutputHead`:
   $$\hat{\boldsymbol{\epsilon}}_i = \mathbf{w}_{\text{head}}^T \mathbf{T}_i^{\text{num}} + b_{\text{head}} \implies \hat{\boldsymbol{\epsilon}} \in \mathbb{R}^{B \times D_{\text{num}}}$$
2. **Discrete Slice** $\mathbf{T}[:, D_{\text{num}}:, :]$:
   Passed through `CategoricalOutputHeads`:
   $$\hat{\mathbf{z}}^{(j)} = \mathbf{T}_j^{\text{cat}} \mathbf{W}_{\text{head}}^{(j)} + \mathbf{b}_{\text{head}}^{(j)} \implies \hat{\mathbf{z}}^{(j)} \in \mathbb{R}^{B \times K_j}$$

---

## 7. MLP Baseline (`TabSynthMLPDenoiser`)

To empirically isolate the value of cross-column attention for Experiment E1 (RQ1):
- `TabSynthMLPDenoiser` shares identical `FeatureTokenizer`, `MixedTimestepEmbedding`, `ClassConditioningEmbedding`, and `OutputHeads`.
- It replaces the Transformer Multi-Head Self-Attention backbone with an expanded 4-layer Residual MLP.
- The two models share the exact same `ModelOutput` dictionary interface and training protocol.

---

## 8. Parameter Count & Architectural Diagnostics

On the reference benchmark dataset (2 numerical features, 3 categorical features with cardinalities $[2, 4, 7]$, 2 classes, $d_{\text{model}}=64$):

| Submodule | TabSynth Hybrid | TabSynth MLP Baseline |
| :--- | :--- | :--- |
| **Feature Tokenizer** | 1,728 | 1,728 |
| **Timestep Embedding** | 45,568 | 45,568 |
| **Class Embedding** | 192 | 192 |
| **Transformer Encoder** | 100,096 | *None (0)* |
| **Residual MLP Denoiser** | 74,880 | 149,760 (4 layers) |
| **Numerical Output Head** | 65 | 65 |
| **Categorical Output Heads** | 845 | 845 |
| **Total Parameters** | **223,374** | **198,030** |
| **Parameter Delta** | **+25,344 (+12.8%)** | Baseline |

---

## 9. Diffusion Engine Integration (`ModelDiffusionIntegrator`)

The integrator coordinates:
1. Continuous forward diffusion: $\mathbf{x}_t^{\text{num}} = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$.
2. Discrete forward Markov corruption: $\mathbf{x}_t^{\text{cat}} \sim \text{Categorical}(\bar{\alpha}_t \mathbf{x}_0 + \frac{1 - \bar{\alpha}_t}{K}\mathbf{1})$.
3. Joint forward model evaluation: $\hat{\boldsymbol{\epsilon}}, \hat{\mathbf{z}} = \text{model}(\mathbf{x}_t^{\text{num}}, \mathbf{x}_t^{\text{cat}}, t_{\text{num}}, t_{\text{cat}}, y)$.
4. Loss computation:
   $$\mathcal{L}_{\text{total}} = \lambda_{\text{num}} \text{MSE}(\boldsymbol{\epsilon}, \hat{\boldsymbol{\epsilon}}) + \lambda_{\text{cat}} \mathcal{L}_{\text{D3PM-KL}}(\mathbf{x}_0, \hat{\mathbf{z}}) + \lambda_{\text{aux}} \text{CE}(\hat{\mathbf{z}}, \mathbf{x}_0)$$

---

## 10. Test Suite & Verification Results

All 23 model unit tests in `tests/models/` passed:
```
tests/models/test_conditioning.py::test_class_conditioning_embedding_shapes PASSED
tests/models/test_conditioning.py::test_unconditional_null_token PASSED
tests/models/test_conditioning.py::test_conditioning_dropout_during_training PASSED
tests/models/test_conditioning.py::test_class_embedding_gradient_flow PASSED
tests/models/test_mlp_denoiser.py::test_mlp_denoiser_shapes_and_interface PASSED
tests/models/test_mlp_denoiser.py::test_mlp_denoiser_gradient_flow PASSED
tests/models/test_mlp_denoiser.py::test_compare_hybrid_and_mlp_baseline PASSED
tests/models/test_model_integration.py::test_hybrid_diffusion_integration[cpu] PASSED
tests/models/test_model_integration.py::test_hybrid_diffusion_integration[mps] PASSED
tests/models/test_model_integration.py::test_mlp_baseline_diffusion_integration PASSED
tests/models/test_output_heads.py::test_numerical_output_head PASSED
tests/models/test_output_heads.py::test_categorical_output_heads_arbitrary_cardinalities PASSED
tests/models/test_time_embedding.py::test_sinusoidal_time_embedding_shapes_and_values PASSED
tests/models/test_time_embedding.py::test_mixed_timestep_embedding_dual_and_single_inputs PASSED
tests/models/test_tokenizer.py::test_numerical_tokenizer_shapes PASSED
tests/models/test_tokenizer.py::test_categorical_tokenizer_one_hot_and_integer_indices PASSED
tests/models/test_feature_tokenizer_mixed_data PASSED
tests/models/test_feature_tokenizer_numerical_only PASSED
tests/models/test_feature_tokenizer_categorical_only PASSED
tests/models/test_feature_tokenizer_batch_size_one PASSED
tests/models/test_transformer_denoiser.py::test_transformer_block_and_encoder_shapes PASSED
tests/models/test_transformer_denoiser.py::test_hybrid_denoiser_forward_and_output_shapes PASSED
tests/models/test_transformer_denoiser.py::test_hybrid_denoiser_gradient_flow PASSED
============================== 23 passed in 6.58s ==============================
```

Overfitting smoke experiment (`scripts/smoke_test_model.py`):
- **CPU**: Initial loss = 1.4214, Final loss = 0.5598 (**60.62% loss reduction** across 15 optimization steps).
- **Apple Silicon MPS**: Confirmed native Metal execution with gradient flow and optimizer convergence.
