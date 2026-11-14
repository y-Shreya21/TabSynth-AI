# Changelog

All notable changes to the **TabSynth AI** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-11-15

### Added
- **Phase 01: Research Foundation & Data Pipeline**
  - Dataset ingestion and preparation scripts for UCI Adult Income, Bank Marketing, and Credit Default.
  - Automated schema inference (`TableSchema`, `ColumnMetadata`) with type classification.
  - Leak-free tabular preprocessing (`TabularPreprocessor`) with bidirectional numerical scaling and categorical encoding.
  - Stratified 70/15/15 partitioning protocol.
- **Phase 02: Mixed-Type Diffusion Engine**
  - Continuous Gaussian DDPM diffusion with Linear, Cosine, and Sigmoid noise schedules.
  - Discrete D3PM multinomial diffusion over categorical simplices with exact analytical Bayes posterior marginalization.
  - Unified `MixedTypeDiffusion` orchestrating continuous and discrete transitions.
- **Phase 03: Hybrid Transformer-MLP Denoising Architecture**
  - Feature-wise numerical and categorical tokenizers with learnable column identity embeddings.
  - Pre-LayerNorm Transformer multi-head self-attention feature encoder.
  - Sinusoidal time embedding and Classifier-Free Guidance (CFG) class conditioning ($p_{\text{uncond}} = 0.15$).
  - Residual MLP denoising backbone with SwiGLU / SiLU non-linearities and dual output heads.
  - TabDDPM-style flat MLP denoiser baseline (`TabSynthMLPDenoiser`) for architectural ablations.
- **Phase 04: Training Pipeline & Sampling with CFG**
  - Parameter grouping optimizer separating weight decay for 2D+ matrices from 1D biases/norms.
  - Warmup + Cosine Annealing learning rate scheduler.
  - Early stopping with best weight restoration and rolling top-$k$ checkpointing (`keep_top_k: 2`).
  - Mixed ancestral reverse sampler (`MixedAncestralSampler`) with Classifier-Free Guidance extrapolation.
  - Class-conditional sampling strategies: `natural`, `balanced`, and `class_specific`.
  - Full Apple Silicon M4 MPS acceleration with CPU fallback.
- **Phase 05: Comprehensive Evaluation, Baselines & Privacy Auditing**
  - Statistical fidelity suite: Kolmogorov-Smirnov test, 1-Wasserstein distance, Total Variation Distance (TVD), Pearson and Spearman correlation Frobenius distance, Theil's U, and 5-fold cross-validated Classifier Two-Sample Test (C2ST).
  - Downstream machine learning utility suite: TRTR, TSTR, relative macro-F1 parity ratios, minority recall gain, and mixed-data minority augmentation across $[0.25\times, 0.5\times, 1.0\times, 2.0\times]$.
  - Empirical privacy auditing suite: Pairwise normalized Gower distance, Distance to Closest Record (DCR) to Train and Test partitions, DCR ratios, 5th percentile ratios, Nearest Neighbor Distance Ratios (NNDR), exact match audits, distance-based Membership Inference Attacks (MIA), Attribute Inference Attacks (AIA), and predicate-based singling out.
  - Standardized baseline registry (`BaseSynthesizer`) implementing Gaussian Copula, native PyTorch TVAE, TabDDPM MLP, and TabSynth Hybrid.
  - Interactive Plotly visualization suite generating standalone HTML and JSON figures.
  - Official research evaluation report (`docs/PHASE_05_EVALUATION_REPORT.md`) derived from empirical runs.
  - Comprehensive 103-test suite across unit, integration, model, diffusion, and evaluation modules.
