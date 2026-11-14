# Phase 04: Training Pipeline & Conditional Data Generation

## 1. Objective
Implement an end-to-end training pipeline, validation protocol, top-$k$ rolling checkpoint manager, and reverse sampling engine supporting Classifier-Free Guidance (CFG) for conditional tabular data synthesis and minority-class steering.

## 2. Methodology & Mathematics
### 2.1 Multi-Objective Composite Loss
$$\mathcal{L}_{\text{total}} = \lambda_{\text{num}} \text{MSE}(\boldsymbol{\epsilon}, \hat{\boldsymbol{\epsilon}}) + \lambda_{\text{cat}} \mathcal{L}_{\text{D3PM-CE}}(\mathbf{x}_0, \hat{\mathbf{x}}_0) + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$
where $\lambda_{\text{num}} = 1.0$, $\lambda_{\text{cat}} = 0.2$, $\lambda_{\text{aux}} = 0.1$. Continuous features optimize MSE on noise predictions; categorical features optimize cross-entropy between true clean one-hot vectors and predicted clean categorical logits $\hat{\mathbf{z}}$.

### 2.2 Optimization & Regularization
- **Parameter Decay Grouping:** 2D+ projection and Transformer weights apply weight decay ($10^{-4}$); 1D biases and LayerNorm parameters have zero weight decay.
- **Learning Rate Schedule:** Warmup (2 epochs) followed by Cosine Annealing decay down to $\eta_{\min} = 10^{-6}$.
- **Gradient Clipping:** Strict norm clipping $\| \mathbf{g} \|_2 \le 1.0$ prevents gradient explosions on mixed categorical boundaries.
- **Early Stopping:** Monitors validation loss with configurable patience (10 epochs) and restores best weights.

### 2.3 Reverse Sampling & Classifier-Free Guidance (CFG)
At each step $t \in [T, 1]$, the reverse sampler computes conditional and unconditional outputs:
$$\tilde{\boldsymbol{\epsilon}} = (1 + w) \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, c, t) - w \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, \emptyset, t)$$
$$\tilde{\mathbf{z}} = (1 + w) \mathbf{z}_{\theta}(\mathbf{x}_t, c, t) - w \mathbf{z}_{\theta}(\mathbf{x}_t, \emptyset, t)$$
Conditioning strategies supported:
- `natural`: Samples class labels matching real training dataset proportions.
- `balanced`: Samples equal quantities across all target classes ($50/50$ for binary).
- `class_specific`: Forces generation of an exact single target class (e.g., 100% minority `>50K`).

## 3. Architecture & Implementation
- **`src/tabsynth/training/optimizer.py`:** Parameter grouping and optimizer builder.
- **`src/tabsynth/training/scheduler.py`:** Cosine Annealing with warmup.
- **`src/tabsynth/training/early_stopping.py`:** Validation loss monitor.
- **`src/tabsynth/training/checkpoint.py`:** `CheckpointManager` maintaining top-$k$ rolling checkpoints (`keep_top_k: 2`) preventing disk saturation.
- **`src/tabsynth/training/trainer.py`:** `TabSynthTrainer` managing epochs, MPS cache flushes, and metrics.
- **`src/tabsynth/sampling/mixed_sampler.py`:** `MixedAncestralSampler`.
- **`src/tabsynth/sampling/conditional_sampler.py`:** `ConditionalSampler`.
- **`src/tabsynth/sampling/generation.py`:** `GenerationPipeline` translating reverse samples into DataFrames via inverse preprocessing.

## 4. Key Design Decisions
- **Chunked Mini-Batch Generation:** Generates synthetic records in chunks (e.g., $B = 256$) to strictly bound peak GPU/MPS memory consumption.
- **Automatic MPS Device Fallback:** Detects Apple Silicon Metal Performance Shaders (`torch.backends.mps.is_available()`) and falls back gracefully to CPU.

## 5. Relevant Source Files
- `src/tabsynth/training/`
- `src/tabsynth/sampling/`
- `scripts/train.py`
- `scripts/generate.py`

## 6. Tests Performed
- `tests/training/test_optimizer.py` (2 tests)
- `tests/training/test_scheduler.py` (2 tests)
- `tests/training/test_early_stopping.py` (2 tests)
- `tests/training/test_checkpoint.py` (2 tests)
- `tests/training/test_train_step.py` (1 test)
- `tests/training/test_trainer.py` (1 test)
- `tests/sampling/test_guidance.py` (3 tests)
- `tests/sampling/test_mixed_sampler.py` (1 test)
- `tests/sampling/test_conditional_sampler.py` (1 test)
- `tests/sampling/test_generation.py` (1 test)

## 7. Actual Results
- Trained `adult_hybrid_best.pt` on UCI Adult Income: validation loss converged from 1.3660 to **0.4544**.
- Trained `adult_mlp_best.pt` baseline: validation loss converged from 1.4932 to **0.4779**.
- Generated 500 balanced synthetic samples at 109.8–205.1 rows/sec with zero missing values and preserved schema.

## 8. Known Limitations
- Very large reverse step counts ($T = 1000$) require significant sampling latency. $T = 100$ steps with Cosine scheduling achieves parity with fractional runtime.

## 9. Reproduction Instructions
```bash
# Train Hybrid model
python scripts/train.py --config configs/experiments/adult_hybrid.yaml

# Generate synthetic records
python scripts/generate.py \
  --checkpoint checkpoints/adult_hybrid_best.pt \
  --num_samples 500 \
  --class_condition balanced \
  --output data/synthetic/adult_hybrid_balanced.csv
```
