# Phase 06 Application Audit: TabSynth AI Production Architecture

## 1. Executive Summary

This document performs a comprehensive pre-development audit of the existing **TabSynth AI** research codebase (Phases 01–05) to design and implement a production-grade service layer, persistent SQLite metadata layer, asynchronous job execution engine, REST API, and Streamlit user interface.

---

## 2. Inventory of Existing Interfaces & ML Components

### 2.1 Phase 01: Ingestion, Schema & Preprocessing
- **Schema Inference**: `tabsynth.data.schema.infer_table_schema(df, dataset_name, target_column)`
  - Returns `TableSchema`: `continuous_columns: List[str]`, `categorical_columns: List[str]`, `cardinalities: List[int]`, `class_distribution: Optional[Dict[str, float]]`.
- **Preprocessor**: `tabsynth.data.preprocessor.TabularPreprocessor(schema, continuous_scaler="quantile", seed=42)`
  - Methods: `.fit(df_train)`, `.transform(df)`, `.inverse_transform(x_num, x_cat)`.
  - Zero-leakage guarantee: fit solely on training partition.
- **Dataset Partitioning & Loaders**:
  - `create_stratified_split(df, target_column, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)`
  - `build_dataloaders(df_train, df_val, df_test, preprocessor, batch_size=256, num_workers=0)`

### 2.2 Phase 02: Diffusion Engine
- **Continuous Diffusion**: `tabsynth.diffusion.gaussian.GaussianDiffusion(num_timesteps, schedule_type="cosine", device=device)`
  - Methods: `.q_sample(x_0, t, noise)`, `.p_sample(model_fn, x_t, t, ...)`.
- **Categorical Diffusion**: `tabsynth.diffusion.multinomial.MultinomialDiffusion(cardinalities, num_timesteps, schedule_type="cosine", device=device)`
  - Methods: `.q_sample(x_0_cat, t)`, `.p_sample(model_fn, x_t_cat, t, ...)`.
- **Mixed Diffusion Wrapper**: `tabsynth.diffusion.mixed.mixed_diffusion.MixedTypeDiffusion`
  - Encapsulates combined continuous and categorical forward/reverse schedules.

### 2.3 Phase 03: Denoising Model Architectures
- **Hybrid Transformer-MLP**: `tabsynth.models.hybrid.TabSynthHybridModel`
  - Parameters: `num_continuous`, `cardinalities`, `num_classes`, `d_model`, `transformer_layers`, `attention_heads`, `mlp_hidden_dim`, `mlp_layers`, `dropout`.
  - Input: `x_num_t (B, D_num)`, `x_cat_t (B, D_cat)`, `t (B,)`, `cond (B,)` (target class).
  - Output: `(pred_eps_num, pred_logits_cat_list)`.
- **MLP Baseline**: `tabsynth.baselines.mlp_diffusion.TabDDPMMLPModel`
  - Serves as the high-throughput, non-attention baseline.

### 2.4 Phase 04: Training & Sampling Pipelines
- **Trainer**: `tabsynth.training.trainer.TabSynthTrainer`
  - Handles loss computation via `TabSynthLoss(multinomial_diffusion, cardinalities, lambda_num, lambda_cat, lambda_aux)`.
  - Implements gradient clipping, early stopping, and step-level tracking.
- **Checkpoint Manager**: `tabsynth.training.checkpoint.CheckpointManager`
  - Saves: `model_state_dict`, `optimizer_state_dict`, `preprocessor` (serialized instance), `schema`, `epoch`, `val_loss`.
  - Top-K pruning and atomic file writes.
- **Sampler**: `tabsynth.sampling.sampler.TabSynthSampler`
  - High-level interface: `.sample(num_samples, batch_size, condition_strategy, specific_class, cfg_scale)`.
  - Reconstructs decoded pandas DataFrame via `preprocessor.inverse_transform()`.

### 2.5 Phase 05: Evaluation & Privacy Auditing
- **Fidelity**: `tabsynth.evaluation.fidelity.evaluate_statistical_fidelity(real_df, synth_df, continuous_cols, categorical_cols)`
  - Returns KS statistics, TVD, Pearson/Wasserstein metrics, and C2ST classifier scores.
- **Utility**: `tabsynth.evaluation.utility.evaluate_downstream_utility(df_train_real, df_train_synth, df_test_real, target_column)`
  - TRTR vs TSTR benchmarks with LogisticRegression and RandomForest.
- **Privacy**: `tabsynth.evaluation.privacy.evaluate_empirical_privacy(df_train, df_test, df_synth, continuous_cols, categorical_cols, subsample_size)`
  - Distance-to-Closest-Record (DCR) ratios, exact matching rates, and shadow Membership Inference Attack (MIA) AUROC.

---

## 3. Existing Serving & API Assessment

Currently, `api/` contains basic prototypes:
- `api/server.py`: basic FastAPI app definition with CORS.
- `api/routes.py`: preliminary route definitions mixing HTTP handling, file I/O, and model training in a single module.
- `api/schemas.py`: basic schema declarations.

### Gaps Identified:
1. **Lack of Persistence**: Training jobs and dataset schemas are held only in memory (`TRAINING_JOBS = {}`), causing all state to be lost on process restart.
2. **Missing Database Layer**: No persistent storage for datasets, models, experiments, or job histories.
3. **Blocking / Primitive Background Execution**: Minimal background execution with no job cancellation, concurrency controls, or persisted task logs.
4. **Monolithic Architecture**: HTTP route handlers directly instantiate PyTorch models and run multi-step algorithms instead of delegating to domain services.
5. **Security Constraints**: Uploads lack strict file size enforcement, MIME checks, path traversal guards, and structured request IDs.

---

## 4. Phase 06 Target Architecture

```
                    +--------------------------------+
                    | Streamlit Frontend Dashboard   |
                    | (TabSynth Analytics - 8 Pages) |
                    +---------------+----------------+
                                    | HTTP / REST
                    +---------------v----------------+
                    |       FastAPI Application      |
                    |  (Request ID, Logging, CORS)   |
                    +---------------+----------------+
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
+---------v---------+     +---------v---------+     +---------v---------+
| Dataset Service   |     | Model & Job Svc   |     | Eval & Synth Svc  |
| - Schema Prof.    |     | - Bounded Queue   |     | - CFG Sampling    |
| - Validation      |     | - State Machine   |     | - Fidelity/Priv.  |
+---------+---------+     +---------+---------+     +---------+---------+
          |                         |                         |
          +-------------------------+-------------------------+
                                    |
                    +---------------v----------------+
                    | SQLite Persistence Repository  |
                    | (data/tabsynth.db)             |
                    +---------------+----------------+
                                    |
                    +---------------v----------------+
                    | Local Filesystem Storage       |
                    | (data/, checkpoints/, exp/)    |
                    +--------------------------------+
```

---

## 5. Implementation Roadmap

1. **Persistence & Configuration**: Implement `api/config.py`, `api/db/session.py`, `api/db/models.py`, `api/db/repositories.py`.
2. **Schema Separation**: Modularize Pydantic v2 schemas in `api/schemas/` with backward-compatible re-exports.
3. **Middleware & Exception Handling**: Add `X-Request-ID` tracing, execution timing, structured logging, and sanitize error responses.
4. **Background Job Manager**: Implement thread-safe `JobManager` with concurrency limiting (preventing MPS OOM) and database-backed lifecycle tracking (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`).
5. **Domain Services**: Construct `DatasetService`, `ModelService`, `GenerationService`, `EvaluationService`, `ExperimentService`.
6. **Modular Route Handlers**: Implement routes for health, datasets, models, generation, evaluation, jobs, and experiments.
7. **Streamlit Frontend ("TabSynth Analytics")**: Implement clean 8-page scientific SaaS UI with API integration and local fallback.
8. **Testing & Verification**: Create unit and integration tests covering the new service layer and endpoints while maintaining 100% pass rate on all 103 existing tests.
9. **Deployment & CI**: Create `Dockerfile`, `docker-compose.yml`, update `.env.example`, and finalize Phase 06 documentation.
