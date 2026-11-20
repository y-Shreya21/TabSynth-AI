# TabSynth Analytics — Application Interface & Screenshot Inventory
## Visual UI/UX Documentation Across 10 Research Views

**Version**: `v0.9.0-rc.1` / Release Candidate  
**Platform**: Streamlit 1.32+ with Enterprise Dark Technical Theme  
**Display Target**: 1920 × 1080 / 1440 × 900 Retina Displays  

---

## 1. Overview of Presentation Layer

TabSynth Analytics provides a 10-view analytical SaaS interface designed for AI researchers, ML engineers, and data privacy officers. The interface uses a dark technical aesthetic (`#0B0F19` background, `#131B2E` container cards, `#8B5CF6` electric violet accents, and `#10B981` emerald telemetry indicators) with responsive containers, real-time status badges, and interactive visualizations.

### Navigation Architecture:
- **Workspace Category**:
  1. `Overview Dashboard` (Platform telemetry, hardware engine health, high-level metrics, system diagram)
  2. `Dataset Manager` (Upload, schema detection, missing value inspection, target distribution)
  3. `Data Explorer` (Marginal distribution overlays, correlation heatmaps, outlier profiling)
- **Generation Category**:
  4. `Model Studio` (Diffusion hyperparameters, architecture selector, live loss curves, job queue)
  5. `Synthetic Data Generator` (Steerable reverse sampling, CFG guidance slider, balanced generation, CSV export)
- **Evaluation Category**:
  6. `Fidelity Analysis` (KS test curves, 1D Wasserstein distances, TVD bars, Pearson Frobenius norm)
  7. `Utility Evaluation` (TSTR vs TRTR downstream classifier benchmarks, macro-F1 parity gauges)
  8. `Privacy Audit` (Gower DCR histograms, NNDR threshold analysis, exact match audit, shadow MIA ROC curve)
- **Research Category**:
  9. `Benchmark Center` (Multi-model radar comparison, ranking matrix, leaderboards across 4 baselines)
  10. `Experiment Explorer` (Historical run tracking, hyperparameter diffs, artifact downloads)

---

## 2. Screenshot & View Inventory

### View 01: Overview Dashboard
- **Route Key**: `Overview Dashboard`
- **Icon**: `📊`
- **Primary Visual Components**:
  - Top header with live engine status badge (`● Engine Online` with pulse animation).
  - Hardware Banner: Apple Silicon MPS / CUDA execution platform indicator.
  - KPI Metrics Grid: 4 stat cards (Total Datasets Ingested, Trained Models Registered, Synthetic Records Synthesized, Best Pearson Frobenius Distance: `0.1117`).
  - System Flow Diagram: Embedded interactive Mermaid generative pipeline.
  - Telemetry Table: Active and recent background worker jobs with progress bars and status badges.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

### View 02: Dataset Manager
- **Route Key**: `Dataset Manager`
- **Icon**: `📁`
- **Primary Visual Components**:
  - Drag-and-drop CSV upload zone with file size enforcement (< 100 MB).
  - Schema feature breakdown table: Name, Data Type, Detected Type (Continuous vs Categorical), Cardinality, Null Count.
  - Target variable class distribution bar chart displaying severe class imbalances (e.g. Adult Income 78% / 22%).
  - First 15 rows preview table with formatted numerical columns.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

### View 03: Data Explorer
- **Route Key**: `Data Explorer`
- **Icon**: `🔍`
- **Primary Visual Components**:
  - Summary KPI cards: Records Analyzed, Features Profiled, Missing Cells, Outliers Flagged.
  - Univariate distribution tabs: Continuous KDE density plots and Categorical frequency bars.
  - Pairwise Pearson correlation heatmap across continuous features with divergent colormap.
  - Outlier detection scatter plots highlighting extreme quantiles.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

### View 04: Model Studio
- **Route Key**: `Model Studio`
- **Icon**: `⚡`
- **Primary Visual Components**:
  - Architecture selector: `TabSynth Hybrid (Transformer + MLP)` vs `TabDDPM MLP Baseline`.
  - Noise schedule dropdown: `cosine`, `linear`, `cosine_log`.
  - Diffusion steps slider (10 to 1,000 steps) and epoch input.
  - Loss weight sliders: $\lambda_{\text{num}}$, $\lambda_{\text{cat}}$, $\lambda_{\text{aux}}$.
  - Asynchronous dispatch button ("Launch Training Job") with job cancellation controls.
  - Real-time training loss curve displaying total, continuous MSE, and categorical cross-entropy loss.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

### View 05: Synthetic Data Generator
- **Route Key**: `Synthetic Data Generator`
- **Icon**: `🧬`
- **Primary Visual Components**:
  - Registered model checkpoint selector.
  - Sample count input and batch size selector.
  - Condition strategy selector: `balanced` (50/50 prior), `natural` (training frequency), or `class_specific`.
  - Classifier-Free Guidance (CFG) slider ($w \in [0.0, 5.0]$).
  - Sampling execution progress bar and elapsed time telemetry.
  - Generated records preview table and one-click "Download Synthetic CSV" button.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

### View 06: Fidelity Analysis
- **Route Key**: `Fidelity Analysis`
- **Icon**: `🎯`
- **Primary Visual Components**:
  - Executive fidelity stat cards: Mean KS (`0.0545`), Mean Wasserstein (`0.2115`), Mean TVD (`0.1719`), Pearson Frobenius Distance (`0.1117`).
  - Marginal distribution overlay: Real vs Synthetic cumulative distribution functions (CDF).
  - Categorical proportion comparison bar charts.
  - 2D Correlation difference matrix ($|\mathbf{R}_{\text{real}} - \mathbf{R}_{\text{synth}}|$) heatmap.
  - Classifier Two-Sample Test (C2ST) ROC-AUC score and confusion matrix.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

### View 07: Utility Evaluation
- **Route Key**: `Utility Evaluation`
- **Icon**: `📈`
- **Primary Visual Components**:
  - Downstream classifier benchmark table: Random Forest, HistGradientBoosting, Logistic Regression.
  - Metric breakdown: Accuracy, Macro-F1, Weighted-F1, Precision, Recall, AUROC.
  - Train on Synthetic, Test on Real (TSTR) vs Train on Real, Test on Real (TRTR) comparison bars.
  - Relative Macro-F1 Parity gauge ($F1_{\text{TSTR}} / F1_{\text{TRTR}}$).
  - Minority class recall restoration indicator.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

### View 08: Privacy Audit
- **Route Key**: `Privacy Audit`
- **Icon**: `🛡️`
- **Primary Visual Components**:
  - Consolidated Privacy Risk Badge (`LOW_RISK`).
  - Distance to Closest Record (DCR) distribution histogram: Synthetic-to-Train vs Synthetic-to-Test vs Real Baseline.
  - DCR Ratio stat card (`1.0155`) and 5th percentile ratio (`2.4997`).
  - Nearest Neighbor Distance Ratio (NNDR) histogram and fraction $< 0.5$ (`3.6%`).
  - Exact match audit: `0 Exact Matches (0.00% Memorization Rate)`.
  - Distance-based shadow Membership Inference Attack (MIA) ROC curve (AUC = `0.4907`).
  - Prominent formal Differential Privacy disclaimer callout.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

### View 09: Benchmark Center
- **Route Key**: `Benchmark Center`
- **Icon**: `🏆`
- **Primary Visual Components**:
  - Multi-Model Comparison Table: Gaussian Copula, TVAE, TabDDPM MLP, TabSynth Hybrid across 15 metrics.
  - Radar Chart: Multi-dimensional comparison across Fidelity, Utility, Privacy, and Speed.
  - Correlation Frobenius norm leaderboards highlighting TabSynth's 29.8% error reduction.
  - Throughput vs Fidelity scatter plot.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

### View 10: Experiment Explorer
- **Route Key**: `Experiment Explorer`
- **Icon**: `📜`
- **Primary Visual Components**:
  - Searchable experiment run registry.
  - Hyperparameter diff comparison between selected runs.
  - Training loss progression overlays across multiple models.
  - Artifact download hub for raw JSON results and CSV summaries.
- **Verification Status**: **VERIFIED (Pass)** via `scripts/run_phase08_acceptance.py`.

---

## 3. UI/UX Verification Summary

All 10 interface views were verified using `streamlit.testing.v1.AppTest` and manual end-to-end traversal:
- **Zero Uncaught Exceptions**: All views render with empty exception lists.
- **High-DPI Responsive Design**: Fluid CSS grid structures adapt cleanly between desktop monitors and laptop screens.
- **Accessible Color Palettes**: Contrast ratios on all typography exceed WCAG 2.1 AA standards ($> 4.5:1$).
