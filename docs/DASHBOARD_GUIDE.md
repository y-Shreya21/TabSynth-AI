# TabSynth Analytics: Streamlit Dashboard User Guide

The **TabSynth Analytics** interactive web dashboard provides a graphical interface for tabular generative modeling, schema inspection, training orchestration, conditional generation, and empirical privacy auditing.

---

## 1. Launching the Dashboard

Ensure your virtual environment is active:
```bash
streamlit run app/main.py --server.port 8501
```
Or use the Makefile shortcut:
```bash
make run-app
```
Access the dashboard at `http://localhost:8501`.

---

## 2. Page Navigation & Features

### Page 1: Overview Dashboard
- **Executive Metric Cards**: Displays total registered datasets, available trained models, background jobs executed, and historical experiments.
- **Hardware Detection**: Displays whether Apple MPS, NVIDIA CUDA, or CPU fallback is active.
- **Recent Jobs Table**: Displays the status, progress percentage, and last updated timestamp for recent jobs.

### Page 2: Dataset Manager
- **Inspect Registered Datasets**: Browse ingested datasets, view dimensional statistics, inspect continuous vs categorical features, and view target class distribution bar charts.
- **Upload New CSV**: Drag-and-drop tabular CSV files up to 50MB. Automatic header validation rejects duplicate column names and validates data integrity.

### Page 3: Model Training
- **Dataset Selection**: Choose from database-registered datasets or local CSVs.
- **Architecture Choice**:
  - *Hybrid Transformer-MLP*: Multi-head self-attention feature interaction backbone + residual MLP denoiser (recommended).
  - *MLP Baseline*: TabDDPM feedforward baseline without attention tokenization.
- **Hyperparameter Controls**: Sliders for training epochs, diffusion timesteps ($T$), batch size, learning rate, and compute device.
- **Asynchronous Execution**: Launches training in a managed background thread. Real-time progress bars and loss curves are updated without blocking user interaction.

### Page 4: Synthetic Data Generator
- **Model Checkpoint Selector**: Select from trained model checkpoints.
- **Conditioning Strategies**:
  - `balanced`: Samples equal quantities of each class label.
  - `natural`: Matches the empirical class distribution of the training set.
  - `class_specific`: Steers generation strictly toward a designated target class (ideal for rare minority-class augmentation).
  - `unconditional`: Generates data ignoring class labels.
- **Classifier-Free Guidance (CFG)**: Adjust guidance scale ($w \in [0.0, 3.0]$) to balance sample diversity against mode fidelity.
- **CSV Download Button**: Direct browser download of generated synthetic tabular data.

### Page 5: Fidelity Dashboard
- **Metric Scorecard**: Displays Mean Kolmogorov-Smirnov (KS) statistic, Mean Total Variation Distance (TVD), Correlation Frobenius distance, and Classifier Two-Sample Test (C2ST) ROC-AUC.
- **1D Marginal Distribution Overlays**: Compares continuous probability densities (Plotly histogram overlays) and categorical frequency distributions (grouped bar charts).

### Page 6: Utility Dashboard
- **TSTR vs TRTR Benchmark**: Evaluates whether synthetic data trains effective downstream models.
- **Classifier Battery**: Benchmarks across Logistic Regression and Random Forest.
- **Key Metrics**: Macro-F1 parity ratio, minority class recall delta, and balanced accuracy.

### Page 7: Privacy Audit
- **Distance-to-Closest-Record (DCR)**: Computes Gower distance ratios between real training records and synthetic samples.
- **Exact Memorization Detection**: Verifies zero verbatim record copies.
- **Shadow Membership Inference Attack (MIA)**: Computes shadow model ROC-AUC score ($\approx 0.50$ represents ideal random guessing).
- **Formal Academic Disclaimer**: Highlights that empirical privacy metrics are not Differential Privacy ($\epsilon, \delta$-DP) guarantees.

### Page 8: Experiment Explorer
- **Experiment Log Browser**: Inspects serialized benchmark results and SQLite records.
- **Metric Inspection**: Full JSON viewer for all evaluation experiments.
