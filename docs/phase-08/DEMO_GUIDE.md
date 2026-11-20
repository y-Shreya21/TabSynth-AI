# TabSynth AI — Technical Demonstration Walkthrough & Script
## 15-Minute Live End-to-End Demonstration Guide

**Audience**: ML Researchers, Principal Architects, Data Science Leaders, Engineering Hiring Committees  
**Duration**: 12–15 Minutes  
**Software Version**: `v0.9.0-rc.1` / Release Candidate  

---

## 1. Pre-Demo Verification Checklist

Complete these preparation steps at least 10 minutes prior to the demonstration:

```bash
# 1. Activate Python 3.11 virtual environment
cd "/Users/shreyayadav/TabSynth AI"
source .venv/bin/activate

# 2. Verify dataset availability
ls -lh data/raw/adult.csv

# 3. Verify pre-trained release checkpoints
ls -lh checkpoints/adult_hybrid_best.pt checkpoints/adult_mlp_best.pt

# 4. Verify benchmark results are present
ls -lh experiments/results/benchmark_summary.csv

# 5. Execute acceptance smoke test to ensure clean state
python scripts/run_phase07_smoke.py

# 6. Launch FastAPI backend (Terminal 1)
make run-api
# Ensure http://localhost:8000/health returns status "healthy"

# 7. Launch Streamlit frontend (Terminal 2)
make run-app
# Open http://localhost:8501 in browser
```

---

## 2. Timed Demonstration Script

### Segment 1: Executive Motivation & Problem Definition (1.5 Minutes)
- **Presenter Action**: Open browser to `http://localhost:8501` on **Overview Dashboard**.
- **Talking Points**:
  - *"Welcome everyone. Today I'm demonstrating TabSynth AI, a generative framework specifically engineered for mixed-type tabular data under severe class imbalance and privacy constraints."*
  - *"In production enterprise domains like banking, healthcare, and fraud detection, data scientists face four interconnected roadblocks: heterogeneous feature types, extreme class imbalance, complex non-linear correlations, and empirical privacy risks."*
  - Point to the **Generative Framework Architecture** diagram on the Overview page.
  - *"TabSynth AI addresses this by coupling continuous Gaussian diffusion with discrete multinomial diffusion (D3PM), unified by a Pre-LN Transformer and residual MLP denoiser with Classifier-Free Guidance."*

### Segment 2: Zero-Leakage Dataset Ingestion & Profiling (2 Minutes)
- **Presenter Action**: Navigate to **Dataset Manager** in the sidebar.
- **Narrated Workflow**:
  - Show the UCI Adult Income dataset already ingested (or drag `data/raw/adult.csv` into the upload zone).
  - Highlight the automated schema inference table:
    - *"Notice how the system immediately classifies 6 numerical features (such as `age`, `hours_per_week`, `capital_gain`) and 8 categorical features (such as `workclass`, `education`, `occupation`) without manual intervention."*
  - Scroll down to the **Target Variable Class Balance** chart:
    - *"Notice the severe imbalance in our target `income`: 78% $\le 50K$ vs only 22% $> 50K$. In traditional GANs, this imbalance leads to mode collapse where the model generates almost no minority instances."*

### Segment 3: Deep Feature Exploration & Correlations (1.5 Minutes)
- **Presenter Action**: Navigate to **Data Explorer**.
- **Narrated Workflow**:
  - Point out the 4 KPI stat cards (32,561 records, 14 features, 0 missing cells, robust quantiles).
  - Click on the **Correlation Matrix** tab to reveal the Pearson correlation heatmap:
    - *"Here we observe complex inter-feature associations—for instance, between `age`, `education_num`, and `hours_per_week`. Capturing these multi-column dependencies without creating impossible combinations is where our Transformer self-attention excels."*

### Segment 4: Model Studio & Training Dispatch (2 Minutes)
- **Presenter Action**: Navigate to **Model Studio**.
- **Narrated Workflow**:
  - Explain the architecture selector:
    - *"We offer two backbones: our proposed TabSynth Hybrid with cross-feature self-attention, and the TabDDPM MLP baseline for comparative ablation."*
  - Show the diffusion hyperparameters:
    - Noise schedule: `cosine`
    - Diffusion steps: `100`
    - Loss weights: $\lambda_{\text{num}} = 1.0, \lambda_{\text{cat}} = 0.2, \lambda_{\text{aux}} = 0.1$
  - Explain the asynchronous architecture:
    - *"Training runs execute in background worker threads managed by our thread-safe Job Manager. Users can monitor loss curves in real-time or cancel long-running jobs gracefully with rollback safety."*

### Segment 5: Controlled Sampling with Classifier-Free Guidance (2.5 Minutes)
- **Presenter Action**: Navigate to **Synthetic Data Generator**.
- **Narrated Workflow**:
  - Select the pre-trained `adult_hybrid_best.pt` checkpoint.
  - Set sample count to `500`.
  - Set Condition Strategy to **`balanced`** (50% $\le 50K$, 50% $> 50K$).
  - Adjust the **Classifier-Free Guidance (CFG) scale** slider to `w = 1.5`:
    - *"CFG allows us to steer reverse trajectories toward the rare minority class without retraining the model. By interpolating between conditional and unconditional score estimates, we amplify minority fidelity while avoiding sample replication."*
  - Click **"Generate Synthetic Records"**.
  - Highlight the generated preview table and click **"Download Synthetic CSV"** to demonstrate instant pipeline export.

### Segment 6: Statistical Fidelity & Downstream Utility (2.5 Minutes)
- **Presenter Action**: Navigate to **Fidelity Analysis**, then **Utility Evaluation**.
- **Narrated Workflow (Fidelity)**:
  - Point to the KPI cards: Mean KS = `0.0545`, Pearson Frobenius Distance = `0.1117`.
  - Highlight the CDF overlays showing continuous variable fidelity.
  - *"Our Pearson correlation Frobenius norm is 0.1117, which represents a 29.8% reduction in correlation error compared to flat MLP diffusion baselines."*
- **Narrated Workflow (Utility)**:
  - Show the TSTR (Train on Synthetic, Test on Real) benchmark table across Random Forest, HistGradientBoosting, and Logistic Regression.
  - Point out the Macro-F1 Parity ratio gauge (`0.5906`):
    - *"In downstream ML utility, models trained purely on synthetic data achieve competitive performance on unseen real test sets."*

### Segment 7: Empirical Privacy Audit & Benchmark Center (2.5 Minutes)
- **Presenter Action**: Navigate to **Privacy Audit**, then **Benchmark Center**.
- **Narrated Workflow (Privacy)**:
  - Highlight the **`LOW_RISK`** rating banner and the prominent Differential Privacy disclaimer.
  - Show the Distance to Closest Record (DCR) histogram:
    - *"Our DCR ratio is 1.0155. This proves that synthetic records are, on average, slightly further from training members than genuine unseen test data is from training data."*
  - Point out **0 Exact Matches**:
    - *"Zero training records were duplicated. Furthermore, our distance-based shadow Membership Inference Attack achieves an ROC-AUC of 0.4907, which is completely ineffective—virtually identical to a random coin flip."*
- **Narrated Workflow (Benchmark Center)**:
  - Show the 4-way comparison table (Gaussian Copula, TVAE, TabDDPM MLP, TabSynth Hybrid).
  - Highlight the multi-dimensional radar chart.
  - *"To summarize: TabSynth AI offers state-of-the-art correlation fidelity, robust empirical privacy protection, and powerful minority-class steering in a production-ready, reproducible package."*

---

## 3. Fallback & Troubleshooting Procedures

| Scenario | Symptom | Immediate Remediation |
| :--- | :--- | :--- |
| **Port 8000 Busy** | `Address already in use` | `lsof -ti :8000 | xargs kill -9` then restart `make run-api` |
| **Port 8501 Busy** | `Port 8501 is already in use` | `lsof -ti :8501 | xargs kill -9` then restart `make run-app` |
| **FastAPI Backend Offline** | Engine status shows `Offline` | Verify API process in Terminal 1; check `curl http://localhost:8000/health` |
| **Generation Takes Too Long** | Live sampling latency $> 10\text{s}$ | Lower sampling steps to `20` or use pre-generated synthetic datasets in `data/synthetic/` |
| **Database Lock Error** | `database is locked` | SQLite WAL mode automatically recovers; execute `python scripts/restore_db.py` if needed |
