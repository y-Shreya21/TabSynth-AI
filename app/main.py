"""TabSynth Analytics: Production-Grade Scientific Dashboard for Tabular Generative Modeling."""

import os
import glob
import json
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api.config import settings
from api.db.repositories import DatasetRepository, ModelRepository, JobRepository, ExperimentRepository
from api.services.dataset_service import DatasetService
from api.services.model_service import ModelService
from api.services.generation_service import GenerationService
from api.services.evaluation_service import EvaluationService
from api.services.job_service import job_manager
from app.api_client import client as api_client
from tabsynth.training.trainer import select_device

# Page Configuration
st.set_page_config(
    page_title="TabSynth Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SaaS Typography and Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .stButton>button {
        border-radius: 6px;
    }
    .badge-status {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🧬 TabSynth Analytics")
st.caption("Production Application: Hybrid Transformer-MLP Diffusion Framework for Privacy-Aware & Class-Conditional Tabular Synthesis")

# Sidebar Status & Navigation
backend_online = api_client.is_online()
if backend_online:
    st.sidebar.success("🟢 API Server: Online (FastAPI)")
else:
    st.sidebar.warning("🟡 API Server: Offline (Direct Engine)")

page = st.sidebar.radio(
    "Navigation",
    [
        "Page 1: Overview Dashboard",
        "Page 2: Dataset Manager",
        "Page 3: Model Training",
        "Page 4: Synthetic Data Generator",
        "Page 5: Fidelity Dashboard",
        "Page 6: Utility Dashboard",
        "Page 7: Privacy Audit",
        "Page 8: Experiment Explorer",
    ],
)


# Helper function to load datasets
@st.cache_data(ttl=60)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# =============================================================================
# Page 1: Overview Dashboard
# =============================================================================
if page == "Page 1: Overview Dashboard":
    st.subheader("System Overview & Scientific Dashboard")

    datasets = DatasetRepository.list_all()
    models = ModelRepository.list_all()
    jobs = JobRepository.list_all(limit=50)
    experiments = ExperimentRepository.list_all()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered Datasets", len(datasets))
    c2.metric("Trained Models", len(models))
    c3.metric("Total Jobs Executed", len(jobs))
    c4.metric("Saved Experiments", len(experiments))

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("#### Hardware & Diffusion Infrastructure")
        device = select_device("auto")
        inf_col1, inf_col2, inf_col3 = st.columns(3)
        inf_col1.metric("Compute Accelerator", str(device).upper())
        inf_col2.metric("Continuous Engine", "Gaussian DDPM")
        inf_col3.metric("Categorical Engine", "Multinomial D3PM")

        st.markdown("""
        **Core Research Capabilities:**
        - **Mixed-Type Diffusion:** Dual continuous and categorical forward noise schedules with analytical reverse transitions.
        - **Transformer-MLP Backbone:** Inter-feature self-attention mechanism capturing complex multi-column dependencies.
        - **Class-Conditional Guidance:** Controllable synthetic distribution steering via Classifier-Free Guidance (CFG).
        - **Zero-Leakage Guarantee:** Training partition isolated with quantile transforms and strict validation splits.
        """)

    with col_right:
        st.markdown("#### Recent Job History")
        if jobs:
            job_rows = [
                {
                    "ID": j.id,
                    "Type": j.job_type.title(),
                    "Status": j.status.upper(),
                    "Progress": f"{int(j.progress * 100)}%",
                    "Updated": j.updated_at[:19] if j.updated_at else "-",
                }
                for j in jobs[:6]
            ]
            st.dataframe(pd.DataFrame(job_rows), use_container_width=True)
        else:
            st.info("No background jobs recorded yet. Ingest data and start training.")


# =============================================================================
# Page 2: Dataset Manager
# =============================================================================
elif page == "Page 2: Dataset Manager":
    st.subheader("Dataset Ingestion & Tabular Schema Profiler")

    tab_inspect, tab_upload = st.tabs(["Inspect Registered Datasets", "Upload New CSV"])

    with tab_upload:
        st.markdown("#### Upload Heterogeneous Tabular Dataset")
        up_file = st.file_uploader("Upload CSV file (Max 50MB)", type=["csv"])
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            dataset_name = st.text_input("Dataset Alias", value="")
        with col_up2:
            target_col = st.text_input("Target Column for Conditioning (Optional)", value="")

        if up_file and st.button("Ingest & Profile Dataset", type="primary"):
            with st.spinner("Validating and profiling schema..."):
                file_bytes = up_file.getvalue()
                try:
                    if backend_online:
                        res = api_client.upload_dataset(
                            file_bytes=file_bytes,
                            filename=up_file.name,
                            name=dataset_name or up_file.name.replace(".csv", ""),
                            target_column=target_col or None,
                        )
                        st.success(f"Uploaded `{res['name']}`! Ingested {res['num_rows']} rows × {res['num_columns']} columns.")
                    else:
                        rec = DatasetService.upload_dataset(
                            file_bytes=file_bytes,
                            filename=up_file.name,
                            name=dataset_name or up_file.name.replace(".csv", ""),
                            target_column=target_col or None,
                        )
                        st.success(f"Saved `{rec.name}` locally! Ingested {rec.num_rows} rows × {rec.num_columns} columns.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    with tab_inspect:
        registered = DatasetRepository.list_all()
        # Fallback to local files if DB is empty
        file_options = {}
        for r in registered:
            file_options[f"{r.name} ({r.id})"] = (r.file_path, r.id, r.target_column)

        for p in sorted(glob.glob("data/raw/*.csv")):
            bname = os.path.basename(p)
            if not any(bname in k for k in file_options):
                file_options[f"Local: {bname}"] = (p, None, None)

        if not file_options:
            st.info("No datasets available. Please upload a dataset in the Upload tab.")
        else:
            selected_label = st.selectbox("Select Dataset to Profile", list(file_options.keys()))
            selected_path, ds_id, ds_target = file_options[selected_label]

            if os.path.exists(selected_path):
                df = load_csv(selected_path)
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Records", f"{len(df):,}")
                m2.metric("Total Columns", len(df.columns))
                m3.metric("Missing Values", int(df.isnull().sum().sum()))

                st.markdown("#### Sample Records (Top 10)")
                st.dataframe(df.head(10), use_container_width=True)

                st.markdown("---")
                col_c, col_d = st.columns(2)
                with col_c:
                    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    st.markdown(f"**Continuous Features ({len(num_cols)}):**")
                    st.write(num_cols)

                with col_d:
                    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
                    st.markdown(f"**Categorical Features ({len(cat_cols)}):**")
                    st.write(cat_cols)

                # Class distribution
                candidate_targets = [c for c in df.columns if df[c].nunique() <= 10]
                if candidate_targets:
                    st.markdown("#### Target Class Distribution")
                    chosen_target = st.selectbox("Target Variable", candidate_targets, index=candidate_targets.index(ds_target) if ds_target in candidate_targets else 0)
                    dist = df[chosen_target].value_counts(normalize=True).reset_index()
                    dist.columns = ["class", "frequency"]
                    fig = px.bar(dist, x="class", y="frequency", title=f"Class Balance for '{chosen_target}'")
                    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Page 3: Model Training
# =============================================================================
elif page == "Page 3: Model Training":
    st.subheader("Model Configuration & Training Pipeline")

    datasets = DatasetRepository.list_all()
    ds_choices = {f"{d.name} ({d.id})": d for d in datasets}
    if not ds_choices:
        for p in sorted(glob.glob("data/raw/*.csv")):
            ds_choices[f"Local: {os.path.basename(p)}"] = p

    if not ds_choices:
        st.warning("No datasets available. Please upload a dataset in Dataset Manager.")
    else:
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            sel_ds_key = st.selectbox("Training Dataset", list(ds_choices.keys()))
            chosen_obj = ds_choices[sel_ds_key]
            if isinstance(chosen_obj, str):
                dpath = chosen_obj
                ds_id = None
                df_temp = load_csv(dpath)
                candidate_targets = [c for c in df_temp.columns if df_temp[c].nunique() <= 10]
            else:
                dpath = chosen_obj.file_path
                ds_id = chosen_obj.id
                candidate_targets = [chosen_obj.target_column] if chosen_obj.target_column else ["income"]

            target_col = st.selectbox("Target Column for Guidance", candidate_targets)
            model_type = st.radio("Denoising Backbone Architecture", ["hybrid_transformer_mlp", "mlp_baseline"], format_func=lambda x: "Hybrid Transformer-MLP (Recommended)" if x == "hybrid_transformer_mlp" else "TabDDPM Residual MLP Baseline")
            model_name = st.text_input("Model Alias", value=f"{model_type}_run")

        with col_cfg2:
            epochs = st.slider("Training Epochs", min_value=1, max_value=100, value=20, step=1)
            steps = st.slider("Diffusion Timesteps (T)", min_value=10, max_value=200, value=50, step=10)
            batch_size = st.select_slider("Batch Size", options=[64, 128, 256, 512], value=256)
            lr = st.select_slider("Learning Rate", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
            dev_choice = st.selectbox("Compute Device", ["auto", "mps", "cpu", "cuda"])

        if st.button("🚀 Launch Asynchronous Training Job", type="primary"):
            with st.spinner("Dispatching background training worker..."):
                train_payload = {
                    "data_path": dpath,
                    "dataset_id": ds_id,
                    "model_name": model_name,
                    "target_column": target_col,
                    "model_type": model_type,
                    "hyperparameters": {
                        "epochs": epochs,
                        "diffusion_steps": steps,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                    },
                    "device": dev_choice,
                }
                try:
                    if backend_online:
                        res = api_client.start_training(train_payload)
                        st.success(f"Job launched successfully! Job ID: `{res['job_id']}`")
                        st.info("Monitor live training progress in the job table below.")
                    else:
                        from api.schemas.models import TrainRequest, TrainHyperparameters
                        req = TrainRequest(
                            data_path=dpath,
                            dataset_id=ds_id,
                            model_name=model_name,
                            target_column=target_col,
                            model_type=model_type,
                            hyperparameters=TrainHyperparameters(
                                epochs=epochs,
                                diffusion_steps=steps,
                                batch_size=batch_size,
                                learning_rate=lr,
                            ),
                            device=dev_choice,
                        )
                        job_id = ModelService.start_training_job(req)
                        st.success(f"Job launched locally! Job ID: `{job_id}`")
                except Exception as exc:
                    st.error(f"Failed to start training: {exc}")

    st.markdown("---")
    st.markdown("#### Active & Recent Training Jobs")
    training_jobs = [j for j in JobRepository.list_all(limit=20) if j.job_type == "training"]
    if training_jobs:
        for j in training_jobs[:5]:
            with st.expander(f"Job {j.id} — Status: {j.status.upper()} (Progress: {int(j.progress * 100)}%)", expanded=(j.status == "running")):
                st.progress(j.progress)
                st.write(f"Step: {j.current_step} / {j.total_steps}")
                if j.result:
                    st.json(j.result)
                if j.error:
                    st.error(f"Error: {j.error}")
                if j.status in ["queued", "running"]:
                    if st.button(f"Cancel Job {j.id}", key=f"btn_cancel_{j.id}"):
                        job_manager.cancel_job(j.id)
                        st.warning("Cancellation requested.")


# =============================================================================
# Page 4: Synthetic Data Generator
# =============================================================================
elif page == "Page 4: Synthetic Data Generator":
    st.subheader("Class-Conditional Synthetic Tabular Generator")

    models = ModelRepository.list_all()
    model_choices = {f"{m.name} ({m.model_type}) - {m.id}": m.checkpoint_path for m in models}

    for ckpt_path in sorted(glob.glob("checkpoints/*/*.pt") + glob.glob("checkpoints/*.pt")):
        bname = os.path.basename(ckpt_path)
        if ckpt_path not in model_choices.values():
            model_choices[f"Checkpoint: {bname} ({os.path.dirname(ckpt_path)})"] = ckpt_path

    if not model_choices:
        st.warning("No model checkpoints available. Train a model in Model Training first.")
    else:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            sel_model_label = st.selectbox("Select Model Checkpoint", list(model_choices.keys()))
            selected_ckpt = model_choices[sel_model_label]
            num_samples = st.slider("Number of Records to Generate", min_value=100, max_value=5000, value=500, step=100)
            condition_mode = st.selectbox("Conditioning Strategy", ["balanced", "natural", "class_specific", "unconditional"])

        with col_g2:
            specific_cls = None
            if condition_mode == "class_specific":
                specific_cls = st.number_input("Target Class Index (e.g. 1 for minority)", min_value=0, max_value=10, value=1)
            cfg_scale = st.slider("Classifier-Free Guidance Scale (w)", min_value=0.0, max_value=3.0, value=1.25, step=0.25)
            steps = st.slider("Reverse Sampling Steps", min_value=10, max_value=200, value=50, step=10)
            output_name = st.text_input("Output CSV Filename", value="data/synthetic/generated_samples.csv")

        if st.button("🚀 Generate Synthetic Records", type="primary"):
            with st.spinner("Executing reverse diffusion trajectory with Classifier-Free Guidance..."):
                gen_payload = {
                    "checkpoint_path": selected_ckpt,
                    "num_samples": num_samples,
                    "condition_strategy": condition_mode,
                    "specific_class": specific_cls,
                    "cfg_scale": cfg_scale,
                    "steps": steps,
                    "output_path": output_name,
                }
                try:
                    if backend_online:
                        res = api_client.generate(gen_payload)
                        out_path = res["output_path"]
                        st.success(f"Successfully generated {res['num_samples']} records! Saved to `{out_path}`.")
                    else:
                        from api.schemas.generation import GenerateRequest
                        req = GenerateRequest(**gen_payload)
                        res = GenerationService.generate(req)
                        out_path = res.output_path
                        st.success(f"Generated {res.num_samples} records! Saved to `{out_path}`.")

                    if os.path.exists(out_path):
                        synth_df = pd.read_csv(out_path)
                        st.dataframe(synth_df.head(10), use_container_width=True)
                        csv_data = synth_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Synthetic CSV",
                            data=csv_data,
                            file_name=os.path.basename(out_path),
                            mime="text/csv",
                        )
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")


# =============================================================================
# Page 5: Fidelity Dashboard
# =============================================================================
elif page == "Page 5: Fidelity Dashboard":
    st.subheader("Statistical Fidelity Benchmark & Marginal Analysis")

    real_csvs = sorted(glob.glob("data/raw/*.csv"))
    synth_csvs = sorted(glob.glob("data/synthetic/*.csv"))

    if not real_csvs or not synth_csvs:
        st.info("Requires at least one real dataset and one synthetic dataset.")
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sel_real = st.selectbox("Ground Truth Real Dataset", real_csvs)
        with col_f2:
            sel_synth = st.selectbox("Generated Synthetic Dataset", synth_csvs)

        if st.button("Compute Fidelity Metrics", type="primary"):
            with st.spinner("Calculating Kolmogorov-Smirnov statistics, TVD, and C2ST..."):
                try:
                    if backend_online:
                        results = api_client.evaluate_fidelity(sel_real, sel_synth)
                    else:
                        results = EvaluationService.evaluate_fidelity(sel_real, sel_synth)

                    st.markdown("#### Fidelity Metric Scorecard")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Mean KS Statistic ↓", f"{results['mean_kolmogorov_smirnov']:.4f}")
                    m2.metric("Mean Total Variation ↓", f"{results['mean_total_variation_distance']:.4f}")
                    m3.metric("Correlation Frobenius Dist ↓", f"{results['correlation_frobenius_distance']:.4f}")
                    m4.metric("C2ST Accuracy (~0.50)", f"{results['c2st_accuracy']:.4f}")

                    st.markdown("---")
                    st.markdown("#### 1D Marginal Distribution Overlays")
                    df_r = load_csv(sel_real)
                    df_s = load_csv(sel_synth)
                    common_cols = [c for c in df_r.columns if c in df_s.columns]
                    col_to_plot = st.selectbox("Feature to Inspect", common_cols)

                    if pd.api.types.is_numeric_dtype(df_r[col_to_plot]):
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=df_r[col_to_plot], name="Real Data", opacity=0.6, histnorm="probability density"))
                        fig.add_trace(go.Histogram(x=df_s[col_to_plot], name="Synthetic Data", opacity=0.6, histnorm="probability density"))
                        fig.update_layout(barmode="overlay", title=f"Continuous Density: {col_to_plot}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        pr = df_r[col_to_plot].value_counts(normalize=True).reset_index()
                        pr.columns = ["cat", "real"]
                        ps = df_s[col_to_plot].value_counts(normalize=True).reset_index()
                        ps.columns = ["cat", "synthetic"]
                        merged = pd.merge(pr, ps, on="cat", how="outer").fillna(0.0)
                        fig = px.bar(merged, x="cat", y=["real", "synthetic"], barmode="group", title=f"Categorical Frequency: {col_to_plot}")
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as exc:
                    st.error(f"Fidelity evaluation failed: {exc}")


# =============================================================================
# Page 6: Utility Dashboard
# =============================================================================
elif page == "Page 6: Utility Dashboard":
    st.subheader("Downstream Machine Learning Utility (TSTR vs TRTR)")
    st.markdown("Evaluates whether machine learning models trained strictly on synthetic data generalize to real test partitions.")

    real_csvs = sorted(glob.glob("data/raw/*.csv"))
    synth_csvs = sorted(glob.glob("data/synthetic/*.csv"))

    if real_csvs and synth_csvs:
        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1:
            u_real = st.selectbox("Real Benchmark Dataset", real_csvs, key="u_real")
        with col_u2:
            u_synth = st.selectbox("Synthetic Training Dataset", synth_csvs, key="u_synth")
        with col_u3:
            df_temp = load_csv(u_real)
            cand_targets = [c for c in df_temp.columns if df_temp[c].nunique() <= 10]
            u_target = st.selectbox("Target Classification Variable", cand_targets)

        if st.button("Run TSTR Benchmark Battery", type="primary"):
            with st.spinner("Training downstream LogisticRegression and RandomForest classifiers..."):
                try:
                    if backend_online:
                        results = api_client.evaluate_utility(u_real, u_synth, u_real, u_target)
                    else:
                        results = EvaluationService.evaluate_utility(u_real, u_synth, u_real, u_target)

                    for clf, metrics in results.items():
                        st.markdown(f"### Classifier: **{clf.replace('_', ' ').title()}**")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("TRTR Macro-F1", f"{metrics['trtr']['macro_f1']:.4f}")
                        c2.metric("TSTR Macro-F1", f"{metrics['tstr']['macro_f1']:.4f}")
                        c3.metric("F1 Parity Ratio", f"{metrics['relative_utility_parity']:.4f}")
                        c4.metric("Minority Recall Delta", f"{metrics['minority_recall_delta']:+.4f}")
                        st.markdown("---")
                except Exception as exc:
                    st.error(f"Utility evaluation failed: {exc}")


# =============================================================================
# Page 7: Privacy Audit
# =============================================================================
elif page == "Page 7: Privacy Audit":
    st.subheader("Empirical Privacy & Training Set Memorization Audit")

    real_csvs = sorted(glob.glob("data/raw/*.csv"))
    synth_csvs = sorted(glob.glob("data/synthetic/*.csv"))

    if real_csvs and synth_csvs:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_real = st.selectbox("Real Benchmark Dataset", real_csvs, key="p_real")
        with col_p2:
            p_synth = st.selectbox("Synthetic Evaluation Dataset", synth_csvs, key="p_synth")

        subsample = st.slider("Privacy Subsample Size (Speed vs Precision)", min_value=100, max_value=1000, value=300, step=100)

        if st.button("Execute Privacy Audit", type="primary"):
            with st.spinner("Computing Distance-to-Closest-Record and shadow MIA..."):
                try:
                    if backend_online:
                        res = api_client.evaluate_privacy(p_real, p_real, p_synth, subsample=subsample)
                    else:
                        res = EvaluationService.evaluate_privacy(p_real, p_real, p_synth, subsample_size=subsample)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("DCR Ratio (Train/Test)", f"{res['dcr_ratio']:.4f}", help="Values near 1.0 indicate identical distance distributions.")
                    c2.metric("Exact Training Matches", f"{res['exact_record_matches']}", help="0 exact matches means zero verbatim record memorization.")
                    c3.metric("MIA AUROC", f"{res['membership_inference_auroc']:.4f}", help="0.50 denotes ideal random guessing.")

                    st.success(f"Empirical Risk Assessment: **{res['privacy_risk_assessment']}**")

                    # Prominent Privacy Disclaimer Alert
                    st.warning("""
                    **⚠️ Research Privacy Disclaimer:**
                    Empirical metrics (DCR, exact matches, shadow MIA) measure privacy under observed test distributions.
                    They do **NOT** provide mathematically provable Differential Privacy ($\\epsilon, \\delta$-DP).
                    For regulated deployments (HIPAA/GDPR), combine with DP-SGD noise mechanisms.
                    """)
                except Exception as exc:
                    st.error(f"Privacy audit failed: {exc}")


# =============================================================================
# Page 8: Experiment Explorer
# =============================================================================
elif page == "Page 8: Experiment Explorer":
    st.subheader("Experiment Registry & Historical Metrics Explorer")

    exps = ExperimentService.list_experiments()

    if not exps:
        st.info("No recorded experiments found. Run evaluations or benchmark runners to populate.")
    else:
        exp_labels = {f"{e.get('name', 'Exp')} ({e['id']})": e['id'] for e in exps}
        sel_exp_label = st.selectbox("Select Experiment", list(exp_labels.keys()))
        sel_id = exp_labels[sel_exp_label]

        details = ExperimentService.get_experiment(sel_id)
        st.markdown(f"#### Experiment: `{details.get('name', sel_id)}`")
        if details.get("description"):
            st.caption(details["description"])

        metrics_data = details.get("metrics", {})
        if metrics_data:
            st.json(metrics_data)
