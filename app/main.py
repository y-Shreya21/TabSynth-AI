"""TabSynth Analytics: Interactive Research Dashboard for Tabular Generative Modeling."""

import os
import glob
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabsynth.data.schema import infer_table_schema
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.training.checkpoint import CheckpointManager
from tabsynth.diffusion.gaussian import GaussianDiffusion
from tabsynth.diffusion.multinomial import MultinomialDiffusion
from tabsynth.models.hybrid import TabSynthHybridModel
from tabsynth.sampling.sampler import TabSynthSampler
from tabsynth.training.trainer import select_device
from tabsynth.evaluation.fidelity import evaluate_statistical_fidelity
from tabsynth.evaluation.utility import evaluate_downstream_utility
from tabsynth.evaluation.privacy import evaluate_empirical_privacy

st.set_page_config(
    page_title="TabSynth Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 TabSynth Analytics")
st.caption("Hybrid Transformer-MLP Diffusion Framework for Privacy-Aware & Class-Conditional Tabular Synthesis")

# Sidebar Navigation
sidebar_nav = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Dataset Manager & Ingestion",
        "Schema Explorer",
        "Conditional Generator",
        "Real vs Synthetic Comparison",
        "Statistical Fidelity",
        "Downstream Utility (TSTR)",
        "Empirical Privacy Audit",
        "Experiment History & Export",
    ],
)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# 1. Dashboard Overview
# -----------------------------------------------------------------------------
if sidebar_nav == "Dashboard Overview":
    st.markdown("### Executive System Summary")
    col1, col2, col3, col4 = st.columns(4)

    device = select_device("auto")
    col1.metric("Compute Device", str(device).upper())
    col2.metric("Architecture", "Transformer-MLP")
    col3.metric("Continuous Engine", "Gaussian DDPM")
    col4.metric("Categorical Engine", "Multinomial D3PM")

    st.markdown("""
    #### 🔬 Three Research Objectives
    1. **High-Fidelity Generation:** Replicates continuous marginal distributions, non-linear cross-column dependencies, and categorical frequencies.
    2. **Class-Conditional Steerability:** Uses Classifier-Free Guidance (CFG) and balanced condition sampling to synthesize minority records on demand.
    3. **Privacy-Aware Modeling:** Suppresses training-set memorization, verified through empirical Distance-to-Closest-Record (DCR) and Membership Inference Attack (MIA) diagnostics.
    """)

    st.info("Select a workflow from the left sidebar to inspect schemas, synthesize data, or run scientific benchmarks.")


# -----------------------------------------------------------------------------
# 2. Dataset Manager & Ingestion
# -----------------------------------------------------------------------------
elif sidebar_nav == "Dataset Manager & Ingestion":
    st.markdown("### Dataset Manager")
    st.write("Inspect existing datasets or upload a new heterogeneous tabular CSV file.")

    available_csvs = sorted(glob.glob("data/raw/*.csv"))
    selected_file = st.selectbox("Select Raw Dataset", available_csvs if available_csvs else ["data/raw/adult.csv"])

    if selected_file and os.path.exists(selected_file):
        df = load_csv(selected_file)
        st.success(f"Loaded `{selected_file}`: **{len(df):,}** rows × **{len(df.columns)}** columns.")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Upload Custom Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        save_path = os.path.join("data/raw", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded to `{save_path}`. Reload or select from above.")


# -----------------------------------------------------------------------------
# 3. Schema Explorer
# -----------------------------------------------------------------------------
elif sidebar_nav == "Schema Explorer":
    st.markdown("### Schema Explorer & Feature Typing")
    available_csvs = sorted(glob.glob("data/raw/*.csv"))
    selected_file = st.selectbox("Select Dataset to Profile", available_csvs if available_csvs else ["data/raw/adult.csv"])

    if selected_file and os.path.exists(selected_file):
        df = load_csv(selected_file)
        target_col = st.selectbox("Select Target Variable", [c for c in df.columns if df[c].nunique() < 10], index=len([c for c in df.columns if df[c].nunique() < 10]) - 1)

        schema = infer_table_schema(df, target_column=target_col)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Continuous Features ({len(schema.continuous_columns)}):**")
            st.write(schema.continuous_columns)
        with col2:
            st.markdown(f"**Categorical Features ({len(schema.categorical_columns)}):**")
            cat_card_dict = {col: card for col, card in zip(schema.categorical_columns, schema.cardinalities)}
            st.json(cat_card_dict)

        if schema.class_distribution:
            st.markdown("#### Target Class Distribution")
            fig = px.bar(
                x=list(schema.class_distribution.keys()),
                y=list(schema.class_distribution.values()),
                labels={"x": "Class", "y": "Empirical Frequency"},
                title=f"Target Distribution for '{target_col}'",
            )
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 4. Conditional Generator
# -----------------------------------------------------------------------------
elif sidebar_nav == "Conditional Generator":
    st.markdown("### Class-Conditional Synthetic Data Generator")

    checkpoints = sorted(glob.glob("checkpoints/*.pt"))
    if not checkpoints:
        st.warning("No trained checkpoints found in `checkpoints/`. Train a model first via `scripts/train.py`.")
    else:
        selected_ckpt = st.selectbox("Select Checkpoint", checkpoints)
        col1, col2 = st.columns(2)
        with col1:
            num_samples = st.slider("Samples to Generate", min_value=100, max_value=5000, value=500, step=100)
            condition_mode = st.selectbox("Conditioning Strategy", ["balanced", "natural", "class_specific", "unconditional"])
        with col2:
            cfg_scale = st.slider("Classifier-Free Guidance Scale (w)", min_value=0.0, max_value=3.0, value=1.25, step=0.25)
            steps = st.slider("Reverse Diffusion Steps", min_value=20, max_value=200, value=50, step=10)

        output_path = st.text_input("Output CSV Path", value="data/synthetic/dashboard_synthetic.csv")

        if st.button("🚀 Synthesize Tabular Records", type="primary"):
            with st.spinner("Executing class-conditional reverse diffusion..."):
                device = select_device("auto")
                ckpt = CheckpointManager.load(checkpoint_path=selected_ckpt, device=device)
                preproc = ckpt["preprocessor"]
                sch = preproc.schema

                num_classes = len(sch.class_distribution) if sch.class_distribution else 2
                model = TabSynthHybridModel(
                    num_continuous=len(sch.continuous_columns),
                    cardinalities=sch.cardinalities,
                    num_classes=num_classes,
                    d_model=64,
                )
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device)

                g_diff = GaussianDiffusion(num_timesteps=steps, device=device)
                m_diff = MultinomialDiffusion(cardinalities=sch.cardinalities, num_timesteps=steps, device=device)

                sampler = TabSynthSampler(
                    model=model,
                    gaussian_diffusion=g_diff,
                    multinomial_diffusion=m_diff,
                    preprocessor=preproc,
                    device=device,
                )

                synth_df = sampler.sample(
                    num_samples=num_samples,
                    batch_size=256,
                    condition_strategy=condition_mode,
                    cfg_scale=cfg_scale,
                )

                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                synth_df.to_csv(output_path, index=False)
                st.success(f"Generated {len(synth_df)} records! Saved to `{output_path}`.")
                st.dataframe(synth_df.head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# 5. Real vs Synthetic Comparison
# -----------------------------------------------------------------------------
elif sidebar_nav == "Real vs Synthetic Comparison":
    st.markdown("### Real vs. Synthetic Feature Distributions")
    real_files = sorted(glob.glob("data/raw/*.csv"))
    synth_files = sorted(glob.glob("data/synthetic/*.csv"))

    if not synth_files:
        st.info("No synthetic CSVs found in `data/synthetic/`. Generate samples first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            real_file = st.selectbox("Real Dataset", real_files)
        with col2:
            synth_file = st.selectbox("Synthetic Dataset", synth_files)

        if real_file and synth_file:
            df_r = load_csv(real_file)
            df_s = load_csv(synth_file)

            common_cols = [c for c in df_r.columns if c in df_s.columns]
            feat = st.selectbox("Select Feature to Compare", common_cols)

            if pd.api.types.is_numeric_dtype(df_r[feat]):
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df_r[feat], name="Real Data", opacity=0.6, histnorm="probability density"))
                fig.add_trace(go.Histogram(x=df_s[feat], name="Synthetic Data", opacity=0.6, histnorm="probability density"))
                fig.update_layout(barmode="overlay", title=f"Continuous Density: {feat}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                p_r = df_r[feat].value_counts(normalize=True).reset_index()
                p_r.columns = ["category", "real_freq"]
                p_s = df_s[feat].value_counts(normalize=True).reset_index()
                p_s.columns = ["category", "synth_freq"]

                merged = pd.merge(p_r, p_s, on="category", how="outer").fillna(0.0)
                fig = px.bar(
                    merged,
                    x="category",
                    y=["real_freq", "synth_freq"],
                    barmode="group",
                    title=f"Categorical Frequency: {feat}",
                )
                st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 6. Statistical Fidelity Dashboard
# -----------------------------------------------------------------------------
elif sidebar_nav == "Statistical Fidelity":
    st.markdown("### Statistical Fidelity Benchmark")
    real_files = sorted(glob.glob("data/raw/*.csv"))
    synth_files = sorted(glob.glob("data/synthetic/*.csv"))

    if not synth_files:
        st.info("Generate synthetic data first.")
    else:
        real_f = st.selectbox("Real Dataset", real_files, key="fid_real")
        synth_f = st.selectbox("Synthetic Dataset", synth_files, key="fid_synth")

        if st.button("Run Fidelity Evaluation"):
            df_r = load_csv(real_f)
            df_s = load_csv(synth_f)
            sch = infer_table_schema(df_r)

            with st.spinner("Computing KS tests, TVD, and C2ST indistinguishability..."):
                res = evaluate_statistical_fidelity(
                    real_df=df_r,
                    synth_df=df_s,
                    continuous_cols=sch.continuous_columns,
                    categorical_cols=sch.categorical_columns,
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean KS Statistic", f"{res['mean_kolmogorov_smirnov']:.4f}")
            m2.metric("Mean TVD", f"{res['mean_total_variation_distance']:.4f}")
            m3.metric("Correlation Frobenius Dist", f"{res['correlation_frobenius_distance']:.4f}")
            m4.metric("C2ST Accuracy", f"{res['c2st_accuracy']:.4f}")

            st.json(res["per_column_ks"])


# -----------------------------------------------------------------------------
# 7. Downstream Utility (TSTR)
# -----------------------------------------------------------------------------
elif sidebar_nav == "Downstream Utility (TSTR)":
    st.markdown("### Train on Synthetic, Test on Real (TSTR) Benchmark")
    st.write("Trains downstream ML models on synthetic data and tests strictly on the real dataset.")

    real_files = sorted(glob.glob("data/raw/*.csv"))
    synth_files = sorted(glob.glob("data/synthetic/*.csv"))

    if synth_files and real_files:
        real_f = st.selectbox("Real Benchmark Dataset", real_files, key="util_real")
        synth_f = st.selectbox("Synthetic Dataset", synth_files, key="util_synth")
        target_c = st.text_input("Target Column", value="income")

        if st.button("Run TSTR vs TRTR Benchmark"):
            df_r = load_csv(real_f)
            df_s = load_csv(synth_f)

            with st.spinner("Training Logistic Regression and Random Forest classifiers..."):
                res = evaluate_downstream_utility(
                    df_train_real=df_r,
                    df_train_synth=df_s,
                    df_test_real=df_r,
                    target_column=target_c,
                )

            for clf_name, data in res.items():
                st.subheader(f"Classifier: {clf_name.replace('_', ' ').title()}")
                col1, col2, col3 = st.columns(3)
                col1.metric("TRTR Macro F1", f"{data['trtr']['macro_f1']:.4f}")
                col2.metric("TSTR Macro F1", f"{data['tstr']['macro_f1']:.4f}")
                col3.metric("F1 Parity Ratio", f"{data['relative_utility_parity']:.4f}", delta=f"{data['minority_recall_delta']:+.4f} Min Recall")


# -----------------------------------------------------------------------------
# 8. Empirical Privacy Audit
# -----------------------------------------------------------------------------
elif sidebar_nav == "Empirical Privacy Audit":
    st.markdown("### Empirical Privacy & Memorization Audit")
    st.write("Audits Distance-to-Closest-Record (DCR) ratios, NNDR, and Membership Inference Attack risk.")

    real_files = sorted(glob.glob("data/raw/*.csv"))
    synth_files = sorted(glob.glob("data/synthetic/*.csv"))

    if synth_files and real_files:
        real_f = st.selectbox("Real Dataset", real_files, key="priv_real")
        synth_f = st.selectbox("Synthetic Dataset", synth_files, key="priv_synth")

        if st.button("Run Privacy Audit"):
            df_r = load_csv(real_f)
            df_s = load_csv(synth_f)
            sch = infer_table_schema(df_r)

            with st.spinner("Calculating Gower distances and shadow MIA..."):
                res = evaluate_empirical_privacy(
                    df_train=df_r,
                    df_test=df_r,
                    df_synth=df_s,
                    continuous_cols=sch.continuous_columns,
                    categorical_cols=sch.categorical_columns,
                    subsample_size=500,
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("DCR Ratio (Train/Test)", f"{res['dcr_ratio']:.4f}")
            c2.metric("Exact Training Matches", f"{res['exact_record_matches']}")
            c3.metric("MIA AUROC", f"{res['membership_inference_auroc']:.4f}", help="0.50 represents ideal random guessing.")

            st.success(f"Assessment: **{res['privacy_risk_assessment']}**")
            st.caption(res["disclaimer"])


# -----------------------------------------------------------------------------
# 9. Experiment History & Export
# -----------------------------------------------------------------------------
elif sidebar_nav == "Experiment History & Export":
    st.markdown("### Experiment History & CSV Export")
    exp_files = sorted(glob.glob("experiments/results/*.json"))

    if not exp_files:
        st.info("No serialized experiment logs found in `experiments/results/`.")
    else:
        selected_exp = st.selectbox("Select Experiment Report", exp_files)
        with open(selected_exp, "r") as f:
            data = json.load(f)
        st.json(data)
