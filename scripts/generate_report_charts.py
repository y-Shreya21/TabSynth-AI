import os
import json
import pandas as pd
import numpy as np

CHART_DIR = "docs/project-report/assets/charts"
DATASET_DIR = "docs/project-report/assets/dataset_visualizations"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Common styling
BG_COLOR = "#0F172A"       # Dark slate
CARD_BG = "#1E293B"        # Slate card
TEXT_COLOR = "#F8FAFC"     # White slate
MUTED_TEXT = "#94A3B8"     # Muted slate
BORDER_COLOR = "#334155"   # Grid/border
ACCENT_PURPLE = "#8B5CF6"  # Electric violet (TabSynth)
ACCENT_GREEN = "#10B981"   # Emerald
ACCENT_BLUE = "#38BDF8"    # Soft blue
ACCENT_AMBER = "#F59E0B"   # Amber
ACCENT_ROSE = "#F43F5E"    # Rose

def create_bar_chart(title, subtitle, categories, series_dict, output_path, y_label="", y_max=None, is_lower_better=True, width=720, height=420):
    """Generates a professional standalone SVG grouped bar chart."""
    n_cats = len(categories)
    n_series = len(series_dict)
    
    # Calculate bounds
    all_vals = []
    for vals in series_dict.values():
        all_vals.extend(vals)
    max_val = y_max if y_max is not None else max(all_vals) * 1.15
    if max_val == 0: max_val = 1.0

    margin_left = 70
    margin_right = 30
    margin_top = 70
    margin_bottom = 60
    
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    
    group_w = chart_w / n_cats
    bar_w = (group_w * 0.75) / n_series
    bar_spacing = 4
    
    colors = [ACCENT_PURPLE, ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_ROSE]
    
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: {BG_COLOR}; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;">')
    
    # Background Card
    svg.append(f'<rect width="{width}" height="{height}" rx="8" fill="{BG_COLOR}"/>')
    svg.append(f'<rect x="10" y="10" width="{width-20}" height="{height-20}" rx="6" fill="{CARD_BG}" stroke="{BORDER_COLOR}" stroke-width="1"/>')
    
    # Title & Subtitle
    direction_tag = " (Lower is Better ↓)" if is_lower_better else " (Higher is Better ↑)"
    svg.append(f'<text x="{margin_left}" y="35" fill="{TEXT_COLOR}" font-size="15" font-weight="700">{title}{direction_tag}</text>')
    svg.append(f'<text x="{margin_left}" y="52" fill="{MUTED_TEXT}" font-size="11">{subtitle}</text>')
    
    # Legend
    legend_x = width - margin_right
    legend_items = list(series_dict.keys())
    # Render legend right aligned
    cur_lx = legend_x
    for i in reversed(range(len(legend_items))):
        s_name = legend_items[i]
        c = colors[i % len(colors)]
        text_w = len(s_name) * 7 + 25
        cur_lx -= text_w
        svg.append(f'<rect x="{cur_lx}" y="38" width="10" height="10" rx="2" fill="{c}"/>')
        svg.append(f'<text x="{cur_lx + 15}" y="47" fill="{TEXT_COLOR}" font-size="11">{s_name}</text>')
    
    # Horizontal Grid Lines & Y Ticks
    n_ticks = 5
    for i in range(n_ticks + 1):
        tick_val = (max_val / n_ticks) * i
        y_pos = margin_top + chart_h - (i * (chart_h / n_ticks))
        svg.append(f'<line x1="{margin_left}" y1="{y_pos}" x2="{width - margin_right}" y2="{y_pos}" stroke="{BORDER_COLOR}" stroke-dasharray="3,3" stroke-width="1"/>')
        val_str = f"{tick_val:.3f}" if max_val < 2 else f"{tick_val:.1f}"
        svg.append(f'<text x="{margin_left - 10}" y="{y_pos + 4}" fill="{MUTED_TEXT}" font-size="10" text-anchor="end">{val_str}</text>')
    
    # Render Bars
    for c_idx, cat in enumerate(categories):
        group_center = margin_left + (c_idx * group_w) + (group_w / 2)
        start_x = group_center - ((n_series * bar_w) / 2)
        
        # Category label
        svg.append(f'<text x="{group_center}" y="{margin_top + chart_h + 20}" fill="{TEXT_COLOR}" font-size="11" font-weight="600" text-anchor="middle">{cat}</text>')
        
        for s_idx, (s_name, s_vals) in enumerate(series_dict.items()):
            val = s_vals[c_idx]
            bar_h = (val / max_val) * chart_h
            bx = start_x + (s_idx * bar_w) + 1
            by = margin_top + chart_h - bar_h
            c = colors[s_idx % len(colors)]
            
            svg.append(f'<rect x="{bx}" y="{by}" width="{bar_w - 2}" height="{bar_h}" rx="3" fill="{c}"/>')
            # Text above bar
            val_label = f"{val:.4f}" if val < 1 else f"{val:.2f}"
            svg.append(f'<text x="{bx + (bar_w - 2)/2}" y="{by - 4}" fill="{c}" font-size="9" font-weight="600" text-anchor="middle">{val_label}</text>')
            
    # X and Y Axes
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="{MUTED_TEXT}" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{width - margin_right}" y2="{margin_top + chart_h}" stroke="{MUTED_TEXT}" stroke-width="1.5"/>')
    
    svg.append('</svg>')
    
    with open(output_path, "w") as f:
        f.write("\n".join(svg))
    print(f"Generated {output_path}")

# 1. Fidelity Comparison
fidelity_df = pd.read_csv("experiments/results/fidelity_summary.csv")
categories = [m.replace("_", " ") for m in fidelity_df["Model"]]
ks_vals = fidelity_df["Mean_KS"].tolist()
wass_vals = fidelity_df["Mean_Wasserstein"].tolist()
tvd_vals = fidelity_df["Mean_TVD"].tolist()

create_bar_chart(
    title="Statistical Fidelity Metrics Comparison",
    subtitle="Mean KS Statistic, 1D Wasserstein Distance, and Total Variation Distance across Baselines",
    categories=categories,
    series_dict={
        "Mean KS": ks_vals,
        "Wasserstein": wass_vals,
        "Mean TVD": tvd_vals
    },
    output_path=f"{CHART_DIR}/fidelity_metrics_comparison.svg",
    is_lower_better=True,
    width=740, height=400
)

# 2. Frobenius Correlation Error
frob_p = fidelity_df["Pearson_Frobenius_Distance"].tolist()
frob_s = fidelity_df["Spearman_Frobenius_Distance"].tolist()
create_bar_chart(
    title="Inter-Feature Correlation Preservation Error",
    subtitle="Frobenius Norm Distance ||R_real - R_synth||_F for Pearson & Spearman Matrices",
    categories=categories,
    series_dict={
        "Pearson Frobenius": frob_p,
        "Spearman Frobenius": frob_s
    },
    output_path=f"{CHART_DIR}/frobenius_correlation_error.svg",
    is_lower_better=True,
    width=740, height=380
)

# 3. Downstream Utility Comparison (TSTR Macro F1)
utility_df = pd.read_csv("experiments/results/utility_summary.csv")
models = ["Gaussian_Copula", "TVAE", "TabDDPM_MLP", "TabSynth_Hybrid"]
clean_models = ["Gaussian Copula", "TVAE", "TabDDPM MLP", "TabSynth Hybrid"]

rf_f1 = [utility_df[(utility_df["Model"] == m) & (utility_df["Classifier"] == "random_forest")]["TSTR_Macro_F1"].values[0] for m in models]
hgb_f1 = [utility_df[(utility_df["Model"] == m) & (utility_df["Classifier"] == "hist_gradient_boosting")]["TSTR_Macro_F1"].values[0] for m in models]
lr_f1 = [utility_df[(utility_df["Model"] == m) & (utility_df["Classifier"] == "logistic_regression")]["TSTR_Macro_F1"].values[0] for m in models]

create_bar_chart(
    title="Downstream ML Utility: TSTR Macro-F1",
    subtitle="Train on Synthetic, Test on Real (TSTR) Macro-F1 across 3 Standard Classifiers",
    categories=clean_models,
    series_dict={
        "Random Forest": rf_f1,
        "HistGradientBoost": hgb_f1,
        "Logistic Regression": lr_f1
    },
    output_path=f"{CHART_DIR}/utility_macro_f1_comparison.svg",
    is_lower_better=False,
    width=740, height=400
)

# 4. F1 Parity Ratio
parity_rf = [utility_df[(utility_df["Model"] == m) & (utility_df["Classifier"] == "random_forest")]["F1_Parity_Ratio"].values[0] for m in models]
parity_hgb = [utility_df[(utility_df["Model"] == m) & (utility_df["Classifier"] == "hist_gradient_boosting")]["F1_Parity_Ratio"].values[0] for m in models]
parity_lr = [utility_df[(utility_df["Model"] == m) & (utility_df["Classifier"] == "logistic_regression")]["F1_Parity_Ratio"].values[0] for m in models]

create_bar_chart(
    title="Utility Parity Ratio (TSTR / TRTR)",
    subtitle="F1 Parity Relative to Ground-Truth Real Data Baseline (TRTR = 1.0)",
    categories=clean_models,
    series_dict={
        "RF Parity": parity_rf,
        "HGB Parity": parity_hgb,
        "LR Parity": parity_lr
    },
    output_path=f"{CHART_DIR}/utility_f1_parity_ratio.svg",
    is_lower_better=False,
    width=740, height=380
)

# 5. Ablation E5 Minority Recall Recovery
create_bar_chart(
    title="Ablation E5: Class Conditioning Strategy Impact",
    subtitle="Minority Class Recall Recovery: Natural Frequency (78/22) vs Balanced Prior (50/50)",
    categories=["Random Forest Minority Recall"],
    series_dict={
        "Natural Conditioning (78/22)": [0.0000],
        "Balanced Conditioning (50/50)": [0.4855]
    },
    output_path=f"{CHART_DIR}/ablation_e5_minority_recall.svg",
    is_lower_better=False,
    width=550, height=350
)

# 6. Ablation E1 Denoiser Correlation Error
create_bar_chart(
    title="Ablation E1: Denoiser Architecture Comparison",
    subtitle="Pearson Correlation Frobenius Error: TabSynth Hybrid vs Pure MLP Backbone",
    categories=["Correlation Frobenius Error"],
    series_dict={
        "TabDDPM MLP (Pure MLP)": [0.1591],
        "TabSynth Hybrid (Trans+MLP)": [0.1117]
    },
    output_path=f"{CHART_DIR}/ablation_e1_denoiser_error.svg",
    is_lower_better=True,
    width=550, height=350
)

# 7. Privacy DCR Ratio
privacy_df = pd.read_csv("experiments/results/privacy_summary.csv")
dcr_ratio = privacy_df["DCR_Ratio"].tolist()
nndr_mean = privacy_df["Mean_NNDR"].tolist()
p_models = [m.replace("_", " ") for m in privacy_df["Model"]]

create_bar_chart(
    title="Empirical Privacy Indicators: DCR Ratio & NNDR",
    subtitle="Distance to Closest Record (DCR) Ratio (~1.0 indicates generalization without memorization)",
    categories=p_models,
    series_dict={
        "DCR Ratio (Train/Test)": dcr_ratio,
        "Mean NNDR": nndr_mean
    },
    output_path=f"{CHART_DIR}/privacy_dcr_ratio.svg",
    is_lower_better=False,
    width=740, height=380
)

# 8. Model Throughput vs Parameters
bench_df = pd.read_json("experiments/results/benchmark_summary.json")
b_models = [m["Model"].replace("_", " ") for m in bench_df.to_dict(orient="records")]
params_k = [m["Params"] / 1000.0 for m in bench_df.to_dict(orient="records")]

create_bar_chart(
    title="Architectural Complexity: Parameter Count",
    subtitle="Trainable Model Parameters (Thousands) across Evaluated Generators",
    categories=b_models,
    series_dict={
        "Parameters (x1,000)": params_k
    },
    output_path=f"{CHART_DIR}/model_throughput_parameters.svg",
    is_lower_better=True,
    width=720, height=360
)

# 9. Dataset Visualizations
# Target Imbalance
create_bar_chart(
    title="UCI Adult Income: Class Imbalance Distribution",
    subtitle="Target Variable (income) Distribution: 32,561 Total Census Records",
    categories=["<=50K (Majority)", ">50K (Minority)"],
    series_dict={
        "Record Count": [24720, 7841],
        "Percentage (%)": [75.92, 24.08]
    },
    output_path=f"{DATASET_DIR}/adult_income_target_distribution.svg",
    is_lower_better=False,
    width=600, height=350
)

# Feature Type Breakdown
create_bar_chart(
    title="Dataset Dimensionality & Feature Composition",
    subtitle="Continuous vs Categorical Feature Breakdown across Evaluation Datasets",
    categories=["Adult Income (15 cols)", "Bank Marketing (17 cols)", "Credit Default (12 cols)"],
    series_dict={
        "Continuous Features": [6, 7, 9],
        "Categorical Features": [8, 9, 2],
        "Target Attribute": [1, 1, 1]
    },
    output_path=f"{DATASET_DIR}/benchmark_datasets_summary.svg",
    is_lower_better=False,
    width=720, height=380
)

print("All charts successfully generated in SVG format!")
