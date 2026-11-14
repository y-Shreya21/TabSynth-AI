"""Interactive Plotly visualization suite for TabSynth AI evaluations."""

import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_marginal_distributions(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    max_continuous: int = 4,
    max_categorical: int = 4,
) -> go.Figure:
    """Overlays real vs synthetic 1D feature marginals."""
    c_cols = continuous_cols[:max_continuous]
    d_cols = categorical_cols[:max_categorical]
    total_plots = len(c_cols) + len(d_cols)

    rows = (total_plots + 1) // 2
    cols = 2 if total_plots > 1 else 1

    titles = [f"Cont: {col}" for col in c_cols] + [f"Cat: {col}" for col in d_cols]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

    idx = 0
    # 1. Continuous Features (Histograms)
    for col in c_cols:
        r = (idx // 2) + 1
        c = (idx % 2) + 1
        fig.add_trace(
            go.Histogram(
                x=real_df[col].dropna(),
                histnorm="probability density",
                name=f"Real ({col})",
                opacity=0.6,
                marker_color="#1f77b4",
                showlegend=(idx == 0),
            ),
            row=r, col=c,
        )
        fig.add_trace(
            go.Histogram(
                x=synth_df[col].dropna(),
                histnorm="probability density",
                name=f"Synth ({col})",
                opacity=0.6,
                marker_color="#ff7f0e",
                showlegend=(idx == 0),
            ),
            row=r, col=c,
        )
        idx += 1

    # 2. Categorical Features (Bar Plots)
    for col in d_cols:
        r = (idx // 2) + 1
        c = (idx % 2) + 1

        cats = sorted(list(set(real_df[col].dropna().astype(str).unique()) | set(synth_df[col].dropna().astype(str).unique())))
        real_counts = real_df[col].astype(str).value_counts(normalize=True).to_dict()
        synth_counts = synth_df[col].astype(str).value_counts(normalize=True).to_dict()

        real_freqs = [real_counts.get(k, 0.0) for k in cats]
        synth_freqs = [synth_counts.get(k, 0.0) for k in cats]

        fig.add_trace(
            go.Bar(
                x=cats,
                y=real_freqs,
                name="Real Frequencies",
                opacity=0.7,
                marker_color="#1f77b4",
                showlegend=(idx == len(c_cols)),
            ),
            row=r, col=c,
        )
        fig.add_trace(
            go.Bar(
                x=cats,
                y=synth_freqs,
                name="Synth Frequencies",
                opacity=0.7,
                marker_color="#ff7f0e",
                showlegend=(idx == len(c_cols)),
            ),
            row=r, col=c,
        )
        idx += 1

    fig.update_layout(
        title="1D Marginal Comparison: Real vs Synthetic",
        barmode="overlay",
        template="plotly_white",
        height=300 * max(1, rows),
        width=900,
    )
    return fig


def plot_correlation_difference(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: List[str],
) -> go.Figure:
    """Heatmap of pairwise Pearson correlation matrix differences |Corr_real - Corr_synth|."""
    cols = [c for c in continuous_cols if c in real_df.columns and c in synth_df.columns]
    if len(cols) < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough continuous columns for correlation matrix.")
        return fig

    real_corr = real_df[cols].corr().fillna(0.0).values
    synth_corr = synth_df[cols].corr().fillna(0.0).values
    diff = np.abs(real_corr - synth_corr)

    fig = go.Figure(
        data=go.Heatmap(
            z=diff,
            x=cols,
            y=cols,
            colorscale="Reds",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="|Diff|"),
        )
    )
    fig.update_layout(
        title="Pairwise Correlation Absolute Error |Corr(Real) - Corr(Synth)|",
        template="plotly_white",
        width=700,
        height=600,
    )
    return fig


def plot_dcr_distributions(
    dcr_train: np.ndarray,
    dcr_test: np.ndarray,
) -> go.Figure:
    """Plots empirical DCR distributions comparing proximity to train vs test partitions."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=dcr_train,
            histnorm="probability density",
            name="DCR(Synth -> Real Train)",
            opacity=0.6,
            marker_color="#d62728",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=dcr_test,
            histnorm="probability density",
            name="DCR(Synth -> Real Test)",
            opacity=0.6,
            marker_color="#2ca02c",
        )
    )
    fig.update_layout(
        title="Distance to Closest Record (DCR) Distributions (Train vs Test)",
        xaxis_title="Gower Distance",
        yaxis_title="Probability Density",
        barmode="overlay",
        template="plotly_white",
        width=800,
        height=500,
    )
    return fig


def plot_augmentation_curves(
    aug_results: Dict[str, Any],
) -> go.Figure:
    """Plots downstream performance gain across minority class augmentation ratios."""
    by_ratio = aug_results.get("by_ratio", {})
    ratios = sorted([float(r) for r in by_ratio.keys()])

    fig = go.Figure()

    # Collect RF metrics across ratios
    rf_recalls = []
    rf_f1s = []

    for r in ratios:
        models = by_ratio[str(r)].get("models", {})
        rf_entry = models.get("random_forest", {})
        # For ratio 0, structure is direct metrics; for >0, metrics is inside 'metrics'
        if "metrics" in rf_entry:
            m = rf_entry["metrics"]
        else:
            m = rf_entry
        rf_recalls.append(m.get("minority_recall", 0.0))
        rf_f1s.append(m.get("macro_f1", 0.0))

    fig.add_trace(
        go.Scatter(
            x=ratios,
            y=rf_recalls,
            mode="lines+markers",
            name="Random Forest Minority Recall",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ratios,
            y=rf_f1s,
            mode="lines+markers",
            name="Random Forest Macro-F1",
            line=dict(color="#2ca02c", width=3, dash="dash"),
        )
    )

    fig.update_layout(
        title="Minority-Class Augmentation Trade-off: Recall vs Macro-F1",
        xaxis_title="Augmentation Ratio (Alpha)",
        yaxis_title="Downstream Score on Untouched Test Partition",
        template="plotly_white",
        width=800,
        height=500,
    )
    return fig


def save_figure(fig: go.Figure, output_dir: str, filename_base: str) -> Dict[str, str]:
    """Saves a Plotly figure as both interactive HTML and JSON data."""
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{filename_base}.html")
    json_path = os.path.join(output_dir, f"{filename_base}.json")

    fig.write_html(html_path, include_plotlyjs="cdn")
    with open(json_path, "w") as f:
        f.write(fig.to_json())

    return {"html": html_path, "json": json_path}
