"""Plotly dark theme templates and specialized charts matching the target visual designs.

Provides:
- Smooth curved spline area charts (Overview Dashboard)
- Multi-colored vertical feature adoption bars (Overview Dashboard)
- Fully annotated 2D correlation matrix with values in every cell (Data Explorer / Fidelity)
- Scientific Box and Violin distribution plots (Experiment Explorer)
- Hyperparameter vs loss scatter plots with side colorbars (Experiment Explorer)
- Performance radar charts (Benchmark Center)
- Multi-model smooth KDE density distribution overlays (Benchmark Center)
- Mini KPI sparkline trend curves
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Reference Target Colors
COLORS = {
    "bg_main": "#0A0D14",
    "bg_sidebar": "#0F1420",
    "bg_card": "#131B2E",
    "bg_elevated": "#182238",
    "border": "#1E283D",
    "border_light": "#2D3D5E",
    "purple": "#8B5CF6",
    "purple_glow": "rgba(139, 92, 246, 0.16)",
    "cyan": "#06B6D4",
    "cyan_glow": "rgba(6, 182, 212, 0.16)",
    "green": "#10B981",
    "green_glow": "rgba(16, 185, 129, 0.16)",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "sky": "#38BDF8",
    "pink": "#EC4899",
    "text_main": "#F8FAFC",
    "text_muted": "#8F9CAE",
}

# Distinct Model Palette
BRAND_PALETTE = [
    "#8B5CF6",  # Purple (TabSynth Hybrid)
    "#06B6D4",  # Cyan (TabDDPM MLP)
    "#10B981",  # Emerald (TVAE)
    "#F59E0B",  # Amber (Gaussian Copula)
    "#38BDF8",  # Sky Blue
    "#EC4899",  # Pink
    "#EF4444",  # Rose
    "#8F9CAE",  # Slate
]


def apply_dark_theme(
    fig: go.Figure,
    height: Optional[int] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> go.Figure:
    """Applies the high-contrast dark theme matching the reference mockups."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A0D14",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=COLORS["text_main"],
            size=12,
        ),
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(size=14, color=COLORS["text_main"]),
            x=0.01,
            xanchor="left",
        ) if title else None,
        margin=dict(l=40, r=25, t=50 if title else 20, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLORS["text_muted"]),
            bgcolor="rgba(0,0,0,0)",
        ) if show_legend else dict(visible=False),
        colorway=BRAND_PALETTE,
    )
    if height:
        fig.update_layout(height=height)

    # Format Axes
    fig.update_xaxes(
        gridcolor=COLORS["border"],
        zerolinecolor=COLORS["border_light"],
        tickfont=dict(color=COLORS["text_muted"], size=11),
        title_font=dict(color=COLORS["text_main"], size=12),
    )
    fig.update_yaxes(
        gridcolor=COLORS["border"],
        zerolinecolor=COLORS["border_light"],
        tickfont=dict(color=COLORS["text_muted"], size=11),
        title_font=dict(color=COLORS["text_main"], size=12),
    )
    return fig


def create_spline_area_chart(
    x: List[Any],
    y: List[float],
    title: str = "Weekly Active Generation",
    color: str = "#8B5CF6",
    height: int = 320,
) -> go.Figure:
    """Creates a smooth curved time-series spline line with glowing area fill underneath."""
    fill_rgba = "rgba(139, 92, 246, 0.16)" if color == "#8B5CF6" else "rgba(6, 182, 212, 0.16)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        line=dict(shape="spline", color=color, width=3),
        fill="tozeroy",
        fillcolor=fill_rgba,
        marker=dict(size=6, color=color, line=dict(width=2, color="#FFFFFF")),
        name="Throughput",
    ))
    return apply_dark_theme(fig, height=height, title=title, show_legend=False)


def create_feature_adoption_bars(
    categories: List[str],
    values: List[float],
    title: str = "Feature Adoption",
    height: int = 320,
) -> go.Figure:
    """Creates multi-colored vertical bars matching the Overview reference."""
    palette = ["#8B5CF6", "#06B6D4", "#38BDF8", "#10B981", "#EC4899"]
    bar_colors = [palette[i % len(palette)] for i in range(len(categories))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(color=bar_colors, cornerradius=4),
        text=[f"{v:,}" if isinstance(v, int) else f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text_muted"], size=11),
    ))
    return apply_dark_theme(fig, height=height, title=title, show_legend=False)


def create_annotated_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "CORRELATION MATRIX",
    height: int = 380,
) -> go.Figure:
    """Creates a 2D correlation matrix with numeric text values labeled in every grid cell."""
    z_vals = corr_matrix.values
    x_labels = list(corr_matrix.columns)
    y_labels = list(corr_matrix.index)

    # Format cell text annotations
    text_vals = [[f"{val:.2f}" for val in row] for row in z_vals]

    # Green-to-dark custom colorscale
    colorscale = [
        [0.0, "#0F1420"],
        [0.3, "#132822"],
        [0.6, "#047857"],
        [1.0, "#10B981"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=10, color="#FFFFFF", family="JetBrains Mono"),
        colorscale=colorscale,
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(
            title=dict(text="r", font=dict(color=COLORS["text_main"], size=11)),
            tickfont=dict(color=COLORS["text_muted"], size=10),
            thickness=12,
            len=0.85,
        ),
    ))
    return apply_dark_theme(fig, height=height, title=title)


def create_box_violin_plots(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    plot_type: str = "box",
    title: str = "Validation Loss Distribution",
    height: int = 340,
) -> go.Figure:
    """Creates scientific Box or Violin distribution plots matching Experiment Explorer."""
    fig = go.Figure()
    unique_groups = df[group_col].unique()
    palette = BRAND_PALETTE

    for idx, grp in enumerate(unique_groups):
        grp_data = df[df[group_col] == grp][metric_col]
        color = palette[idx % len(palette)]
        if plot_type == "violin":
            fig.add_trace(go.Violin(
                y=grp_data,
                name=str(grp),
                box_visible=True,
                meanline_visible=True,
                line_color=color,
                fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.25])}",
            ))
        else:
            fig.add_trace(go.Box(
                y=grp_data,
                name=str(grp),
                marker_color=color,
                boxpoints="outliers",
                line=dict(width=1.5),
            ))

    return apply_dark_theme(fig, height=height, title=title)


def create_scatter_parameter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str = "Hyperparameter vs Validation Loss",
    height: int = 340,
) -> go.Figure:
    """Creates a multi-variable scatter plot with side colorbar matching Experiment Explorer."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        color_continuous_scale=[
            (0.0, COLORS["purple"]),
            (0.5, COLORS["cyan"]),
            (1.0, COLORS["green"]),
        ],
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color="#FFFFFF")))
    return apply_dark_theme(fig, height=height, title=title)


def create_score_distribution_kde(
    model_curves: Dict[str, Dict[str, Any]],
    title: str = "SCORE DISTRIBUTION",
    height: int = 320,
) -> go.Figure:
    """Creates 3 smooth KDE curves (green, blue, purple) matching Benchmark Center."""
    fig = go.Figure()
    palette = [COLORS["green"], COLORS["cyan"], COLORS["purple"]]

    idx = 0
    for name, data in model_curves.items():
        x_vals = data["x"]
        y_vals = data["y"]
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=name,
            line=dict(shape="spline", color=color, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.12])}",
        ))
        idx += 1

    return apply_dark_theme(fig, height=height, title=title)


def create_radar_chart(
    categories: List[str],
    series_dict: Dict[str, List[float]],
    title: str = "PERFORMANCE RADAR",
    height: int = 320,
) -> go.Figure:
    """Creates a multi-axis spider radar chart matching Benchmark Center."""
    fig = go.Figure()
    palette = [COLORS["green"], COLORS["cyan"], COLORS["purple"], COLORS["amber"]]

    idx = 0
    for name, values in series_dict.items():
        color = palette[idx % len(palette)]
        vals_closed = list(values) + [values[0]]
        cats_closed = list(categories) + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            name=name,
            line=dict(color=color, width=2),
            opacity=0.65,
        ))
        idx += 1

    fig.update_layout(
        polar=dict(
            bgcolor="#0A0D14",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                tickfont=dict(size=9, color=COLORS["text_muted"]),
            ),
            angularaxis=dict(
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                tickfont=dict(size=10, color=COLORS["text_main"]),
            ),
        ),
        showlegend=True,
    )
    return apply_dark_theme(fig, title=title, height=height)


def create_sparkline(
    values: List[float],
    color: str = "#10B981",
    height: int = 40,
) -> go.Figure:
    """Creates a mini sparkline trend curve without axes or legends."""
    fig = go.Figure()
    fill_rgba = "rgba(16, 185, 129, 0.15)" if color == "#10B981" else "rgba(239, 68, 68, 0.15)"
    fig.add_trace(go.Scatter(
        x=list(range(len(values))),
        y=values,
        mode="lines",
        line=dict(shape="spline", color=color, width=2),
        fill="tozeroy",
        fillcolor=fill_rgba,
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# Backwards-compatible alias
create_correlation_heatmap = create_annotated_correlation_heatmap

__all__ = [
    "COLORS",
    "BRAND_PALETTE",
    "apply_dark_theme",
    "create_spline_area_chart",
    "create_feature_adoption_bars",
    "create_annotated_correlation_heatmap",
    "create_correlation_heatmap",
    "create_box_violin_plots",
    "create_scatter_parameter_plot",
    "create_score_distribution_kde",
    "create_radar_chart",
    "create_sparkline",
]
