"""Reusable presentation components for TabSynth Analytics.

Provides styled metric cards, status badges, headers, disclaimer boxes,
empty states, hardware banners, glowing model benchmark cards, horizontal
score progress bars, and insight banners using st.html and textwrap.dedent.
"""

import textwrap
from typing import Optional, List, Dict, Any
import streamlit as st


def render_page_header(
    title: str,
    subtitle: Optional[str] = None,
    category: Optional[str] = None,
    breadcrumbs: Optional[List[str]] = None,
) -> None:
    """Renders the top breadcrumb and title header for a navigation view."""
    crumbs_html = ""
    if breadcrumbs:
        crumbs_str = " / ".join(breadcrumbs)
        crumbs_html = f'<div class="ts-breadcrumb">🧬 TABSYNTH / {crumbs_str}</div>'
    elif category:
        crumbs_html = f'<div class="ts-breadcrumb">🧬 TABSYNTH / {category.upper()}</div>'

    sub_html = f'<div class="ts-page-subtitle">{subtitle}</div>' if subtitle else ""

    html = f"""
    <div class="ts-page-header">
        {crumbs_html}
        <h1 class="ts-page-title">{title}</h1>
        {sub_html}
    </div>
    """
    st.html(textwrap.dedent(html).strip())


def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """Renders a section title with optional subtitle and icon."""
    icon_str = f"{icon} " if icon else ""
    sub_html = f'<div class="ts-section-subtitle">{subtitle}</div>' if subtitle else ""
    html = f"""
    <div class="ts-section-header">{icon_str}{title}</div>
    {sub_html}
    """
    st.html(textwrap.dedent(html).strip())


def render_metric_card(
    title: str,
    value: Any,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    icon: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> None:
    """Renders an individual stat card matching the Overview reference."""
    delta_html = ""
    if delta:
        delta_class = "ts-delta-pill-green" if delta_positive else "ts-delta-pill-red"
        arrow = "↑" if delta_positive else "↓"
        delta_html = f'<span class="ts-delta-pill {delta_class}">{arrow} {delta}</span>'

    caption_html = f'<div class="ts-metric-caption">{subtitle}</div>' if subtitle else ""

    html = f"""
    <div class="ts-metric-card">
        <div class="ts-metric-label">
            <span>{title}</span>
            <span>{icon or ""}</span>
        </div>
        <div class="ts-metric-value-row">
            <span class="ts-metric-value">{value}</span>
            {delta_html}
        </div>
        {caption_html}
    </div>
    """
    st.html(textwrap.dedent(html).strip())


def render_benchmark_rank_card(
    rank: int,
    model_name: str,
    model_type: str,
    score: float,
    performance_tag: str,
    color_theme: str = "purple",
) -> None:
    """Renders a 3D-styled glowing model benchmark card matching the Benchmark Center reference."""
    circle_cls = f"ts-score-circle-{color_theme}"
    card_cls = f"ts-model-card-{color_theme}"
    badge_colors = {
        "purple": ("#8B5CF6", "rgba(139,92,246,0.15)"),
        "cyan": ("#06B6D4", "rgba(6,182,212,0.15)"),
        "green": ("#10B981", "rgba(16,185,129,0.15)"),
    }
    b_color, b_bg = badge_colors.get(color_theme, ("#8B5CF6", "rgba(139,92,246,0.15)"))

    html = f"""
    <div class="ts-model-card {card_cls}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.75rem; font-weight:700; color:{b_color}; letter-spacing:0.05em;">RANK {rank}</span>
            <span style="font-size:0.7rem; color:#8F9CAE; background:#182238; padding:2px 6px; border-radius:4px;">{model_type}</span>
        </div>
        <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-top:0.5rem;">{model_name}</div>
        <div class="ts-score-circle {circle_cls}">
            <div style="font-size:1.4rem; font-weight:700; line-height:1;">{score:.1f}</div>
            <div style="font-size:0.62rem; color:#8F9CAE; margin-top:2px;">SCORE</div>
        </div>
        <div style="font-size:0.75rem; font-weight:600; color:{b_color}; background:{b_bg}; padding:3px 8px; border-radius:12px; display:inline-block;">
            {performance_tag}
        </div>
    </div>
    """
    st.html(textwrap.dedent(html).strip())


def render_score_bar_row(
    dimension_name: str,
    score_pct: float,
    fill_color: str = "purple",
) -> None:
    """Renders a single horizontal progress row for benchmark capability breakdown."""
    fill_cls = f"ts-fill-{fill_color}"
    html = f"""
    <div class="ts-score-row">
        <div class="ts-score-name">{dimension_name}</div>
        <div class="ts-progress-track">
            <div class="ts-progress-fill {fill_cls}" style="width: {min(100.0, max(0.0, score_pct)):.1f}%;"></div>
        </div>
        <div style="font-family:'JetBrains Mono'; font-size:0.8rem; font-weight:600; color:#F8FAFC; text-align:right;">{score_pct:.1f}</div>
    </div>
    """
    st.html(textwrap.dedent(html).strip())


def render_insight_banner(
    message: str,
    icon: str = "💡",
    highlight_text: Optional[str] = None,
) -> None:
    """Renders the highlighted analytical insight banner matching the Laptop Mockup reference."""
    html = f"""
    <div class="ts-insight-box">
        <div class="ts-insight-icon">{icon}</div>
        <div class="ts-insight-text">
            <strong>{highlight_text or 'Key Finding'}:</strong> {message}
        </div>
    </div>
    """
    st.html(textwrap.dedent(html).strip())


def get_status_badge_html(status: str, pulse: bool = False) -> str:
    """Returns HTML string for a status pill."""
    status_lower = status.lower()
    if status_lower in ["completed", "success", "low_risk", "online", "active"]:
        badge_cls = "ts-badge-success"
        dot = "●"
    elif status_lower in ["running", "in_progress", "training", "generating"]:
        badge_cls = "ts-badge-running"
        dot = '<span class="ts-pulse-dot"></span>' if pulse else "●"
    elif status_lower in ["queued", "pending", "medium_risk", "offline"]:
        badge_cls = "ts-badge-warning"
        dot = "●"
    elif status_lower in ["failed", "error", "high_risk"]:
        badge_cls = "ts-badge-error"
        dot = "●"
    elif status_lower in ["cancelled"]:
        badge_cls = "ts-badge-neutral"
        dot = "○"
    else:
        badge_cls = "ts-badge-purple"
        dot = "●"

    return f'<span class="ts-badge {badge_cls}">{dot} {status.upper()}</span>'


def render_status_badge(status: str, pulse: bool = False) -> None:
    """Renders a standalone status badge pill in Streamlit using st.html."""
    st.html(get_status_badge_html(status, pulse=pulse))


def render_dataset_card(
    name: str,
    dataset_id: str,
    num_rows: int,
    num_cols: int,
    continuous_count: int,
    categorical_count: int,
    target_col: Optional[str] = None,
    missing_count: int = 0,
) -> None:
    """Renders a metadata card for a registered dataset."""
    target_badge = (
        f'<span class="ts-badge ts-badge-purple" style="font-size:0.68rem;">Target: {target_col}</span>'
        if target_col
        else '<span class="ts-badge ts-badge-neutral" style="font-size:0.68rem;">Unsupervised</span>'
    )
    missing_badge = (
        '<span class="ts-badge ts-badge-success" style="font-size:0.68rem;">Zero Missing</span>'
        if missing_count == 0
        else f'<span class="ts-badge ts-badge-warning" style="font-size:0.68rem;">{missing_count:,} Missing Cells</span>'
    )

    html = f"""
    <div class="ts-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.6rem;">
            <div>
                <div style="font-weight:700; font-size:1.05rem; color:#F8FAFC;">{name}</div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#8F9CAE;">ID: {dataset_id}</div>
            </div>
            <div style="display:flex; gap:0.4rem; align-items:center;">
                {target_badge}
                {missing_badge}
            </div>
        </div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid #1E283D;">
            <div>
                <div style="font-size:0.72rem; color:#8F9CAE;">ROWS</div>
                <div style="font-family:'JetBrains Mono'; font-weight:600; color:#F8FAFC;">{num_rows:,}</div>
            </div>
            <div>
                <div style="font-size:0.72rem; color:#8F9CAE;">COLUMNS</div>
                <div style="font-family:'JetBrains Mono'; font-weight:600; color:#F8FAFC;">{num_cols}</div>
            </div>
            <div>
                <div style="font-size:0.72rem; color:#8F9CAE;">CONTINUOUS</div>
                <div style="font-family:'JetBrains Mono'; font-weight:600; color:#06B6D4;">{continuous_count}</div>
            </div>
            <div>
                <div style="font-size:0.72rem; color:#8F9CAE;">CATEGORICAL</div>
                <div style="font-family:'JetBrains Mono'; font-weight:600; color:#8B5CF6;">{categorical_count}</div>
            </div>
        </div>
    </div>
    """
    st.html(textwrap.dedent(html).strip())


def render_hardware_banner(
    device_str: str,
    engine_continuous: str = "Gaussian DDPM",
    engine_categorical: str = "Multinomial D3PM",
    precision: str = "Float32 (Local-First)",
    backend_status: str = "FastAPI Online",
    compute_status: str = "Ready / Operational",
) -> None:
    """Renders the top infrastructure card displaying accelerator and diffusion engines."""
    dev_badge = "ts-badge-success" if "mps" in device_str.lower() or "cuda" in device_str.lower() else "ts-badge-cyan"
    html = f"""
    <div class="ts-card" style="background: linear-gradient(135deg, #131B2E 0%, #0F1420 100%);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
            <div style="display:flex; align-items:center; gap:0.5rem;">
                <span style="font-size:1.2rem;">⚡</span>
                <span style="font-weight:700; font-size:0.95rem; color:#F8FAFC; text-transform:uppercase; letter-spacing:0.04em;">Compute & Diffusion Infrastructure</span>
            </div>
            <div style="display:flex; gap:0.5rem;">
                <span class="ts-badge {dev_badge}">Compute: {device_str.upper()}</span>
                <span class="ts-badge ts-badge-purple">{compute_status}</span>
            </div>
        </div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top:0.5rem; padding-top:0.75rem; border-top: 1px solid #1E283D;">
            <div>
                <div style="font-size:0.72rem; color:#8F9CAE; text-transform:uppercase;">Accelerator Device</div>
                <div style="font-family:'JetBrains Mono'; font-weight:600; color:#F8FAFC; margin-top:0.2rem;">{device_str.upper()}</div>
            </div>
            <div>
                <div style="font-size:0.72rem; color:#8F9CAE; text-transform:uppercase;">Continuous Engine</div>
                <div style="font-family:'JetBrains Mono'; font-weight:600; color:#06B6D4; margin-top:0.2rem;">{engine_continuous}</div>
            </div>
            <div>
                <div style="font-size:0.72rem; color:#8F9CAE; text-transform:uppercase;">Categorical Engine</div>
                <div style="font-family:'JetBrains Mono'; font-weight:600; color:#8B5CF6; margin-top:0.2rem;">{engine_categorical}</div>
            </div>
            <div>
                <div style="font-size:0.72rem; color:#8F9CAE; text-transform:uppercase;">Execution Precision</div>
                <div style="font-family:'JetBrains Mono'; font-weight:600; color:#10B981; margin-top:0.2rem;">{precision}</div>
            </div>
        </div>
    </div>
    """
    st.html(textwrap.dedent(html).strip())


def render_disclaimer(
    title: str = "Research Privacy Disclaimer",
    message: Optional[str] = None,
    icon: str = "🛡️",
) -> None:
    """Renders a prominent, high-visibility empirical privacy disclaimer callout."""
    default_msg = (
        "Empirical metrics (Distance-to-Closest-Record, exact match rate, shadow MIA AUROC) evaluate "
        "re-identification resistance under empirical test distributions. They do <strong>NOT</strong> provide "
        "mathematically provable Differential Privacy (ε, δ)-DP guarantees. "
        "For regulated enterprise environments (HIPAA, GDPR, CCPA), DP-SGD noise addition during training is required."
    )
    body_text = message or default_msg
    html = f"""
    <div class="ts-disclaimer-box">
        <div class="ts-disclaimer-title">
            <span>{icon}</span> {title}
        </div>
        <div class="ts-disclaimer-text">{body_text}</div>
    </div>
    """
    st.html(textwrap.dedent(html).strip())


def render_empty_state(
    title: str,
    description: str,
    icon: str = "🔍",
) -> None:
    """Renders a clean empty-state placeholder when no data exists."""
    html = f"""
    <div class="ts-empty-state">
        <div class="ts-empty-icon">{icon}</div>
        <div class="ts-empty-title">{title}</div>
        <div class="ts-empty-desc">{description}</div>
    </div>
    """
    st.html(textwrap.dedent(html).strip())


def render_error_callout(
    title: str,
    message: str,
    hint: Optional[str] = None,
) -> None:
    """Renders a structured error callout with actionable resolution hints."""
    hint_html = f'<div style="margin-top:0.5rem; font-size:0.8rem; color:#8F9CAE;">💡 <strong>Hint:</strong> {hint}</div>' if hint else ""
    html = f"""
    <div style="background:rgba(239, 68, 68, 0.08); border-left:4px solid #EF4444; border-top:1px solid #1E283D; border-right:1px solid #1E283D; border-bottom:1px solid #1E283D; border-radius:0 8px 8px 0; padding:1rem 1.25rem; margin:1rem 0;">
        <div style="color:#EF4444; font-weight:700; font-size:0.92rem; display:flex; align-items:center; gap:0.4rem;">
            <span>⚠️</span> {title}
        </div>
        <div style="color:#F8FAFC; font-size:0.85rem; margin-top:0.35rem; line-height:1.4;">{message}</div>
        {hint_html}
    </div>
    """
    st.html(textwrap.dedent(html).strip())
