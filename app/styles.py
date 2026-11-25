"""Custom CSS design system and styling tokens for TabSynth Analytics.

Implements an enterprise-grade dark SaaS visual hierarchy directly inspired
by the modern technical dashboard aesthetics (PulseMetrics / Tabular AI).
"""

CUSTOM_CSS = """
<style>
/* =========================================================================
   TabSynth Analytics CSS Design System
   Exact Reference Palette:
     Main BG: #0A0D14 | Sidebar: #0F1420 | Cards: #131B2E | Elevated: #182238
     Borders: #1E283D | Border-Hover: #2D3D5E
     Primary Accent: #8B5CF6 (Purple) | Cyan: #06B6D4 | Emerald: #10B981
     Amber: #F59E0B | Rose: #EF4444 | Sky: #38BDF8
     Text Main: #F8FAFC | Text Secondary: #8F9CAE | Text Muted: #64748B
   ========================================================================= */

:root {
    --ts-bg-main: #0A0D14;
    --ts-bg-sidebar: #0F1420;
    --ts-bg-card: #131B2E;
    --ts-bg-elevated: #182238;
    --ts-border: #1E283D;
    --ts-border-hover: #2D3D5E;
    --ts-purple: #8B5CF6;
    --ts-cyan: #06B6D4;
    --ts-emerald: #10B981;
    --ts-amber: #F59E0B;
    --ts-rose: #EF4444;
    --ts-sky: #38BDF8;
    --ts-text-main: #F8FAFC;
    --ts-text-muted: #8F9CAE;
}

/* Global scrollbar styling */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0A0D14;
}
::-webkit-scrollbar-thumb {
    background: #1E283D;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #8B5CF6;
}

/* Page Header Block */
.ts-page-header {
    margin-bottom: 1.5rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid #1E283D;
}
.ts-breadcrumb {
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8B5CF6;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.ts-page-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.ts-page-subtitle {
    font-size: 0.88rem;
    color: #8F9CAE;
    margin-top: 0.3rem;
    line-height: 1.45;
}

/* Section Header */
.ts-section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-top: 1.4rem;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
}
.ts-section-subtitle {
    font-size: 0.8rem;
    color: #8F9CAE;
    margin-top: -0.4rem;
    margin-bottom: 1rem;
}

/* Stat & Metric Cards (Matching Overview Reference) */
.ts-metric-card {
    background: #131B2E;
    border: 1px solid #1E283D;
    border-radius: 10px;
    padding: 1.15rem 1.25rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}
.ts-metric-card:hover {
    border-color: #2D3D5E;
    box-shadow: 0 6px 24px rgba(139, 92, 246, 0.12);
}
.ts-metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #8F9CAE;
    margin-bottom: 0.45rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ts-metric-value-row {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.3rem;
}
.ts-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
}
.ts-delta-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
}
.ts-delta-pill-green {
    color: #10B981;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.ts-delta-pill-red {
    color: #EF4444;
    background: rgba(239, 68, 68, 0.14);
    border: 1px solid rgba(239, 68, 68, 0.25);
}
.ts-metric-caption {
    font-size: 0.74rem;
    color: #64748B;
    margin-top: 0.25rem;
}

/* General Container Card */
.ts-card {
    background: #131B2E;
    border: 1px solid #1E283D;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: border-color 0.15s ease;
}
.ts-card:hover {
    border-color: #2D3D5E;
}

/* Benchmark Model Rank Card (Matching Benchmark Center Reference) */
.ts-model-card {
    background: #131B2E;
    border: 1px solid #1E283D;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease;
}
.ts-model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
}
.ts-model-card-purple {
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.15);
}
.ts-model-card-cyan {
    border-color: rgba(6, 182, 212, 0.4);
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.15);
}
.ts-model-card-green {
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
}
.ts-score-circle {
    width: 76px;
    height: 76px;
    border-radius: 50%;
    margin: 0.8rem auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 3px solid;
    font-family: 'JetBrains Mono', monospace;
}
.ts-score-circle-purple {
    border-color: #8B5CF6;
    color: #F8FAFC;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%);
}
.ts-score-circle-cyan {
    border-color: #06B6D4;
    color: #F8FAFC;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.2) 0%, transparent 70%);
}
.ts-score-circle-green {
    border-color: #10B981;
    color: #F8FAFC;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.2) 0%, transparent 70%);
}

/* Horizontal Score Progress Bar */
.ts-score-row {
    display: grid;
    grid-template-columns: 140px 1fr 60px;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1E283D;
}
.ts-score-name {
    font-size: 0.8rem;
    color: #8F9CAE;
    font-weight: 500;
}
.ts-progress-track {
    background: #182238;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    position: relative;
}
.ts-progress-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}
.ts-fill-green { background: linear-gradient(90deg, #10B981 0%, #34D399 100%); }
.ts-fill-cyan { background: linear-gradient(90deg, #06B6D4 0%, #38BDF8 100%); }
.ts-fill-purple { background: linear-gradient(90deg, #8B5CF6 0%, #A78BFA 100%); }

/* Insight Callout Banner (Matching Laptop Mockup Reference) */
.ts-insight-box {
    background: #131B2E;
    border: 1px solid #1E283D;
    border-left: 4px solid #10B981;
    border-radius: 8px;
    padding: 0.85rem 1.15rem;
    margin: 1rem 0;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}
.ts-insight-icon {
    font-size: 1.25rem;
    color: #10B981;
    line-height: 1;
}
.ts-insight-text {
    font-size: 0.84rem;
    color: #8F9CAE;
    line-height: 1.45;
}
.ts-insight-text strong {
    color: #F8FAFC;
}

/* Status Badges */
.ts-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.ts-badge-success {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.ts-badge-running {
    background: rgba(6, 182, 212, 0.15);
    color: #06B6D4;
    border: 1px solid rgba(6, 182, 212, 0.3);
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.2);
}
.ts-badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.ts-badge-error {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.ts-badge-purple {
    background: rgba(139, 92, 246, 0.15);
    color: #C084FC;
    border: 1px solid rgba(139, 92, 246, 0.35);
}
.ts-badge-neutral {
    background: rgba(148, 163, 184, 0.12);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

/* Pulsing Dot Animation */
.ts-pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: currentColor;
    animation: tsPulse 1.8s infinite;
}
@keyframes tsPulse {
    0% { opacity: 0.4; transform: scale(0.9); }
    50% { opacity: 1.0; transform: scale(1.2); }
    100% { opacity: 0.4; transform: scale(0.9); }
}

/* Disclaimer & Compliance Alert */
.ts-disclaimer-box {
    background: #131B2E;
    border-left: 4px solid #F59E0B;
    border-top: 1px solid #1E283D;
    border-right: 1px solid #1E283D;
    border-bottom: 1px solid #1E283D;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 1.25rem 0;
}
.ts-disclaimer-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #F59E0B;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.35rem;
}
.ts-disclaimer-text {
    font-size: 0.82rem;
    color: #8F9CAE;
    line-height: 1.45;
    margin: 0;
}

/* Empty State Box */
.ts-empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    background: #131B2E;
    border: 1px dashed #1E283D;
    border-radius: 10px;
    margin: 1.5rem 0;
}
.ts-empty-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.8;
}
.ts-empty-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-bottom: 0.35rem;
}
.ts-empty-desc {
    font-size: 0.85rem;
    color: #8F9CAE;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.45;
}

/* Sidebar branding box */
.ts-brand-box {
    padding: 0.85rem 1rem;
    background: linear-gradient(180deg, #131B2E 0%, #0F1420 100%);
    border: 1px solid #1E283D;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.ts-brand-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.ts-brand-subtitle {
    font-size: 0.72rem;
    color: #8F9CAE;
    margin-top: 0.2rem;
    font-weight: 500;
}

/* Streamlit Button enhancement */
div.stButton > button {
    background: #131B2E !important;
    border: 1px solid #1E283D !important;
    color: #F8FAFC !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
    transition: all 0.15s ease !important;
}
div.stButton > button:hover {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.25) !important;
    color: #F8FAFC !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%) !important;
    border: 1px solid #A78BFA !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 10px rgba(139, 92, 246, 0.3) !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #9333EA 0%, #6D28D9 100%) !important;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.45) !important;
}

/* Streamlit Form Input Border styling */
div[data-baseweb="input"], div[data-baseweb="select"] {
    border-color: #1E283D !important;
    background-color: #0F1420 !important;
}

/* Code block font overrides */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* =========================================================================
   Top Navigation Bar & Header System
   ========================================================================= */
.ts-top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(90deg, #131B2E 0%, #0F1420 100%);
    border: 1px solid #1E283D;
    border-radius: 8px;
    padding: 0.65rem 1.15rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
.ts-top-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.ts-top-brand-icon {
    font-size: 1.6rem;
    line-height: 1;
}
.ts-top-brand-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.01em;
}
.ts-top-brand-subtitle {
    font-size: 0.74rem;
    color: #8F9CAE;
    font-weight: 500;
}
.ts-top-meta {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.ts-top-chip {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: #94A3B8;
    background: #182238;
    border: 1px solid #1E283D;
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-family: 'JetBrains Mono', monospace;
}
.ts-chip-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #8B5CF6;
}
.ts-top-status-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: #8F9CAE;
    background: #182238;
    border: 1px solid #1E283D;
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
}
.ts-top-status-label {
    font-weight: 600;
    color: #8F9CAE;
}

/* Top Navigation Pills Styling */
.st-key-top_navbar_pills {
    margin-bottom: 1.25rem !important;
    padding-bottom: 0.75rem !important;
    border-bottom: 1px solid #1E283D !important;
}
.st-key-top_navbar_pills [role="radiogroup"],
.st-key-top_navbar_pills div[data-testid="stPills"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.45rem !important;
    align-items: center !important;
}
.st-key-top_navbar_pills button {
    background-color: #131B2E !important;
    border: 1px solid #1E283D !important;
    color: #8F9CAE !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 0.35rem 0.75rem !important;
    transition: all 0.15s ease-in-out !important;
}
.st-key-top_navbar_pills button:hover {
    border-color: #8B5CF6 !important;
    color: #F8FAFC !important;
    background-color: #182238 !important;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.25) !important;
}
.st-key-top_navbar_pills button[aria-checked="true"],
.st-key-top_navbar_pills button[data-checked="true"] {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.25) 0%, rgba(124, 58, 237, 0.35) 100%) !important;
    border: 1px solid #8B5CF6 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 0 14px rgba(139, 92, 246, 0.35) !important;
}
</style>
"""
