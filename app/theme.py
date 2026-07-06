"""Shared dark/gold visual theme for the Streamlit dashboard.

Matches the portfolio site's palette: near-black background, warm gold accent,
thin hairline borders, subtle fade-in animation. Plotly charts get a registered
default template so every chart in the app picks up the theme automatically;
explicit red/green semantic colors (risk up/down) are left untouched since
they carry meaning, not just decoration.
"""
from __future__ import annotations

BG = "#0A0A0A"
SURFACE = "#131313"
ACCENT = "#C6A15B"
ACCENT_2 = "#8B7355"
TEXT = "#F2EFE9"
TEXT_MUTED = "#9C9791"
BORDER = "rgba(255,255,255,0.14)"
GRID = "rgba(255,255,255,0.08)"

COLORWAY = [ACCENT, "#5A8F8C", ACCENT_2, "#D4AF37", "#6B8E9E", "#4A3A20"]


def apply_plotly_theme(pio) -> None:
    """Register and activate the dark/gold Plotly template as the default."""
    import plotly.graph_objects as go

    pio.templates["portfolio_dark"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=BG,
            plot_bgcolor=SURFACE,
            font=dict(color=TEXT, family="Inter, -apple-system, sans-serif"),
            colorway=COLORWAY,
            xaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER),
            yaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            transition=dict(duration=500, easing="cubic-in-out"),
            margin=dict(t=50, b=40),
        )
    )
    pio.templates.default = "portfolio_dark"


def plotly_chart(fig, **kwargs) -> None:
    """Drop-in replacement for st.plotly_chart.

    Streamlit's plotly_chart applies its own default theme on top of a
    figure's template unless theme=None is passed, and in practice some
    layout properties (paper/plot background) still don't reliably survive
    that path — so force them directly onto the figure here rather than
    relying solely on the registered default template. Per-trace colors
    (e.g. explicit red/green risk-factor bars) are untouched since this only
    sets layout-level properties, not trace marker colors.
    """
    import streamlit as st

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=SURFACE,
        font_color=TEXT,
        xaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    st.plotly_chart(fig, theme=None, **kwargs)


THEME_CSS = f"""
<style>
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.block-container {{ padding-top: 2.2rem; max-width: 1150px; animation: fadeInUp 0.6s ease-out; }}
.stApp {{ background-color: {BG}; }}
[data-testid="stHeader"] {{ background-color: transparent; }}
[data-testid="stSidebar"] {{
    background-color: #050505;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

.hero {{
    background: linear-gradient(120deg, #0A0A0A 0%, #1a1408 60%, #0A0A0A 100%);
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 24px 30px;
    margin-bottom: 20px;
    color: {TEXT};
    box-shadow: 0 10px 30px rgba(198,161,91,0.15);
    animation: fadeInUp 0.7s ease-out;
}}
.hero h1 {{ font-size: 1.6rem; font-weight: 600; margin: 0 0 6px 0; color: {TEXT}; }}
.hero p {{ margin: 0; color: {TEXT_MUTED}; }}

[data-testid="stVerticalBlockBorderWrapper"] {{ animation: fadeInUp 0.5s ease-out; }}
h1, h2, h3 {{ color: {TEXT} !important; font-weight: 500 !important; }}
p, span, label, .stMarkdown, .stCaption {{ color: #E5E2DC; }}

[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 12px 16px;
    transition: border-color 0.3s ease, transform 0.3s ease;
}}
[data-testid="stMetric"]:hover {{ border-color: {ACCENT}; transform: translateY(-2px); }}
[data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-size: 1.5rem; }}
[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}

.stTabs [data-baseweb="tab"] {{ color: {TEXT_MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}

.stButton > button {{
    background-color: transparent;
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 2px;
    transition: all 0.3s ease;
}}
.stButton > button:hover {{ background-color: {ACCENT}; color: {BG}; }}

hr {{ border-color: {BORDER} !important; }}
[data-testid="stExpander"] {{ background-color: {SURFACE}; border: 1px solid {BORDER}; }}
input, textarea, select, [data-baseweb="select"] > div {{
    background-color: {SURFACE} !important;
    color: {TEXT} !important;
    border-color: {BORDER} !important;
}}
[data-testid="stDataFrame"] {{ animation: fadeInUp 0.5s ease-out; }}
</style>
"""
