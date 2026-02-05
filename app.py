#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliteserien DSS – Professional Football Analytics Dashboard
@author: kulmiyearab

Profesjonelt beslutningsstøttesystem for talentidentifisering og rekruttering.
Refaktorert til kurert, klubb-klar informasjonsarkitektur:
- Forside (produktpresentasjon)
- Dashboard (Talent Map + shortlist-inngang)
- Spillerkort (rolleprofil + bars som default)
- Duell (A vs B + differanser)
- Scout report (unge talenter)
- Spillerstiler (klynger med forklaring + PCA som støtte)
- Transfer tracker (watchlist)

Teknologi: Streamlit + Plotly + Pandas
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# KONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

PARQUET_PATH = BASE_DIR / "data" / "eliteserien_2025_enriched.parquet"
LOGO_DIR = BASE_DIR / "assets" / "logos"   # valgfritt
KIT_DIR = BASE_DIR / "assets" / "kits"     # valgfritt



# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None,
)


# =============================================================================
# STYLING
# =============================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg0:#0b0f14;
  --bg1:#0f141b;
  --panel: rgba(255,255,255,0.04);
  --panel2: rgba(255,255,255,0.055);
  --border: rgba(255,255,255,0.10);
  --border2: rgba(102,126,234,0.28);
  --text:#eaeef5;
  --muted:#a7b0c0;

  --brand1:#667eea;
  --brand2:#764ba2;

  --pos:#3ddc97;
  --warn:#ffcc66;
  --neg:#ff5c6c;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

html, body, [class*="css"]  {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(102,126,234,0.16), transparent 60%),
              radial-gradient(900px 500px at 80% 10%, rgba(118,75,162,0.14), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
  color: var(--text) !important;
}

.block-container{
  max-width: 1600px;
  padding-top: 1.2rem;
  padding-bottom: 2.0rem;
  padding-left: 2.0rem;
  padding-right: 2.0rem;
}

section[data-testid="stSidebar"]{
  width: 310px !important;
  background: linear-gradient(180deg, #0c1117 0%, #0c1117 60%, #0b0f14 100%);
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] > div { padding-top: 1.6rem; }

h1{
  font-size: 2.5rem !important;
  font-weight: 800 !important;
  letter-spacing: -1.2px !important;
  background: linear-gradient(135deg, var(--brand1) 0%, var(--brand2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.35rem !important;
  line-height: 1.15 !important;
}
h2{
  font-size: 1.55rem !important;
  font-weight: 750 !important;
  letter-spacing: -0.6px !important;
  color: #ffffff !important;
  margin-top: 1.3rem !important;
  margin-bottom: 0.75rem !important;
}
h3{
  font-size: 1.15rem !important;
  font-weight: 650 !important;
  color: #eef2f8 !important;
  margin-bottom: 0.6rem !important;
  letter-spacing: -0.2px !important;
}

/* Panels */
.pane{
  background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 18px;
  padding: 1.35rem 1.35rem;
  box-shadow: 0 12px 26px rgba(0,0,0,0.22);
  margin-bottom: 1.0rem;
}
.pane-soft{
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 1.15rem 1.15rem;
  box-shadow: 0 10px 22px rgba(0,0,0,0.18);
}

.hero{
  background: linear-gradient(135deg, rgba(102,126,234,0.18), rgba(118,75,162,0.16));
  border: 1px solid rgba(102,126,234,0.28);
  border-radius: 22px;
  padding: 2.4rem 2.2rem;
  box-shadow: 0 14px 30px rgba(0,0,0,0.22);
  margin-bottom: 1.2rem;
}

.divider{
  border-top: 1px solid rgba(255,255,255,0.08);
  margin: 1.8rem 0;
}

/* Metric cards */
div[data-testid="stMetric"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 1.05rem;
  border-radius: 16px;
  box-shadow: 0 10px 20px rgba(0,0,0,0.18);
}
div[data-testid="stMetric"] label{
  color: var(--muted) !important;
}

/* Buttons (less “cheap”, no uppercase) */
.stButton > button, .stDownloadButton > button{
  border-radius: 12px !important;
  padding: 0.70rem 1.1rem !important;
  font-weight: 650 !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.2px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  background: linear-gradient(135deg, rgba(102,126,234,0.88), rgba(118,75,162,0.88)) !important;
  color: white !important;
  box-shadow: 0 10px 20px rgba(0,0,0,0.20);
  transition: all 0.16s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  transform: translateY(-1px);
  border-color: rgba(102,126,234,0.45) !important;
  box-shadow: 0 14px 26px rgba(0,0,0,0.26);
}

/* Chips */
.chips{
  display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;
}
.chip{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 650;
  color: rgba(255,255,255,0.92);
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
}
.chip-strong{
  background: rgba(102,126,234,0.16);
  border: 1px solid rgba(102,126,234,0.30);
}
.chip-warn{
  background: rgba(255,204,102,0.14);
  border: 1px solid rgba(255,204,102,0.28);
}
.chip-neg{
  background: rgba(255,92,108,0.14);
  border: 1px solid rgba(255,92,108,0.28);
}
.chip-ok{
  background: rgba(61,220,151,0.12);
  border: 1px solid rgba(61,220,151,0.26);
}

/* Table container: softer, less “excel” */
div[data-testid="stDataFrame"]{
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 12px 26px rgba(0,0,0,0.22);
}

/* Small helper text */
.small{
  color: var(--muted);
  font-size: 0.92rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def safe_col(df_: pd.DataFrame, col: str) -> bool:
    return col in df_.columns


def slugify(text: str) -> str:
    t = str(text).lower().strip()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    return t.strip("-")


def find_logo(team_name: str) -> Optional[Path]:
    if not team_name or not LOGO_DIR.exists():
        return None
    p = LOGO_DIR / f"{slugify(team_name)}.png"
    return p if p.exists() else None


def find_kit(team_name: str) -> Optional[Path]:
    if not team_name or not KIT_DIR.exists():
        return None
    p = KIT_DIR / f"{slugify(team_name)}.png"
    return p if p.exists() else None


def percentil(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return float((s <= float(value)).mean() * 100)


def _pct_color(pct: float) -> str:
    if np.isnan(pct):
        return "#8a93a3"
    if pct >= 85:
        return "#3ddc97"
    if pct >= 70:
        return "#7ae7b8"
    if pct >= 55:
        return "#ffcc66"
    if pct >= 40:
        return "#ff9f6e"
    return "#ff5c6c"


def pick_player_row(df_: pd.DataFrame, player_name: str) -> dict:
    cand = df_[df_["player_name"].astype(str) == str(player_name)].copy()
    if len(cand) == 0:
        return {}
    if safe_col(cand, "minutes"):
        cand = cand.sort_values("minutes", ascending=False)
    return cand.iloc[0].to_dict()


def fmt_int(n) -> str:
    try:
        return f"{int(round(float(n))):,}".replace(",", " ")
    except Exception:
        return "—"


def fmt_float(x, decimals=2) -> str:
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"


def chip_row(items: List[Tuple[str, str]]) -> None:
    """
    items: list of (label, kind)
      kind in {"strong","ok","warn","neg",""}
    """
    html = '<div class="chips">'
    for text, kind in items:
        cls = "chip"
        if kind:
            cls += f" chip-{kind}"
        html += f'<span class="{cls}">{text}</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def panel_open() -> None:
    st.markdown('<div class="pane">', unsafe_allow_html=True)


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: Optional[List[Tuple[str, str]]] = None) -> None:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="small">{subtitle}</div>', unsafe_allow_html=True)
    if chips:
        chip_row(chips)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# DOMAIN CONFIG
# =============================================================================

# Variabler der lavere er bedre (percentil inverteres)
INVERT_PERCENTILE_COLS = {
    "z_risk_rate_pos",
    "z_fouls_committed_per90_pos",
}

ROLE_FEATURES: Dict[str, List[Tuple[str, str]]] = {
    "playmaker": [
        ("z_passes_key_per90_pos", "Nøkkelpasninger"),
        ("z_passes_accuracy_pos", "Pasningspresisjon"),
        ("z_pass_efficiency_pos", "Pasningseffektivitet"),
        ("z_assists_per90_pos", "Assists"),
        ("z_intensity_pos", "Arbeidsrate"),
    ],
    "ballwinner": [
        ("z_tackles_total_per90_pos", "Taklinger"),
        ("z_interceptions_per90_pos", "Brytninger"),
        ("z_duels_won_per90_pos", "Dueller vunnet"),
        ("z_intensity_pos", "Press/intensitet"),
        ("z_duel_efficiency_pos", "Duelleffektivitet"),
    ],
    "box_to_box": [
        ("z_duels_total_per90_pos", "Dueller totalt"),
        ("z_passes_total_per90_pos", "Pasninger"),
        ("z_tackles_total_per90_pos", "Taklinger"),
        ("z_goals_per90_pos", "Mål"),
        ("z_intensity_pos", "Arbeidsrate"),
    ],
    "finisher": [
        ("z_goals_per90_pos", "Mål per 90"),
        ("z_shot_efficiency_pos", "Måleffektivitet"),
        ("z_shots_on_per90_pos", "Skudd på mål"),
        ("z_shots_total_per90_pos", "Skuddvolum"),
        ("z_dribbles_success_per90_pos", "Driblinger"),
    ],
    "creator_att": [
        ("z_assists_per90_pos", "Assists"),
        ("z_passes_key_per90_pos", "Nøkkelpasninger"),
        ("z_dribbles_success_per90_pos", "Driblinger"),
        ("z_pass_efficiency_pos", "Pasningseffektivitet"),
        ("z_dribble_efficiency_pos", "Driblingeffektivitet"),
    ],
    "pressplayer": [
        ("z_intensity_pos", "Press/intensitet"),
        ("z_duels_total_per90_pos", "Dueller"),
        ("z_tackles_total_per90_pos", "Taklinger"),
        ("z_interceptions_per90_pos", "Brytninger"),
        ("z_fouls_committed_per90_pos", "Feil begått"),
    ],
    "ballplaying_def": [
        ("z_passes_accuracy_pos", "Pasningspresisjon"),
        ("z_passes_total_per90_pos", "Pasningsvolum"),
        ("z_tackles_total_per90_pos", "Taklinger"),
        ("z_interceptions_per90_pos", "Brytninger"),
        ("z_duels_won_per90_pos", "Dueller vunnet"),
    ],
    "stopper": [
        ("z_duels_won_per90_pos", "Dueller vunnet"),
        ("z_tackles_total_per90_pos", "Taklinger"),
        ("z_blocks_per90_pos", "Blokkeringer"),
        ("z_interceptions_per90_pos", "Brytninger"),
        ("z_duel_efficiency_pos", "Duelleffektivitet"),
    ],
    "press_def": [
        ("z_intensity_pos", "Press/intensitet"),
        ("z_tackles_total_per90_pos", "Taklinger"),
        ("z_duels_total_per90_pos", "Dueller"),
        ("z_interceptions_per90_pos", "Brytninger"),
        ("z_blocks_per90_pos", "Blokkeringer"),
    ],
    "shotstopper": [
        ("rating", "Rating"),
    ],
}

KEY_METRICS = [
    ("fair_score", "Prestasjon (fair)", 2),
    ("forecast_score", "Potensial (forecast)", 2),
    ("total_score", "Total score", 2),
    ("reliability", "Reliability", 2),
    ("age_factor", "Aldersfaktor", 2),
]


# =============================================================================
# PLOT BUILDERS
# =============================================================================

def _base_layout(fig: go.Figure, height: int = 560) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#eaeef5"),
    )
    return fig


def create_talent_map(
    df_: pd.DataFrame,
    x_col: str = "fair_score",
    y_col: str = "forecast_score",
    size_col: Optional[str] = "minutes",
    color_col: Optional[str] = "age",
    hover_name: str = "player_name",
) -> go.Figure:
    """
    Hero scatter med kvadranter + labels.
    """
    plot_df = df_.copy()
    # robust size
    if size_col and size_col in plot_df.columns:
        plot_df["_size"] = pd.to_numeric(plot_df[size_col], errors="coerce").fillna(0).clip(lower=0)
    else:
        plot_df["_size"] = 1.0

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        size="_size",
        color=color_col if color_col in plot_df.columns else None,
        hover_name=hover_name if hover_name in plot_df.columns else None,
        hover_data={
            "team_name": True if "team_name" in plot_df.columns else False,
            "pos_group": True if "pos_group" in plot_df.columns else False,
            "minutes": True if "minutes" in plot_df.columns else False,
            "age": True if "age" in plot_df.columns else False,
            x_col: ":.2f",
            y_col: ":.2f",
        },
        color_continuous_scale="Viridis",
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=1, color="rgba(255,255,255,0.25)"),
            opacity=0.90,
        )
    )

    # kvadranter: bruk medianer for robusthet
    x_med = float(pd.to_numeric(plot_df[x_col], errors="coerce").dropna().median()) if x_col in plot_df.columns else 0.0
    y_med = float(pd.to_numeric(plot_df[y_col], errors="coerce").dropna().median()) if y_col in plot_df.columns else 0.0

    fig.add_vline(x=x_med, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.22)")
    fig.add_hline(y=y_med, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.22)")

    # kvadrant labels
    fig.add_annotation(x=x_med, y=plot_df[y_col].max(), text="Høy prestasjon", showarrow=False,
                       yshift=10, font=dict(size=12, color="rgba(255,255,255,0.70)"))
    fig.add_annotation(x=plot_df[x_col].min(), y=y_med, text="Lav prestasjon", showarrow=False,
                       xshift=-10, font=dict(size=12, color="rgba(255,255,255,0.70)"))
    fig.add_annotation(x=plot_df[x_col].max(), y=y_med, text="Høy prestasjon", showarrow=False,
                       xshift=10, font=dict(size=12, color="rgba(255,255,255,0.70)"))
    fig.add_annotation(x=x_med, y=plot_df[y_col].min(), text="Lavt potensial", showarrow=False,
                       yshift=-10, font=dict(size=12, color="rgba(255,255,255,0.70)"))
    fig.add_annotation(
        x=x_med + (plot_df[x_col].max() - plot_df[x_col].min()) * 0.22,
        y=y_med + (plot_df[y_col].max() - plot_df[y_col].min()) * 0.22,
        text="Stabil topp",
        showarrow=False,
        font=dict(size=13, color="rgba(255,255,255,0.78)"),
    )
    fig.add_annotation(
        x=x_med - (plot_df[x_col].max() - plot_df[x_col].min()) * 0.30,
        y=y_med + (plot_df[y_col].max() - plot_df[y_col].min()) * 0.22,
        text="Utviklingscase",
        showarrow=False,
        font=dict(size=13, color="rgba(255,255,255,0.78)"),
    )

    fig.update_xaxes(title_text="Prestasjon (fair score)", zeroline=False, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="Potensial (forecast score)", zeroline=False, gridcolor="rgba(255,255,255,0.06)")

    fig = _base_layout(fig, height=620)
    return fig


def create_percentile_bars(
    ref_df: pd.DataFrame,
    player_row: dict,
    feats: List[Tuple[str, str]],
    title: str,
    max_axes: int = 7,
) -> go.Figure:
    """
    “Club style” alternativ til radar: horisontale percentilbars (0–100).
    """
    items = []
    for col, label in feats[:max_axes]:
        if col not in ref_df.columns:
            continue
        v = player_row.get(col)
        pct = percentil(ref_df[col], v)
        if col in INVERT_PERCENTILE_COLS and pd.notna(pct):
            pct = 100 - pct
        if pd.isna(pct):
            continue
        items.append((label, float(pct)))

    if not items:
        fig = go.Figure()
        fig.add_annotation(text="Ingen rolledata tilgjengelig.", showarrow=False, font=dict(color="rgba(255,255,255,0.7)"))
        return _base_layout(fig, height=360)

    labels = [i[0] for i in items][::-1]
    values = [i[1] for i in items][::-1]

    colors = [_pct_color(v) for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Percentil: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_xaxes(range=[0, 100], gridcolor="rgba(255,255,255,0.06)", zeroline=False, title_text="")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.00)", zeroline=False, title_text="")

    fig.update_layout(
        title=dict(text=title, x=0.0, y=0.98, xanchor="left", font=dict(size=14, color="#eaeef5")),
        bargap=0.35,
        margin=dict(l=10, r=10, t=46, b=10),
        height=360,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig


def create_radar_chart_pro(
    ref_df: pd.DataFrame,
    player_a: dict,
    feats: List[Tuple[str, str]],
    name_a: str,
    player_b: Optional[dict] = None,
    name_b: Optional[str] = None,
    max_axes: int = 7,
) -> go.Figure:
    """
    Pro radar:
    - færre akser (max_axes)
    - lav støy i grid
    - valgfri referanse (posisjon-median) i bakgrunn
    """
    used = []
    for col, label in feats:
        if col in ref_df.columns:
            used.append((col, label))
    used = used[:max_axes]

    if not used:
        fig = go.Figure()
        fig.add_annotation(text="Ingen rolledata tilgjengelig.", showarrow=False)
        return _base_layout(fig, height=520)

    labels = []
    a_vals = []
    b_vals = []
    ref_vals = []

    for col, label in used:
        labels.append(label)

        pct_a = percentil(ref_df[col], player_a.get(col))
        if col in INVERT_PERCENTILE_COLS and pd.notna(pct_a):
            pct_a = 100 - pct_a
        a_vals.append(0.0 if pd.isna(pct_a) else float(pct_a))

        # referanse: median percentil = 50 (men vi kan bruke faktisk median-verdi -> percentil ~50)
        ref_vals.append(50.0)

        if player_b is not None:
            pct_b = percentil(ref_df[col], player_b.get(col))
            if col in INVERT_PERCENTILE_COLS and pd.notna(pct_b):
                pct_b = 100 - pct_b
            b_vals.append(0.0 if pd.isna(pct_b) else float(pct_b))

    theta = labels + [labels[0]]
    a_r = a_vals + [a_vals[0]]
    ref_r = ref_vals + [ref_vals[0]]

    fig = go.Figure()

    # referanse i bakgrunn
    fig.add_trace(
        go.Scatterpolar(
            r=ref_r,
            theta=theta,
            name="Referanse (50p)",
            line=dict(width=1.5, color="rgba(255,255,255,0.22)", dash="dot"),
            fill=None,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=a_r,
            theta=theta,
            name=name_a,
            fill="toself",
            line=dict(width=3.0, color="rgba(102,126,234,0.95)"),
            fillcolor="rgba(102,126,234,0.22)",
            marker=dict(size=6, color="rgba(102,126,234,0.95)"),
        )
    )

    if player_b is not None and name_b:
        b_r = b_vals + [b_vals[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=b_r,
                theta=theta,
                name=name_b,
                fill="toself",
                line=dict(width=3.0, color="rgba(255,92,108,0.95)"),
                fillcolor="rgba(255,92,108,0.18)",
                marker=dict(size=6, color="rgba(255,92,108,0.95)"),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=45, r=45, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 100],
                tickmode="array",
                tickvals=[0, 25, 50, 75, 100],
                tickfont=dict(size=10, color="rgba(255,255,255,0.55)"),
                gridcolor="rgba(255,255,255,0.07)",
                linecolor="rgba(255,255,255,0.10)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="rgba(255,255,255,0.90)"),
                gridcolor="rgba(255,255,255,0.07)",
                linecolor="rgba(255,255,255,0.10)",
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.25)",
            bordercolor="rgba(255,255,255,0.10)",
            borderwidth=1,
            font=dict(size=12),
        ),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# =============================================================================
# UI: TOP FILTER BAR
# =============================================================================

@dataclass
class Filters:
    datasett: Optional[List[str]]
    pos: str
    min_minutes: int
    age_range: Optional[Tuple[int, int]]
    team_query: str
    player_query: str
    role: str


def render_top_filters(df: pd.DataFrame) -> Filters:
    """
    Topbar filters (kompakt). Sidebar er kun navigasjon.
    """
    with st.container():
        panel_open()
        st.markdown("### Filtre")
        c1, c2, c3, c4, c5, c6 = st.columns([1.25, 0.8, 1.0, 1.2, 1.3, 1.0])

        # datasett
        datasett_valg = None
        if safe_col(df, "datasett"):
            alle = sorted(df["datasett"].dropna().unique().tolist())
            with c1:
                datasett_valg = st.multiselect("Datasett", alle, default=alle)

        # pos
        with c2:
            pos_filter = st.selectbox("Posisjon", ["Alle", "MID", "ATT", "DEF", "GK"], index=0)

        # minutes
        with c3:
            if safe_col(df, "minutes"):
                min_minutes = st.slider("Minutter (min)", 0, int(df["minutes"].max()), 450, step=50)
            else:
                min_minutes = 0

        # age
        with c4:
            if safe_col(df, "age"):
                age_min = int(df["age"].min())
                age_max = int(df["age"].max())
                age_range = st.slider("Alder", age_min, age_max, (age_min, age_max))
            else:
                age_range = None

        # team search
        with c5:
            team_filter = st.text_input("Lag", value="", placeholder="Søk lag…")

        # player search
        with c6:
            player_search = st.text_input("Spiller", value="", placeholder="Søk spiller…")

        # rolleprofil (egen rad)
        r1, r2 = st.columns([1.2, 3.0])
        with r1:
            role = st.selectbox(
                "Rolleprofil",
                list(ROLE_FEATURES.keys()),
                format_func=lambda x: x.replace("_", " ").title(),
            )
        with r2:
            st.markdown(
                '<div class="small">Tips: bruk Rolleprofil for å rangere og sammenligne spillere på taktisk profil. '
                "Percentil-bars er standard (raskest å lese). Radar finnes som alternativ.</div>",
                unsafe_allow_html=True,
            )

        panel_close()

    return Filters(
        datasett=datasett_valg,
        pos=pos_filter,
        min_minutes=min_minutes,
        age_range=age_range,
        team_query=team_filter.strip(),
        player_query=player_search.strip(),
        role=role,
    )


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()

    if f.datasett and safe_col(out, "datasett"):
        out = out[out["datasett"].isin(f.datasett)].copy()

    if f.pos != "Alle" and safe_col(out, "pos_group"):
        out = out[out["pos_group"] == f.pos].copy()

    if safe_col(out, "minutes"):
        out = out[out["minutes"] >= int(f.min_minutes)].copy()

    if f.age_range and safe_col(out, "age"):
        out = out[(out["age"] >= f.age_range[0]) & (out["age"] <= f.age_range[1])].copy()

    if f.team_query and safe_col(out, "team_name"):
        out = out[out["team_name"].astype(str).str.contains(f.team_query, case=False, na=False)].copy()

    if f.player_query and safe_col(out, "player_name"):
        out = out[out["player_name"].astype(str).str.contains(f.player_query, case=False, na=False)].copy()

    return out


def filter_chips(f: Filters, n_players: int) -> List[Tuple[str, str]]:
    chips = [("Utvalg", "strong"), (f"{n_players} spillere", "")]
    if f.pos != "Alle":
        chips.append((f"Pos: {f.pos}", "strong"))
    if f.age_range:
        chips.append((f"Alder: {f.age_range[0]}–{f.age_range[1]}", ""))
    chips.append((f"Min: {f.min_minutes}", ""))
    chips.append((f"Rolle: {f.role.replace('_', ' ').title()}", ""))
    return chips


# =============================================================================
# TABLE HELPERS (less excel)
# =============================================================================

def scouting_table(
    df_: pd.DataFrame,
    title: str,
    subtitle: str,
    cols: List[str],
    height: int = 420,
    sort_col: Optional[str] = None,
    descending: bool = True,
    show_search_hint: bool = True,
) -> None:
    panel_open()
    st.markdown(f"### {title}")
    st.markdown(f'<div class="small">{subtitle}</div>', unsafe_allow_html=True)

    view = df_.copy()
    if sort_col and sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=not descending)

    view = view[cols].copy()

    # basic formatting
    col_config = {}
    if "fair_score" in view.columns:
        col_config["fair_score"] = st.column_config.NumberColumn("Fair", format="%.2f")
    if "forecast_score" in view.columns:
        col_config["forecast_score"] = st.column_config.NumberColumn("Forecast", format="%.2f")
    if "total_score" in view.columns:
        col_config["total_score"] = st.column_config.NumberColumn("Total", format="%.2f")
    if "reliability" in view.columns:
        col_config["reliability"] = st.column_config.NumberColumn("Rel.", format="%.2f")
    if "minutes" in view.columns:
        col_config["minutes"] = st.column_config.NumberColumn("Min", format="%.0f")
    if "age" in view.columns:
        col_config["age"] = st.column_config.NumberColumn("Alder", format="%.0f")

    st.dataframe(
        view.reset_index(drop=True),
        use_container_width=True,
        height=height,
        hide_index=True,
        column_config=col_config,
    )

    if show_search_hint:
        st.markdown('<div class="small">Tips: bruk spiller-/lag-søk i filterlinjen for å fokusere.</div>', unsafe_allow_html=True)

    panel_close()


# =============================================================================
# SESSION STATE
# =============================================================================

if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = []  # list of tuples (player_name, team_name)


def add_to_shortlist(player_name: str, team_name: str) -> None:
    key = (str(player_name), str(team_name))
    if key not in st.session_state["shortlist"]:
        st.session_state["shortlist"].append(key)
        st.toast(f"Lagt til: {player_name}", icon="✅")


def remove_from_shortlist(player_name: str, team_name: str) -> None:
    key = (str(player_name), str(team_name))
    st.session_state["shortlist"] = [k for k in st.session_state["shortlist"] if k != key]


# =============================================================================
# LOAD DATA
# =============================================================================

if not PARQUET_PATH.exists():
    st.error(
        "Datagrunnlag mangler.\n\n"
        f"Finner ikke: {PARQUET_PATH}\n\n"
        "Løsning: bygg parquet-filen først og last inn siden på nytt."
    )
    st.stop()

try:
    df = load_data(PARQUET_PATH)
except Exception as e:
    st.error(f"Kunne ikke laste data: {e}")
    st.stop()

# =============================================================================
# SIDEBAR NAV (kun navigasjon)
# =============================================================================

with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Profesjonelt scouting-verktøy")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigasjon",
        [
            "Forside",
            "Dashboard",
            "Spillerkort",
            "Duell",
            "Scout report",
            "Spillerstiler",
            "Transfer tracker",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption(f"Shortlist: {len(st.session_state.get('shortlist', []))}")
    if safe_col(df, "team_name"):
        st.caption(f"Lag i datasett: {df['team_name'].nunique():,}".replace(",", " "))
    st.caption(f"Spillere i datasett: {len(df):,}".replace(",", " "))


# =============================================================================
# TOP FILTERS (for alle sider som trenger det)
# =============================================================================

filters = render_top_filters(df)
filtered_df = apply_filters(df, filters)

# =============================================================================
# PAGE: FORSIDE (produktpresentasjon)
# =============================================================================

if page == "Forside":
    hero(
        "Eliteserien DSS",
        "Beslutningsstøtte for talentidentifisering, rolle-match og rekruttering – kurert for klubbbruk.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Spillere (utvalg)", f"{len(filtered_df):,}".replace(",", " "))
    if safe_col(filtered_df, "team_name"):
        k2.metric("Lag", f"{filtered_df['team_name'].nunique():,}".replace(",", " "))
    if safe_col(filtered_df, "minutes") and len(filtered_df) > 0:
        k3.metric("Snitt minutter", fmt_int(filtered_df["minutes"].mean()))
    if safe_col(filtered_df, "age") and len(filtered_df) > 0:
        k4.metric("Snitt alder", f"{filtered_df['age'].mean():.1f} år")
    if safe_col(filtered_df, "total_score") and len(filtered_df) > 0:
        k5.metric("Snitt total", f"{filtered_df['total_score'].mean():.2f}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Curated shortlists
    left, right = st.columns([1.05, 1.05])

    with left:
        if safe_col(filtered_df, "fair_score"):
            view = filtered_df.copy()
            cols = [c for c in ["player_name", "team_name", "pos_group", "minutes", "fair_score", "reliability"] if safe_col(view, c)]
            scouting_table(
                view.nlargest(5, "fair_score"),
                "Rekrutteringsshortlist – prestasjon (nå)",
                "Top 5 fair score i utvalget. Bruk spillerkort for dyp evaluering.",
                cols=cols,
                height=260,
                sort_col="fair_score",
            )
        else:
            panel_open()
            st.markdown("### Rekrutteringsshortlist – prestasjon (nå)")
            st.info("Mangler kolonnen 'fair_score' i datasettet.")
            panel_close()

    with right:
        if safe_col(filtered_df, "forecast_score") and safe_col(filtered_df, "age"):
            view = filtered_df[filtered_df["age"] <= 23].copy()
            cols = [c for c in ["player_name", "age", "team_name", "pos_group", "minutes", "forecast_score", "reliability"] if safe_col(view, c)]
            scouting_table(
                view.nlargest(5, "forecast_score"),
                "Rekrutteringsshortlist – potensial (U23)",
                "Top 5 forecast score (U23) i utvalget.",
                cols=cols,
                height=260,
                sort_col="forecast_score",
            )
        else:
            panel_open()
            st.markdown("### Rekrutteringsshortlist – potensial (U23)")
            st.info("Mangler 'forecast_score' og/eller 'age' i datasettet.")
            panel_close()

    # Hero visual: Talent map
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    panel_open()
    st.markdown("### Markedskart (Talent Map)")
    st.markdown('<div class="small">Prestasjon vs potensial. Kvadranter brukes til rask sortering av kandidattyper.</div>', unsafe_allow_html=True)

    if safe_col(filtered_df, "fair_score") and safe_col(filtered_df, "forecast_score"):
        fig = create_talent_map(
            filtered_df,
            x_col="fair_score",
            y_col="forecast_score",
            size_col="minutes" if safe_col(filtered_df, "minutes") else None,
            color_col="age" if safe_col(filtered_df, "age") else None,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="small">Tolkning: <b>Utviklingscase</b> = lav prestasjon / høyt potensial. '
            '<b>Stabil topp</b> = høy/høy. Bruk Duell for sammenligning og Scout report for U23-søk.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Mangler 'fair_score' og/eller 'forecast_score' i datasettet.")

    panel_close()


# =============================================================================
# PAGE: DASHBOARD (analytiker: map + shortlist)
# =============================================================================

elif page == "Dashboard":
    hero(
        "Dashboard",
        "Finn kandidater raskt, og gå videre til spillerkort eller duell for beslutning.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    if len(filtered_df) == 0:
        panel_open()
        st.warning("Ingen spillere i utvalget. Juster filtre.")
        panel_close()
        st.stop()

    left, right = st.columns([1.75, 1.0], gap="large")

    with left:
        panel_open()
        st.markdown("### Talent Map")
        st.markdown('<div class="small">Klikk i kartet (hover) for detaljer. Bruk filter for fokus.</div>', unsafe_allow_html=True)

        if safe_col(filtered_df, "fair_score") and safe_col(filtered_df, "forecast_score"):
            fig = create_talent_map(
                filtered_df,
                x_col="fair_score",
                y_col="forecast_score",
                size_col="minutes" if safe_col(filtered_df, "minutes") else None,
                color_col="age" if safe_col(filtered_df, "age") else None,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Mangler 'fair_score' og/eller 'forecast_score' i datasettet.")
        panel_close()

    with right:
        # Shortlist summary + curated lists
        panel_open()
        st.markdown("### Inngang til evaluering")

        a, b = st.columns(2)
        with a:
            st.metric("Shortlist", f"{len(st.session_state.get('shortlist', []))}")
        with b:
            if safe_col(filtered_df, "reliability"):
                st.metric("Snitt reliability", f"{filtered_df['reliability'].mean():.2f}")
            else:
                st.metric("Snitt reliability", "—")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Value candidates: høy forecast, lav fair
        if safe_col(filtered_df, "fair_score") and safe_col(filtered_df, "forecast_score"):
            value = filtered_df.copy()
            value["value_gap"] = value["forecast_score"] - value["fair_score"]
            cols = [c for c in ["player_name", "team_name", "pos_group", "age", "minutes", "fair_score", "forecast_score", "value_gap"] if safe_col(value, c)]
            scouting_table(
                value.nlargest(8, "value_gap"),
                "Utviklingscase (value gap)",
                "Høyt potensial relativt til prestasjon. Bruk spillerkort for rolleprofil.",
                cols=cols,
                height=320,
                sort_col="value_gap",
            )
        else:
            st.info("Mangler 'fair_score'/'forecast_score' for value-gap.")

        panel_close()

    # Role match leaderboard
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    panel_open()
    st.markdown("### Rolle-match (rangering)")
    st.markdown(f'<div class="small">Rangert etter rolleprofil: <b>{filters.role.replace("_"," ").title()}</b> (percentil-basert).</div>', unsafe_allow_html=True)

    feats = ROLE_FEATURES.get(filters.role, [])
    feat_cols = [c for c, _ in feats if safe_col(filtered_df, c)]

    if len(feat_cols) == 0:
        st.info("Mangler nødvendige rollekolonner i datasettet.")
        panel_close()
    else:
        # reference group: pos if valgt, ellers alt
        ref_df = df.copy()
        if filters.pos != "Alle" and safe_col(ref_df, "pos_group"):
            ref_df = ref_df[ref_df["pos_group"] == filters.pos].copy()

        def role_score(row: pd.Series) -> float:
            pts = []
            for col, _label in feats[:7]:
                if col not in ref_df.columns:
                    continue
                pct = percentil(ref_df[col], row.get(col))
                if col in INVERT_PERCENTILE_COLS and pd.notna(pct):
                    pct = 100 - pct
                if pd.notna(pct):
                    pts.append(float(pct))
            return float(np.mean(pts)) if pts else np.nan

        ranked = filtered_df.copy()
        ranked["role_match"] = ranked.apply(role_score, axis=1)

        show_cols = [c for c in [
            "player_name", "team_name", "pos_group", "age", "minutes",
            "role_match", "fair_score", "forecast_score", "reliability"
        ] if safe_col(ranked, c)]

        if "role_match" in ranked.columns:
            ranked = ranked.sort_values("role_match", ascending=False)

        st.dataframe(
            ranked.head(15)[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=480,
            hide_index=True,
            column_config={
                "role_match": st.column_config.NumberColumn("Rolle-match", format="%.1f"),
                "minutes": st.column_config.NumberColumn("Min", format="%.0f"),
                "fair_score": st.column_config.NumberColumn("Fair", format="%.2f"),
                "forecast_score": st.column_config.NumberColumn("Forecast", format="%.2f"),
                "reliability": st.column_config.NumberColumn("Rel.", format="%.2f"),
                "age": st.column_config.NumberColumn("Alder", format="%.0f"),
            }
        )

        panel_close()


# =============================================================================
# PAGE: SPILLERKORT
# =============================================================================

elif page == "Spillerkort":
    hero(
        "Spillerkort",
        "Detaljert spilleranalyse med rolleprofil, percentiler og nøkkelmetrikker.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    if not safe_col(filtered_df, "player_name") or len(filtered_df) == 0:
        panel_open()
        st.warning("Ingen spillere i utvalget. Juster filtre.")
        panel_close()
        st.stop()

    spillerliste = filtered_df["player_name"].dropna().astype(str).sort_values().unique().tolist()
    selected_player = st.selectbox("Velg spiller", spillerliste, index=0)

    player_data = pick_player_row(filtered_df, selected_player)
    if not player_data:
        st.warning("Kunne ikke hente data for valgt spiller.")
        st.stop()

    # Header panel
    panel_open()
    c1, c2, c3 = st.columns([0.9, 2.1, 1.2], gap="large")

    with c1:
        kit_path = find_kit(player_data.get("team_name", ""))
        logo_path = find_logo(player_data.get("team_name", "")) if kit_path is None else None
        if kit_path:
            st.image(str(kit_path), width=160)
        elif logo_path:
            st.image(str(logo_path), width=160)
        else:
            st.markdown("")

    with c2:
        st.markdown(f"## {player_data.get('player_name', 'Ukjent')}")
        chips = []
        if safe_col(df, "team_name"):
            chips.append((str(player_data.get("team_name", "—")), "strong"))
        if safe_col(df, "pos_group") and pd.notna(player_data.get("pos_group")):
            chips.append((str(player_data.get("pos_group", "—")), ""))
        if safe_col(df, "age") and pd.notna(player_data.get("age")):
            age = int(player_data.get("age"))
            kind = "ok" if age <= 21 else ("warn" if age <= 25 else "")
            chips.append((f"{age} år", kind))
        if safe_col(df, "best_role") and pd.notna(player_data.get("best_role")):
            chips.append((str(player_data.get("best_role")).replace("_", " ").title(), ""))

        chip_row(chips)

        st.markdown(
            f'<div class="small"><b>Minutter:</b> {fmt_int(player_data.get("minutes", 0))} '
            f'&nbsp;&nbsp; <b>Kamper:</b> {player_data.get("apps", "—")} '
            f'&nbsp;&nbsp; <b>Starter:</b> {player_data.get("starts", "—")}</div>',
            unsafe_allow_html=True,
        )

    with c3:
        if st.button("Legg til i shortlist", use_container_width=True):
            add_to_shortlist(player_data.get("player_name"), player_data.get("team_name"))

        csv_data = pd.DataFrame([player_data]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Eksporter spillerdata (CSV)",
            csv_data,
            file_name=f"{slugify(selected_player)}_spillerkort.csv",
            mime="text/csv",
            use_container_width=True,
        )

    panel_close()

    # Key metrics
    panel_open()
    st.markdown("### Nøkkelmetrikker")
    mcols = st.columns(5)
    for i, (col, label, dec) in enumerate(KEY_METRICS):
        if safe_col(df, col):
            val = player_data.get(col)
            if pd.notna(val):
                mcols[i].metric(label, fmt_float(val, dec))
            else:
                mcols[i].metric(label, "—")
    panel_close()

    # Role profile: bars default + radar optional
    feats = ROLE_FEATURES.get(filters.role, [])
    feat_cols = [c for c, _ in feats if safe_col(df, c)]

    # reference group: same pos_group as player (best practice)
    ref_df = df.copy()
    if safe_col(ref_df, "pos_group") and pd.notna(player_data.get("pos_group")):
        ref_df = ref_df[ref_df["pos_group"] == player_data.get("pos_group")].copy()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        panel_open()
        st.markdown("### Rolleprofil (lesbar default)")
        st.markdown(f'<div class="small">{filters.role.replace("_"," ").title()} – percentiler vs samme posisjon.</div>', unsafe_allow_html=True)

        if len(feat_cols) == 0 or len(ref_df) == 0:
            st.info("Mangler rolledata for valgt rolle/posisjon.")
        else:
            # Tabs: Bars first, Radar second
            t1, t2 = st.tabs(["Percentil-bars", "Radar (alternativ)"])

            with t1:
                fig = create_percentile_bars(ref_df, player_data, feats, title="", max_axes=7)
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                fig = create_radar_chart_pro(
                    ref_df=ref_df,
                    player_a=player_data,
                    feats=feats,
                    name_a=selected_player,
                    max_axes=7,
                )
                st.plotly_chart(fig, use_container_width=True)

        panel_close()

    with right:
        panel_open()
        st.markdown("### Rask vurdering")
        st.markdown('<div class="small">Takeaways basert på sterkeste/svakeste percentiler (rolle).</div>', unsafe_allow_html=True)

        take = []
        for col, label in feats[:7]:
            if col not in ref_df.columns:
                continue
            pct = percentil(ref_df[col], player_data.get(col))
            if col in INVERT_PERCENTILE_COLS and pd.notna(pct):
                pct = 100 - pct
            if pd.notna(pct):
                take.append((label, float(pct)))

        if take:
            take_sorted = sorted(take, key=lambda x: x[1], reverse=True)
            top3 = take_sorted[:3]
            bot3 = take_sorted[-3:][::-1]

            st.markdown("**Styrker**")
            for lbl, pct in top3:
                st.markdown(f"- {lbl}: **{pct:.1f}%**")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Utviklingsområder**")
            for lbl, pct in bot3:
                st.markdown(f"- {lbl}: **{pct:.1f}%**")
        else:
            st.info("Ingen takeaways tilgjengelig.")

        panel_close()


# =============================================================================
# PAGE: DUELL
# =============================================================================

elif page == "Duell":
    hero(
        "Duell",
        "Head-to-head sammenligning med rolleprofil og differanser.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    if len(filtered_df) < 2 or not safe_col(filtered_df, "player_name"):
        panel_open()
        st.warning("Minst 2 spillere kreves i utvalget for sammenligning.")
        panel_close()
        st.stop()

    players = filtered_df["player_name"].dropna().astype(str).sort_values().unique().tolist()
    if len(players) < 2:
        panel_open()
        st.warning("For få spillere i utvalg.")
        panel_close()
        st.stop()

    a_col, b_col = st.columns(2)
    with a_col:
        player_a = st.selectbox("Spiller A", players, index=0, key="duel_a")
    with b_col:
        player_b = st.selectbox("Spiller B", players, index=min(1, len(players) - 1), key="duel_b")

    data_a = pick_player_row(filtered_df, player_a)
    data_b = pick_player_row(filtered_df, player_b)

    # cards
    c1, c2 = st.columns(2, gap="large")
    for col, name, row in [(c1, player_a, data_a), (c2, player_b, data_b)]:
        with col:
            panel_open()
            st.markdown(f"### {name}")
            chip_items = []
            if safe_col(df, "team_name"):
                chip_items.append((str(row.get("team_name", "—")), "strong"))
            if safe_col(df, "pos_group"):
                chip_items.append((str(row.get("pos_group", "—")), ""))
            if safe_col(df, "age") and pd.notna(row.get("age")):
                age = int(row.get("age"))
                kind = "ok" if age <= 21 else ("warn" if age <= 25 else "")
                chip_items.append((f"{age} år", kind))
            chip_row(chip_items)

            st.markdown(f'<div class="small"><b>Minutter:</b> {fmt_int(row.get("minutes", 0))}</div>', unsafe_allow_html=True)
            if safe_col(df, "fair_score"):
                st.markdown(f'<div class="small"><b>Fair:</b> {fmt_float(row.get("fair_score", np.nan), 2)}</div>', unsafe_allow_html=True)
            if safe_col(df, "forecast_score"):
                st.markdown(f'<div class="small"><b>Forecast:</b> {fmt_float(row.get("forecast_score", np.nan), 2)}</div>', unsafe_allow_html=True)
            panel_close()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    feats = ROLE_FEATURES.get(filters.role, [])
    # reference group: same pos_group as A (assume same pos comparison)
    ref_df = df.copy()
    if safe_col(ref_df, "pos_group") and pd.notna(data_a.get("pos_group")):
        ref_df = ref_df[ref_df["pos_group"] == data_a.get("pos_group")].copy()

    # compute pct and diffs
    labels = []
    vals_a = []
    vals_b = []
    diff_rows = []

    for col, label in feats[:7]:
        if col not in ref_df.columns:
            continue

        pct_a = percentil(ref_df[col], data_a.get(col))
        pct_b = percentil(ref_df[col], data_b.get(col))

        if col in INVERT_PERCENTILE_COLS:
            if pd.notna(pct_a): pct_a = 100 - pct_a
            if pd.notna(pct_b): pct_b = 100 - pct_b

        if pd.isna(pct_a) or pd.isna(pct_b):
            continue

        labels.append(label)
        vals_a.append(float(pct_a))
        vals_b.append(float(pct_b))

        diff_rows.append({
            "Metrikk": label,
            f"{player_a} (pct)": round(float(pct_a), 1),
            f"{player_b} (pct)": round(float(pct_b), 1),
            "Differanse (A-B)": round(float(pct_a - pct_b), 1),
        })

    rcol, dcol = st.columns([1.4, 1.0], gap="large")

    with rcol:
        panel_open()
        st.markdown("### Rolleprofil – sammenligning")
        st.markdown(f'<div class="small">{filters.role.replace("_"," ").title()} (maks 7 akser) + referanse.</div>', unsafe_allow_html=True)

        if labels:
            fig = create_radar_chart_pro(
                ref_df=ref_df,
                player_a=data_a,
                player_b=data_b,
                feats=feats,
                name_a=player_a,
                name_b=player_b,
                max_axes=7,
            )
            st.plotly_chart(fig, use_container_width=True)

            # add quick takeaways
            diffs = [(lbl, a - b) for lbl, a, b in zip(labels, vals_a, vals_b)]
            diffs_sorted = sorted(diffs, key=lambda x: x[1], reverse=True)
            best = diffs_sorted[:2]
            worst = diffs_sorted[-2:][::-1]

            st.markdown("**Takeaways**")
            if best:
                st.markdown(f"- {player_a} over {player_b} på: " + ", ".join([f"**{lbl}**" for lbl, _ in best]))
            if worst:
                st.markdown(f"- {player_b} over {player_a} på: " + ", ".join([f"**{lbl}**" for lbl, _ in worst]))
        else:
            st.info("Ingen rolledata tilgjengelig for sammenligning.")
        panel_close()

    with dcol:
        panel_open()
        st.markdown("### Differanser")
        st.markdown('<div class="small">Positiv verdi betyr at Spiller A er bedre på metrikken (percentil).</div>', unsafe_allow_html=True)

        if diff_rows:
            diff_df = pd.DataFrame(diff_rows).sort_values("Differanse (A-B)", ascending=False)
            st.dataframe(
                diff_df,
                use_container_width=True,
                height=520,
                hide_index=True,
                column_config={
                    "Differanse (A-B)": st.column_config.NumberColumn("Diff (A-B)", format="%.1f"),
                }
            )
        else:
            st.info("Ingen differanser å vise.")
        panel_close()


# =============================================================================
# PAGE: SCOUT REPORT
# =============================================================================

elif page == "Scout report":
    hero(
        "Scout report",
        "Finn unge talenter (U25) med styrt sortering og tydelig shortlist.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    panel_open()
    st.markdown("### Kriterier")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_age = st.slider("Maks alder", 17, 25, 21)
    with c2:
        min_mins = st.slider("Min minutter", 0, 2500, 450, step=50)
    with c3:
        sort_map = {"forecast_score": "Potensial (forecast)", "fair_score": "Prestasjon (fair)", "total_score": "Total"}
        sort_by = st.selectbox("Sorter etter", list(sort_map.keys()), format_func=lambda x: sort_map[x])
    panel_close()

    talent_df = filtered_df.copy()
    if safe_col(talent_df, "age"):
        talent_df = talent_df[talent_df["age"] <= max_age].copy()
    if safe_col(talent_df, "minutes"):
        talent_df = talent_df[talent_df["minutes"] >= min_mins].copy()

    if len(talent_df) == 0:
        panel_open()
        st.warning("Ingen talenter funnet. Juster kriteriene.")
        panel_close()
        st.stop()

    if safe_col(talent_df, sort_by):
        talent_df = talent_df.sort_values(sort_by, ascending=False).copy()

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Talenter funnet", f"{len(talent_df):,}".replace(",", " "))
    k2.metric("Snitt alder", f"{talent_df['age'].mean():.1f} år" if safe_col(talent_df, "age") else "—")
    k3.metric("Snitt forecast", f"{talent_df['forecast_score'].mean():.2f}" if safe_col(talent_df, "forecast_score") else "—")
    k4.metric("Lag", f"{talent_df['team_name'].nunique():,}".replace(",", " ") if safe_col(talent_df, "team_name") else "—")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    cols = [c for c in [
        "player_name", "age", "team_name", "pos_group", "best_role",
        "minutes", "fair_score", "forecast_score", "reliability"
    ] if safe_col(talent_df, c)]

    scouting_table(
        talent_df.head(25),
        "Topp-talenter",
        f"Sortert etter: {sort_map.get(sort_by, sort_by)}",
        cols=cols,
        height=620,
        sort_col=sort_by,
    )

    # Visual: Talent map for talents
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    panel_open()
    st.markdown("### Talent Map – U25")
    st.markdown('<div class="small">Fokus på U25-utvalget. Bruk kvadranter for å se utviklingscase vs stabile topp.</div>', unsafe_allow_html=True)
    if safe_col(talent_df, "fair_score") and safe_col(talent_df, "forecast_score"):
        fig = create_talent_map(
            talent_df.head(150),
            x_col="fair_score",
            y_col="forecast_score",
            size_col="minutes" if safe_col(talent_df, "minutes") else None,
            color_col="reliability" if safe_col(talent_df, "reliability") else None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Mangler 'fair_score'/'forecast_score'.")
    panel_close()


# =============================================================================
# PAGE: SPILLERSTILER (klynger)
# =============================================================================

elif page == "Spillerstiler":
    hero(
        "Spillerstiler",
        "Klynger brukes for å finne spillertyper og lignende profiler. PCA er kun støtte.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    if not safe_col(df, "cluster"):
        panel_open()
        st.info("Spillerstiler krever at klyngeanalyse er lagret i datasettet (kolonnen 'cluster').")
        panel_close()
        st.stop()

    # Position for cluster analysis
    panel_open()
    st.markdown("### Velg analyse")
    p1, p2 = st.columns([1.0, 2.0])
    with p1:
        pos_for_cluster = st.selectbox("Posisjon", ["MID", "ATT", "DEF"], index=0)
    with p2:
        st.markdown(
            '<div class="small">Klynger tolkes som stilrom innen posisjon (ikke kvalitet). '
            "Bruk listen for å identifisere kandidater, og spillerkort/duell for evaluering.</div>",
            unsafe_allow_html=True,
        )
    panel_close()

    cluster_df = filtered_df.copy()
    if safe_col(cluster_df, "pos_group"):
        cluster_df = cluster_df[cluster_df["pos_group"] == pos_for_cluster].copy()

    if len(cluster_df) == 0:
        panel_open()
        st.warning("Ingen spillere funnet for valgt posisjon.")
        panel_close()
        st.stop()

    clusters = sorted([c for c in cluster_df["cluster"].dropna().unique().tolist()])
    if len(clusters) == 0:
        panel_open()
        st.warning("Ingen klynger tilgjengelig i filtrert utvalg.")
        panel_close()
        st.stop()

    panel_open()
    st.markdown("### Klynge")
    selected_cluster = st.selectbox("Velg klynge", clusters, format_func=lambda x: f"Klynge {int(x)}")
    panel_close()

    members = cluster_df[cluster_df["cluster"] == selected_cluster].copy()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Spillere", f"{len(members):,}".replace(",", " "))
    k2.metric("Snitt alder", f"{members['age'].mean():.1f} år" if safe_col(members, "age") else "—")
    k3.metric("Snitt minutter", fmt_int(members["minutes"].mean()) if safe_col(members, "minutes") else "—")
    k4.metric("Snitt fair", f"{members['fair_score'].mean():.2f}" if safe_col(members, "fair_score") else "—")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Members table
    cols = [c for c in ["player_name", "team_name", "age", "minutes", "best_role", "fair_score", "forecast_score", "reliability"] if safe_col(members, c)]
    if safe_col(members, "fair_score"):
        members = members.sort_values("fair_score", ascending=False)

    scouting_table(
        members,
        f"Spillere i klynge {int(selected_cluster)}",
        "Bruk dette som “similar players”-liste. Klikk videre via spiller-søk/Spillerkort.",
        cols=cols,
        height=560,
        sort_col="fair_score" if safe_col(members, "fair_score") else None,
    )

    # PCA as support (collapsed)
    with st.expander("PCA (støttevisual)", expanded=False):
        panel_open()
        st.markdown("### Stilrom (PCA)")
        st.markdown('<div class="small">Dette er en støttevisning – ikke en “wow gimmick”. Bruk for å se avstander i stilrom.</div>', unsafe_allow_html=True)

        try:
            from sklearn.decomposition import PCA

            cluster_features = [
                c for c in [
                    "z_passes_key_per90_pos",
                    "z_passes_accuracy_pos",
                    "z_duels_total_per90_pos",
                    "z_interceptions_per90_pos",
                    "z_intensity_pos",
                    "z_goals_per90_pos",
                    "z_assists_per90_pos",
                ]
                if safe_col(cluster_df, c)
            ]

            if len(cluster_features) >= 2:
                X = cluster_df[cluster_features].dropna()
                idx = X.index

                pca = PCA(n_components=2, random_state=42)
                Z = pca.fit_transform(X.values)

                plot_df = pd.DataFrame(
                    {
                        "PC1": Z[:, 0],
                        "PC2": Z[:, 1],
                        "Klynge": cluster_df.loc[idx, "cluster"].astype(int).astype(str),
                        "Spiller": cluster_df.loc[idx, "player_name"].astype(str),
                    }
                )

                fig = px.scatter(
                    plot_df,
                    x="PC1",
                    y="PC2",
                    color="Klynge",
                    hover_data=["Spiller"],
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                )
                fig.update_traces(marker=dict(size=10, line=dict(width=1, color="rgba(255,255,255,0.25)")))
                fig.update_layout(
                    template="plotly_dark",
                    height=560,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Taktisk dimensjon 1",
                    yaxis_title="Taktisk dimensjon 2",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Ikke nok features tilgjengelig for PCA.")
        except Exception as e:
            st.warning(f"Kunne ikke lage PCA: {e}")

        panel_close()


# =============================================================================
# PAGE: TRANSFER TRACKER (Shortlist)
# =============================================================================

elif page == "Transfer tracker":
    hero(
        "Transfer tracker",
        "Shortlist for rekrutteringsarbeid. Eksporter til CSV/Excel.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    if len(st.session_state["shortlist"]) == 0:
        panel_open()
        st.info("Shortlist er tom. Legg til spillere fra Spillerkort.")
        panel_close()
        st.stop()

    shortlist_keys = st.session_state["shortlist"]
    shortlist_df = df.copy()

    mask = shortlist_df.apply(
        lambda r: (str(r.get("player_name")), str(r.get("team_name"))) in shortlist_keys,
        axis=1,
    )
    shortlist_df = shortlist_df[mask].copy()

    if len(shortlist_df) == 0:
        panel_open()
        st.warning("Ingen spillere i shortlist (match ikke funnet i datasett).")
        panel_close()
        st.stop()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Spillere", f"{len(shortlist_df):,}".replace(",", " "))
    k2.metric("Snitt alder", f"{shortlist_df['age'].mean():.1f} år" if safe_col(shortlist_df, "age") else "—")
    k3.metric("Snitt fair", f"{shortlist_df['fair_score'].mean():.2f}" if safe_col(shortlist_df, "fair_score") else "—")
    k4.metric("Snitt forecast", f"{shortlist_df['forecast_score'].mean():.2f}" if safe_col(shortlist_df, "forecast_score") else "—")
    k5.metric("Lag", f"{shortlist_df['team_name'].nunique():,}".replace(",", " ") if safe_col(shortlist_df, "team_name") else "—")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    cols = [c for c in [
        "player_name", "age", "team_name", "pos_group", "best_role",
        "minutes", "fair_score", "forecast_score", "reliability"
    ] if safe_col(shortlist_df, c)]

    if safe_col(shortlist_df, "forecast_score"):
        shortlist_df = shortlist_df.sort_values("forecast_score", ascending=False)

    scouting_table(
        shortlist_df,
        "Shortlist",
        "Dette er din operative målliste. Bruk Duell/Spillerkort for dyp vurdering.",
        cols=cols,
        height=560,
        sort_col="forecast_score" if safe_col(shortlist_df, "forecast_score") else None,
    )

    panel_open()
    a1, a2, a3 = st.columns([1.2, 1.2, 0.8])
    with a1:
        csv_data = shortlist_df[cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Eksporter (CSV)",
            csv_data,
            file_name="transfer_tracker.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with a2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            shortlist_df[cols].to_excel(writer, index=False, sheet_name="TransferTracker")
        st.download_button(
            "Eksporter (Excel)",
            data=output.getvalue(),
            file_name="transfer_tracker.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with a3:
        if st.button("Tøm", use_container_width=True):
            st.session_state["shortlist"] = []
            st.rerun()
    panel_close()


# =============================================================================
# FOOTER
# =============================================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div style="text-align:center; color: rgba(255,255,255,0.45); font-size:0.9rem; padding: 0.6rem 0;">
  <div><strong>{APP_TITLE}</strong> © 2025 | Masteroppgave | kulmiyearab</div>
  <div style="font-size:0.82rem; margin-top: 2px;">Beslutningsstøtte for scouting og rekruttering</div>
</div>
""",
    unsafe_allow_html=True,
)
