#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliteserien DSS – demo-klar (masteroppgave)

Mål for denne versjonen:
- Alt på norsk / fotballspråk (ingen emojier).
- Filtre i sidepanel.
- Ingen “døde” knapper: knapper skal åpne Spillerkort eller Duell.
- Duell åpnes automatisk når 2/2 spillere er valgt i sidebaren.
- Strammere tabeller og mer profesjonelt uttrykk (så langt Streamlit tillater).

Kjør:
  streamlit run eliteserien_dss_demo.py

Viktig:
- Juster BASE_DIR til riktig mappe hos deg (der parquet-fila og Logo-mappa ligger).
"""

from __future__ import annotations

import io
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# KONFIG – JUSTER DISSE HOS DEG
# =============================================================================
BASE_DIR = Path("/Users/kulmiyearab/Documents")  # <-- endre om nødvendig
PARQUET_PATH = BASE_DIR / "eliteserien_2025_enriched.parquet"
LOGO_DIR = BASE_DIR / "Logo"

APP_TITLE = "Eliteserien DSS"

PAGES = [
    "Forside",
    "Dashboard",
    "Spillerkort",
    "Duell",
    "Scout-rapport",
]


# =============================================================================
# SIDEOPPSETT
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None,
)


# =============================================================================
# STYLING – “silk”/premium innenfor Streamlit
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
  --text:#eaeef5;
  --muted:#a7b0c0;

  --brand:#6f8cff;
  --brand2:#8a6bff;

  --pos:#3ddc97;
  --warn:#ffcc66;
  --neg:#ff5c6c;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

html, body, [class*="css"]  {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(111,140,255,0.14), transparent 60%),
              radial-gradient(900px 500px at 80% 10%, rgba(138,107,255,0.12), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
  color: var(--text) !important;
}

.block-container{
  max-width: 1650px;
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
  padding-left: 2.0rem;
  padding-right: 2.0rem;
}

section[data-testid="stSidebar"]{
  width: 330px !important;
  background: linear-gradient(180deg, #0c1117 0%, #0b0f14 100%);
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] > div { padding-top: 1.3rem; }

h1{
  font-size: 2.35rem !important;
  font-weight: 820 !important;
  letter-spacing: -1.1px !important;
  color: #ffffff !important;
  margin-bottom: 0.25rem !important;
  line-height: 1.15 !important;
}
h2{
  font-size: 1.45rem !important;
  font-weight: 780 !important;
  letter-spacing: -0.5px !important;
  color: #ffffff !important;
  margin-top: 1.15rem !important;
  margin-bottom: 0.7rem !important;
}
h3{
  font-size: 1.08rem !important;
  font-weight: 700 !important;
  color: #eef2f8 !important;
  margin-bottom: 0.55rem !important;
  letter-spacing: -0.2px !important;
}

.pane{
  background: linear-gradient(135deg, rgba(255,255,255,0.050), rgba(255,255,255,0.030));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 1.25rem 1.25rem;
  box-shadow: 0 14px 30px rgba(0,0,0,0.24);
  margin-bottom: 1.0rem;
}

.hero{
  background: linear-gradient(135deg, rgba(111,140,255,0.16), rgba(138,107,255,0.12));
  border: 1px solid rgba(111,140,255,0.26);
  border-radius: 22px;
  padding: 2.2rem 2.0rem;
  box-shadow: 0 16px 32px rgba(0,0,0,0.24);
  margin-bottom: 1.1rem;
}

.divider{
  border-top: 1px solid rgba(255,255,255,0.08);
  margin: 1.6rem 0;
}

.small{
  color: var(--muted);
  font-size: 0.93rem;
}

/* Knapper: tykkere, roligere */
.stButton > button, .stDownloadButton > button{
  border-radius: 14px !important;
  padding: 0.78rem 1.05rem !important;
  font-weight: 720 !important;
  font-size: 0.95rem !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.06) !important;
  color: rgba(255,255,255,0.95) !important;
  box-shadow: 0 10px 22px rgba(0,0,0,0.22);
  transition: all 0.14s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  transform: translateY(-1px);
  background: rgba(255,255,255,0.09) !important;
  border-color: rgba(111,140,255,0.35) !important;
  box-shadow: 0 14px 28px rgba(0,0,0,0.26);
}
button[kind="primary"]{
  background: linear-gradient(135deg, rgba(111,140,255,0.92), rgba(138,107,255,0.88)) !important;
  border-color: rgba(111,140,255,0.40) !important;
}

/* Chips */
.chips{
  display:flex; flex-wrap:wrap; gap:8px; margin-top:10px;
}
.chip{
  display:inline-flex;
  align-items:center;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 700;
  color: rgba(255,255,255,0.92);
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
}
.chip-strong{
  background: rgba(111,140,255,0.16);
  border: 1px solid rgba(111,140,255,0.28);
}
.chip-warn{
  background: rgba(255,204,102,0.14);
  border: 1px solid rgba(255,204,102,0.26);
}
.chip-ok{
  background: rgba(61,220,151,0.12);
  border: 1px solid rgba(61,220,151,0.22);
}

/* Tabeller: mindre “Excel” */
div[data-testid="stDataFrame"]{
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 14px 30px rgba(0,0,0,0.22);
}
div[data-testid="stDataFrame"] thead tr th{
  background: rgba(255,255,255,0.05) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HJELPERE
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def safe_col(df_: pd.DataFrame, col: str) -> bool:
    return col in df_.columns


def slugify(text: str) -> str:
    t = str(text).strip().lower()
    t = t.replace("ø", "o").replace("å", "a").replace("æ", "ae")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-z0-9]+", "-", t)
    return t.strip("-")


def kort_navn(navn: str) -> str:
    if not navn:
        return "—"
    deler = str(navn).strip().split()
    if len(deler) == 1:
        return deler[0]
    return f"{deler[0]} {deler[-1]}"


@st.cache_data(show_spinner=False)
def build_logo_index(logo_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    if not logo_dir.exists():
        return idx
    for p in logo_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".png":
            idx[slugify(p.stem)] = p
    return idx


def find_logo(team_name: str, logo_idx: dict[str, Path]) -> Optional[Path]:
    if not team_name:
        return None
    needle = slugify(team_name)
    direct = LOGO_DIR / f"{needle}.png"
    if direct.exists():
        return direct
    if needle in logo_idx:
        return logo_idx[needle]
    for key, path in logo_idx.items():
        if needle in key or key in needle:
            return path
    return None


def pick_player_row(df_: pd.DataFrame, player_name: str) -> dict:
    if not safe_col(df_, "player_name"):
        return {}
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
    st.markdown(f"# {title}")
    st.markdown(f'<div class="small">{subtitle}</div>', unsafe_allow_html=True)
    if chips:
        chip_row(chips)
    st.markdown("</div>", unsafe_allow_html=True)


def percentil(referanse, verdi) -> Optional[float]:
    """Percentil (0–100) for verdi relativt til referansegruppe. Midtrank ved like verdier."""
    if verdi is None or pd.isna(verdi):
        return None
    s = pd.to_numeric(pd.Series(referanse), errors="coerce").dropna()
    if len(s) == 0:
        return None
    v = float(pd.to_numeric(verdi, errors="coerce"))
    if np.isnan(v):
        return None
    under = (s < v).sum()
    lik = (s == v).sum()
    return float((under + 0.5 * lik) / len(s) * 100.0)


def _pct_farge(p: float) -> str:
    if p >= 80:
        return "rgba(61,220,151,0.90)"
    if p >= 60:
        return "rgba(111,140,255,0.85)"
    if p >= 40:
        return "rgba(255,255,255,0.55)"
    if p >= 20:
        return "rgba(255,204,102,0.85)"
    return "rgba(255,92,108,0.90)"


# =============================================================================
# DOMENE / METRIKKER
# =============================================================================
INVERT_PERCENTILE_COLS = {
    "z_risk_rate_pos",
    "z_fouls_committed_per90_pos",
}

ROLE_FEATURES: Dict[str, List[Tuple[str, str]]] = {
    "playmaker": [
        ("z_passes_key_per90_pos", "Nøkkelpasninger"),
        ("z_passes_accuracy_pos", "Pasningspresisjon"),
        ("z_pass_efficiency_pos", "Pasningseffektivitet"),
        ("z_assists_per90_pos", "Målgivende"),
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
        ("z_dribbles_success_per90_pos", "Vellykkede driblinger"),
    ],
    "creator_att": [
        ("z_assists_per90_pos", "Målgivende"),
        ("z_passes_key_per90_pos", "Nøkkelpasninger"),
        ("z_dribbles_success_per90_pos", "Vellykkede driblinger"),
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
        ("rating", "Keeperrating"),
    ],
}

KEY_METRICS = [
    ("fair_score", "Prestasjon (fair)", 2),
    ("forecast_score", "Potensial (forecast)", 2),
    ("total_score", "Total", 2),
    ("reliability", "Datagrunnlag", 2),
    ("age_factor", "Aldersfaktor", 2),
]


# =============================================================================
# FIGURER
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
    plot_df = df_.copy()

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
            x_col: ":.2f" if x_col in plot_df.columns else False,
            y_col: ":.2f" if y_col in plot_df.columns else False,
        },
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=1, color="rgba(255,255,255,0.22)"),
            opacity=0.90,
        )
    )

    x_med = float(pd.to_numeric(plot_df[x_col], errors="coerce").dropna().median()) if x_col in plot_df.columns else 0.0
    y_med = float(pd.to_numeric(plot_df[y_col], errors="coerce").dropna().median()) if y_col in plot_df.columns else 0.0

    fig.add_vline(x=x_med, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.22)")
    fig.add_hline(y=y_med, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.22)")

    fig.update_xaxes(title_text="Prestasjon (fair)", zeroline=False, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="Potensial (forecast)", zeroline=False, gridcolor="rgba(255,255,255,0.06)")

    return _base_layout(fig, height=620)


def create_percentile_bars(
    ref_df: pd.DataFrame,
    player_row: dict,
    feats: List[Tuple[str, str]],
    title: str = "",
    max_axes: int = 7,
) -> go.Figure:
    items: List[Tuple[str, float]] = []
    for col, label in feats[:max_axes]:
        if col not in ref_df.columns:
            continue
        v = player_row.get(col)
        pct = percentil(ref_df[col], v)
        if pct is None:
            continue
        if col in INVERT_PERCENTILE_COLS:
            pct = 100 - pct
        items.append((label, float(pct)))

    if not items:
        fig = go.Figure()
        fig.add_annotation(text="Ingen rolledata tilgjengelig i utvalget.", showarrow=False)
        return _base_layout(fig, height=360)

    labels = [i[0] for i in items][::-1]
    values = [i[1] for i in items][::-1]
    colors = [_pct_farge(v) for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Percentil: %{x:.1f}p<extra></extra>",
        )
    )

    fig.update_xaxes(range=[0, 100], gridcolor="rgba(255,255,255,0.06)", zeroline=False, title_text="")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.00)", zeroline=False, title_text="")

    fig.update_layout(
        title=dict(text=title, x=0.0, y=0.98, xanchor="left", font=dict(size=14, color="#eaeef5")),
        bargap=0.35,
        margin=dict(l=10, r=10, t=40, b=10),
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
    used = [(c, l) for c, l in feats if c in ref_df.columns][:max_axes]
    if not used:
        fig = go.Figure()
        fig.add_annotation(text="Ingen rolledata tilgjengelig i utvalget.", showarrow=False)
        return _base_layout(fig, height=520)

    labels, a_vals, b_vals, ref_vals = [], [], [], []
    for col, label in used:
        labels.append(label)

        pct_a = percentil(ref_df[col], player_a.get(col))
        pct_a = 0.0 if pct_a is None else float(pct_a)
        if col in INVERT_PERCENTILE_COLS:
            pct_a = 100 - pct_a
        a_vals.append(pct_a)

        ref_vals.append(50.0)

        if player_b is not None:
            pct_b = percentil(ref_df[col], player_b.get(col))
            pct_b = 0.0 if pct_b is None else float(pct_b)
            if col in INVERT_PERCENTILE_COLS:
                pct_b = 100 - pct_b
            b_vals.append(pct_b)

    theta = labels + [labels[0]]
    a_r = a_vals + [a_vals[0]]
    ref_r = ref_vals + [ref_vals[0]]

    fig = go.Figure()
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
            line=dict(width=3.0, color="rgba(111,140,255,0.95)"),
            fillcolor="rgba(111,140,255,0.22)",
            marker=dict(size=6, color="rgba(111,140,255,0.95)"),
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
                line=dict(width=3.0, color="rgba(255,92,108,0.92)"),
                fillcolor="rgba(255,92,108,0.18)",
                marker=dict(size=6, color="rgba(255,92,108,0.92)"),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=45, r=45, t=35, b=40),
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
# FILTER / APPLY
# =============================================================================
@dataclass
class Filters:
    datasett: Optional[List[str]]
    pos: str
    min_minutes: int
    age_range: Optional[Tuple[int, int]]
    team: str
    player_query: str
    role: str


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()

    if f.datasett and safe_col(out, "datasett"):
        out = out[out["datasett"].isin(f.datasett)].copy()

    if f.pos != "Alle" and safe_col(out, "pos_group"):
        out = out[out["pos_group"] == f.pos].copy()

    if safe_col(out, "minutes"):
        out = out[pd.to_numeric(out["minutes"], errors="coerce").fillna(0) >= int(f.min_minutes)].copy()

    if f.age_range and safe_col(out, "age"):
        age = pd.to_numeric(out["age"], errors="coerce")
        out = out[(age >= f.age_range[0]) & (age <= f.age_range[1])].copy()

    if f.team != "Alle" and safe_col(out, "team_name"):
        out = out[out["team_name"].astype(str) == str(f.team)].copy()

    if f.player_query and safe_col(out, "player_name"):
        out = out[out["player_name"].astype(str).str.contains(f.player_query, case=False, na=False)].copy()

    return out


def filter_chips(f: Filters, n_players: int) -> List[Tuple[str, str]]:
    chips = [("Utvalg", "strong"), (f"{n_players} spillere", "")]
    if f.pos != "Alle":
        chips.append((f"Pos: {f.pos}", "strong"))
    if f.team != "Alle":
        chips.append((f"Lag: {f.team}", ""))
    if f.age_range:
        chips.append((f"Alder: {f.age_range[0]}–{f.age_range[1]}", ""))
    chips.append((f"Min: {f.min_minutes}", ""))
    chips.append((f"Rolle: {f.role.replace('_', ' ').title()}", ""))
    return chips


def scouting_table(
    df_: pd.DataFrame,
    title: str,
    subtitle: str,
    cols: List[str],
    height: int = 420,
    sort_col: Optional[str] = None,
    descending: bool = True,
) -> None:
    panel_open()
    st.markdown(f"### {title}")
    st.markdown(f'<div class="small">{subtitle}</div>', unsafe_allow_html=True)

    view = df_.copy()
    if sort_col and sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=not descending)

    view = view[cols].copy()

    col_config = {}
    if "fair_score" in view.columns:
        col_config["fair_score"] = st.column_config.NumberColumn("Prestasjon", format="%.2f")
    if "forecast_score" in view.columns:
        col_config["forecast_score"] = st.column_config.NumberColumn("Potensial", format="%.2f")
    if "total_score" in view.columns:
        col_config["total_score"] = st.column_config.NumberColumn("Total", format="%.2f")
    if "reliability" in view.columns:
        col_config["reliability"] = st.column_config.NumberColumn("Datagrunnlag", format="%.2f")
    if "minutes" in view.columns:
        col_config["minutes"] = st.column_config.NumberColumn("Minutter", format="%.0f")
    if "age" in view.columns:
        col_config["age"] = st.column_config.NumberColumn("Alder", format="%.0f")

    st.dataframe(
        view.reset_index(drop=True),
        use_container_width=True,
        height=height,
        hide_index=True,
        column_config=col_config,
    )
    panel_close()


# =============================================================================
# SESJON / NAVIGASJON
# =============================================================================
def sett_side(navn: str) -> None:
    st.session_state["side"] = navn
    st.rerun()


if "side" not in st.session_state:
    st.session_state["side"] = "Forside"

if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = []  # list of tuples (player_name, team_name)

if "duell_spillere" not in st.session_state:
    st.session_state["duell_spillere"] = []  # list of player_name (lengde 0–2)

if "spillerkort_spiller" not in st.session_state:
    st.session_state["spillerkort_spiller"] = None


def add_to_shortlist(player_name: str, team_name: str) -> None:
    key = (str(player_name), str(team_name))
    if key not in st.session_state["shortlist"]:
        st.session_state["shortlist"].append(key)


def sett_spillerkort(player_name: str) -> None:
    st.session_state["spillerkort_spiller"] = str(player_name)
    sett_side("Spillerkort")


def legg_til_i_duell(player_name: str) -> None:
    navn = str(player_name)
    lst = list(st.session_state.get("duell_spillere", []))
    if navn in lst:
        return
    if len(lst) < 2:
        lst.append(navn)
    else:
        lst = [lst[1], navn]
    st.session_state["duell_spillere"] = lst

    if len(lst) == 2:
        sett_side("Duell")
    else:
        st.rerun()


def fjern_fra_duell(player_name: str) -> None:
    navn = str(player_name)
    lst = [p for p in st.session_state.get("duell_spillere", []) if p != navn]
    st.session_state["duell_spillere"] = lst
    st.rerun()


# =============================================================================
# LAST DATA
# =============================================================================
if not PARQUET_PATH.exists():
    st.error(
        "Datagrunnlag mangler.\n\n"
        f"Finner ikke: {PARQUET_PATH}\n\n"
        "Løsning: sjekk BASE_DIR / filnavn og last siden på nytt."
    )
    st.stop()

try:
    df = load_data(PARQUET_PATH)
except Exception as e:
    st.error(f"Kunne ikke laste data: {e}")
    st.stop()

logo_index = build_logo_index(LOGO_DIR)

for col in ["player_name", "team_name", "pos_group"]:
    if col not in df.columns:
        st.error(f"Mangler nødvendig kolonne i datasettet: {col}")
        st.stop()


# =============================================================================
# SIDEBAR: NAV + FILTRE + DUELL
# =============================================================================
with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Beslutningsstøtte for scouting og rekruttering")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Navigasjon
    valgt = st.radio(
        "Navigasjon",
        PAGES,
        index=PAGES.index(st.session_state["side"]) if st.session_state["side"] in PAGES else 0,
        label_visibility="collapsed",
    )
    if valgt != st.session_state["side"]:
        st.session_state["side"] = valgt

    # Filtre i sidebar
    with st.expander("Filtre", expanded=True):
        datasett_valg = None
        if safe_col(df, "datasett"):
            alle = sorted(df["datasett"].dropna().unique().tolist())
            datasett_valg = st.multiselect("Datasett", alle, default=alle)

        pos_filter = st.selectbox("Posisjon", ["Alle", "MID", "ATT", "DEF", "GK"], index=0)

        if safe_col(df, "minutes") and df["minutes"].notna().any():
            min_minutes = st.slider("Minutter (min)", 0, int(pd.to_numeric(df["minutes"], errors="coerce").max()), 450, step=50)
        else:
            min_minutes = 0

        if safe_col(df, "age") and df["age"].notna().any():
            age_min = int(pd.to_numeric(df["age"], errors="coerce").dropna().min())
            age_max = int(pd.to_numeric(df["age"], errors="coerce").dropna().max())
            age_range = st.slider("Alder", age_min, age_max, (age_min, age_max))
        else:
            age_range = None

        team_valg = "Alle"
        if safe_col(df, "team_name"):
            lag = sorted(df["team_name"].dropna().astype(str).unique().tolist())
            team_valg = st.selectbox("Lag", ["Alle"] + lag, index=0)

        player_search = st.text_input("Spillersøk", value="", placeholder="Søk navn…")

        role = st.selectbox(
            "Rolleprofil",
            list(ROLE_FEATURES.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )

        st.markdown(
            '<div class="small">Rolleprofil viser percentiler innen samme posisjon. Bars er standard. Radar er alternativ.</div>',
            unsafe_allow_html=True,
        )

    # Sett filters-objekt
    filters = Filters(
        datasett=datasett_valg,
        pos=pos_filter,
        min_minutes=min_minutes,
        age_range=age_range,
        team=team_valg,
        player_query=player_search.strip(),
        role=role,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Duell i sidebar
    st.markdown("### Sammenligning (duell)")
    alle_spillere = sorted(df["player_name"].dropna().astype(str).unique().tolist())

    duell_valg = list(st.session_state.get("duell_spillere", []))

    s1 = st.selectbox(
        "Spiller A",
        ["—"] + alle_spillere,
        index=(["—"] + alle_spillere).index(duell_valg[0]) if len(duell_valg) >= 1 and duell_valg[0] in alle_spillere else 0,
        format_func=lambda x: kort_navn(x) if x != "—" else "—",
    )
    s2 = st.selectbox(
        "Spiller B",
        ["—"] + alle_spillere,
        index=(["—"] + alle_spillere).index(duell_valg[1]) if len(duell_valg) >= 2 and duell_valg[1] in alle_spillere else 0,
        format_func=lambda x: kort_navn(x) if x != "—" else "—",
    )

    valgt_ny = [x for x in [s1, s2] if x != "—" and x is not None]
    if len(valgt_ny) == 2 and valgt_ny[0] == valgt_ny[1]:
        valgt_ny = [valgt_ny[0]]

    st.session_state["duell_spillere"] = valgt_ny
    st.caption(f"Valgt: {len(valgt_ny)}/2")

    # AUTO: når 2/2 → gå til Duell
    if len(valgt_ny) == 2 and st.session_state.get("side") != "Duell":
        st.session_state["side"] = "Duell"
        st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption(f"Shortlist: {len(st.session_state.get('shortlist', []))}")
    st.caption(f"Lag i datasett: {df['team_name'].nunique():,}".replace(",", " "))
    st.caption(f"Spillere i datasett: {len(df):,}".replace(",", " "))


# =============================================================================
# FILTERED DF (etter sidebar)
# =============================================================================
filtered_df = apply_filters(df, filters)


# =============================================================================
# SIDE: FORSIDE
# =============================================================================
if st.session_state["side"] == "Forside":
    hero(
        "Eliteserien DSS",
        "Beslutningsstøtte for talentidentifisering, rolle-match og rekruttering – tilpasset klubbbruk.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Spillere (utvalg)", f"{len(filtered_df):,}".replace(",", " "))
    k2.metric("Lag", f"{filtered_df['team_name'].nunique():,}".replace(",", " ") if safe_col(filtered_df, "team_name") else "—")
    k3.metric("Snitt minutter", fmt_int(pd.to_numeric(filtered_df["minutes"], errors="coerce").mean()) if safe_col(filtered_df, "minutes") and len(filtered_df) else "—")
    k4.metric("Snitt alder", f"{pd.to_numeric(filtered_df['age'], errors='coerce').mean():.1f} år" if safe_col(filtered_df, "age") and len(filtered_df) else "—")
    k5.metric("Snitt total", f"{pd.to_numeric(filtered_df['total_score'], errors='coerce').mean():.2f}" if safe_col(filtered_df, "total_score") and len(filtered_df) else "—")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.05])

    with left:
        if safe_col(filtered_df, "fair_score") and len(filtered_df):
            view = filtered_df.copy()
            cols = [c for c in ["player_name", "team_name", "pos_group", "minutes", "fair_score", "reliability"] if safe_col(view, c)]
            scouting_table(
                view.nlargest(8, "fair_score"),
                "Rekrutteringsliste – prestasjon (nå)",
                "Spillere med sterk prestasjon i utvalget. Åpne spillerkort for vurdering i rolle.",
                cols=cols,
                height=300,
                sort_col="fair_score",
            )
            top_player = view.nlargest(1, "fair_score")["player_name"].iloc[0]
            if st.button("Åpne beste kandidat (prestasjon)", use_container_width=True, type="primary"):
                sett_spillerkort(top_player)
        else:
            panel_open()
            st.markdown("### Rekrutteringsliste – prestasjon (nå)")
            st.info("Mangler 'fair_score' eller utvalget er tomt.")
            panel_close()

    with right:
        if safe_col(filtered_df, "forecast_score") and safe_col(filtered_df, "age") and len(filtered_df):
            view = filtered_df[pd.to_numeric(filtered_df["age"], errors="coerce") <= 23].copy()
            cols = [c for c in ["player_name", "age", "team_name", "pos_group", "minutes", "forecast_score", "reliability"] if safe_col(view, c)]
            if len(view):
                scouting_table(
                    view.nlargest(8, "forecast_score"),
                    "Rekrutteringsliste – potensial (U23)",
                    "Spillere med høyt potensial i utvalget. Bruk spillerkort for rolleprofil og datagrunnlag.",
                    cols=cols,
                    height=300,
                    sort_col="forecast_score",
                )
                top_u23 = view.nlargest(1, "forecast_score")["player_name"].iloc[0]
                if st.button("Åpne beste kandidat (potensial)", use_container_width=True, type="primary"):
                    sett_spillerkort(top_u23)
            else:
                panel_open()
                st.markdown("### Rekrutteringsliste – potensial (U23)")
                st.info("Ingen U23-kandidater i utvalget.")
                panel_close()
        else:
            panel_open()
            st.markdown("### Rekrutteringsliste – potensial (U23)")
            st.info("Mangler 'forecast_score'/'age' eller utvalget er tomt.")
            panel_close()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    panel_open()
    st.markdown("### Markedskart (Talent Map)")
    st.markdown('<div class="small">Prestasjon vs potensial. Bruk filter for å snevre inn til relevant marked.</div>', unsafe_allow_html=True)

    if safe_col(filtered_df, "fair_score") and safe_col(filtered_df, "forecast_score") and len(filtered_df) > 0:
        fig = create_talent_map(
            filtered_df,
            x_col="fair_score",
            y_col="forecast_score",
            size_col="minutes" if safe_col(filtered_df, "minutes") else None,
            color_col="age" if safe_col(filtered_df, "age") else None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Mangler 'fair_score' og/eller 'forecast_score', eller tomt utvalg.")
    panel_close()


# =============================================================================
# SIDE: DASHBOARD
# =============================================================================
elif st.session_state["side"] == "Dashboard":
    hero(
        "Dashboard",
        "Finn kandidater raskt, og gå videre til spillerkort eller duell.",
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
        st.markdown("### Markedskart")
        st.markdown('<div class="small">Bruk filter for marked og rolle. Hover gir rask kontekst.</div>', unsafe_allow_html=True)

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
        panel_open()
        st.markdown("### Hurtigvalg")
        st.markdown('<div class="small">Åpne spillerkort, legg til duell, eller legg til shortlist.</div>', unsafe_allow_html=True)

        spillerliste = sorted(filtered_df["player_name"].dropna().astype(str).unique().tolist())
        valgt_spiller = st.selectbox("Velg spiller", spillerliste, format_func=kort_navn)

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Spillerkort", use_container_width=True, type="primary"):
                sett_spillerkort(valgt_spiller)
        with b2:
            if st.button("Legg i duell", use_container_width=True):
                legg_til_i_duell(valgt_spiller)
        with b3:
            team = pick_player_row(df, valgt_spiller).get("team_name", "")
            if st.button("Shortlist", use_container_width=True):
                add_to_shortlist(valgt_spiller, team)

        panel_close()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    panel_open()
    st.markdown("### Rolle-match (rangering)")
    st.markdown(
        f'<div class="small">Rangert etter rolleprofil: <b>{filters.role.replace("_"," ").title()}</b>. '
        "Percentiler er beregnet mot referansegruppe innen posisjon.</div>",
        unsafe_allow_html=True,
    )

    feats = ROLE_FEATURES.get(filters.role, [])
    feat_cols = [c for c, _ in feats if safe_col(filtered_df, c)]

    if len(feat_cols) == 0:
        st.info("Mangler nødvendige rollekolonner i datasettet for valgt rolle.")
        panel_close()
    else:
        ref_df = df.copy()
        if filters.pos != "Alle" and safe_col(ref_df, "pos_group"):
            ref_df = ref_df[ref_df["pos_group"] == filters.pos].copy()

        def role_score(row: pd.Series) -> float:
            pts = []
            for col, _label in feats[:7]:
                if col not in ref_df.columns:
                    continue
                pct = percentil(ref_df[col], row.get(col))
                if pct is None:
                    continue
                if col in INVERT_PERCENTILE_COLS:
                    pct = 100 - pct
                pts.append(float(pct))
            return float(np.mean(pts)) if pts else np.nan

        ranked = filtered_df.copy()
        ranked["rolle_match"] = ranked.apply(role_score, axis=1)
        ranked = ranked.sort_values("rolle_match", ascending=False)

        show_cols = [c for c in [
            "player_name", "team_name", "pos_group", "age", "minutes",
            "rolle_match", "fair_score", "forecast_score", "reliability"
        ] if safe_col(ranked, c)]

        st.dataframe(
            ranked.head(20)[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=520,
            hide_index=True,
            column_config={
                "rolle_match": st.column_config.NumberColumn("Rolle-match", format="%.1f"),
                "minutes": st.column_config.NumberColumn("Minutter", format="%.0f"),
                "fair_score": st.column_config.NumberColumn("Prestasjon", format="%.2f"),
                "forecast_score": st.column_config.NumberColumn("Potensial", format="%.2f"),
                "reliability": st.column_config.NumberColumn("Datagrunnlag", format="%.2f"),
                "age": st.column_config.NumberColumn("Alder", format="%.0f"),
            },
        )

        panel_close()


# =============================================================================
# SIDE: SPILLERKORT
# =============================================================================
elif st.session_state["side"] == "Spillerkort":
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

    age_min, age_max = (filters.age_range if filters.age_range else (None, None))
    utvalg_key = f"{len(spillerliste)}_{filters.role}_{filters.min_minutes}_{age_min}_{age_max}_{filters.pos}_{filters.team}"

    if st.session_state.get("spillerkort_utvalg_key") != utvalg_key:
        st.session_state["spillerkort_utvalg_key"] = utvalg_key
        st.session_state["spillerkort_default"] = random.choice(spillerliste) if spillerliste else None

    forhånd = st.session_state.get("spillerkort_spiller")
    if forhånd in spillerliste:
        default_player = forhånd
        st.session_state["spillerkort_default"] = forhånd
    else:
        default_player = st.session_state.get("spillerkort_default")

    default_index = spillerliste.index(default_player) if default_player in spillerliste else 0

    selected_player = st.selectbox(
        "Velg spiller",
        spillerliste,
        index=default_index,
        format_func=kort_navn,
    )

    player_data = pick_player_row(filtered_df, selected_player)
    if not player_data:
        st.warning("Kunne ikke hente data for valgt spiller.")
        st.stop()

    panel_open()
    c1, c2, c3 = st.columns([0.85, 2.15, 1.15], gap="large")

    with c1:
        team_name = str(player_data.get("team_name", "")).strip()
        logo_path = find_logo(team_name, logo_index)
        if logo_path:
            st.image(str(logo_path), width=105)

    with c2:
        st.markdown(f"## {kort_navn(player_data.get('player_name', 'Ukjent'))}")
        chips = []
        chips.append((str(player_data.get("team_name", "—")), "strong"))
        if pd.notna(player_data.get("pos_group")):
            chips.append((str(player_data.get("pos_group", "—")), ""))
        if safe_col(df, "age") and pd.notna(player_data.get("age")):
            age = int(float(player_data.get("age")))
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
        team = str(player_data.get("team_name", "")).strip()

        if st.button("Legg til i shortlist", use_container_width=True, type="primary"):
            add_to_shortlist(player_data.get("player_name"), team)

        if st.button("Legg til i duell", use_container_width=True):
            legg_til_i_duell(player_data.get("player_name"))

        csv_data = pd.DataFrame([player_data]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Eksporter spillerdata (CSV)",
            csv_data,
            file_name=f"{slugify(kort_navn(selected_player))}_spillerkort.csv",
            mime="text/csv",
            use_container_width=True,
        )

    panel_close()

    panel_open()
    st.markdown("### Nøkkelmetrikker")
    mcols = st.columns(5)
    for i, (col, label, dec) in enumerate(KEY_METRICS):
        if safe_col(df, col):
            val = player_data.get(col)
            mcols[i].metric(label, fmt_float(val, dec) if pd.notna(val) else "—")
    panel_close()

    feats = ROLE_FEATURES.get(filters.role, [])
    feat_cols = [c for c, _ in feats if safe_col(df, c)]

    ref_df = df.copy()
    if safe_col(ref_df, "pos_group") and pd.notna(player_data.get("pos_group")):
        ref_df = ref_df[ref_df["pos_group"] == player_data.get("pos_group")].copy()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        panel_open()
        st.markdown("### Rolleprofil")
        st.markdown(
            f'<div class="small">{filters.role.replace("_"," ").title()} – percentiler mot samme posisjon.</div>',
            unsafe_allow_html=True,
        )

        if len(feat_cols) == 0 or len(ref_df) == 0:
            st.info("Mangler rolledata for valgt rolle eller posisjon.")
        else:
            t1, t2 = st.tabs(["Percentil-bars", "Radar"])

            with t1:
                fig = create_percentile_bars(ref_df, player_data, feats, title="", max_axes=7)
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                fig = create_radar_chart_pro(
                    ref_df=ref_df,
                    player_a=player_data,
                    feats=feats,
                    name_a=kort_navn(selected_player),
                    max_axes=7,
                )
                st.plotly_chart(fig, use_container_width=True)

        panel_close()

    with right:
        panel_open()
        st.markdown("### Rask vurdering (rolle)")
        st.markdown(
            '<div class="small">Basert på sterkeste og svakeste percentiler i valgt rolleprofil.</div>',
            unsafe_allow_html=True,
        )

        take = []
        for col, label in feats[:7]:
            if col not in ref_df.columns:
                continue
            pct = percentil(ref_df[col], player_data.get(col))
            if pct is None:
                continue
            if col in INVERT_PERCENTILE_COLS:
                pct = 100 - pct
            take.append((label, float(pct)))

        if take:
            take_sorted = sorted(take, key=lambda x: x[1], reverse=True)
            top3 = take_sorted[:3]
            bot3 = take_sorted[-3:][::-1]

            st.markdown("**Styrker**")
            for lbl, pct in top3:
                st.markdown(f"- {lbl}: **{pct:.1f}p**")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Utviklingsområder**")
            for lbl, pct in bot3:
                st.markdown(f"- {lbl}: **{pct:.1f}p**")
        else:
            st.info("Ingen rolle-takeaways tilgjengelig for dette utvalget.")

        panel_close()


# =============================================================================
# SIDE: DUELL
# =============================================================================
elif st.session_state["side"] == "Duell":
    hero(
        "Duell",
        "Sammenlign to spillere i samme rolleprofil og se forskjeller i percentiler.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    duell = st.session_state.get("duell_spillere", [])
    if len(duell) != 2:
        panel_open()
        st.info("Velg to spillere i sidebaren under «Sammenligning (duell)».")
        panel_close()
        st.stop()

    pA, pB = duell[0], duell[1]
    a = pick_player_row(df, pA)
    b = pick_player_row(df, pB)

    if not a or not b:
        panel_open()
        st.warning("Kunne ikke hente data for én eller begge spillere.")
        panel_close()
        st.stop()

    ref_df = df.copy()
    if safe_col(ref_df, "pos_group") and pd.notna(a.get("pos_group")):
        ref_df = ref_df[ref_df["pos_group"] == a.get("pos_group")].copy()

    panel_open()
    h1, h2, h3 = st.columns([1.0, 1.0, 1.0], gap="large")

    with h1:
        st.markdown("### Spiller A")
        st.markdown(f"**{kort_navn(pA)}**")
        st.markdown(f'<div class="small">{a.get("team_name","—")} | {a.get("pos_group","—")}</div>', unsafe_allow_html=True)
        if st.button("Åpne spillerkort A", use_container_width=True, type="primary"):
            sett_spillerkort(pA)
        if st.button("Fjern A fra duell", use_container_width=True):
            fjern_fra_duell(pA)

    with h2:
        st.markdown("### Spiller B")
        st.markdown(f"**{kort_navn(pB)}**")
        st.markdown(f'<div class="small">{b.get("team_name","—")} | {b.get("pos_group","—")}</div>', unsafe_allow_html=True)
        if st.button("Åpne spillerkort B", use_container_width=True, type="primary"):
            sett_spillerkort(pB)
        if st.button("Fjern B fra duell", use_container_width=True):
            fjern_fra_duell(pB)

    with h3:
        st.markdown("### Kontroll")
        if st.button("Tøm duell", use_container_width=True):
            st.session_state["duell_spillere"] = []
            st.rerun()

    panel_close()

    panel_open()
    st.markdown("### Nøkkelmetrikker (side ved side)")

    colA, colB = st.columns(2, gap="large")

    def _metric_grid(data: dict):
        m = st.columns(5)
        for i, (col, label, dec) in enumerate(KEY_METRICS):
            if safe_col(df, col):
                v = data.get(col)
                m[i].metric(label, fmt_float(v, dec) if pd.notna(v) else "—")

    with colA:
        _metric_grid(a)
    with colB:
        _metric_grid(b)

    panel_close()

    feats = ROLE_FEATURES.get(filters.role, [])
    feat_cols = [c for c, _ in feats if safe_col(ref_df, c)]

    panel_open()
    st.markdown("### Rolleprofil – sammenligning")
    st.markdown(
        f'<div class="small">Rolle: <b>{filters.role.replace("_"," ").title()}</b>. '
        "Percentiler mot referanse innen posisjon.</div>",
        unsafe_allow_html=True,
    )

    if len(feat_cols) == 0 or len(ref_df) == 0:
        st.info("Mangler rolledata for valgt rolle eller referansegruppe.")
        panel_close()
    else:
        t1, t2 = st.tabs(["Radar", "Forskjell per metrikk"])

        with t1:
            fig = create_radar_chart_pro(
                ref_df=ref_df,
                player_a=a,
                feats=feats,
                name_a=kort_navn(pA),
                player_b=b,
                name_b=kort_navn(pB),
                max_axes=7,
            )
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            rows = []
            for col, label in feats[:7]:
                if col not in ref_df.columns:
                    continue
                p_a = percentil(ref_df[col], a.get(col))
                p_b = percentil(ref_df[col], b.get(col))
                if p_a is None or p_b is None:
                    continue
                if col in INVERT_PERCENTILE_COLS:
                    p_a = 100 - p_a
                    p_b = 100 - p_b
                rows.append({"Metrikk": label, "A (p)": p_a, "B (p)": p_b, "Forskjell (A-B)": p_a - p_b})

            if rows:
                diff = pd.DataFrame(rows).sort_values("Forskjell (A-B)", ascending=False)
                st.dataframe(
                    diff,
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                    column_config={
                        "A (p)": st.column_config.NumberColumn("Spiller A", format="%.1f"),
                        "B (p)": st.column_config.NumberColumn("Spiller B", format="%.1f"),
                        "Forskjell (A-B)": st.column_config.NumberColumn("Forskjell", format="%.1f"),
                    },
                )
            else:
                st.info("Ingen differanser tilgjengelig for valgt rolleprofil.")

    panel_close()


# =============================================================================
# SIDE: SCOUT-RAPPORT
# =============================================================================
elif st.session_state["side"] == "Scout-rapport":
    hero(
        "Scout-rapport",
        "Finn unge kandidater med sortering og tydelig datagrunnlag.",
        chips=filter_chips(filters, len(filtered_df)),
    )

    panel_open()
    st.markdown("### Kriterier")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_age = st.slider("Maks alder", 17, 25, 21)
    with c2:
        min_mins = st.slider("Minutter (min)", 0, 2500, 450, step=50)
    with c3:
        sort_map = {
            "forecast_score": "Potensial (forecast)",
            "fair_score": "Prestasjon (fair)",
            "total_score": "Total",
        }
        sort_by = st.selectbox("Sorter etter", list(sort_map.keys()), format_func=lambda x: sort_map[x])
    panel_close()

    talent_df = filtered_df.copy()
    if safe_col(talent_df, "age"):
        age = pd.to_numeric(talent_df["age"], errors="coerce")
        talent_df = talent_df[age <= max_age].copy()
    if safe_col(talent_df, "minutes"):
        mins = pd.to_numeric(talent_df["minutes"], errors="coerce").fillna(0)
        talent_df = talent_df[mins >= min_mins].copy()

    if len(talent_df) == 0:
        panel_open()
        st.warning("Ingen kandidater i utvalget. Juster kriteriene.")
        panel_close()
        st.stop()

    if safe_col(talent_df, sort_by):
        talent_df = talent_df.sort_values(sort_by, ascending=False).copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Kandidater", f"{len(talent_df):,}".replace(",", " "))
    k2.metric("Snitt alder", f"{pd.to_numeric(talent_df['age'], errors='coerce').mean():.1f} år" if safe_col(talent_df, "age") else "—")
    k3.metric("Snitt potensial", f"{pd.to_numeric(talent_df['forecast_score'], errors='coerce').mean():.2f}" if safe_col(talent_df, "forecast_score") else "—")
    k4.metric("Lag", f"{talent_df['team_name'].nunique():,}".replace(",", " ") if safe_col(talent_df, "team_name") else "—")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    cols = [c for c in [
        "player_name", "age", "team_name", "pos_group", "best_role",
        "minutes", "fair_score", "forecast_score", "reliability"
    ] if safe_col(talent_df, c)]

    scouting_table(
        talent_df.head(30),
        "Liste – unge kandidater",
        f"Sortert etter: {sort_map.get(sort_by, sort_by)}. Åpne spillerkort for rolleprofil.",
        cols=cols,
        height=620,
        sort_col=sort_by,
    )

    panel_open()
    st.markdown("### Handlinger")
    kandidat = st.selectbox("Velg kandidat", talent_df["player_name"].astype(str).unique().tolist(), format_func=kort_navn)
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Åpne spillerkort", use_container_width=True, type="primary"):
            sett_spillerkort(kandidat)
    with b2:
        if st.button("Legg i duell", use_container_width=True):
            legg_til_i_duell(kandidat)
    with b3:
        team = pick_player_row(df, kandidat).get("team_name", "")
        if st.button("Shortlist", use_container_width=True):
            add_to_shortlist(kandidat, team)
    panel_close()


# =============================================================================
# FOOTER
# =============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div style="text-align:center; color: rgba(255,255,255,0.45); font-size:0.9rem; padding: 0.6rem 0;">
  <div><strong>{APP_TITLE}</strong> | Beslutningsstøtte for scouting og rekruttering</div>
</div>
""",
    unsafe_allow_html=True,
)
