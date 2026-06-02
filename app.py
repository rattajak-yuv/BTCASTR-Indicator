import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import swisseph as swe
import json
from io import StringIO
from datetime import datetime, timezone

st.set_page_config(page_title="Bitcoin Astro Indicator", layout="wide")

# =========================================================
# BASIC CONFIG
# =========================================================
NATAL_DT = datetime(2009, 1, 12, 3, 30, 25, tzinfo=timezone.utc)
LAT = 34.1070
LON = -118.0570
swe.set_sid_mode(swe.SIDM_LAHIRI)

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
}

PLANET_SYMBOLS = {
    "Sun": "☉",
    "Moon": "☽",
    "Mercury": "☿",
    "Venus": "♀",
    "Mars": "♂",
    "Jupiter": "♃",
    "Saturn": "♄",
    "Uranus": "♅",
    "Neptune": "♆",
    "Pluto": "♇",
    "Asc": "ASC",
    "MC": "MC",
}

ZODIAC = [
    ("Aries", "♈"), ("Taurus", "♉"), ("Gemini", "♊"), ("Cancer", "♋"),
    ("Leo", "♌"), ("Virgo", "♍"), ("Libra", "♎"), ("Scorpio", "♏"),
    ("Sagittarius", "♐"), ("Capricorn", "♑"), ("Aquarius", "♒"), ("Pisces", "♓")
]

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.stApp {
    background: #F6F8FB;
    color: #111827;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1550px;
}
.metric-card {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 16px 18px;
    min-height: 130px;
}
.metric-label {
    color: #9ca3af;
    font-size: 0.86rem;
}
.metric-value {
    color: white;
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 6px;
}
.metric-sub {
    color: #60a5fa;
    font-size: 0.86rem;
    margin-top: 6px;
}
.explain-box {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 14px 16px;
    margin: 10px 0 16px 0;
    color: #cbd5e1;
    line-height: 1.55;
}
.taxonomy-card {
    border-radius: 14px;
    padding: 16px 18px;
    min-height: 120px;
    border: 1px solid rgba(255,255,255,0.10);
}
.taxonomy-title {
    color: rgba(255,255,255,0.72);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.taxonomy-value {
    color: white;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 8px;
}
.taxonomy-sub {
    color: rgba(255,255,255,0.86);
    font-size: 0.88rem;
    margin-top: 8px;
    line-height: 1.45;
}
.dashboard-chip {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 6px;
    color: white;
}
.terminal-shell {
    background: radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 28%),
                linear-gradient(180deg, #06111f 0%, #07131d 100%);
    border: 1px solid #132334;
    border-radius: 22px;
    padding: 18px 20px 10px 20px;
    margin-bottom: 18px;
    box-shadow: 0 18px 50px rgba(2, 6, 23, 0.45);
}
.terminal-kicker {
    color: #7dd3fc;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    font-weight: 700;
}
.terminal-headline {
    color: #f8fafc;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 4px;
}
.terminal-subhead {
    color: #94a3b8;
    font-size: 0.96rem;
    line-height: 1.6;
    margin-top: 8px;
    margin-bottom: 8px;
}
.terminal-stat {
    background: linear-gradient(180deg, rgba(15,23,42,0.96) 0%, rgba(2,6,23,0.96) 100%);
    border: 1px solid #1f3347;
    border-radius: 16px;
    padding: 16px 16px 14px 16px;
    min-height: 118px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.terminal-stat-label {
    color: #7c93ae;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.terminal-stat-value {
    color: #f8fafc;
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 10px;
}
.terminal-stat-sub {
    color: #cbd5e1;
    font-size: 0.84rem;
    line-height: 1.45;
    margin-top: 10px;
}
.terminal-section-title {
    color: #e2e8f0;
    font-size: 1.02rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}
.terminal-note {
    color: #8aa0b8;
    font-size: 0.9rem;
    margin-top: 4px;
}
.event-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.96) 0%, rgba(3,7,18,0.96) 100%);
    border: 1px solid #1f3347;
    border-radius: 14px;
    padding: 14px 14px 12px 14px;
    min-height: 138px;
    margin-bottom: 10px;
}
.event-date {
    color: #f8fafc;
    font-weight: 700;
    font-size: 0.98rem;
}
.event-type {
    font-size: 0.78rem;
    color: #7dd3fc;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 6px;
}
.event-direction {
    font-size: 1.2rem;
    font-weight: 800;
    margin-top: 8px;
}
.event-body {
    color: #cbd5e1;
    font-size: 0.83rem;
    line-height: 1.5;
    margin-top: 8px;
}
.risk-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.96) 0%, rgba(3,7,18,0.96) 100%);
    border: 1px solid #1f3347;
    border-left: 4px solid #475569;
    border-radius: 14px;
    padding: 14px 14px 12px 14px;
    min-height: 132px;
}
.light-header {
    background: #ffffff;
    border: 1px solid #E5E7EB;
    border-radius: 20px;
    padding: 24px 26px 18px 26px;
    margin-bottom: 18px;
}
.light-kicker {
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.73rem;
    font-weight: 700;
}
.light-headline {
    color: #111827;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 6px;
}
.light-subhead {
    color: #6B7280;
    font-size: 0.96rem;
    line-height: 1.65;
    margin-top: 8px;
}
.light-disclaimer {
    color: #6B7280;
    font-size: 0.82rem;
    margin-top: 10px;
}
.summary-strip {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 18px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
}
.summary-item {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 14px 14px 12px 14px;
    min-height: 102px;
}
.summary-label {
    color: #6B7280;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.summary-value {
    color: #111827;
    font-size: 1.1rem;
    font-weight: 800;
    margin-top: 10px;
}
.summary-sub {
    color: #6B7280;
    font-size: 0.82rem;
    margin-top: 8px;
    line-height: 1.45;
}
.light-metric {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-top: 4px solid #CBD5E1;
    border-radius: 16px;
    padding: 18px 18px 16px 18px;
    min-height: 144px;
}
.light-metric-label {
    color: #6B7280;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.light-metric-value {
    color: #111827;
    font-size: 1.28rem;
    font-weight: 800;
    margin-top: 12px;
}
.light-metric-sub {
    color: #6B7280;
    font-size: 0.86rem;
    line-height: 1.5;
    margin-top: 10px;
}
.section-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    margin-bottom: 18px;
    min-height: 214px;
}
.section-title {
    color: #111827;
    font-size: 1.02rem;
    font-weight: 800;
    letter-spacing: 0.01em;
}
.section-note {
    color: #6B7280;
    font-size: 0.88rem;
    margin-top: 4px;
    margin-bottom: 12px;
    line-height: 1.5;
}
.light-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.77rem;
    font-weight: 700;
    color: #FFFFFF;
}
.turning-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 16px 16px 14px 16px;
    min-height: 158px;
}
.turning-date {
    color: #111827;
    font-weight: 800;
    font-size: 0.96rem;
}
.turning-type {
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-size: 0.74rem;
    margin-top: 7px;
}
.turning-direction {
    font-weight: 800;
    font-size: 1rem;
    margin-top: 10px;
}
.turning-body {
    color: #6B7280;
    font-size: 0.84rem;
    line-height: 1.5;
    margin-top: 10px;
}
.mini-risk-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 4px solid #CBD5E1;
    border-radius: 16px;
    padding: 16px 16px 14px 16px;
    min-height: 150px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOADERS
# =========================================================
@st.cache_data(ttl=3600)
def load_csv_safe(path, default=None, parse_date_cols=None):
    if default is None:
        default = pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            cleaned_text = "\n".join(
                line for line in handle.read().splitlines()
                if not line.startswith(("<<<<<<<", "=======", ">>>>>>>"))
            )
        df = pd.read_csv(StringIO(cleaned_text), on_bad_lines="skip")
        for col in parse_date_cols or []:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception:
        return default.copy() if isinstance(default, pd.DataFrame) else default

@st.cache_data(ttl=3600)
def load_json_safe(path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default.copy() if isinstance(default, dict) else default

@st.cache_data(ttl=3600)
def load_daily():
    return load_csv_safe("data/bitcoin_astro_daily_score.csv", parse_date_cols=["date"])

@st.cache_data(ttl=3600)
def load_raw_aspects():
    return load_csv_safe("data/astro_aspects_raw.csv", parse_date_cols=["date"])

@st.cache_data(ttl=3600)
def load_ml_predictions():
    return load_csv_safe("data/ml_predictions.csv", parse_date_cols=["date"])

@st.cache_data(ttl=3600)
def load_ml_summary():
    return load_csv_safe("data/ml_model_summary.csv")

@st.cache_data(ttl=3600)
def load_ml_importance():
    return load_csv_safe("data/ml_feature_importance.csv")

@st.cache_data(ttl=3600)
def load_optimization_v2():
    return load_csv_safe("data/model_optimization_v2_results.csv")

@st.cache_data(ttl=3600)
def load_dashboard_current_state():
    return load_json_safe("data/dashboard_current_state.json")

@st.cache_data(ttl=3600)
def load_dashboard_timeline():
    return load_json_safe("data/dashboard_timeline.json")

@st.cache_data(ttl=3600)
def load_dashboard_risk_calendar():
    return load_json_safe("data/dashboard_risk_calendar.json")

@st.cache_data(ttl=3600)
def load_dashboard_summary():
    return load_json_safe("data/dashboard_summary.json")

@st.cache_data(ttl=3600)
def load_taxonomy_attribution():
    return load_csv_safe("data/taxonomy_attribution.csv")

@st.cache_data(ttl=3600)
def load_taxonomy_feature_importance():
    return load_csv_safe("data/taxonomy_feature_importance.csv")

@st.cache_data(ttl=3600)
def load_forecast_intelligence_v4():
    return load_csv_safe(
        "data/forecast_intelligence_v4.csv",
        parse_date_cols=["start_date", "end_date"],
    )

@st.cache_data(ttl=3600)
def load_future_forecast_timeline():
    return load_csv_safe(
        "data/future_forecast_timeline.csv",
        parse_date_cols=["date"],
    )

@st.cache_data(ttl=3600)
def load_allocation_timeline_v2():
    return load_csv_safe(
        "data/allocation_timeline_v2.csv",
        parse_date_cols=["start_date", "end_date"],
    )

@st.cache_data(ttl=3600)
def load_current_allocation_v2():
    return load_json_safe("data/current_allocation_v2.json")

@st.cache_data(ttl=3600)
def load_taxonomy_performance_validation():
    return load_csv_safe("data/taxonomy_performance_validation.csv")

@st.cache_data(ttl=3600)
def load_taxonomy_performance_by_year():
    return load_csv_safe("data/taxonomy_performance_by_year.csv")

@st.cache_data(ttl=3600)
def load_taxonomy_regime_audit():
    return load_csv_safe("data/taxonomy_regime_audit.csv")

@st.cache_data(ttl=3600)
def load_allocation_variant_results():
    return load_csv_safe("data/allocation_variant_results.csv")

@st.cache_data(ttl=3600)
def load_allocation_variant_annual():
    return load_csv_safe("data/allocation_variant_annual.csv")

@st.cache_data(ttl=3600)
def load_allocation_variant_stress_test():
    return load_csv_safe("data/allocation_variant_stress_test.csv")

@st.cache_data(ttl=3600)
def load_allocation_grid_search():
    return load_csv_safe("data/allocation_grid_search.csv")

# =========================================================
# HELPERS
# =========================================================
def fmt_money(x):
    return "N/A" if pd.isna(x) else f"${x:,.0f}"

def fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}"

def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2%}"

def format_pct(x):
    return fmt_pct(x)

def format_return(x):
    return "N/A" if pd.isna(x) else f"{x:+.2%}"

def format_date(x):
    if pd.isna(x) or x in [None, ""]:
        return "N/A"
    return pd.to_datetime(x).strftime("%Y-%m-%d")

def classify_regime(score):
    if score >= 3:
        return "strong_uptrend"
    if score >= 1.5:
        return "uptrend"
    if score <= -3:
        return "crash_risk"
    if score <= -1.5:
        return "downtrend"
    return "sideways"

def regime_color(regime):
    return {
        "strong_uptrend": "#22c55e",
        "uptrend": "#84cc16",
        "sideways": "#94a3b8",
        "downtrend": "#f59e0b",
        "crash_risk": "#ef4444",
        "reversal_zone": "#a855f7",
        "exhaustion_zone": "#f97316",
        "compression_zone": "#facc15",
    }.get(regime, "#94a3b8")

def signal_color(sig):
    return {
        "strong_buy": "#22c55e",
        "buy": "#84cc16",
        "neutral": "#94a3b8",
        "sell": "#f59e0b",
        "strong_sell": "#ef4444",
    }.get(sig, "#94a3b8")

def taxonomy_color(label):
    return {
        "High Conviction Expansion": "#D97706",
        "High Momentum Expansion": "#D97706",
        "Constructive Drift": "#2E7D32",
        "Transition / Low Conviction": "#C9A227",
        "Tactical Neutral": "#C9A227",
        "Recovery / Reversal Setup": "#7C3AED",
        "Volatility Caution": "#7F1D1D",
        "High Volatility Risk": "#7F1D1D",
        "Defensive / Weak Trend": "#C62828",
        "Constructive / Positive Drift": "#2E7D32",
        "Neutral / Tactical": "#C9A227",
        "High Risk": "#7F1D1D",
        "Bearish": "#C62828",
        "False Bull / Exhaustion Risk": "#D97706",
    }.get(label, "#475569")

def taxonomy_bg(label):
    return {
        "High Conviction Expansion": "linear-gradient(135deg, #432510 0%, #D97706 100%)",
        "High Momentum Expansion": "linear-gradient(135deg, #432510 0%, #D97706 100%)",
        "Constructive Drift": "linear-gradient(135deg, #16351d 0%, #2E7D32 100%)",
        "Transition / Low Conviction": "linear-gradient(135deg, #453f18 0%, #C9A227 100%)",
        "Tactical Neutral": "linear-gradient(135deg, #453f18 0%, #C9A227 100%)",
        "Recovery / Reversal Setup": "linear-gradient(135deg, #2c1a53 0%, #7C3AED 100%)",
        "Volatility Caution": "linear-gradient(135deg, #2b0a0a 0%, #7F1D1D 100%)",
        "High Volatility Risk": "linear-gradient(135deg, #2b0a0a 0%, #7F1D1D 100%)",
        "Defensive / Weak Trend": "linear-gradient(135deg, #3b0a0f 0%, #C62828 100%)",
        "Constructive / Positive Drift": "linear-gradient(135deg, #16351d 0%, #2E7D32 100%)",
        "Neutral / Tactical": "linear-gradient(135deg, #453f18 0%, #C9A227 100%)",
        "High Risk": "linear-gradient(135deg, #2b0a0a 0%, #7F1D1D 100%)",
        "Bearish": "linear-gradient(135deg, #3b0a0f 0%, #C62828 100%)",
        "False Bull / Exhaustion Risk": "linear-gradient(135deg, #432510 0%, #D97706 100%)",
    }.get(label, "linear-gradient(135deg, #1f2937 0%, #334155 100%)")

def taxonomy_rgba(label, alpha=0.18):
    hex_color = taxonomy_color(label).lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(71,85,105,{alpha})"
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"

def taxonomy_badge_html(label):
    label = label or "Unknown"
    return (
        f"<span class='light-badge' "
        f"style='background:{taxonomy_rgba(label, 0.12)};"
        f" color:{taxonomy_color(label)};"
        f" border:1px solid {taxonomy_rgba(label, 0.28)};'>"
        f"{label}</span>"
    )

TAXONOMY_ATTRIBUTION_LABEL_MAP = {
    "Constructive Drift": "Constructive Drift",
    "High Conviction Expansion": "High Momentum Expansion",
    "Transition / Low Conviction": "Tactical Neutral",
    "Recovery / Reversal Setup": "Defensive / Weak Trend",
    "Volatility Caution": "High Volatility Risk",
}

def risk_level_color(level):
    return {
        "Low": "#2E7D32",
        "Medium": "#C9A227",
        "High": "#C62828",
    }.get(level, "#64748b")

def safe_datetime(value):
    if pd.isna(value) or value in [None, ""]:
        return None
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None

def add_safe_vertical_marker(fig, x_value, label, line_color="#6B7280"):
    x_dt = safe_datetime(x_value)
    if x_dt is None:
        return
    try:
        fig.add_shape(
            type="line",
            x0=x_dt,
            x1=x_dt,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=line_color, width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=x_dt,
            y=1,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color=line_color, size=11),
            bgcolor="rgba(255,255,255,0.9)",
        )
    except Exception:
        return

def card_container():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()

def safe_plotly_chart(fig, key=None):
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as exc:
        st.warning(f"Unable to render chart safely: {exc}")

def parse_ranked_items(value, limit=5):
    if pd.isna(value) or value in [None, ""]:
        return []
    items = []
    for raw_item in str(value).split(", "):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        if " (" in raw_item and raw_item.endswith(")"):
            name, score_text = raw_item.rsplit(" (", 1)
            score_text = score_text[:-1]
            try:
                score = float(score_text)
            except ValueError:
                score = np.nan
        else:
            name = raw_item
            score = np.nan
        items.append({"name": name, "score": score})
    return items[:limit]

def safe_record(value):
    return value if isinstance(value, dict) else {}

def normalize_taxonomy_columns(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()
    df_norm = frame.copy()
    for source_col in ["taxonomy_v4", "taxonomy_v2", "dominant_taxonomy", "current_taxonomy"]:
        if source_col in df_norm.columns:
            df_norm["taxonomy_label"] = df_norm[source_col]
            break
    if "taxonomy_label" not in df_norm.columns:
        df_norm["taxonomy_label"] = "N/A"
    if "start_date" in df_norm.columns:
        df_norm["start_date"] = pd.to_datetime(df_norm["start_date"], errors="coerce")
    if "end_date" in df_norm.columns:
        df_norm["end_date"] = pd.to_datetime(df_norm["end_date"], errors="coerce")
    if "date" in df_norm.columns:
        df_norm["date"] = pd.to_datetime(df_norm["date"], errors="coerce")
    return df_norm

def merge_historical_and_future_frames(historical_df, future_df, on="date"):
    historical_df = historical_df.copy() if historical_df is not None else pd.DataFrame()
    future_df = future_df.copy() if future_df is not None else pd.DataFrame()
    if not historical_df.empty and on in historical_df.columns:
        historical_df[on] = pd.to_datetime(historical_df[on], errors="coerce")
    if not future_df.empty and on in future_df.columns:
        future_df[on] = pd.to_datetime(future_df[on], errors="coerce")
    if historical_df.empty:
        merged = future_df
    elif future_df.empty:
        merged = historical_df
    else:
        merged = pd.concat([historical_df, future_df], ignore_index=True, sort=False)
    if not merged.empty and on in merged.columns:
        merged = merged.dropna(subset=[on]).sort_values(on).reset_index(drop=True)
    return merged

def julday(dt):
    return swe.julday(
        dt.year, dt.month, dt.day,
        dt.hour + dt.minute / 60 + dt.second / 3600
    )

def get_planet_lon(jd, pid):
    xx, _ = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
    return xx[0]

def get_houses(jd, lat, lon):
    cusps, ascmc = swe.houses_ex(jd, lat, lon, b"P", swe.FLG_SIDEREAL)
    return cusps, ascmc

def degree_to_sign(lon):
    idx = int(lon // 30)
    deg = lon % 30
    return ZODIAC[idx][0], ZODIAC[idx][1], deg

def fmt_deg(lon):
    sign, sym, deg = degree_to_sign(lon)
    d = int(deg)
    m = int((deg - d) * 60)
    return f"{sym} {sign} {d}°{m:02d}′"

def get_natal_chart():
    jd = julday(NATAL_DT)
    _, ascmc = get_houses(jd, LAT, LON)
    natal = {}
    for name, pid in PLANETS.items():
        natal[name] = get_planet_lon(jd, pid)
    natal["Asc"] = ascmc[0]
    natal["MC"] = ascmc[1]
    return natal

def make_natal_wheel(natal):
    fig = go.Figure()
    theta = np.linspace(0, 360, 361)

    fig.add_trace(go.Scatterpolar(
        r=[1] * len(theta), theta=theta, mode="lines",
        line=dict(color="#64748b", width=2), showlegend=False
    ))

    for i, (_, sym) in enumerate(ZODIAC):
        deg = i * 30
        fig.add_trace(go.Scatterpolar(
            r=[0, 1.05], theta=[deg, deg], mode="lines",
            line=dict(color="#475569", width=1), showlegend=False
        ))
        fig.add_trace(go.Scatterpolar(
            r=[1.16], theta=[deg + 15], mode="text",
            text=[sym], textfont=dict(size=22, color="#e5e7eb"),
            showlegend=False
        ))

    r_levels = {
        "Sun": 0.88, "Moon": 0.82, "Mercury": 0.76, "Venus": 0.70,
        "Mars": 0.64, "Jupiter": 0.58, "Saturn": 0.52,
        "Uranus": 0.46, "Neptune": 0.40, "Pluto": 0.34,
        "Asc": 0.96, "MC": 0.96,
    }

    for name, lon in natal.items():
        fig.add_trace(go.Scatterpolar(
            r=[r_levels.get(name, 0.55)],
            theta=[lon],
            mode="markers+text",
            marker=dict(size=12, color="#60a5fa" if name not in ["Asc", "MC"] else "#f59e0b"),
            text=[PLANET_SYMBOLS.get(name, name)],
            textposition="top center",
            textfont=dict(size=14, color="white"),
            hovertemplate=f"{name}<br>{fmt_deg(lon)}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        template="plotly_dark",
        height=620,
        polar=dict(
            bgcolor="#020617",
            angularaxis=dict(direction="clockwise", rotation=90, showticklabels=False),
            radialaxis=dict(showticklabels=False, ticks="", range=[0, 1.22]),
        ),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig

def metric_card(label, value, sub="", color="white"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def dashboard_card(label, value, sub="", taxonomy_label=None):
    bg = taxonomy_bg(taxonomy_label) if taxonomy_label else "linear-gradient(135deg, #111827 0%, #1f2937 100%)"
    st.markdown(f"""
    <div class="taxonomy-card" style="background:{bg};">
        <div class="taxonomy-title">{label}</div>
        <div class="taxonomy-value">{value}</div>
        <div class="taxonomy-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def taxonomy_chip(label):
    color = taxonomy_color(label)
    return f"<span class='dashboard-chip' style='background:{color};'>{label}</span>"

def render_taxonomy_text(label):
    color = taxonomy_color(label)
    return f"<span style='color:{color};font-weight:700;'>{label}</span>"

def terminal_stat_card(label, value, sub="", accent="#3b82f6"):
    st.markdown(f"""
    <div class="terminal-stat" style="border-top:3px solid {accent};">
        <div class="terminal-stat-label">{label}</div>
        <div class="terminal-stat-value">{value}</div>
        <div class="terminal-stat-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def event_card_html(date_text, event_type, direction_text, direction_color, body):
    return f"""
    <div class="event-card">
        <div class="event-date">{date_text}</div>
        <div class="event-type">{event_type}</div>
        <div class="event-direction" style="color:{direction_color};">{direction_text}</div>
        <div class="event-body">{body}</div>
    </div>
    """

def risk_window_card_html(title, date_text, taxonomy, body):
    return f"""
    <div class="risk-card" style="border-left-color:{taxonomy_color(taxonomy)};">
        <div class="event-type">{title}</div>
        <div class="event-date" style="margin-top:8px;">{date_text}</div>
        <div class="event-direction" style="font-size:1rem;color:{taxonomy_color(taxonomy)};">{taxonomy}</div>
        <div class="event-body">{body}</div>
    </div>
    """

# =========================================================
# DATA
# =========================================================
df = load_daily()
raw_aspects = load_raw_aspects()
ml_pred = load_ml_predictions()
ml_summary = load_ml_summary()
ml_importance = load_ml_importance()
opt_v2 = load_optimization_v2()
dashboard_current = load_dashboard_current_state()
dashboard_timeline = load_dashboard_timeline()
dashboard_risk_calendar = load_dashboard_risk_calendar()
dashboard_summary = load_dashboard_summary()
taxonomy_attribution = load_taxonomy_attribution()
taxonomy_feature_importance = load_taxonomy_feature_importance()
forecast_intelligence_v4 = load_forecast_intelligence_v4()
future_forecast_timeline = load_future_forecast_timeline()
allocation_timeline_v2 = load_allocation_timeline_v2()
current_allocation_v2 = load_current_allocation_v2()
taxonomy_performance_validation = load_taxonomy_performance_validation()
taxonomy_performance_by_year = load_taxonomy_performance_by_year()
taxonomy_regime_audit = load_taxonomy_regime_audit()
allocation_variant_results = load_allocation_variant_results()
allocation_variant_annual = load_allocation_variant_annual()
allocation_variant_stress_test = load_allocation_variant_stress_test()
allocation_grid_search = load_allocation_grid_search()

price_df = df.dropna(subset=["price"]).copy()
if price_df.empty:
    st.error("ยังไม่มีข้อมูลราคา BTC")
    st.stop()

last_price_date = price_df["date"].max()
latest = df[df["date"] == last_price_date].iloc[-1]
latest_price = price_df.iloc[-1]["price"]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Forecast Dashboard Controls")

range_option = st.sidebar.selectbox(
    "Historical Chart Range",
    ["3M", "6M", "1Y", "2Y", "4Y", "ALL"],
    index=2
)

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon",
    [30, 90, 180, 365],
    index=3
)

taxonomy_filter_options = [
    "All",
    "Constructive Drift",
    "High Conviction Expansion",
    "Transition / Low Conviction",
    "Recovery / Reversal Setup",
    "Volatility Caution",
]
taxonomy_filter = st.sidebar.selectbox("Taxonomy Filter", taxonomy_filter_options, index=0)
show_turning_points = st.sidebar.checkbox("Show Turning Points", value=True)
show_taxonomy_overlay = st.sidebar.checkbox("Show Taxonomy Overlay", value=True)
show_ml_probability = st.sidebar.checkbox("Show ML Probability", value=True)
show_allocation_panel = st.sidebar.checkbox("Show Allocation Panel", value=True)
show_astro_environment = st.sidebar.checkbox("Show Astro Environment", value=True)
show_future_forecast_zone = st.sidebar.checkbox("Show Future Forecast Zone", value=True)
show_detail_tables = st.sidebar.checkbox("Show Detailed Tables", value=True)

with st.sidebar.expander("Research Chart Controls", expanded=False):
    indicator_options = {
        "Astro Momentum v2": "astro_momentum_v2",
        "Astro Momentum v2 Smooth": "astro_momentum_v2_smooth",
        "Bullish Score": "astro_bullish_score",
        "Bearish Score": "astro_bearish_score",
        "Reversal Score": "astro_reversal_score",
        "Volatility Score": "astro_volatility_score",
        "Compression Score": "astro_compression_score",
        "Trend Start Score": "astro_trend_start_score",
        "Trend End Score": "astro_trend_end_score",
    }
    indicator_label = st.selectbox("Astro Indicator", list(indicator_options.keys()), index=1)
    show_signals = st.checkbox("Show Rule-Based Buy/Sell Markers", value=True)

indicator_col = indicator_options[indicator_label]

min_date = price_df["date"].min()
if range_option == "ALL":
    start_date = min_date
elif range_option == "4Y":
    start_date = last_price_date - pd.Timedelta(days=365 * 4)
elif range_option == "2Y":
    start_date = last_price_date - pd.Timedelta(days=365 * 2)
elif range_option == "1Y":
    start_date = last_price_date - pd.Timedelta(days=365)
elif range_option == "6M":
    start_date = last_price_date - pd.Timedelta(days=183)
else:
    start_date = last_price_date - pd.Timedelta(days=92)

start_date = max(start_date, min_date)

hist = df[(df["date"] >= start_date) & (df["date"] <= last_price_date)].copy()
hist_price = hist.dropna(subset=["price"]).copy()
future = df[(df["date"] > last_price_date) & (df["date"] <= last_price_date + pd.Timedelta(days=forecast_days))].copy()

# =========================================================
# HEADER
# =========================================================
st.title("Bitcoin Astro Quant Dashboard")

if dashboard_current:
    def render_light_metric(label, value, helper_text="", accent="#2563EB", badge_label=None):
        with card_container():
            st.markdown(
                f"<div style='height:4px;background:{accent};border-radius:999px;margin-bottom:12px;'></div>",
                unsafe_allow_html=True,
            )
            st.caption(label.upper())
            if badge_label:
                st.markdown(taxonomy_badge_html(badge_label), unsafe_allow_html=True)
            st.markdown(f"**{value}**")
            if helper_text:
                st.caption(helper_text)

    def render_outlook_card(title, data):
        taxonomy = data.get("dominant_taxonomy", "N/A")
        with card_container():
            st.caption(title.upper())
            st.markdown(taxonomy_badge_html(taxonomy), unsafe_allow_html=True)
            st.markdown(f"**{taxonomy}**")
            st.caption(f"Average confidence {format_pct(data.get('average_confidence', np.nan))}")
            st.caption(f"Average probability {format_pct(data.get('average_probability', np.nan))}")
            st.write(data.get("summary", "No calibrated summary available."))

    def first_window_by_taxonomy(window_df, taxonomy_label):
        if window_df.empty:
            return {}
        match = window_df[window_df["taxonomy_v2"] == taxonomy_label]
        return match.iloc[0].to_dict() if not match.empty else {}

    def get_taxonomy_attribution_row(taxonomy_label):
        if taxonomy_attribution.empty or "taxonomy_v3" not in taxonomy_attribution.columns:
            return {}
        attribution_label = TAXONOMY_ATTRIBUTION_LABEL_MAP.get(taxonomy_label, taxonomy_label)
        match = taxonomy_attribution[taxonomy_attribution["taxonomy_v3"] == attribution_label]
        return match.iloc[0].to_dict() if not match.empty else {}

    def get_top_feature_rows(taxonomy_label, limit=5):
        if taxonomy_feature_importance.empty or "taxonomy_v3" not in taxonomy_feature_importance.columns:
            return pd.DataFrame()
        attribution_label = TAXONOMY_ATTRIBUTION_LABEL_MAP.get(taxonomy_label, taxonomy_label)
        match = taxonomy_feature_importance[taxonomy_feature_importance["taxonomy_v3"] == attribution_label].copy()
        if match.empty:
            return match
        sort_col = "abs_zscore_diff" if "abs_zscore_diff" in match.columns else "differential"
        match = match.sort_values(sort_col, ascending=False).head(limit).copy()
        return match

    def render_driver_list(title, items, empty_text="No strong drivers detected."):
        st.caption(title.upper())
        if not items:
            st.write(empty_text)
            return
        for item in items:
            score = item.get("score", np.nan)
            score_text = "" if pd.isna(score) else f" ({score:+.2f})"
            st.write(f"• {item.get('name', 'Unknown')}{score_text}")

    def render_contribution_bars(feature_df, key_suffix):
        if feature_df.empty or "feature" not in feature_df.columns:
            st.info("Contribution bars are unavailable for this taxonomy.")
            return
        plot_df = feature_df.copy()
        if "zscore_diff" in plot_df.columns:
            plot_df["contribution_value"] = plot_df["zscore_diff"].fillna(plot_df.get("differential", 0.0))
        else:
            plot_df["contribution_value"] = plot_df.get("differential", 0.0)
        plot_df["bar_color"] = np.where(plot_df["contribution_value"] >= 0, "#2E7D32", "#C62828")
        plot_df = plot_df.sort_values("contribution_value", ascending=True)

        fig = go.Figure(
            go.Bar(
                x=plot_df["contribution_value"],
                y=plot_df["feature"],
                orientation="h",
                marker=dict(color=plot_df["bar_color"]),
                text=plot_df["contribution_value"].map(lambda x: f"{x:+.2f}"),
                textposition="outside",
                hovertemplate=(
                    "Feature: %{y}<br>"
                    "Contribution: %{x:+.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            height=max(240, 52 * len(plot_df)),
            margin=dict(l=12, r=12, t=8, b=8),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            xaxis_title="Relative Contribution",
            yaxis_title="",
            font=dict(color="#111827"),
            showlegend=False,
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zeroline=True, zerolinecolor="#CBD5E1")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True, key=f"contribution_{key_suffix}")

    def render_state_explanation_card(title, taxonomy_label, confidence_value, probability_value, summary_text, key_suffix):
        attribution_row = get_taxonomy_attribution_row(taxonomy_label)
        feature_rows = get_top_feature_rows(taxonomy_label, limit=5)
        planets = parse_ranked_items(attribution_row.get("most_influential_planets"), limit=3)
        aspects = parse_ranked_items(attribution_row.get("most_influential_aspects"), limit=3)
        positive_features = parse_ranked_items(attribution_row.get("top_positive_astro_features"), limit=3)
        with card_container():
            st.caption(title.upper())
            st.markdown(taxonomy_badge_html(taxonomy_label), unsafe_allow_html=True)
            st.markdown(f"**{taxonomy_label}**")
            st.caption(
                f"Confidence {format_pct(confidence_value)} | Probability {format_pct(probability_value)}"
            )
            if summary_text:
                st.write(summary_text)
            if positive_features:
                st.caption("DOMINANT DRIVERS")
                st.write(", ".join(item["name"] for item in positive_features))
            if planets:
                st.caption("DOMINANT PLANETS")
                st.write(", ".join(item["name"] for item in planets))
            if aspects:
                st.caption("DOMINANT ASPECTS")
                st.write(", ".join(item["name"] for item in aspects))
            if not feature_rows.empty:
                render_contribution_bars(feature_rows.head(3), key_suffix)

    top_dashboard_windows = pd.DataFrame(dashboard_timeline.get("windows", []))
    top_turning_points = pd.DataFrame(dashboard_risk_calendar.get("turning_points", []))
    top_risk_windows = pd.DataFrame(dashboard_risk_calendar.get("risk_windows", []))
    top_current_window = safe_record(dashboard_current.get("current_window"))
    top_next_turning = safe_record(dashboard_current.get("next_turning_point"))
    top_next_constructive = safe_record(dashboard_current.get("next_constructive_window"))
    top_next_high_risk = safe_record(dashboard_current.get("next_high_risk_window"))
    top_30d = safe_record(dashboard_summary.get("30D Outlook"))
    top_90d = safe_record(dashboard_summary.get("90D Outlook"))
    top_365d = safe_record(dashboard_summary.get("365D Outlook"))
    prev_price = price_df[price_df["date"] < last_price_date]["price"].iloc[-1] if len(price_df) > 1 else latest_price
    price_delta = latest_price - prev_price
    price_delta_pct = (price_delta / prev_price) if prev_price else 0.0
    forecast_generated_date = format_date(dashboard_current.get("forecast_date"))
    forecast_cutoff = last_price_date + pd.Timedelta(days=forecast_days)

    if not top_dashboard_windows.empty:
        top_dashboard_windows["start_date"] = pd.to_datetime(top_dashboard_windows["start_date"])
        top_dashboard_windows["end_date"] = pd.to_datetime(top_dashboard_windows["end_date"])
        top_dashboard_windows = top_dashboard_windows.sort_values("start_date").reset_index(drop=True)
        top_dashboard_windows = top_dashboard_windows[top_dashboard_windows["start_date"] <= forecast_cutoff].copy()

    if not top_turning_points.empty:
        top_turning_points["turning_point_date"] = pd.to_datetime(top_turning_points["turning_point_date"])
        top_turning_points = top_turning_points.sort_values("turning_point_date").reset_index(drop=True)
        top_turning_points = top_turning_points[top_turning_points["turning_point_date"] <= forecast_cutoff].copy()

    if not top_risk_windows.empty:
        top_risk_windows["start_date"] = pd.to_datetime(top_risk_windows["start_date"])
        top_risk_windows["end_date"] = pd.to_datetime(top_risk_windows["end_date"])
        top_risk_windows = top_risk_windows.sort_values("start_date").reset_index(drop=True)
        top_risk_windows = top_risk_windows[top_risk_windows["start_date"] <= forecast_cutoff].copy()

    if taxonomy_filter != "All" and not top_dashboard_windows.empty:
        top_dashboard_windows = top_dashboard_windows[top_dashboard_windows["taxonomy_v2"] == taxonomy_filter].copy()

    if top_dashboard_windows.empty and taxonomy_filter != "All":
        st.warning(f"No forecast windows match the taxonomy filter `{taxonomy_filter}` for the selected horizon.")

    next_risk_window = top_risk_windows.iloc[0].to_dict() if not top_risk_windows.empty else {}
    next_recovery_window = first_window_by_taxonomy(top_dashboard_windows, "Recovery / Reversal Setup")
    next_momentum_window = first_window_by_taxonomy(top_dashboard_windows, "High Conviction Expansion")
    st.subheader("Evidence-based BTC forecast timeline powered by Robust Astro Engine v1")
    st.caption(f"Forecast generated: {forecast_generated_date}")
    st.caption("Research tool only. Not financial advice.")

    st.markdown("#### Executive Summary")
    summary_cols = st.columns(6)
    summary_cards = [
        ("Current Market View", dashboard_current.get("market_view", "N/A"), dashboard_current.get("current_signal", "N/A"), taxonomy_color(dashboard_current.get("current_taxonomy")), None),
        ("Current Taxonomy", dashboard_current.get("current_taxonomy", "N/A"), top_current_window.get("v2_posture", "N/A"), taxonomy_color(dashboard_current.get("current_taxonomy")), dashboard_current.get("current_taxonomy")),
        ("Confidence", format_pct(dashboard_current.get("current_confidence", np.nan)), f"Probability {format_pct(dashboard_current.get('current_probability', np.nan))}", "#2563EB", None),
        ("Bias", dashboard_current.get("recommended_bias", "N/A"), top_current_window.get("taxonomy_reason", top_current_window.get("narrative_v2", "No narrative available."))[:110], taxonomy_color(dashboard_current.get("current_taxonomy")), None),
        ("Risk", dashboard_current.get("risk_level", "N/A"), f"Next high-risk window {format_date(top_next_high_risk.get('start_date'))}", risk_level_color(dashboard_current.get("risk_level")), None),
        ("BTC Price", fmt_money(latest_price), f"{price_delta_pct:+.2%} vs prior close", "#2563EB" if price_delta >= 0 else "#DC2626", None),
    ]
    for col, (label, value, helper_text, accent, badge_label) in zip(summary_cols, summary_cards):
        with col:
            render_light_metric(label, value, helper_text, accent=accent, badge_label=badge_label)

    st.markdown("#### Integrated BTC Market Regime Map")
    st.caption(
        "A unified market-environment map combining BTC price history, ML probability, Taxonomy v4, Portfolio Allocation v2, and the future forecast regime."
    )

    integrated_current_allocation = safe_record(current_allocation_v2)
    integrated_forecast_windows = normalize_taxonomy_columns(forecast_intelligence_v4)
    if integrated_forecast_windows.empty:
        integrated_forecast_windows = normalize_taxonomy_columns(top_dashboard_windows)
    integrated_allocation_windows = allocation_timeline_v2.copy() if not allocation_timeline_v2.empty else pd.DataFrame()
    integrated_future_forecast = future_forecast_timeline.copy() if not future_forecast_timeline.empty else pd.DataFrame()
    integrated_missing_inputs = []
    if ml_pred.empty:
        integrated_missing_inputs.append("historical ML probability")
    if integrated_forecast_windows.empty:
        integrated_missing_inputs.append("forecast taxonomy windows")
    if integrated_allocation_windows.empty:
        integrated_missing_inputs.append("allocation_timeline_v2")
    if integrated_future_forecast.empty:
        integrated_missing_inputs.append("future_forecast_timeline")
    if integrated_missing_inputs:
        st.warning(
            "Integrated view is using partial data because these inputs are missing or malformed: "
            + ", ".join(integrated_missing_inputs)
            + "."
        )

    current_summary_cols = st.columns(7)
    current_summary_cards = [
        ("Current Taxonomy v4", integrated_current_allocation.get("current_taxonomy", dashboard_current.get("current_taxonomy", "N/A")), dashboard_current.get("market_view", "N/A"), taxonomy_color(integrated_current_allocation.get("current_taxonomy", dashboard_current.get("current_taxonomy", "N/A"))), integrated_current_allocation.get("current_taxonomy", dashboard_current.get("current_taxonomy", "N/A"))),
        ("Current Signal", integrated_current_allocation.get("current_signal", dashboard_current.get("current_signal", "N/A")), dashboard_current.get("recommended_bias", "N/A"), "#2563EB", None),
        ("Current ML Probability", format_pct(integrated_current_allocation.get("current_ml_probability", dashboard_current.get("current_probability", np.nan))), "Robust Astro Engine v1 probability", "#2563EB", None),
        ("Current Confidence", format_pct(integrated_current_allocation.get("current_confidence", dashboard_current.get("current_confidence", np.nan))), dashboard_current.get("risk_level", "N/A"), "#2563EB", None),
        ("Current BTC Allocation v2", format_pct((integrated_current_allocation.get("adjusted_btc_allocation", np.nan) or 0) / 100.0), integrated_current_allocation.get("allocation_posture", "N/A"), "#2E7D32", None),
        ("Current Cash Allocation", format_pct((integrated_current_allocation.get("cash_allocation", np.nan) or 0) / 100.0), integrated_current_allocation.get("recommended_variant", "N/A"), "#6B7280", None),
        ("Next Review Date", format_date(integrated_current_allocation.get("next_review_date", top_next_turning.get("turning_point_date"))), "Allocation review checkpoint", "#7C3AED", None),
    ]
    for col, (label, value, helper_text, accent, badge_label) in zip(current_summary_cols, current_summary_cards):
        with col:
            render_light_metric(label, value, helper_text, accent=accent, badge_label=badge_label)

    recommended_variant_key = integrated_allocation_windows["variant_key"].iloc[0] if not integrated_allocation_windows.empty else None
    if integrated_current_allocation.get("recommended_variant") and not integrated_allocation_windows.empty:
        recommended_matches = integrated_allocation_windows[
            integrated_allocation_windows["variant_key"].astype(str).str.contains(
                str(integrated_current_allocation.get("recommended_variant", "")).split()[0],
                case=False,
                na=False,
            )
        ]
        if not recommended_matches.empty:
            recommended_variant_key = recommended_matches["variant_key"].iloc[0]
    if recommended_variant_key and not integrated_allocation_windows.empty:
        integrated_allocation_windows = integrated_allocation_windows[
            integrated_allocation_windows["variant_key"] == recommended_variant_key
        ].copy()

    if not integrated_forecast_windows.empty:
        integrated_forecast_windows = integrated_forecast_windows[
            integrated_forecast_windows["start_date"] <= forecast_cutoff
        ].copy()
    if not integrated_allocation_windows.empty:
        integrated_allocation_windows = integrated_allocation_windows[
            integrated_allocation_windows["start_date"] <= forecast_cutoff
        ].copy()
    if not integrated_future_forecast.empty:
        integrated_future_forecast = integrated_future_forecast[
            integrated_future_forecast["date"] <= forecast_cutoff
        ].copy()

    historical_ml = pd.DataFrame()
    if not ml_pred.empty and {"date", "horizon", "ml_prob_up"}.issubset(ml_pred.columns):
        historical_ml = ml_pred.copy()
        historical_ml["date"] = pd.to_datetime(historical_ml["date"], errors="coerce")
        historical_ml["ml_prob_up"] = pd.to_numeric(historical_ml["ml_prob_up"], errors="coerce")
        historical_ml = historical_ml.dropna(subset=["date", "ml_prob_up"])
        historical_ml = historical_ml[historical_ml["horizon"].isin([7, 14, 30, 60])]
        if historical_ml.empty:
            historical_ml = ml_pred.copy()
            historical_ml["date"] = pd.to_datetime(historical_ml["date"], errors="coerce")
            historical_ml["ml_prob_up"] = pd.to_numeric(historical_ml["ml_prob_up"], errors="coerce")
            historical_ml = historical_ml.dropna(subset=["date", "ml_prob_up"])
        historical_ml = historical_ml.groupby("date", as_index=False)["ml_prob_up"].mean()

    future_taxonomy_daily_rows = []
    if not integrated_forecast_windows.empty:
        for _, row in integrated_forecast_windows.iterrows():
            if pd.isna(row.get("start_date")) or pd.isna(row.get("end_date")):
                continue
            daily_range = pd.date_range(row["start_date"], row["end_date"], freq="D")
            for dt in daily_range:
                future_taxonomy_daily_rows.append(
                    {
                        "date": dt,
                        "taxonomy_label": row.get("taxonomy_label", "N/A"),
                        "confidence": row.get("average_confidence", np.nan),
                        "ml_probability": row.get("average_ml_probability", np.nan),
                        "astro_score": row.get("average_astro_score", np.nan),
                        "window_explanation": row.get("narrative_v4", row.get("taxonomy_reason", "")),
                    }
                )
    future_taxonomy_daily = pd.DataFrame(future_taxonomy_daily_rows)

    future_allocation_daily_rows = []
    if not integrated_allocation_windows.empty:
        for _, row in integrated_allocation_windows.iterrows():
            if pd.isna(row.get("start_date")) or pd.isna(row.get("end_date")):
                continue
            daily_range = pd.date_range(row["start_date"], row["end_date"], freq="D")
            for dt in daily_range:
                future_allocation_daily_rows.append(
                    {
                        "date": dt,
                        "btc_allocation": row.get("btc_allocation", np.nan),
                        "cash_allocation": row.get("cash_allocation", np.nan),
                        "allocation_posture": row.get("allocation_posture", ""),
                        "allocation_explanation": row.get("explanation", ""),
                    }
                )
    future_allocation_daily = pd.DataFrame(future_allocation_daily_rows)

    integrated_context = merge_historical_and_future_frames(
        historical_ml.rename(columns={"ml_prob_up": "historical_ml_probability"}),
        integrated_future_forecast.rename(
            columns={
                "ml_probability": "future_ml_probability",
                "confidence_score": "future_confidence",
                "astro_score": "future_astro_score",
            }
        ),
    )
    if not future_taxonomy_daily.empty:
        integrated_context = integrated_context.merge(future_taxonomy_daily, on="date", how="outer")
    if not future_allocation_daily.empty:
        integrated_context = integrated_context.merge(future_allocation_daily, on="date", how="outer")
    integrated_context = integrated_context.sort_values("date").drop_duplicates(subset=["date"], keep="last") if not integrated_context.empty else pd.DataFrame()

    panel_flags = [
        ("BTCUSD Price", True),
        ("Taxonomy v4 Regime", True),
        ("ML Probability", show_ml_probability),
        ("Portfolio Allocation v2", show_allocation_panel),
        ("Astro / Forecast Environment", show_astro_environment),
    ]
    active_panels = [name for name, flag in panel_flags if flag]
    row_height_map = {
        "BTCUSD Price": 0.34,
        "Taxonomy v4 Regime": 0.12,
        "ML Probability": 0.18,
        "Portfolio Allocation v2": 0.18,
        "Astro / Forecast Environment": 0.18,
    }
    total_height_units = sum(row_height_map[name] for name in active_panels)
    row_heights = [row_height_map[name] / total_height_units for name in active_panels]
    row_lookup = {name: idx + 1 for idx, name in enumerate(active_panels)}

    chart_price = price_df[(price_df["date"] >= start_date) & (price_df["date"] <= last_price_date)].copy()
    if chart_price.empty:
        st.warning("Historical BTC price data is unavailable for the integrated regime map.")
    else:
        integrated_fig = make_subplots(
            rows=len(active_panels),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=active_panels,
        )
        integrated_chart_end = safe_datetime(max(forecast_cutoff, last_price_date)) or safe_datetime(last_price_date)
        forecast_start_dt = safe_datetime(last_price_date)
        forecast_end_dt = safe_datetime(forecast_cutoff)

        if show_future_forecast_zone and forecast_start_dt and forecast_end_dt:
            integrated_fig.add_shape(
                type="rect",
                x0=forecast_start_dt,
                x1=forecast_end_dt,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                fillcolor="rgba(148,163,184,0.10)",
                line=dict(width=0),
                layer="below",
            )

        if "BTCUSD Price" in row_lookup:
            price_context = chart_price[["date", "price"]].copy()
            if not integrated_context.empty:
                price_context = price_context.merge(integrated_context, on="date", how="left")
            price_context["taxonomy_label"] = price_context.get("taxonomy_label", pd.Series(index=price_context.index)).fillna("Historical")
            price_context["display_probability"] = price_context.get("historical_ml_probability", pd.Series(index=price_context.index)).fillna(price_context.get("future_ml_probability", np.nan))
            price_context["display_confidence"] = price_context.get("future_confidence", pd.Series(index=price_context.index))
            price_context["window_explanation"] = price_context.get("window_explanation", pd.Series(index=price_context.index)).fillna("Historical BTC price history.")
            price_custom = np.column_stack(
                [
                    price_context["taxonomy_label"].astype(str),
                    price_context["display_probability"].map(format_pct),
                    price_context["display_confidence"].map(format_pct),
                    price_context["window_explanation"].astype(str),
                ]
            )
            integrated_fig.add_trace(
                go.Scatter(
                    x=price_context["date"].dt.to_pydatetime(),
                    y=price_context["price"],
                    mode="lines",
                    name="BTCUSD Price",
                    line=dict(color="#2563EB", width=2.4),
                    customdata=price_custom,
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "BTC Price: $%{y:,.0f}<br>"
                        "Taxonomy: %{customdata[0]}<br>"
                        "ML Probability: %{customdata[1]}<br>"
                        "Confidence: %{customdata[2]}<br>"
                        "Explanation: %{customdata[3]}<extra></extra>"
                    ),
                ),
                row=row_lookup["BTCUSD Price"],
                col=1,
            )
            if show_taxonomy_overlay and not integrated_forecast_windows.empty:
                for _, row in integrated_forecast_windows.iterrows():
                    x0 = safe_datetime(max(row["start_date"], start_date))
                    x1 = safe_datetime(min(row["end_date"], forecast_cutoff))
                    if x0 is None or x1 is None:
                        continue
                    integrated_fig.add_vrect(
                        x0=x0,
                        x1=x1,
                        fillcolor=taxonomy_rgba(row["taxonomy_label"], 0.10),
                        layer="below",
                        line_width=0,
                        row=row_lookup["BTCUSD Price"],
                        col=1,
                    )

        if "Taxonomy v4 Regime" in row_lookup:
            if not integrated_forecast_windows.empty:
                for _, row in integrated_forecast_windows.iterrows():
                    integrated_fig.add_trace(
                        go.Scatter(
                            x=[row["start_date"], row["end_date"]],
                            y=[1, 1],
                            mode="lines",
                            line=dict(color=taxonomy_color(row["taxonomy_label"]), width=20),
                            name=row["taxonomy_label"],
                            customdata=[[row.get("narrative_v4", row.get("taxonomy_reason", ""))]] * 2,
                            hovertemplate=(
                                "Window: %{x|%Y-%m-%d}<br>"
                                "Taxonomy: " + str(row["taxonomy_label"]) + "<br>"
                                "Explanation: %{customdata[0]}<extra></extra>"
                            ),
                            showlegend=False,
                        ),
                        row=row_lookup["Taxonomy v4 Regime"],
                        col=1,
                    )
            integrated_fig.update_yaxes(
                row=row_lookup["Taxonomy v4 Regime"],
                col=1,
                range=[0.7, 1.3],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            )

        if "ML Probability" in row_lookup:
            if not historical_ml.empty:
                integrated_fig.add_trace(
                    go.Scatter(
                        x=historical_ml["date"].dt.to_pydatetime(),
                        y=historical_ml["ml_prob_up"],
                        mode="lines",
                        name="Historical ML Probability",
                        line=dict(color="#0F766E", width=2),
                        hovertemplate=(
                            "Date: %{x|%Y-%m-%d}<br>"
                            "ML Probability: %{y:.2%}<extra></extra>"
                        ),
                    ),
                    row=row_lookup["ML Probability"],
                    col=1,
                )
            if not integrated_future_forecast.empty:
                integrated_fig.add_trace(
                    go.Scatter(
                        x=integrated_future_forecast["date"].dt.to_pydatetime(),
                        y=integrated_future_forecast["ml_probability"],
                        mode="lines",
                        name="Future ML Probability",
                        line=dict(color="#14B8A6", width=2, dash="dot"),
                        customdata=np.column_stack(
                            [
                                integrated_future_forecast["confidence_score"].map(format_pct),
                                integrated_future_forecast["signal"].astype(str),
                                integrated_future_forecast["risk_level"].astype(str),
                            ]
                        ),
                        hovertemplate=(
                            "Date: %{x|%Y-%m-%d}<br>"
                            "ML Probability: %{y:.2%}<br>"
                            "Confidence: %{customdata[0]}<br>"
                            "Signal: %{customdata[1]}<br>"
                            "Risk: %{customdata[2]}<extra></extra>"
                        ),
                    ),
                    row=row_lookup["ML Probability"],
                    col=1,
                )
            for threshold, color in [(0.45, "#C9A227"), (0.50, "#64748B"), (0.60, "#D97706")]:
                try:
                    integrated_fig.add_hline(
                        y=threshold,
                        line_width=1,
                        line_dash="dot",
                        line_color=color,
                        row=row_lookup["ML Probability"],
                        col=1,
                    )
                except Exception:
                    pass
            integrated_fig.update_yaxes(
                row=row_lookup["ML Probability"],
                col=1,
                range=[0, 1],
                tickformat=".0%",
            )

        if "Portfolio Allocation v2" in row_lookup:
            allocation_start_line = pd.DataFrame(
                [
                    {
                        "date": last_price_date,
                        "btc_allocation": integrated_current_allocation.get("adjusted_btc_allocation", np.nan),
                        "cash_allocation": integrated_current_allocation.get("cash_allocation", np.nan),
                        "allocation_posture": integrated_current_allocation.get("allocation_posture", "N/A"),
                        "allocation_explanation": integrated_current_allocation.get("explanation", ""),
                    }
                ]
            )
            allocation_plot = merge_historical_and_future_frames(allocation_start_line, future_allocation_daily)
            if not allocation_plot.empty:
                integrated_fig.add_trace(
                    go.Scatter(
                        x=allocation_plot["date"].dt.to_pydatetime(),
                        y=allocation_plot["btc_allocation"],
                        mode="lines",
                        name="BTC Allocation %",
                        line=dict(color="#7C3AED", width=2.2, shape="hv"),
                        customdata=np.column_stack(
                            [
                                allocation_plot["cash_allocation"].fillna(100 - allocation_plot["btc_allocation"]).map(lambda x: "N/A" if pd.isna(x) else f"{x:.0f}%"),
                                allocation_plot["allocation_posture"].fillna("N/A").astype(str),
                                allocation_plot["allocation_explanation"].fillna("").astype(str),
                            ]
                        ),
                        hovertemplate=(
                            "Date: %{x|%Y-%m-%d}<br>"
                            "BTC Allocation: %{y:.0f}%<br>"
                            "Cash Allocation: %{customdata[0]}<br>"
                            "Posture: %{customdata[1]}<br>"
                            "Explanation: %{customdata[2]}<extra></extra>"
                        ),
                    ),
                    row=row_lookup["Portfolio Allocation v2"],
                    col=1,
                )
            integrated_fig.update_yaxes(
                row=row_lookup["Portfolio Allocation v2"],
                col=1,
                range=[0, 100],
                ticksuffix="%",
            )

        if "Astro / Forecast Environment" in row_lookup:
            astro_hist = hist.copy()
            astro_hist["astro_env_value"] = pd.to_numeric(
                astro_hist.get("astro_momentum_v2_smooth", astro_hist.get("astro_score")),
                errors="coerce",
            )
            astro_hist = astro_hist.dropna(subset=["date", "astro_env_value"])
            if not astro_hist.empty:
                integrated_fig.add_trace(
                    go.Scatter(
                        x=astro_hist["date"].dt.to_pydatetime(),
                        y=astro_hist["astro_env_value"],
                        mode="lines",
                        name="Historical Astro Environment",
                        line=dict(color="#8B5CF6", width=2),
                        hovertemplate=(
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Astro Environment: %{y:.2f}<extra></extra>"
                        ),
                    ),
                    row=row_lookup["Astro / Forecast Environment"],
                    col=1,
                )
            if not integrated_future_forecast.empty:
                integrated_fig.add_trace(
                    go.Scatter(
                        x=integrated_future_forecast["date"].dt.to_pydatetime(),
                        y=integrated_future_forecast["astro_score"],
                        mode="lines",
                        name="Future Astro Environment",
                        line=dict(color="#A78BFA", width=2, dash="dot"),
                        customdata=np.column_stack(
                            [
                                integrated_future_forecast["signal"].astype(str),
                                integrated_future_forecast["risk_level"].astype(str),
                                integrated_future_forecast["confidence_score"].map(format_pct),
                            ]
                        ),
                        hovertemplate=(
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Astro Score: %{y:.2f}<br>"
                            "Signal: %{customdata[0]}<br>"
                            "Risk: %{customdata[1]}<br>"
                            "Confidence: %{customdata[2]}<extra></extra>"
                        ),
                    ),
                    row=row_lookup["Astro / Forecast Environment"],
                    col=1,
                )

        if forecast_start_dt:
            try:
                integrated_fig.add_shape(
                    type="line",
                    x0=forecast_start_dt,
                    x1=forecast_start_dt,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color="#6B7280", width=1.5, dash="dash"),
                )
                integrated_fig.add_annotation(
                    x=forecast_start_dt,
                    y=1,
                    xref="x",
                    yref="paper",
                    text="Forecast Start",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="#6B7280", size=11),
                    bgcolor="rgba(255,255,255,0.92)",
                )
            except Exception:
                pass

        integrated_fig.update_layout(
            height=920,
            margin=dict(l=24, r=24, t=42, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(255,255,255,0.9)"),
            font=dict(color="#111827"),
        )
        integrated_fig.update_xaxes(
            range=[safe_datetime(start_date), integrated_chart_end],
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
        )
        integrated_fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zeroline=False)
        st.plotly_chart(integrated_fig, use_container_width=True)

    st.caption(
        "This chart does not forecast BTC price directly. It forecasts the market environment and recommended BTC/Cash exposure. Future BTC price is intentionally not drawn."
    )

    st.markdown("#### Future Regime Roadmap")
    roadmap_df = integrated_allocation_windows.copy()
    if not roadmap_df.empty:
        if not integrated_forecast_windows.empty:
            roadmap_df = roadmap_df.merge(
                integrated_forecast_windows[
                    [
                        "start_date",
                        "end_date",
                        "taxonomy_label",
                        "average_confidence",
                        "average_ml_probability",
                        "narrative_v4",
                    ]
                ],
                on=["start_date", "end_date"],
                how="left",
            )
        roadmap_df["display_taxonomy"] = roadmap_df.get("taxonomy_v4", roadmap_df.get("taxonomy_label", "N/A"))
        roadmap_df["display_confidence"] = roadmap_df.get("confidence", roadmap_df.get("average_confidence", np.nan))
        roadmap_df["display_probability"] = roadmap_df.get("ml_probability", roadmap_df.get("average_ml_probability", np.nan))
        roadmap_df["display_explanation"] = roadmap_df.get("narrative_v4", roadmap_df.get("explanation", ""))
        roadmap_display = roadmap_df[
            [
                "start_date",
                "end_date",
                "display_taxonomy",
                "btc_allocation",
                "cash_allocation",
                "display_probability",
                "display_confidence",
                "allocation_posture",
                "display_explanation",
            ]
        ].copy()
        roadmap_display.columns = [
            "start_date",
            "end_date",
            "taxonomy_v4",
            "BTC allocation",
            "Cash allocation",
            "ML probability",
            "confidence",
            "allocation posture",
            "explanation",
        ]
        roadmap_display["start_date"] = roadmap_display["start_date"].map(format_date)
        roadmap_display["end_date"] = roadmap_display["end_date"].map(format_date)
        roadmap_display["BTC allocation"] = roadmap_display["BTC allocation"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.0f}%")
        roadmap_display["Cash allocation"] = roadmap_display["Cash allocation"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.0f}%")
        roadmap_display["ML probability"] = roadmap_display["ML probability"].map(format_pct)
        roadmap_display["confidence"] = roadmap_display["confidence"].map(format_pct)
        st.dataframe(roadmap_display, use_container_width=True, hide_index=True)
    else:
        st.info("Future allocation roadmap is unavailable because `allocation_timeline_v2.csv` could not be loaded.")

    st.markdown("#### BTC Price + Forecast Taxonomy Overlay")
    st.caption("Historical range comes from the sidebar control. Forecast horizon controls future taxonomy windows and turning points.")

    chart_windows = top_dashboard_windows.copy()
    chart_turning_points = top_turning_points.copy()
    chart_start = safe_datetime(start_date) or safe_datetime(min_date)
    chart_end = safe_datetime(max(forecast_cutoff, last_price_date)) or safe_datetime(last_price_date)
    chart_price = price_df[(price_df["date"] >= start_date) & (price_df["date"] <= last_price_date)].copy()

    if chart_price.empty:
        st.warning("BTC price history is unavailable for the selected range.")
    else:
        chart_price["date"] = pd.to_datetime(chart_price["date"], errors="coerce")
        chart_price = chart_price.dropna(subset=["date", "price"]).copy()

        try:
            price_fig = go.Figure()

            if show_taxonomy_overlay and not chart_windows.empty:
                for _, row in chart_windows.iterrows():
                    x0 = safe_datetime(max(row["start_date"], start_date))
                    x1 = safe_datetime(min(row["end_date"], forecast_cutoff))
                    if x0 is None or x1 is None:
                        continue
                    price_fig.add_vrect(
                        x0=x0,
                        x1=x1,
                        fillcolor=taxonomy_rgba(row["taxonomy_v2"], 0.10 if row["taxonomy_v2"] not in {"High Volatility Risk", "Volatility Caution"} else 0.15),
                        layer="below",
                        line_width=0,
                    )

            forecast_start_dt = safe_datetime(last_price_date)
            forecast_end_dt = safe_datetime(forecast_cutoff)
            if forecast_start_dt and forecast_end_dt:
                price_fig.add_vrect(
                    x0=forecast_start_dt,
                    x1=forecast_end_dt,
                    fillcolor="rgba(148,163,184,0.08)",
                    layer="below",
                    line_width=0,
                    annotation_text="Forecast Zone",
                    annotation_position="top left",
                    annotation_font_color="#6B7280",
                )

            hist_custom = np.column_stack(
                [
                    np.full(len(chart_price), "Historical / Live"),
                    np.full(len(chart_price), "N/A"),
                    np.full(len(chart_price), "N/A"),
                ]
            )
            price_fig.add_trace(
                go.Scatter(
                    x=chart_price["date"].dt.to_pydatetime(),
                    y=chart_price["price"],
                    mode="lines",
                    name="BTCUSD Price",
                    line=dict(color="#2563EB", width=2.6),
                    customdata=hist_custom,
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "BTC Price: $%{y:,.0f}<br>"
                        "Taxonomy: %{customdata[0]}<br>"
                        "Confidence: %{customdata[1]}<br>"
                        "Probability: %{customdata[2]}<extra></extra>"
                    ),
                )
            )

            forecast_context_rows = []
            if not chart_windows.empty:
                for _, row in chart_windows.iterrows():
                    date_range = pd.date_range(max(row["start_date"], last_price_date), min(row["end_date"], forecast_cutoff), freq="D")
                    for dt in date_range:
                        forecast_context_rows.append(
                            {
                                "date": dt,
                                "taxonomy_v2": row["taxonomy_v2"],
                                "average_confidence": row["average_confidence"],
                                "average_ml_probability": row["average_ml_probability"],
                            }
                        )

            forecast_context = pd.DataFrame(forecast_context_rows)
            if not forecast_context.empty:
                forecast_context["date"] = pd.to_datetime(forecast_context["date"], errors="coerce")
                forecast_context = forecast_context.dropna(subset=["date"]).groupby("date", as_index=False).last()
                forecast_context["reference_price"] = latest_price
                future_custom = np.column_stack(
                    [
                        forecast_context["taxonomy_v2"].astype(str),
                        forecast_context["average_confidence"].map(format_pct),
                        forecast_context["average_ml_probability"].map(format_pct),
                    ]
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=forecast_context["date"].dt.to_pydatetime(),
                        y=forecast_context["reference_price"],
                        mode="lines",
                        name="Forecast Baseline",
                        line=dict(color="#94A3B8", width=1.5, dash="dot"),
                        customdata=future_custom,
                        hovertemplate=(
                            "Date: %{x|%Y-%m-%d}<br>"
                            "BTC Price: $%{y:,.0f} (last close reference)<br>"
                            "Taxonomy: %{customdata[0]}<br>"
                            "Confidence: %{customdata[1]}<br>"
                            "Probability: %{customdata[2]}<extra></extra>"
                        ),
                    )
                )

            if show_turning_points and not chart_turning_points.empty:
                chart_turning_points = chart_turning_points[chart_turning_points["turning_point_date"] >= start_date].copy()
                if not chart_turning_points.empty:
                    chart_turning_points["turning_point_date"] = pd.to_datetime(chart_turning_points["turning_point_date"], errors="coerce")
                    chart_turning_points = chart_turning_points.dropna(subset=["turning_point_date"])
                    marker_map = {
                        "bullish": ("triangle-up", "#16A34A"),
                        "bearish": ("triangle-down", "#DC2626"),
                    }
                    chart_turning_points["marker_symbol"] = chart_turning_points["new_signal"].str.lower().map(lambda x: marker_map.get(x, ("circle", "#6B7280"))[0])
                    chart_turning_points["marker_color"] = chart_turning_points["new_signal"].str.lower().map(lambda x: marker_map.get(x, ("circle", "#6B7280"))[1])
                    chart_turning_points["plot_y"] = chart_price["price"].max() * 1.02
                    price_fig.add_trace(
                        go.Scatter(
                            x=chart_turning_points["turning_point_date"].dt.to_pydatetime(),
                            y=chart_turning_points["plot_y"],
                            mode="markers",
                            name="Turning Points",
                            marker=dict(
                                symbol=chart_turning_points["marker_symbol"],
                                size=12,
                                color=chart_turning_points["marker_color"],
                                line=dict(width=1, color="#FFFFFF"),
                            ),
                            customdata=np.column_stack(
                                [
                                    chart_turning_points["turning_point_type"].astype(str),
                                    chart_turning_points["old_signal"].astype(str),
                                    chart_turning_points["new_signal"].astype(str),
                                    chart_turning_points["confidence"].map(format_pct),
                                ]
                            ),
                            hovertemplate=(
                                "Date: %{x|%Y-%m-%d}<br>"
                                "Turning point: %{customdata[0]}<br>"
                                "Signal shift: %{customdata[1]} -> %{customdata[2]}<br>"
                                "Confidence: %{customdata[3]}<br>"
                                "Probability: N/A<extra></extra>"
                            ),
                        )
                    )

            add_safe_vertical_marker(price_fig, last_price_date, "Forecast Start", "#6B7280")
            price_fig.update_layout(
                height=580,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.03, x=0, bgcolor="rgba(255,255,255,0.9)"),
                xaxis_title="Date",
                yaxis_title="BTCUSD Price",
                font=dict(color="#111827"),
            )
            price_fig.update_xaxes(
                range=[chart_start, chart_end],
                showgrid=True,
                gridcolor="rgba(148,163,184,0.18)",
                zeroline=False,
            )
            price_fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.18)",
                zeroline=False,
                range=[chart_price["price"].min() * 0.95, max(chart_price["price"].max(), latest_price) * 1.08],
            )
            st.plotly_chart(price_fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Unable to render the forecast price chart safely: {exc}")

    st.markdown("#### Outlook Cards")
    outlook_cols = st.columns(3)
    with outlook_cols[0]:
        render_outlook_card("30D Outlook", top_30d)
    with outlook_cols[1]:
        render_outlook_card("90D Outlook", top_90d)
    with outlook_cols[2]:
        render_outlook_card("365D Outlook", top_365d)

    st.markdown("#### Research Validation Layer")
    st.caption(
        "Institutional validation evidence for Taxonomy v4 edge, Portfolio Allocation Engine v2 behavior, and ML probability calibration."
    )

    validation_missing_inputs = []
    if taxonomy_performance_validation.empty:
        validation_missing_inputs.append("taxonomy_performance_validation.csv")
    if allocation_variant_results.empty:
        validation_missing_inputs.append("allocation_variant_results.csv")
    if ml_pred.empty:
        validation_missing_inputs.append("ml_predictions.csv")
    if validation_missing_inputs:
        st.warning(
            "Some research validation inputs are missing or empty: "
            + ", ".join(validation_missing_inputs)
            + ". The section below will show partial evidence only."
        )

    st.markdown("##### Taxonomy Performance Analytics")
    taxonomy_perf = taxonomy_performance_validation.copy()
    taxonomy_audit = taxonomy_regime_audit.copy()
    taxonomy_year = taxonomy_performance_by_year.copy()
    strongest_edge_taxonomy = "N/A"
    weakest_edge_taxonomy = "N/A"
    most_stable_taxonomy = "N/A"
    most_fragile_taxonomy = "N/A"
    taxonomy_edge_positive = False

    if not taxonomy_perf.empty and {"taxonomy_v3", "horizon_days", "average_return"}.issubset(taxonomy_perf.columns):
        taxonomy_perf = taxonomy_perf.rename(columns={"taxonomy_v3": "taxonomy_label"})
        return_pivot = taxonomy_perf.pivot_table(
            index="taxonomy_label",
            columns="horizon_days",
            values="average_return",
            aggfunc="first",
        )
        win_rate_30 = taxonomy_perf[taxonomy_perf["horizon_days"] == 30][["taxonomy_label", "win_rate", "sample_count", "volatility", "return_volatility_ratio"]].copy()
        perf_table = return_pivot.rename(columns={
            7: "7D avg return",
            14: "14D avg return",
            30: "30D avg return",
            60: "60D avg return",
            90: "90D avg return",
        }).reset_index()
        perf_table = perf_table.merge(win_rate_30, on="taxonomy_label", how="left")
        perf_table = perf_table.rename(columns={
            "taxonomy_label": "Taxonomy",
            "win_rate": "30D win rate",
            "sample_count": "sample count",
            "volatility": "volatility",
            "return_volatility_ratio": "return / volatility",
        })
        perf_display = perf_table.copy()
        for col in ["7D avg return", "14D avg return", "30D avg return", "60D avg return", "90D avg return", "30D win rate", "volatility", "return / volatility"]:
            if col in perf_display.columns:
                perf_display[col] = perf_display[col].map(lambda x: "N/A" if pd.isna(x) else (f"{x:.2f}" if col == "return / volatility" else format_return(x) if "return" in col or col == "volatility" else format_pct(x)))
        if "sample count" in perf_display.columns:
            perf_display["sample count"] = perf_display["sample count"].map(lambda x: "N/A" if pd.isna(x) else f"{int(x):,}")
        st.dataframe(perf_display, use_container_width=True, hide_index=True)

        perf_30 = taxonomy_perf[taxonomy_perf["horizon_days"] == 30].copy()
        if not perf_30.empty:
            strongest_row = perf_30.sort_values("average_return", ascending=False).iloc[0]
            weakest_row = perf_30.sort_values("average_return", ascending=True).iloc[0]
            strongest_edge_taxonomy = strongest_row["taxonomy_label"]
            weakest_edge_taxonomy = weakest_row["taxonomy_label"]
            taxonomy_edge_positive = float(strongest_row["average_return"]) > 0

            if not taxonomy_audit.empty and "taxonomy_v3" in taxonomy_audit.columns:
                taxonomy_audit = taxonomy_audit.rename(columns={"taxonomy_v3": "taxonomy_label"})
                audit_stable = taxonomy_audit.copy()
                audit_stable["stability_rank"] = np.where(
                    audit_stable.get("stability_assessment", "").astype(str).str.contains("Fragile", case=False, na=False),
                    0,
                    1,
                )
                audit_stable = audit_stable.sort_values(
                    ["stability_rank", "positive_year_share", "sample_count"],
                    ascending=[False, False, False],
                )
                most_stable_taxonomy = audit_stable.iloc[0]["taxonomy_label"]
                most_fragile_taxonomy = audit_stable.sort_values(
                    ["stability_rank", "sample_count", "active_years"],
                    ascending=[True, True, True],
                ).iloc[0]["taxonomy_label"]

            chart_cols = st.columns(3)
            chart_specs = [
                ("30D Average Return by Taxonomy", "average_return", "#2563EB", "30d_return"),
                ("30D Win Rate by Taxonomy", "win_rate", "#0F766E", "30d_win_rate"),
                ("Sample Count by Taxonomy", "sample_count", "#7C3AED", "sample_count"),
            ]
            for col, (title, metric_col, color, key_suffix) in zip(chart_cols, chart_specs):
                with col:
                    plot_df = perf_30.sort_values(metric_col, ascending=False).copy()
                    fig = go.Figure(
                        go.Bar(
                            x=plot_df["taxonomy_label"],
                            y=plot_df[metric_col],
                            marker=dict(color=[taxonomy_color(label) if metric_col != "sample_count" else color for label in plot_df["taxonomy_label"]]),
                            text=plot_df[metric_col].map(lambda x: f"{x:.0f}" if metric_col == "sample_count" else format_pct(x)),
                            textposition="outside",
                            hovertemplate="Taxonomy: %{x}<br>Value: %{y}<extra></extra>",
                        )
                    )
                    fig.update_layout(
                        title=title,
                        height=320,
                        margin=dict(l=12, r=12, t=50, b=70),
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        font=dict(color="#111827"),
                        xaxis_title="",
                        yaxis_title="",
                        showlegend=False,
                    )
                    fig.update_xaxes(tickangle=-25, showgrid=False)
                    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
                    safe_plotly_chart(fig, key=f"taxonomy_{key_suffix}")

        insight_cols = st.columns(4)
        insight_cards = [
            ("Strongest Edge Taxonomy", strongest_edge_taxonomy, "Highest 30D average return", taxonomy_color(strongest_edge_taxonomy)),
            ("Weakest Edge Taxonomy", weakest_edge_taxonomy, "Lowest 30D average return", taxonomy_color(weakest_edge_taxonomy)),
            ("Most Stable Taxonomy", most_stable_taxonomy, "Best stability profile from regime audit", taxonomy_color(most_stable_taxonomy)),
            ("Most Fragile Taxonomy", most_fragile_taxonomy, "Most sample/regime fragility", taxonomy_color(most_fragile_taxonomy)),
        ]
        for col, (label, value, helper, accent) in zip(insight_cols, insight_cards):
            with col:
                render_light_metric(label, value, helper, accent=accent, badge_label=value if value != "N/A" else None)
    else:
        st.warning("Taxonomy performance analytics are unavailable because the validation table is missing required columns.")

    st.markdown("##### Portfolio Allocation v2 Backtest")
    allocation_results = allocation_variant_results.copy()
    allocation_annual = allocation_variant_annual.copy()
    allocation_stress = allocation_variant_stress_test.copy()
    recommended_variant_label = "Trend-Preserving v2"
    recommended_variant_key = None
    allocation_ready_for_paper = "Not execution-ready"
    if not allocation_results.empty and {"variant_label", "buy_hold_total_return"}.issubset(allocation_results.columns):
        recommended_match = allocation_results[allocation_results["variant_label"] == recommended_variant_label].copy()
        if not recommended_match.empty:
            recommended_row = recommended_match.iloc[0]
            recommended_variant_key = recommended_row["variant_key"]
            metric_cols = st.columns(7)
            metric_cards = [
                ("Variant", recommended_variant_label, "Recommended allocation research variant", "#7C3AED", None),
                ("Total Return", format_return(recommended_row.get("total_return", np.nan)), f"Buy & Hold {format_return(recommended_row.get('buy_hold_total_return', np.nan))}", "#2563EB", None),
                ("CAGR", format_return(recommended_row.get("CAGR", np.nan)), f"Buy & Hold {format_return(recommended_row.get('buy_hold_CAGR', np.nan))}", "#2563EB", None),
                ("Max Drawdown", format_return(recommended_row.get("max_drawdown", np.nan)), f"Improvement {format_return(recommended_row.get('drawdown_improvement_points', np.nan))}", "#C62828", None),
                ("Sharpe", "N/A" if pd.isna(recommended_row.get("Sharpe ratio", np.nan)) else f"{recommended_row.get('Sharpe ratio', np.nan):.2f}", f"Buy & Hold {recommended_row.get('buy_hold_sharpe', np.nan):.2f}", "#2E7D32", None),
                ("Sortino", "N/A" if pd.isna(recommended_row.get("Sortino ratio", np.nan)) else f"{recommended_row.get('Sortino ratio', np.nan):.2f}", f"Buy & Hold {recommended_row.get('buy_hold_sortino', np.nan):.2f}", "#2E7D32", None),
                ("Return Capture", format_pct(recommended_row.get("return_capture_ratio_vs_buy_hold", np.nan)), "Upside captured vs Buy & Hold", "#D97706", None),
            ]
            for col, (label, value, helper, accent, badge) in zip(metric_cols, metric_cards):
                with col:
                    render_light_metric(label, value, helper, accent=accent, badge_label=badge)

            if not allocation_annual.empty and recommended_variant_key:
                annual_curve = allocation_annual[
                    allocation_annual["variant_key"] == recommended_variant_key
                ].copy()
                annual_curve["year"] = pd.to_numeric(annual_curve["year"], errors="coerce")
                annual_curve["total_return"] = pd.to_numeric(annual_curve["total_return"], errors="coerce")
                annual_curve = annual_curve.dropna(subset=["year", "total_return"])
                if not annual_curve.empty:
                    annual_curve = annual_curve.sort_values(["strategy", "year"])
                    annual_curve["equity"] = annual_curve.groupby("strategy")["total_return"].transform(lambda s: (1 + s).cumprod())
                    equity_fig = go.Figure()
                    for strategy_name, line_color in [("Allocation Strategy", "#7C3AED"), ("Buy & Hold", "#2563EB")]:
                        sub = annual_curve[annual_curve["strategy"] == strategy_name]
                        if sub.empty:
                            continue
                        equity_fig.add_trace(
                            go.Scatter(
                                x=sub["year"].astype(int).astype(str),
                                y=sub["equity"],
                                mode="lines+markers",
                                name=strategy_name,
                                line=dict(color=line_color, width=2.4),
                                hovertemplate="Year: %{x}<br>Equity: %{y:.2f}x<extra></extra>",
                            )
                        )
                    equity_fig.update_layout(
                        title="Allocation v2 Equity Curve vs Buy & Hold",
                        height=360,
                        margin=dict(l=12, r=12, t=50, b=20),
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        font=dict(color="#111827"),
                        yaxis_title="Cumulative equity",
                        xaxis_title="Year",
                    )
                    equity_fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
                    safe_plotly_chart(equity_fig, key="allocation_equity_curve")
                else:
                    st.info("Allocation equity curve could not be derived from the annual backtest table.")
            else:
                st.info("Allocation equity curve is unavailable from the current outputs, so the backtest is shown through summary metrics only.")

            allocation_ready_for_paper = (
                "Ready for paper trading only"
                if (
                    recommended_row.get("Sharpe ratio", -np.inf) > recommended_row.get("buy_hold_sharpe", np.inf)
                    and recommended_row.get("drawdown_improvement_points", 0) > 0
                )
                else "Needs monitoring"
            )
    else:
        st.warning("Allocation variant results are unavailable, so the backtest section is partially disabled.")

    st.markdown("##### Stress Test")
    if not allocation_stress.empty and recommended_variant_key:
        stress_df = allocation_stress[allocation_stress["variant_key"] == recommended_variant_key].copy()
        stress_display = stress_df[
            [
                "period",
                "strategy_total_return",
                "buy_hold_total_return",
                "strategy_max_drawdown",
                "buy_hold_max_drawdown",
                "strategy_sharpe",
                "buy_hold_sharpe",
            ]
        ].copy()
        for col in ["strategy_total_return", "buy_hold_total_return", "strategy_max_drawdown", "buy_hold_max_drawdown"]:
            stress_display[col] = stress_display[col].map(format_return)
        for col in ["strategy_sharpe", "buy_hold_sharpe"]:
            stress_display[col] = stress_display[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
        stress_display.columns = [
            "Period",
            "Strategy return",
            "Buy & Hold return",
            "Strategy max drawdown",
            "Buy & Hold max drawdown",
            "Strategy Sharpe",
            "Buy & Hold Sharpe",
        ]
        st.dataframe(stress_display, use_container_width=True, hide_index=True)

        helps_periods = stress_df[stress_df["strategy_max_drawdown"] > stress_df["buy_hold_max_drawdown"]]["period"].tolist()
        underperform_periods = stress_df[stress_df["strategy_total_return"] < stress_df["buy_hold_total_return"]]["period"].tolist()
        stress_cols = st.columns(3)
        with stress_cols[0]:
            render_light_metric(
                "Where Allocation Helps",
                helps_periods[0] if helps_periods else "N/A",
                ", ".join(helps_periods[:3]) if helps_periods else "No drawdown improvement periods identified.",
                accent="#2E7D32",
            )
        with stress_cols[1]:
            render_light_metric(
                "Where It Underperforms",
                underperform_periods[0] if underperform_periods else "N/A",
                ", ".join(underperform_periods[:3]) if underperform_periods else "No underperformance periods identified.",
                accent="#D97706",
            )
        with stress_cols[2]:
            render_light_metric(
                "Behavior Profile",
                "Risk-managed BTC",
                "Drawdown control improves materially, but the strategy still trails full BTC upside in stronger trend years.",
                accent="#7C3AED",
            )
    else:
        st.warning("Stress-test evidence is unavailable for the recommended allocation variant.")

    st.markdown("##### ML Probability Calibration")
    calibration_df = ml_pred.copy()
    calibration_summary = ml_summary.copy()
    best_horizon = None
    calibration_table = pd.DataFrame()
    brier_score = np.nan
    expected_calibration_error = np.nan
    ml_calibration_status = "Needs monitoring"

    if not calibration_summary.empty and {"feature_set", "balanced_score", "horizon_days"}.issubset(calibration_summary.columns):
        best_summary = calibration_summary[calibration_summary["feature_set"].astype(str).str.contains("selected", case=False, na=False)].copy()
        if best_summary.empty:
            best_summary = calibration_summary.copy()
        best_summary = best_summary.sort_values("balanced_score", ascending=False)
        if not best_summary.empty:
            best_horizon = int(best_summary.iloc[0]["horizon_days"])

    if not calibration_df.empty and {"date", "horizon", "ml_prob_up", "price"}.issubset(calibration_df.columns) and best_horizon is not None:
        calibration_df = calibration_df[calibration_df["horizon"] == best_horizon].copy()
        calibration_df["date"] = pd.to_datetime(calibration_df["date"], errors="coerce")
        calibration_df["ml_prob_up"] = pd.to_numeric(calibration_df["ml_prob_up"], errors="coerce")
        calibration_df["price"] = pd.to_numeric(calibration_df["price"], errors="coerce")
        calibration_df = calibration_df.dropna(subset=["date", "ml_prob_up", "price"]).sort_values("date").drop_duplicates("date")
        calibration_df["forward_return"] = calibration_df["price"].shift(-best_horizon) / calibration_df["price"] - 1.0
        if "actual_direction" in calibration_df.columns:
            calibration_df["actual_hit"] = pd.to_numeric(calibration_df["actual_direction"], errors="coerce")
        else:
            calibration_df["actual_hit"] = np.where(calibration_df["forward_return"] > 0, 1.0, 0.0)
        calibration_df = calibration_df.dropna(subset=["forward_return", "actual_hit"])
        bucket_bins = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
        bucket_labels = ["40-45%", "45-50%", "50-55%", "55-60%", "60-65%", "65-70%", "70%+"]
        calibration_df["probability_bucket"] = pd.cut(
            calibration_df["ml_prob_up"],
            bins=bucket_bins,
            labels=bucket_labels,
            include_lowest=True,
            right=False,
        )
        calibration_table = calibration_df.dropna(subset=["probability_bucket"]).groupby("probability_bucket", observed=False).agg(
            sample_count=("ml_prob_up", "size"),
            average_predicted_probability=("ml_prob_up", "mean"),
            actual_hit_rate=("actual_hit", "mean"),
            average_forward_return=("forward_return", "mean"),
        ).reset_index()
        calibration_table["calibration_error"] = (
            calibration_table["average_predicted_probability"] - calibration_table["actual_hit_rate"]
        ).abs()
        brier_score = float(np.mean((calibration_df["ml_prob_up"] - calibration_df["actual_hit"]) ** 2))
        expected_calibration_error = float(
            (
                calibration_table["sample_count"] * calibration_table["calibration_error"]
            ).sum() / max(calibration_table["sample_count"].sum(), 1)
        ) if not calibration_table.empty else np.nan
        ml_calibration_status = (
            "Research validated" if not pd.isna(expected_calibration_error) and expected_calibration_error <= 0.05
            else "Needs monitoring"
        )

    if not calibration_table.empty:
        cal_cols = st.columns(3)
        with cal_cols[0]:
            render_light_metric(
                "Calibration Horizon",
                f"{best_horizon}D" if best_horizon is not None else "N/A",
                "Best selected-features model by balanced score.",
                accent="#2563EB",
            )
        with cal_cols[1]:
            render_light_metric(
                "Brier Score",
                "N/A" if pd.isna(brier_score) else f"{brier_score:.4f}",
                "Lower is better.",
                accent="#0F766E",
            )
        with cal_cols[2]:
            render_light_metric(
                "Expected Calibration Error",
                "N/A" if pd.isna(expected_calibration_error) else f"{expected_calibration_error:.4f}",
                ml_calibration_status,
                accent="#D97706",
            )

        reliability_fig = go.Figure()
        reliability_fig.add_trace(
            go.Scatter(
                x=calibration_table["average_predicted_probability"],
                y=calibration_table["actual_hit_rate"],
                mode="lines+markers",
                name="Observed",
                line=dict(color="#2563EB", width=2.3),
                marker=dict(size=9),
                hovertemplate="Predicted: %{x:.2%}<br>Observed: %{y:.2%}<extra></extra>",
            )
        )
        reliability_fig.add_trace(
            go.Scatter(
                x=[0.40, 0.80],
                y=[0.40, 0.80],
                mode="lines",
                name="Perfect Calibration",
                line=dict(color="#94A3B8", width=1.5, dash="dash"),
                hoverinfo="skip",
            )
        )
        reliability_fig.update_layout(
            title="ML Probability Reliability Curve",
            height=340,
            margin=dict(l=12, r=12, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#111827"),
            xaxis_title="Average predicted probability",
            yaxis_title="Observed hit rate",
        )
        reliability_fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor="rgba(148,163,184,0.18)")
        reliability_fig.update_yaxes(tickformat=".0%", showgrid=True, gridcolor="rgba(148,163,184,0.18)")
        safe_plotly_chart(reliability_fig, key="ml_reliability_curve")

        calibration_display = calibration_table.copy()
        for col in ["average_predicted_probability", "actual_hit_rate", "average_forward_return", "calibration_error"]:
            calibration_display[col] = calibration_display[col].map(
                lambda x: "N/A" if pd.isna(x) else (format_return(x) if col == "average_forward_return" else format_pct(x))
            )
        calibration_display["sample_count"] = calibration_display["sample_count"].map(lambda x: f"{int(x):,}")
        st.dataframe(calibration_display, use_container_width=True, hide_index=True)
    else:
        st.warning("ML probability calibration could not be computed from the current prediction file and available target labels.")

    st.markdown("##### Research Verdict")
    allocation_capture_ok = False
    allocation_risk_adjusted_ok = False
    if not allocation_results.empty and "variant_label" in allocation_results.columns:
        verdict_match = allocation_results[allocation_results["variant_label"] == recommended_variant_label]
        if not verdict_match.empty:
            verdict_row = verdict_match.iloc[0]
            allocation_capture_ok = verdict_row.get("return_capture_ratio_vs_buy_hold", 0) >= 0.60
            allocation_risk_adjusted_ok = (
                verdict_row.get("Sharpe ratio", -np.inf) > verdict_row.get("buy_hold_sharpe", np.inf)
                and verdict_row.get("Sortino ratio", -np.inf) > verdict_row.get("buy_hold_sortino", np.inf)
                and verdict_row.get("drawdown_improvement_points", 0) > 0
            )

    verdict_cols = st.columns(5)
    verdict_cards = [
        ("A. Taxonomy v4 Edge", "Research validated" if taxonomy_edge_positive else "Needs monitoring", strongest_edge_taxonomy if strongest_edge_taxonomy != "N/A" else "No strong positive edge identified.", taxonomy_color(strongest_edge_taxonomy if strongest_edge_taxonomy != "N/A" else "Constructive Drift")),
        ("B. Allocation v2 Risk-Adjusted Return", "Research validated" if allocation_risk_adjusted_ok else "Needs monitoring", "Sharpe / Sortino and drawdown are better than Buy & Hold." if allocation_risk_adjusted_ok else "Risk-adjusted improvement is partial or inconsistent.", "#7C3AED"),
        ("C. Allocation v2 Upside Capture", "Needs monitoring" if allocation_capture_ok else "Not execution-ready", "Return capture clears the 60% threshold." if allocation_capture_ok else "Upside capture still trails a robust BTC substitute target.", "#D97706"),
        ("D. ML Probability Calibration", ml_calibration_status, f"ECE {expected_calibration_error:.4f}" if not pd.isna(expected_calibration_error) else "Calibration evidence unavailable.", "#2563EB"),
        ("E. Paper Trading Readiness", "Ready for paper trading only" if allocation_risk_adjusted_ok and taxonomy_edge_positive else "Not execution-ready", "Use for monitored research deployment, not live execution." if allocation_risk_adjusted_ok and taxonomy_edge_positive else "Evidence is promising but still needs tighter monitoring and alignment.", "#2E7D32" if allocation_risk_adjusted_ok and taxonomy_edge_positive else "#C62828"),
    ]
    for col, (label, value, helper, accent) in zip(verdict_cols, verdict_cards):
        with col:
            render_light_metric(label, value, helper, accent=accent)

    st.markdown("#### Why The Model Thinks This")
    current_taxonomy = dashboard_current.get("current_taxonomy", "N/A")
    current_attr = get_taxonomy_attribution_row(current_taxonomy)
    current_features = get_top_feature_rows(current_taxonomy, limit=5)
    current_planets = parse_ranked_items(current_attr.get("most_influential_planets"), limit=5)
    current_aspects = parse_ranked_items(current_attr.get("most_influential_aspects"), limit=5)

    explain_cols = st.columns([1.1, 1.3])
    with explain_cols[0]:
        with card_container():
            st.caption("CURRENT STATE".upper())
            st.markdown(taxonomy_badge_html(current_taxonomy), unsafe_allow_html=True)
            st.markdown(f"**{current_taxonomy}**")
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    "Astro Momentum",
                    fmt_num(latest.get("astro_momentum_v2_smooth", np.nan)),
                    help="Latest smoothed astro momentum from the live feature frame.",
                )
            with metric_cols[1]:
                st.metric(
                    "ML Probability",
                    format_pct(dashboard_current.get("current_probability", np.nan)),
                    help="Current forecast probability from the Robust Astro Engine v1 ML layer.",
                )
            st.caption(
                f"Typical taxonomy momentum range: {current_attr.get('typical_momentum_range', 'N/A')}"
            )
            st.caption(
                f"Typical taxonomy probability range: {current_attr.get('typical_probability_range', 'N/A')}"
            )
            render_driver_list("Top 5 Contributing Features", [
                {
                    "name": row["feature"],
                    "score": row["zscore_diff"] if "zscore_diff" in row else row.get("differential", np.nan),
                }
                for _, row in current_features.iterrows()
            ])
            driver_cols = st.columns(2)
            with driver_cols[0]:
                render_driver_list("Top Planets", current_planets, empty_text="No strong planetary concentration.")
            with driver_cols[1]:
                render_driver_list("Top Aspects", current_aspects, empty_text="No strong aspect concentration.")

    with explain_cols[1]:
        with card_container():
            st.caption("CONTRIBUTION BARS")
            st.write("These show the strongest feature-level pushes behind the current taxonomy state.")
            render_contribution_bars(current_features, "current_state")

    st.markdown("##### Outlook Drivers")
    outlook_driver_cols = st.columns(3)
    with outlook_driver_cols[0]:
        render_state_explanation_card(
            "30D Outlook",
            top_30d.get("dominant_taxonomy", "N/A"),
            top_30d.get("average_confidence", np.nan),
            top_30d.get("average_probability", np.nan),
            top_30d.get("summary", ""),
            "30d",
        )
    with outlook_driver_cols[1]:
        render_state_explanation_card(
            "90D Outlook",
            top_90d.get("dominant_taxonomy", "N/A"),
            top_90d.get("average_confidence", np.nan),
            top_90d.get("average_probability", np.nan),
            top_90d.get("summary", ""),
            "90d",
        )
    with outlook_driver_cols[2]:
        render_state_explanation_card(
            "365D Outlook",
            top_365d.get("dominant_taxonomy", "N/A"),
            top_365d.get("average_confidence", np.nan),
            top_365d.get("average_probability", np.nan),
            top_365d.get("summary", ""),
            "365d",
        )

    if show_detail_tables:
        st.markdown("#### Forecast Windows Detail")
        st.caption("Detailed forecast windows remain available for auditability, with taxonomy and risk filters.")
        detail_df = top_dashboard_windows.copy()
        if not detail_df.empty:
            filter_cols = st.columns([1, 1, 1])
            with filter_cols[0]:
                only_risk = st.checkbox("Risk windows only", value=False, key="detail_only_risk")
            with filter_cols[1]:
                only_constructive = st.checkbox("Constructive only", value=False, key="detail_only_constructive")
            with filter_cols[2]:
                st.caption(f"Taxonomy filter: {taxonomy_filter}")

            if only_risk:
                detail_df = detail_df[detail_df["taxonomy_v2"].isin(["Volatility Caution", "High Volatility Risk"])].copy()
            if only_constructive:
                detail_df = detail_df[detail_df["taxonomy_v2"].isin(["Constructive Drift", "High Conviction Expansion", "Recovery / Reversal Setup"])].copy()

            detail_display = detail_df.copy()
            detail_display["start_date"] = detail_display["start_date"].map(format_date)
            detail_display["end_date"] = detail_display["end_date"].map(format_date)
            detail_display["average_confidence"] = detail_display["average_confidence"].map(format_pct)
            detail_display["average_ml_probability"] = detail_display["average_ml_probability"].map(format_pct)
            if "average_astro_score" in detail_display.columns:
                detail_display["average_astro_score"] = detail_display["average_astro_score"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
            detail_display["taxonomy_reason"] = detail_display["taxonomy_reason"].fillna(detail_display.get("narrative_v2", ""))
            detail_display = detail_display[
                [
                    "start_date",
                    "end_date",
                    "taxonomy_v2",
                    "duration_days",
                    "average_confidence",
                    "average_ml_probability",
                    "average_astro_score",
                    "v2_posture",
                    "taxonomy_reason",
                ]
            ]
            st.dataframe(detail_display, use_container_width=True, hide_index=True)
        else:
            st.info("No forecast windows are available for the current filters.")

    st.markdown("#### Upcoming Turning Points")
    st.caption("The next three turning points are surfaced as quick-read cards, with the full table available on demand.")
    if not top_turning_points.empty:
        turn_cards = top_turning_points.head(3).copy()
        turn_cols = st.columns(3)
        for idx, (_, row) in enumerate(turn_cards.iterrows()):
            new_signal = str(row["new_signal"]).strip()
            direction_color = taxonomy_color("Constructive Drift") if new_signal.lower() == "bullish" else taxonomy_color("Volatility Caution") if new_signal.lower() == "bearish" else "#6B7280"
            with turn_cols[idx]:
                with card_container():
                    st.markdown(
                        f"<div style='height:4px;background:{direction_color};border-radius:999px;margin-bottom:12px;'></div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(format_date(row["turning_point_date"]))
                    st.markdown(f"**{str(row['turning_point_type']).replace('_', ' ').title()}**")
                    st.markdown(f"**{row['old_signal']} -> {row['new_signal']}**")
                    st.caption(f"Confidence {format_pct(row['confidence'])}")
                    st.write(row["explanation"])

        with st.expander("Show full turning point table", expanded=False):
            tp_display = top_turning_points.copy()
            tp_display["turning_point_date"] = tp_display["turning_point_date"].map(format_date)
            tp_display["confidence"] = tp_display["confidence"].map(format_pct)
            st.dataframe(
                tp_display[
                    [
                        "turning_point_date",
                        "turning_point_type",
                        "old_signal",
                        "new_signal",
                        "confidence",
                        "explanation",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No turning points are available for the selected horizon.")

    st.markdown("#### Risk Calendar")
    st.caption("Compact risk cards keep the next major caution windows visible without repeating the same information unnecessarily.")
    risk_card_specs = [
        ("Next Risk Window", next_risk_window, next_risk_window.get("taxonomy_v2", dashboard_current.get("current_taxonomy", "Transition / Low Conviction"))),
        ("Next Constructive Window", top_next_constructive, top_next_constructive.get("taxonomy_v2", "Constructive Drift")),
        ("Next High Conviction Expansion", next_momentum_window, next_momentum_window.get("taxonomy_v2", "High Conviction Expansion")),
        ("Next Recovery / Reversal Window", next_recovery_window, next_recovery_window.get("taxonomy_v2", "Recovery / Reversal Setup")),
        ("Next Volatility Caution", top_next_high_risk, top_next_high_risk.get("taxonomy_v2", "Volatility Caution")),
    ]
    risk_cols = st.columns(5)
    for idx, (title, row, taxonomy_label) in enumerate(risk_card_specs):
        start_text = format_date(row.get("start_date", row.get("turning_point_date")))
        end_text = format_date(row.get("end_date"))
        date_text = start_text if end_text == "N/A" else f"{start_text} to {end_text}"
        body_text = row.get("taxonomy_reason", row.get("explanation", "No additional context available."))
        with risk_cols[idx]:
            with card_container():
                st.markdown(
                    f"<div style='height:4px;background:{taxonomy_color(taxonomy_label)};border-radius:999px;margin-bottom:12px;'></div>",
                    unsafe_allow_html=True,
                )
                st.caption(title.upper())
                st.markdown(taxonomy_badge_html(taxonomy_label), unsafe_allow_html=True)
                st.markdown(f"**{date_text}**")
                st.caption(body_text)

    st.markdown("---")
else:
    st.warning("Forecast dashboard files are missing or empty. The research tabs below are still available.")
    st.markdown("---")

if dashboard_current and False:
    top_dashboard_windows = pd.DataFrame(dashboard_timeline.get("windows", []))
    top_turning_points = pd.DataFrame(dashboard_risk_calendar.get("turning_points", []))
    top_risk_windows = pd.DataFrame(dashboard_risk_calendar.get("risk_windows", []))
    top_current_window = dashboard_current.get("current_window", {})
    top_next_turning = dashboard_current.get("next_turning_point", {})
    top_next_constructive = dashboard_current.get("next_constructive_window", {})
    top_next_high_risk = dashboard_current.get("next_high_risk_window", {})
    top_30d = dashboard_summary.get("30D Outlook", {})
    top_90d = dashboard_summary.get("90D Outlook", {})
    top_365d = dashboard_summary.get("365D Outlook", {})
    prev_price = price_df[price_df["date"] < last_price_date]["price"].iloc[-1] if len(price_df) > 1 else latest_price
    price_delta = latest_price - prev_price
    price_delta_pct = (price_delta / prev_price) if prev_price else 0.0

    if not top_dashboard_windows.empty:
        top_dashboard_windows["start_date"] = pd.to_datetime(top_dashboard_windows["start_date"])
        top_dashboard_windows["end_date"] = pd.to_datetime(top_dashboard_windows["end_date"])
        top_dashboard_windows = top_dashboard_windows.sort_values("start_date").reset_index(drop=True)

    if not top_turning_points.empty:
        top_turning_points["turning_point_date"] = pd.to_datetime(top_turning_points["turning_point_date"])
        top_turning_points = top_turning_points.sort_values("turning_point_date").reset_index(drop=True)

    if not top_risk_windows.empty:
        top_risk_windows["start_date"] = pd.to_datetime(top_risk_windows["start_date"])
        top_risk_windows["end_date"] = pd.to_datetime(top_risk_windows["end_date"])
        top_risk_windows = top_risk_windows.sort_values("start_date").reset_index(drop=True)

    st.markdown(
        f"""
        <div class="terminal-shell">
            <div class="terminal-kicker">Institutional Forecast Terminal</div>
            <div class="terminal-headline">BTC Macro Regime & Forecast Dashboard</div>
            <div class="terminal-subhead">
                {dashboard_current.get("market_view", "Market view unavailable")} •
                Current window runs from {top_current_window.get("start_date", "N/A")} to {top_current_window.get("end_date", "N/A")}.
                The forecast engine remains unchanged; this layer is a visual redesign for faster investment reads.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_cols = st.columns(6)
    with hero_cols[0]:
        terminal_stat_card(
            "Current Signal",
            dashboard_current.get("current_signal", "N/A"),
            dashboard_current.get("market_view", ""),
            taxonomy_color(dashboard_current.get("current_taxonomy")),
        )
    with hero_cols[1]:
        terminal_stat_card(
            "Taxonomy",
            dashboard_current.get("current_taxonomy", "N/A"),
            top_current_window.get("v2_posture", ""),
            taxonomy_color(dashboard_current.get("current_taxonomy")),
        )
    with hero_cols[2]:
        terminal_stat_card(
            "Confidence",
            fmt_pct(dashboard_current.get("current_confidence", np.nan)),
            f"Probability {fmt_pct(dashboard_current.get('current_probability', np.nan))}",
            "#38bdf8",
        )
    with hero_cols[3]:
        terminal_stat_card(
            "Risk",
            dashboard_current.get("risk_level", "N/A"),
            top_next_high_risk.get("start_date", "No high-risk window queued"),
            risk_level_color(dashboard_current.get("risk_level")),
        )
    with hero_cols[4]:
        terminal_stat_card(
            "Bias",
            dashboard_current.get("recommended_bias", "N/A"),
            top_current_window.get("narrative_v2", "")[:82] + ("..." if len(top_current_window.get("narrative_v2", "")) > 82 else ""),
            taxonomy_color(dashboard_current.get("current_taxonomy")),
        )
    with hero_cols[5]:
        terminal_stat_card(
            "BTC Price",
            fmt_money(latest_price),
            f"{price_delta:+,.0f} ({price_delta_pct:+.2%}) vs prior close",
            "#22c55e" if price_delta >= 0 else "#ef4444",
        )

    st.markdown('<div class="terminal-section-title">Primary Market Chart</div>', unsafe_allow_html=True)
    st.markdown('<div class="terminal-note">Historical BTCUSD price with forecast taxonomy windows, turning-point markers, and daily hover context.</div>', unsafe_allow_html=True)

    terminal_range = st.radio(
        "Terminal Timeframe",
        ["1M", "3M", "6M", "1Y", "2Y", "4Y", "ALL"],
        index=3,
        horizontal=True,
        label_visibility="collapsed",
    )

    lookback_days = {
        "1M": 31,
        "3M": 92,
        "6M": 183,
        "1Y": 365,
        "2Y": 730,
        "4Y": 1460,
    }
    terminal_start = price_df["date"].min() if terminal_range == "ALL" else last_price_date - pd.Timedelta(days=lookback_days[terminal_range])
    terminal_start = max(terminal_start, price_df["date"].min())
    terminal_price = price_df[price_df["date"] >= terminal_start].copy()

    overlay_rows = []
    if not top_dashboard_windows.empty:
        for _, row in top_dashboard_windows.iterrows():
            row_start = max(row["start_date"], last_price_date.normalize() if hasattr(last_price_date, "normalize") else last_price_date)
            for dt in pd.date_range(row_start, row["end_date"], freq="D"):
                overlay_rows.append(
                    {
                        "date": dt,
                        "taxonomy_v2": row["taxonomy_v2"],
                        "average_confidence": row["average_confidence"],
                        "average_ml_probability": row["average_ml_probability"],
                        "v2_posture": row["v2_posture"],
                    }
                )
    future_overlay = pd.DataFrame(overlay_rows)
    if not future_overlay.empty:
        future_overlay = future_overlay.groupby("date", as_index=False).last()
        future_overlay["reference_price"] = latest_price

    price_fig = go.Figure()
    if not top_dashboard_windows.empty:
        for _, row in top_dashboard_windows.iterrows():
            if row["end_date"] < terminal_start:
                continue
            price_fig.add_vrect(
                x0=max(row["start_date"], terminal_start),
                x1=row["end_date"],
                fillcolor=taxonomy_rgba(row["taxonomy_v2"], 0.18),
                layer="below",
                line_width=0,
            )

    hist_custom = np.column_stack(
        [
            np.full(len(terminal_price), "Historical / Live Price"),
            np.full(len(terminal_price), "N/A"),
            np.full(len(terminal_price), "N/A"),
        ]
    )
    price_fig.add_trace(
        go.Scatter(
            x=terminal_price["date"],
            y=terminal_price["price"],
            mode="lines",
            name="BTCUSD",
            line=dict(color="#60a5fa", width=2.4),
            customdata=hist_custom,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "BTC Price: $%{y:,.0f}<br>"
                "Taxonomy: %{customdata[0]}<br>"
                "Confidence: %{customdata[1]}<br>"
                "Probability: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    if not future_overlay.empty:
        future_custom = np.column_stack(
            [
                future_overlay["taxonomy_v2"].astype(str),
                future_overlay["average_confidence"].map(fmt_pct),
                future_overlay["average_ml_probability"].map(fmt_pct),
            ]
        )
        price_fig.add_trace(
            go.Scatter(
                x=future_overlay["date"],
                y=future_overlay["reference_price"],
                mode="lines+markers",
                name="Forecast Context",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
                marker=dict(size=5, color="#cbd5e1"),
                customdata=future_custom,
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "BTC Price: $%{y:,.0f} (last close reference)<br>"
                    "Taxonomy: %{customdata[0]}<br>"
                    "Confidence: %{customdata[1]}<br>"
                    "Probability: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    if not top_turning_points.empty:
        chart_turns = top_turning_points[top_turning_points["turning_point_date"] >= terminal_start].copy()
        if not chart_turns.empty:
            turn_y = np.full(len(chart_turns), terminal_price["price"].max() * 1.05)
            chart_turns["marker_text"] = np.where(
                chart_turns["new_signal"].str.lower() == "bullish",
                "▲",
                np.where(chart_turns["new_signal"].str.lower() == "bearish", "▼", "•"),
            )
            chart_turns["marker_color"] = np.where(
                chart_turns["new_signal"].str.lower() == "bullish",
                "#22c55e",
                np.where(chart_turns["new_signal"].str.lower() == "bearish", "#ef4444", "#cbd5e1"),
            )
            price_fig.add_trace(
                go.Scatter(
                    x=chart_turns["turning_point_date"],
                    y=turn_y,
                    mode="text",
                    text=chart_turns["marker_text"],
                    textfont=dict(size=17, color=chart_turns["marker_color"]),
                    name="Turning Points",
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Turning Point: %{customdata[0]}<br>"
                        "Signal Shift: %{customdata[1]} → %{customdata[2]}<br>"
                        "Confidence: %{customdata[3]}<br>"
                        "Probability: N/A<extra></extra>"
                    ),
                    customdata=np.column_stack(
                        [
                            chart_turns["turning_point_type"].astype(str),
                            chart_turns["old_signal"].astype(str),
                            chart_turns["new_signal"].astype(str),
                            chart_turns["confidence"].map(fmt_pct),
                        ]
                    ),
                    showlegend=False,
                )
            )

    add_safe_vertical_marker(price_fig, last_price_date, "Forecast Start", "#cbd5e1")
    price_fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=24, r=24, t=24, b=24),
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        legend=dict(orientation="h", y=1.03, x=0),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="BTCUSD",
    )
    price_fig.update_yaxes(range=[terminal_price["price"].min() * 0.94, terminal_price["price"].max() * 1.11])
    st.plotly_chart(price_fig, use_container_width=True)

    st.markdown('<div class="terminal-section-title">Forward Outlook Grid</div>', unsafe_allow_html=True)
    st.markdown('<div class="terminal-note">Three calibrated horizons for institutional positioning, from tactical 30-day risk to 365-day regime drift.</div>', unsafe_allow_html=True)
    outlook_cols = st.columns(3)
    with outlook_cols[0]:
        terminal_stat_card(
            "30D Outlook",
            top_30d.get("dominant_taxonomy", "N/A"),
            top_30d.get("summary", ""),
            taxonomy_color(top_30d.get("dominant_taxonomy")),
        )
    with outlook_cols[1]:
        terminal_stat_card(
            "90D Outlook",
            top_90d.get("dominant_taxonomy", "N/A"),
            top_90d.get("summary", ""),
            taxonomy_color(top_90d.get("dominant_taxonomy")),
        )
    with outlook_cols[2]:
        terminal_stat_card(
            "365D Outlook",
            top_365d.get("dominant_taxonomy", "N/A"),
            top_365d.get("summary", ""),
            taxonomy_color(top_365d.get("dominant_taxonomy")),
        )

    st.markdown('<div class="terminal-section-title">Upcoming Turning Points</div>', unsafe_allow_html=True)
    st.markdown('<div class="terminal-note">Near-term inflection dates ranked by forecast direction and confidence, replacing the old raw table.</div>', unsafe_allow_html=True)
    if not top_turning_points.empty:
        turn_cards = top_turning_points.head(6).copy()
        turn_card_cols = st.columns(3)
        for idx, (_, row) in enumerate(turn_cards.iterrows()):
            direction_text = "▲ Bullish" if str(row["new_signal"]).lower() == "bullish" else "▼ Bearish" if str(row["new_signal"]).lower() == "bearish" else "• Neutral"
            direction_color = "#22c55e" if "Bullish" in direction_text else "#ef4444" if "Bearish" in direction_text else "#cbd5e1"
            with turn_card_cols[idx % 3]:
                st.markdown(
                    event_card_html(
                        row["turning_point_date"].strftime("%Y-%m-%d"),
                        str(row["turning_point_type"]).replace("_", " "),
                        direction_text,
                        direction_color,
                        f"{row['old_signal']} → {row['new_signal']} • Confidence {fmt_pct(row['confidence'])}<br>{row['explanation']}",
                    ),
                    unsafe_allow_html=True,
                )
    else:
        st.info("No turning-point events are currently available.")

    st.markdown('<div class="terminal-section-title">Forecast Taxonomy Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="terminal-note">Forward windows grouped into institution-friendly regimes with duration, confidence, and probability in hover.</div>', unsafe_allow_html=True)
    if not top_dashboard_windows.empty:
        timeline_windows = top_dashboard_windows.head(14).copy()
        window_fig = go.Figure()
        y_labels = []
        for idx, row in timeline_windows.iterrows():
            y_labels.append(f"W{idx + 1}")
            window_fig.add_trace(
                go.Scatter(
                    x=[row["start_date"], row["end_date"]],
                    y=[idx, idx],
                    mode="lines+markers",
                    line=dict(color=taxonomy_color(row["taxonomy_v2"]), width=18),
                    marker=dict(size=7, color=taxonomy_color(row["taxonomy_v2"])),
                    name=row["taxonomy_v2"],
                    customdata=np.array(
                        [[
                            row["taxonomy_v2"],
                            fmt_pct(row["average_confidence"]),
                            fmt_pct(row["average_ml_probability"]),
                            row["v2_posture"],
                            row["duration_days"],
                        ]] * 2
                    ),
                    hovertemplate=(
                        "Taxonomy: %{customdata[0]}<br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Confidence: %{customdata[1]}<br>"
                        "Probability: %{customdata[2]}<br>"
                        "Posture: %{customdata[3]}<br>"
                        "Duration: %{customdata[4]} days<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
        window_fig.update_layout(
            template="plotly_dark",
            height=460,
            margin=dict(l=24, r=24, t=24, b=24),
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            xaxis_title="Date",
            yaxis=dict(
                title="Window Sequence",
                tickmode="array",
                tickvals=list(range(len(y_labels))),
                ticktext=y_labels,
            ),
        )
        st.plotly_chart(window_fig, use_container_width=True)
    else:
        st.info("No forecast windows are available yet.")

    st.markdown('<div class="terminal-section-title">Risk Calendar</div>', unsafe_allow_html=True)
    st.markdown('<div class="terminal-note">High-conviction constructive and defensive windows summarized as decision-ready cards instead of raw tables.</div>', unsafe_allow_html=True)
    risk_cols = st.columns(3)
    with risk_cols[0]:
        st.markdown(
            risk_window_card_html(
                "Next Turning Point",
                top_next_turning.get("turning_point_date", "N/A"),
                dashboard_current.get("current_taxonomy"),
                top_next_turning.get("explanation", "No next turning-point explanation available."),
            ),
            unsafe_allow_html=True,
        )
    with risk_cols[1]:
        st.markdown(
            risk_window_card_html(
                "Next Constructive Window",
                f"{top_next_constructive.get('start_date', 'N/A')} to {top_next_constructive.get('end_date', 'N/A')}",
                top_next_constructive.get("taxonomy_v2", "Constructive Drift"),
                top_next_constructive.get("taxonomy_reason", "Constructive window details unavailable."),
            ),
            unsafe_allow_html=True,
        )
    with risk_cols[2]:
        st.markdown(
            risk_window_card_html(
                "Next High-Risk Window",
                f"{top_next_high_risk.get('start_date', 'N/A')} to {top_next_high_risk.get('end_date', 'N/A')}",
                top_next_high_risk.get("taxonomy_v2", "Volatility Caution"),
                top_next_high_risk.get("taxonomy_reason", "High-risk window details unavailable."),
            ),
            unsafe_allow_html=True,
        )

    if not top_risk_windows.empty:
        more_risk_cols = st.columns(min(3, len(top_risk_windows.head(3))))
        for idx, (_, row) in enumerate(top_risk_windows.head(3).iterrows()):
            with more_risk_cols[idx]:
                st.markdown(
                    risk_window_card_html(
                        "Risk Window",
                        f"{row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}",
                        row["taxonomy_v2"],
                        f"Confidence {fmt_pct(row['average_confidence'])} • Probability {fmt_pct(row['average_ml_probability'])}<br>{row['taxonomy_reason']}",
                    ),
                    unsafe_allow_html=True,
                )

    st.markdown("---")

tabs = st.tabs([
    "1. Overview",
    "2. Forecast",
    "3. ML Intelligence",
    "4. Astro Research",
    "5. Raw Aspects"
])

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tabs[0]:
    st.markdown("""
    <div class="explain-box">
    <b>Reading flow:</b> เริ่มจากดูสถานะปัจจุบัน → ดูราคาและ Astro Momentum → ดูว่า Rule-Based Strategy ชนะหรือแพ้ Buy & Hold
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Latest BTC Price", fmt_money(latest_price), "Yahoo Finance BTC-USD")
    with c2:
        metric_card("Selected Astro", fmt_num(latest[indicator_col]), indicator_label)
    with c3:
        regime = latest["astro_regime_v2"] if "astro_regime_v2" in latest.index else classify_regime(latest["astro_momentum_v2"])
        metric_card("Astro Regime", regime, "Astro v2", regime_color(regime))
    with c4:
        sig = latest["signal"] if "signal" in latest.index else "N/A"
        metric_card("Rule Signal", sig, "Rule-based signal", signal_color(sig))
    with c5:
        if not opt_v2.empty:
            best = opt_v2.iloc[0]
            metric_card("Best Optimized", f"{best['indicator']} / span {int(best['span'])}", "Optimizer v2")
        else:
            metric_card("Best Optimized", "N/A", "No optimizer file")

    # Main overview chart
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.48, 0.27, 0.25],
        subplot_titles=("BTC Price", indicator_label, "Rule-Based Strategy vs Buy & Hold")
    )

    fig.add_trace(go.Scatter(
        x=hist_price["date"], y=hist_price["price"],
        mode="lines", name="BTC Price",
        line=dict(color="#3b82f6", width=2)
    ), row=1, col=1)

    if show_signals and "signal" in hist_price.columns:
        sm = hist_price[["date", "price", "signal"]].copy()
        sm["prev"] = sm["signal"].shift(1)
        sm = sm[sm["signal"] != sm["prev"]]
        buys = sm[sm["signal"].isin(["buy", "strong_buy"])]
        sells = sm[sm["signal"].isin(["sell", "strong_sell"])]

        fig.add_trace(go.Scatter(
            x=buys["date"], y=buys["price"],
            mode="markers", name="Buy",
            marker=dict(color="#22c55e", size=9, symbol="triangle-up")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=sells["date"], y=sells["price"],
            mode="markers", name="Sell",
            marker=dict(color="#ef4444", size=9, symbol="triangle-down")
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist[indicator_col],
        mode="lines", name=indicator_label,
        line=dict(color="#93c5fd", width=2)
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=2, col=1)

    if "strategy_equity" in hist.columns:
        bt = hist.dropna(subset=["strategy_equity", "buy_hold_equity"])
        fig.add_trace(go.Scatter(
            x=bt["date"], y=bt["strategy_equity"],
            mode="lines", name="Rule Strategy",
            line=dict(color="#22c55e", width=2)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=bt["date"], y=bt["buy_hold_equity"],
            mode="lines", name="Buy & Hold",
            line=dict(color="#e5e7eb", width=2, dash="dash")
        ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=1050,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 — FORECAST
# =========================================================
with tabs[1]:
    st.markdown("""
    <div class="explain-box">
    <b>Reading flow:</b> เส้นฟ้า = Astro history, เส้นส้มประ = Astro forecast. ใช้ดู momentum โหรในอนาคต ไม่ใช่ราคาในอนาคต
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist[indicator_col],
        mode="lines", name="Historical Astro",
        line=dict(color="#93c5fd", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future["date"], y=future[indicator_col],
        mode="lines", name="Future Astro",
        line=dict(color="#f59e0b", width=2, dash="dash")
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    add_safe_vertical_marker(fig, last_price_date, "Forecast Start", "#e5e7eb")
    fig.update_layout(
        template="plotly_dark",
        height=620,
        title=f"{indicator_label} — Future Forecast",
        xaxis_title="Date",
        yaxis_title=indicator_label
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Future Astro Table")
    cols = ["date", indicator_col, "astro_regime_v2", "astro_bullish_score", "astro_bearish_score",
            "astro_reversal_score", "astro_volatility_score", "astro_trend_start_score", "astro_trend_end_score"]
    cols = [c for c in cols if c in future.columns]
    show_future = future[cols].copy()
    show_future["date"] = show_future["date"].dt.date
    st.dataframe(show_future.head(120), use_container_width=True)

# =========================================================
# TAB 3 — ML INTELLIGENCE
# =========================================================
# =========================================================
# TAB 3 — ML INTELLIGENCE
# =========================================================
with tabs[2]:
    st.markdown("""
    <div class="explain-box">
    <b>Reading flow:</b><br>
    1) เลือก horizon ที่ต้องการดู เช่น 3D, 14D, 60D<br>
    2) ดูว่า ML Strategy ชนะ Buy & Hold หรือไม่<br>
    3) ดู Probability Up ว่าโมเดลเอียงขึ้นหรือลง<br>
    4) ดู Feature Importance ว่า Astro factor ไหนมีผลกับ horizon นั้นมากที่สุด
    </div>
    """, unsafe_allow_html=True)

    if ml_summary.empty or ml_pred.empty:
        st.warning("ยังไม่มีไฟล์ ML predictions / ML summary")
    else:
        available_horizons = sorted(ml_summary["horizon_days"].dropna().unique().astype(int).tolist())

        selected_horizon = st.selectbox(
            "Select ML Horizon",
            available_horizons,
            index=available_horizons.index(14) if 14 in available_horizons else 0
        )

        summary_h = ml_summary[ml_summary["horizon_days"] == selected_horizon].iloc[0]
        pred_h = ml_pred[ml_pred["horizon"] == selected_horizon].copy()

        pred_h = pred_h.sort_values("date").reset_index(drop=True)

        # -----------------------------
        # SUMMARY CARDS
        # -----------------------------
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            metric_card(
                f"ML Return {selected_horizon}D",
                fmt_pct(summary_h.get("ml_total_return", np.nan)),
                "Walk-forward"
            )

        with c2:
            metric_card(
                "Buy & Hold",
                fmt_pct(summary_h.get("buy_hold_return_same_period", np.nan)),
                "Same ML period"
            )

        with c3:
            metric_card(
                "ML Max DD",
                fmt_pct(summary_h.get("ml_max_drawdown", np.nan)),
                "Drawdown risk"
            )

        with c4:
            metric_card(
                "Accuracy",
                fmt_pct(summary_h.get("direction_accuracy", np.nan)),
                f"{selected_horizon}D direction"
            )

        with c5:
            metric_card(
                "Trades",
                int(summary_h.get("number_of_trades", 0)),
                "Position changes"
            )

        ml_return = summary_h.get("ml_total_return", np.nan)
        bh_return = summary_h.get("buy_hold_return_same_period", np.nan)

        if pd.notna(ml_return) and pd.notna(bh_return):
            if ml_return > bh_return:
                interpretation = f"ML {selected_horizon}D is outperforming Buy & Hold in the walk-forward test."
            else:
                interpretation = f"ML {selected_horizon}D is underperforming Buy & Hold in the walk-forward test."
        else:
            interpretation = "Not enough data to compare ML vs Buy & Hold."

        st.markdown(f"""
        <div class="explain-box">
        <b>Current ML Interpretation:</b><br>
        {interpretation}<br><br>
        Probability Up > upper threshold = Long bias<br>
        Probability Up < lower threshold = Short bias<br>
        Middle zone = Flat / unclear
        </div>
        """, unsafe_allow_html=True)

        # -----------------------------
        # MAIN ML CHART
        # -----------------------------
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.42, 0.30, 0.28],
            subplot_titles=(
                f"ML Strategy vs Buy & Hold — {selected_horizon}D",
                f"ML Probability Up — {selected_horizon}D",
                "ML Position"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pred_h["date"],
                y=pred_h["ml_strategy_equity"],
                mode="lines",
                name="ML Strategy",
                line=dict(color="#22c55e", width=2)
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=pred_h["date"],
                y=pred_h["buy_hold_equity_ml_period"],
                mode="lines",
                name="Buy & Hold",
                line=dict(color="#e5e7eb", width=2, dash="dash")
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=pred_h["date"],
                y=pred_h["ml_prob_up"],
                mode="lines",
                name="Probability Up",
                line=dict(color="#f59e0b", width=2)
            ),
            row=2,
            col=1
        )

        long_th = summary_h.get("long_probability_threshold", np.nan)
        short_th = summary_h.get("short_probability_threshold", np.nan)

        if pd.notna(long_th):
            fig.add_hline(
                y=long_th,
                line_dash="dash",
                line_color="#22c55e",
                row=2,
                col=1
            )

        if pd.notna(short_th):
            fig.add_hline(
                y=short_th,
                line_dash="dash",
                line_color="#ef4444",
                row=2,
                col=1
            )

        if "ml_position" in pred_h.columns:
            pos_col = "ml_position"
        else:
            pos_col = "ml_position_raw"

        fig.add_trace(
            go.Scatter(
                x=pred_h["date"],
                y=pred_h[pos_col],
                mode="lines",
                name="ML Position",
                line=dict(color="#60a5fa", width=2),
                line_shape="hv"
            ),
            row=3,
            col=1
        )

        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=3, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=900,
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Prob Up", row=2, col=1)
        fig.update_yaxes(title_text="Position", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # HORIZON COMPARISON TABLE
        # -----------------------------
        st.subheader("Horizon Comparison")

        summary_cols = [
            "horizon_days",
            "ml_total_return",
            "buy_hold_return_same_period",
            "ml_max_drawdown",
            "buy_hold_max_drawdown_same_period",
            "ml_sharpe_like",
            "buy_hold_sharpe_like",
            "return_drawdown_ratio",
            "balanced_score",
            "number_of_trades",
            "direction_accuracy",
            "direction_precision",
            "direction_recall",
        ]

        summary_cols = [c for c in summary_cols if c in ml_summary.columns]
        show_summary = ml_summary[summary_cols].copy()

        pct_cols = [
            "ml_total_return",
            "buy_hold_return_same_period",
            "ml_max_drawdown",
            "buy_hold_max_drawdown_same_period",
            "direction_accuracy",
            "direction_precision",
            "direction_recall",
        ]

        for c in pct_cols:
            if c in show_summary.columns:
                show_summary[c] = show_summary[c].apply(fmt_pct)

        num_cols = [
            "ml_sharpe_like",
            "buy_hold_sharpe_like",
            "return_drawdown_ratio",
            "balanced_score",
        ]

        for c in num_cols:
            if c in show_summary.columns:
                show_summary[c] = pd.to_numeric(show_summary[c], errors="coerce").round(3)

        st.dataframe(show_summary, use_container_width=True)

        # -----------------------------
        # FEATURE IMPORTANCE BY HORIZON
        # -----------------------------
        st.subheader(f"Top Feature Importance — {selected_horizon}D")

        if ml_importance.empty:
            st.warning("ยังไม่มีไฟล์ ml_feature_importance.csv")
        else:
            imp_h = ml_importance[ml_importance["horizon"] == selected_horizon].copy()

            if imp_h.empty:
                st.info(f"ไม่พบ feature importance สำหรับ horizon {selected_horizon}D")
            else:
                top_n = st.slider("Top N Features", 10, 50, 25, step=5)

                top_imp = (
                    imp_h.sort_values("importance", ascending=False)
                    .head(top_n)
                    .copy()
                )

                fig_imp = go.Figure()
                fig_imp.add_trace(
                    go.Bar(
                        x=top_imp["importance"],
                        y=top_imp["feature"],
                        orientation="h",
                        marker_color="#60a5fa"
                    )
                )

                fig_imp.update_layout(
                    template="plotly_dark",
                    height=760,
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    title=f"Top {top_n} Features — {selected_horizon}D"
                )

                st.plotly_chart(fig_imp, use_container_width=True)

                st.dataframe(
                    top_imp[["feature", "importance"]].reset_index(drop=True),
                    use_container_width=True
                )

        # -----------------------------
        # LATEST ML SIGNAL BY HORIZON
        # -----------------------------
        st.subheader("Latest ML Signals by Horizon")

        latest_signals = []

        for h in available_horizons:
            ph = ml_pred[ml_pred["horizon"] == h].copy()
            if ph.empty:
                continue

            last = ph.sort_values("date").iloc[-1]

            pos = last.get("ml_position_raw", np.nan)
            prob = last.get("ml_prob_up", np.nan)

            if pos == 1:
                signal_label = "LONG"
            elif pos == -1:
                signal_label = "SHORT"
            else:
                signal_label = "FLAT"

            latest_signals.append({
                "horizon": f"{h}D",
                "date": last["date"],
                "prob_up": prob,
                "signal": signal_label,
                "price": last["price"],
            })

        latest_signal_df = pd.DataFrame(latest_signals)

        if not latest_signal_df.empty:
            latest_signal_df["date"] = pd.to_datetime(latest_signal_df["date"]).dt.date
            latest_signal_df["prob_up"] = pd.to_numeric(latest_signal_df["prob_up"], errors="coerce").round(4)
            latest_signal_df["price"] = pd.to_numeric(latest_signal_df["price"], errors="coerce").round(2)
            st.dataframe(latest_signal_df, use_container_width=True)

# =========================================================
# TAB 4 — ASTRO RESEARCH
# =========================================================
with tabs[3]:
    st.markdown("""
    <div class="explain-box">
    <b>Reading flow:</b> ดู Natal Chart ก่อนเพื่อเข้าใจดวงกำเนิด Bitcoin จากนั้นดูตำแหน่งดาวกำเนิดและแกน Asc/MC
    </div>
    """, unsafe_allow_html=True)

    natal = get_natal_chart()
    left, right = st.columns([1.0, 1.1])

    with left:
        st.subheader("Bitcoin Natal Chart Wheel")
        st.plotly_chart(make_natal_wheel(natal), use_container_width=True)

    with right:
        st.subheader("Natal Positions")
        rows = []
        for body, lon in natal.items():
            rows.append({
                "body": body,
                "symbol": PLANET_SYMBOLS.get(body, ""),
                "longitude": round(lon, 4),
                "position": fmt_deg(lon)
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Optimized Model v2 Results")
    if opt_v2.empty:
        st.info("ยังไม่มี optimization v2 results")
    else:
        st.dataframe(opt_v2.head(30), use_container_width=True)

# =========================================================
# TAB 5 — RAW ASPECTS
# =========================================================
with tabs[4]:
    st.markdown("""
    <div class="explain-box">
    <b>Reading flow:</b> ใช้ panel นี้เพื่อ debug ว่า Astro Score มาจากดาวจรอะไร ทำมุมกับจุดกำเนิดไหน และสร้างคะแนนประเภทใด
    </div>
    """, unsafe_allow_html=True)

    if raw_aspects.empty:
        st.warning("ยังไม่มี data/astro_aspects_raw.csv")
    else:
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            raw_start = st.date_input("Start", value=last_price_date.date() - pd.Timedelta(days=30), key="raw_start")
        with f2:
            raw_end = st.date_input("End", value=last_price_date.date(), key="raw_end")
        with f3:
            score_type = st.selectbox(
                "Score Type",
                ["bullish", "bearish", "reversal", "volatility", "compression", "trend_start", "trend_end"],
                index=0
            )
        with f4:
            min_score = st.number_input("Min abs score", min_value=0.0, value=0.1, step=0.1)

        p1, p2, p3 = st.columns(3)
        with p1:
            planets = st.multiselect("Transit Planet", sorted(raw_aspects["transit_planet"].dropna().unique()))
        with p2:
            aspects = st.multiselect("Aspect", sorted(raw_aspects["aspect"].dropna().unique()))
        with p3:
            targets = st.multiselect("Target", sorted(raw_aspects["target"].dropna().unique()))

        rv = raw_aspects[
            (raw_aspects["date"].dt.date >= raw_start) &
            (raw_aspects["date"].dt.date <= raw_end)
        ].copy()

        if planets:
            rv = rv[rv["transit_planet"].isin(planets)]
        if aspects:
            rv = rv[rv["aspect"].isin(aspects)]
        if targets:
            rv = rv[rv["target"].isin(targets)]

        rv[score_type] = pd.to_numeric(rv[score_type], errors="coerce").fillna(0)
        rv = rv[rv[score_type].abs() >= min_score]

        if rv.empty:
            st.info("ไม่พบ raw aspects ตาม filter ที่เลือก")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Aspect Count", f"{len(rv):,}")
            c2.metric(f"Total {score_type}", f"{rv[score_type].sum():.2f}")
            c3.metric("Max Single Score", f"{rv[score_type].max():.2f}")

            planet_impact = (
                rv.groupby("transit_planet")[score_type]
                .sum()
                .reset_index()
                .sort_values(score_type, ascending=False)
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=planet_impact["transit_planet"],
                y=planet_impact[score_type],
                marker_color="#60a5fa"
            ))
            fig.update_layout(
                template="plotly_dark",
                height=420,
                title=f"Planet Contribution — {score_type}",
                xaxis_title="Transit Planet",
                yaxis_title="Score"
            )
            st.plotly_chart(fig, use_container_width=True)

            show_cols = [
                "date", "source", "rule_name", "transit_planet", "target", "aspect",
                "orb", "orb_factor", "multiplier",
                "bullish", "bearish", "reversal", "volatility", "compression", "trend_start", "trend_end"
            ]
            show_cols = [c for c in show_cols if c in rv.columns]
            table = rv[show_cols].copy()
            table["date"] = table["date"].dt.date
            st.dataframe(table.sort_values(score_type, ascending=False), use_container_width=True, height=540)
