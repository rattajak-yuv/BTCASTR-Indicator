import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="BTC Astro Quant Terminal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")

TAXONOMY_COLORS = {
    "Constructive / Positive Drift": "#2E7D32",
    "Neutral / Tactical": "#B59B2B",
    "False Bull / Exhaustion Risk": "#D97706",
    "Bearish": "#C62828",
    "High Risk": "#7F1D1D",
    "Unknown": "#6B7280",
}

SIGNAL_COLORS = {
    "Bullish": "#2E7D32",
    "Neutral": "#9CA3AF",
    "Bearish": "#C62828",
}

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
:root {
  --bg: #F6F8FB;
  --panel: #FFFFFF;
  --ink: #202331;
  --muted: #6B7280;
  --line: #E6EAF0;
  --navy: #111827;
  --card: #FFFFFF;
  --shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--ink);
}

[data-testid="stSidebar"] {
  background: #EEF2F7;
  border-right: 1px solid #E1E7EF;
}

.block-container {
  padding-top: 2.2rem;
  padding-bottom: 4rem;
  max-width: 1320px;
}

h1, h2, h3 {
  letter-spacing: -0.03em;
}

.terminal-hero {
  background: linear-gradient(135deg, #0B1220 0%, #172033 55%, #0B1220 100%);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 22px 55px rgba(15,23,42,0.18);
  border-radius: 28px;
  padding: 34px 38px;
  color: white;
  margin: 0 0 26px 0;
}
.terminal-kicker {
  color: #93C5FD;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.22em;
  font-weight: 700;
  margin-bottom: 10px;
}
.terminal-title {
  font-size: 2.25rem;
  font-weight: 850;
  letter-spacing: -0.04em;
  margin-bottom: 10px;
}
.terminal-subtitle {
  font-size: 1.02rem;
  line-height: 1.65;
  color: #C8D1E1;
  max-width: 960px;
}

.section-kicker {
  color: #94A3B8;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  font-size: 0.78rem;
  margin: 26px 0 8px 0;
}
.section-title {
  color: #1F2937;
  font-size: 1.55rem;
  font-weight: 850;
  margin-bottom: 6px;
}
.section-caption {
  color: #64748B;
  font-size: 0.96rem;
  margin-bottom: 16px;
}

.metric-card {
  height: 168px;
  background: var(--card);
  border-radius: 22px;
  padding: 22px 22px;
  border: 1px solid #E5EAF2;
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  overflow: hidden;
}
.metric-card.dark {
  background: linear-gradient(180deg, #101827 0%, #121827 100%);
  color: #FFFFFF;
  border: 1px solid rgba(255,255,255,0.07);
}
.metric-label {
  color: #8EA0B8;
  text-transform: uppercase;
  letter-spacing: 0.13em;
  font-size: 0.72rem;
  font-weight: 750;
  line-height: 1.2;
}
.metric-value {
  font-size: 1.58rem;
  font-weight: 850;
  letter-spacing: -0.03em;
  line-height: 1.15;
  word-break: break-word;
}
.metric-footnote {
  color: #8EA0B8;
  font-size: 0.88rem;
  line-height: 1.4;
}

.summary-panel {
  background: #FFFFFF;
  border: 1px solid #E5EAF2;
  box-shadow: var(--shadow);
  border-radius: 24px;
  padding: 22px 24px;
  margin: 20px 0 24px 0;
  line-height: 1.65;
  color: #334155;
}
.summary-panel strong { color: #111827; }

.badge {
  display: inline-block;
  padding: 6px 11px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 750;
  color: white;
  margin-right: 6px;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}
.info-card {
  background: #FFFFFF;
  border: 1px solid #E5EAF2;
  box-shadow: var(--shadow);
  border-radius: 22px;
  padding: 20px 22px;
  min-height: 180px;
}
.info-card-title {
  color: #64748B;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-weight: 800;
  font-size: 0.75rem;
  margin-bottom: 12px;
}
.info-card-main {
  font-weight: 850;
  font-size: 1.28rem;
  line-height: 1.2;
  margin-bottom: 12px;
}
.info-card-body {
  color: #475569;
  line-height: 1.55;
  font-size: 0.92rem;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 16px;
  border-bottom: 1px solid #E5EAF2;
}
.stTabs [data-baseweb="tab"] {
  height: 42px;
  padding: 8px 4px;
  color: #64748B;
}
.stTabs [aria-selected="true"] {
  color: #EF4444 !important;
  border-bottom: 2px solid #EF4444;
}

[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid #E5EAF2;
}

@media (max-width: 900px) {
  .terminal-title { font-size: 1.7rem; }
  .card-grid { grid-template-columns: 1fr; }
  .metric-card { height: auto; min-height: 140px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def read_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def read_csv(path: str):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_btc_price(start="2014-01-01"):
    candidates = [
        DATA_DIR / "bitcoin_astro_daily_score.csv",
        DATA_DIR / "ml_dataset.csv",
        DATA_DIR / "ml_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                cols = {c.lower(): c for c in df.columns}
                date_col = cols.get("date") or cols.get("datetime")
                price_col = None
                for name in ["price", "close", "btc_price", "btc_close", "adj close"]:
                    if name in cols:
                        price_col = cols[name]
                        break
                if date_col and price_col:
                    out = df[[date_col, price_col]].copy()
                    out.columns = ["date", "price"]
                    out["date"] = pd.to_datetime(out["date"], errors="coerce")
                    out["price"] = pd.to_numeric(out["price"], errors="coerce")
                    out = out.dropna().drop_duplicates("date").sort_values("date")
                    if not out.empty:
                        return out
            except Exception:
                pass

    if yf is not None:
        try:
            px = yf.download("BTC-USD", start=start, progress=False, auto_adjust=True)
            if isinstance(px.columns, pd.MultiIndex):
                px.columns = px.columns.get_level_values(0)
            px = px.reset_index()
            if "Date" in px.columns and "Close" in px.columns:
                out = px[["Date", "Close"]].copy()
                out.columns = ["date", "price"]
                out["date"] = pd.to_datetime(out["date"], errors="coerce")
                out["price"] = pd.to_numeric(out["price"], errors="coerce")
                return out.dropna().sort_values("date")
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "price"])


def pct(x):
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"


def money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"


def date_fmt(x):
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return "—"


def clamp_text(text, n=120):
    if text is None:
        return "—"
    text = str(text)
    return text if len(text) <= n else text[: n - 3] + "..."


def confidence_bucket(x):
    try:
        x = float(x)
    except Exception:
        return "Unknown"
    if x >= 0.60:
        return "High"
    if x >= 0.40:
        return "Medium"
    return "Low"


def taxonomy_color(taxonomy):
    return TAXONOMY_COLORS.get(str(taxonomy), TAXONOMY_COLORS["Unknown"])


def metric_card(label, value, footnote="", taxonomy=None):
    color = taxonomy_color(taxonomy) if taxonomy else "#334155"
    st.markdown(
        f"""
<div class="metric-card dark" style="border-top: 3px solid {color};">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}</div>
  <div class="metric-footnote">{footnote}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def info_card(title, main, body="", color="#334155"):
    st.markdown(
        f"""
<div class="info-card" style="border-left: 4px solid {color};">
  <div class="info-card-title">{title}</div>
  <div class="info-card-main" style="color:{color};">{main}</div>
  <div class="info-card-body">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def prepare_windows(timeline_json):
    rows = []
    if timeline_json and isinstance(timeline_json, dict):
        rows = timeline_json.get("windows", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        # fallback to csv if available
        for path in [DATA_DIR / "forecast_intelligence_v2.csv", DATA_DIR / "forecast_windows.csv"]:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    break
                except Exception:
                    continue
    if df.empty:
        return df
    # normalize columns
    rename = {}
    if "taxonomy" in df.columns and "taxonomy_v2" not in df.columns:
        rename["taxonomy"] = "taxonomy_v2"
    if "days" in df.columns and "duration_days" not in df.columns:
        rename["days"] = "duration_days"
    if "avg_confidence" in df.columns and "average_confidence" not in df.columns:
        rename["avg_confidence"] = "average_confidence"
    if "avg_probability" in df.columns and "average_ml_probability" not in df.columns:
        rename["avg_probability"] = "average_ml_probability"
    df = df.rename(columns=rename)
    for c in ["start_date", "end_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.dropna(subset=["start_date", "end_date"]).sort_values("start_date")


def prepare_turning_points(risk_json):
    rows = []
    if risk_json and isinstance(risk_json, dict):
        rows = risk_json.get("turning_points", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        for path in [DATA_DIR / "turning_points.csv"]:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    break
                except Exception:
                    continue
    if df.empty:
        return df
    if "turning_point_date" in df.columns:
        df["turning_point_date"] = pd.to_datetime(df["turning_point_date"], errors="coerce")
    return df.dropna(subset=["turning_point_date"]).sort_values("turning_point_date")


def range_start(price_df, choice):
    if price_df.empty:
        return pd.Timestamp.today() - pd.Timedelta(days=365)
    max_d = price_df["date"].max()
    if choice == "1M":
        return max_d - pd.Timedelta(days=31)
    if choice == "3M":
        return max_d - pd.Timedelta(days=92)
    if choice == "6M":
        return max_d - pd.Timedelta(days=183)
    if choice == "1Y":
        return max_d - pd.Timedelta(days=365)
    if choice == "2Y":
        return max_d - pd.Timedelta(days=730)
    if choice == "4Y":
        return max_d - pd.Timedelta(days=1460)
    return price_df["date"].min()


def build_price_taxonomy_chart(price_df, windows_df, turning_df, range_choice, show_windows=True, show_turns=True):
    fig = go.Figure()

    if price_df.empty:
        fig.add_annotation(text="BTC price data not available", showarrow=False)
        return fig

    start_d = range_start(price_df, range_choice)
    end_d = max(price_df["date"].max(), windows_df["end_date"].max() if not windows_df.empty else price_df["date"].max())
    visible_px = price_df[(price_df["date"] >= start_d) & (price_df["date"] <= end_d)].copy()
    if visible_px.empty:
        visible_px = price_df.tail(365).copy()

    fig.add_trace(
        go.Scatter(
            x=visible_px["date"],
            y=visible_px["price"],
            mode="lines",
            name="BTCUSD",
            line=dict(color="#4F83FF", width=2.4),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>BTC: $%{y:,.0f}<extra></extra>",
        )
    )

    # Future / historical taxonomy windows as background bands
    if show_windows and not windows_df.empty:
        view_windows = windows_df[(windows_df["end_date"] >= start_d) & (windows_df["start_date"] <= end_d)].copy()
        for _, r in view_windows.iterrows():
            tax = r.get("taxonomy_v2", "Unknown")
            c = taxonomy_color(tax)
            fig.add_vrect(
                x0=r["start_date"],
                x1=r["end_date"],
                fillcolor=c,
                opacity=0.13,
                line_width=0,
                layer="below",
                annotation_text=None,
            )

    # Today line
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    fig.add_vline(
        x=today,
        line_width=1.5,
        line_dash="dot",
        line_color="#94A3B8",
        annotation_text="Today",
        annotation_position="top",
    )

    if show_turns and not turning_df.empty:
        tdf = turning_df[(turning_df["turning_point_date"] >= start_d) & (turning_df["turning_point_date"] <= end_d)].copy()
        if not tdf.empty:
            px_for_join = price_df.set_index("date").sort_index()["price"]
            markers = []
            for _, r in tdf.iterrows():
                d = r["turning_point_date"]
                # nearest prior price
                try:
                    idx = px_for_join.index.searchsorted(d, side="right") - 1
                    y = float(px_for_join.iloc[max(idx, 0)])
                except Exception:
                    y = float(visible_px["price"].median())
                new_sig = str(r.get("new_signal", "Neutral"))
                symbol = "triangle-up" if new_sig == "Bullish" else "triangle-down" if new_sig == "Bearish" else "circle"
                color = SIGNAL_COLORS.get(new_sig, "#CBD5E1")
                markers.append({
                    "date": d,
                    "price": y,
                    "new_signal": new_sig,
                    "symbol": symbol,
                    "color": color,
                    "type": r.get("turning_point_type", ""),
                    "confidence": r.get("confidence", np.nan),
                    "explanation": r.get("explanation", ""),
                })
            mdf = pd.DataFrame(markers)
            for sig, g in mdf.groupby("new_signal"):
                fig.add_trace(
                    go.Scatter(
                        x=g["date"],
                        y=g["price"],
                        mode="markers",
                        name=f"Turning Point: {sig}",
                        marker=dict(
                            size=10,
                            symbol=g["symbol"].iloc[0],
                            color=g["color"].iloc[0],
                            line=dict(width=1, color="#FFFFFF"),
                        ),
                        customdata=np.stack([g["type"], g["confidence"], g["explanation"]], axis=-1),
                        hovertemplate=(
                            "<b>%{x|%Y-%m-%d}</b><br>Signal: " + sig +
                            "<br>Type: %{customdata[0]}<br>Confidence: %{customdata[1]:.2%}<br>%{customdata[2]}<extra></extra>"
                        ),
                    )
                )

    fig.update_layout(
        height=640,
        margin=dict(l=20, r=20, t=36, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="", gridcolor="#EEF2F7", rangeslider=dict(visible=False)),
        yaxis=dict(title="BTCUSD", gridcolor="#EEF2F7", tickprefix="$", separatethousands=True),
        font=dict(color="#334155", family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"),
    )
    return fig


def timeline_chart(windows_df):
    if windows_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Forecast windows not available", showarrow=False)
        return fig
    df = windows_df.copy().reset_index(drop=True)
    df["window_id"] = [f"W{i+1}" for i in range(len(df))]
    fig = go.Figure()
    for _, r in df.iterrows():
        tax = r.get("taxonomy_v2", "Unknown")
        fig.add_trace(
            go.Scatter(
                x=[r["start_date"], r["end_date"]],
                y=[r["window_id"], r["window_id"]],
                mode="lines+markers",
                line=dict(color=taxonomy_color(tax), width=12),
                marker=dict(size=6, color=taxonomy_color(tax)),
                name=tax,
                showlegend=False,
                hovertemplate=(
                    f"<b>{tax}</b><br>{date_fmt(r['start_date'])} → {date_fmt(r['end_date'])}"
                    f"<br>Confidence: {pct(r.get('average_confidence'))}"
                    f"<br>Probability: {pct(r.get('average_ml_probability'))}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(title="", gridcolor="#EEF2F7"),
        yaxis=dict(title="Forecast Sequence", gridcolor="#EEF2F7"),
        font=dict(color="#334155", family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"),
    )
    return fig

# ============================================================
# LOAD DATA
# ============================================================

current_state = read_json(str(DATA_DIR / "dashboard_current_state.json")) or {}
timeline_json = read_json(str(DATA_DIR / "dashboard_timeline.json")) or {}
risk_json = read_json(str(DATA_DIR / "dashboard_risk_calendar.json")) or {}
summary_json = read_json(str(DATA_DIR / "dashboard_summary.json")) or {}

price_df = load_btc_price()
windows_df = prepare_windows(timeline_json)
turning_df = prepare_turning_points(risk_json)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## Controls")
range_choice = st.sidebar.radio(
    "Price / Overlay Range",
    ["1M", "3M", "6M", "1Y", "2Y", "4Y", "ALL"],
    index=4,
    horizontal=False,
)
show_taxonomy_bands = st.sidebar.checkbox("Show taxonomy background", value=True)
show_turning_markers = st.sidebar.checkbox("Show turning-point markers", value=True)
show_detail_tables = st.sidebar.checkbox("Show detailed tables", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Forecast Data")
st.sidebar.caption(f"Forecast date: {current_state.get('forecast_date', summary_json.get('forecast_date', '—'))}")
st.sidebar.caption("Model: Robust Astro Engine v1")
st.sidebar.caption("Mode: Forecast Dashboard")

# ============================================================
# HEADER
# ============================================================

st.title("Bitcoin Astro Quant Dashboard")

forecast_date = current_state.get("forecast_date", summary_json.get("forecast_date", "—"))
current_signal = current_state.get("current_signal", summary_json.get("Current Signal", "—"))
current_tax = current_state.get("current_taxonomy", summary_json.get("Current Taxonomy", "—"))
current_conf = current_state.get("current_confidence", summary_json.get("Current Confidence", np.nan))
current_prob = current_state.get("current_probability", np.nan)
market_view = current_state.get("market_view", "—")
risk_level = current_state.get("risk_level", "—")
recommended_bias = current_state.get("recommended_bias", "—")

latest_price = price_df["price"].iloc[-1] if not price_df.empty else np.nan
prev_price = price_df["price"].iloc[-2] if len(price_df) > 1 else np.nan
price_chg = (latest_price / prev_price - 1) if np.isfinite(latest_price) and np.isfinite(prev_price) and prev_price else np.nan

st.markdown(
    f"""
<div class="terminal-hero">
  <div class="terminal-kicker">BTC ASTRO FORECAST TERMINAL</div>
  <div class="terminal-title">Evidence-Based Bitcoin Forecast</div>
  <div class="terminal-subtitle">
    Generated: <b>{forecast_date}</b> · Current state: <b>{current_signal}</b> · Calibrated taxonomy: <b>{current_tax}</b>.<br>
    This dashboard prioritizes robust out-of-sample interpretation, forecast windows, and turning-point visibility.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# EXECUTIVE CARDS
# ============================================================

cols = st.columns([1, 1.25, 1, 1, 1, 1])
with cols[0]:
    metric_card("Current Signal", current_signal, market_view, current_tax)
with cols[1]:
    metric_card("Taxonomy", current_tax, current_state.get("current_window", {}).get("v2_posture", ""), current_tax)
with cols[2]:
    metric_card("Confidence", pct(current_conf), f"{confidence_bucket(current_conf)} · Probability {pct(current_prob)}", current_tax)
with cols[3]:
    metric_card("Risk", risk_level, "Current market risk framing", current_tax)
with cols[4]:
    metric_card("Bias", recommended_bias, market_view, current_tax)
with cols[5]:
    metric_card("BTC Price", money(latest_price), f"{price_chg:+.2%} vs prior close" if np.isfinite(price_chg) else "", current_tax)

cw = current_state.get("current_window", {}) or {}
if cw:
    st.markdown(
        f"""
<div class="summary-panel">
  <strong>Current window:</strong> <span class="badge" style="background:{taxonomy_color(current_tax)}">{current_tax}</span>
  {date_fmt(cw.get('start_date'))} → {date_fmt(cw.get('end_date'))}<br>
  <strong>Investor interpretation:</strong> {cw.get('narrative_v2', cw.get('taxonomy_reason', '—'))}
</div>
""",
        unsafe_allow_html=True,
    )

# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(["1. Forecast Terminal", "2. Window Details", "3. Turning Points", "4. Research Tables"])

with tab1:
    st.markdown('<div class="section-kicker">Primary Market Chart</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">BTCUSD price with calibrated forecast taxonomy bands and turning-point markers.</div>', unsafe_allow_html=True)
    fig = build_price_taxonomy_chart(price_df, windows_df, turning_df, range_choice, show_taxonomy_bands, show_turning_markers)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-kicker">Forward Outlook</div>', unsafe_allow_html=True)
    outlooks = (timeline_json or {}).get("outlooks", {}) or summary_json
    c1, c2, c3 = st.columns(3)
    for col, key, label in [(c1, "30d", "30D Outlook"), (c2, "90d", "90D Outlook"), (c3, "365d", "365D Outlook")]:
        data = outlooks.get(key, {}) if isinstance(outlooks, dict) else {}
        if not data and isinstance(summary_json, dict):
            data = summary_json.get(label, {}) or {}
        tax = data.get("dominant_taxonomy", "—")
        with col:
            info_card(
                label,
                tax,
                data.get("summary", f"Confidence {pct(data.get('average_confidence'))} · Probability {pct(data.get('average_probability'))}"),
                taxonomy_color(tax),
            )

    st.markdown('<div class="section-kicker">12-Month Forecast Timeline</div>', unsafe_allow_html=True)
    st.plotly_chart(timeline_chart(windows_df), use_container_width=True)

    st.markdown('<div class="section-kicker">Risk Calendar</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    ntp = current_state.get("next_turning_point", {}) or {}
    ncw = current_state.get("next_constructive_window", {}) or {}
    nhw = current_state.get("next_high_risk_window", {}) or {}
    with r1:
        info_card("Next Turning Point", date_fmt(ntp.get("turning_point_date")), ntp.get("explanation", "—"), "#3B82F6")
    with r2:
        info_card("Next Constructive Window", f"{date_fmt(ncw.get('start_date'))} → {date_fmt(ncw.get('end_date'))}", ncw.get("taxonomy_reason", "—"), "#2E7D32")
    with r3:
        info_card("Next High-Risk Window", f"{date_fmt(nhw.get('start_date'))} → {date_fmt(nhw.get('end_date'))}", nhw.get("taxonomy_reason", "—"), "#C62828")

with tab2:
    st.markdown('<div class="section-title">Forecast Window Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Detailed window-level interpretation is kept here for investors who want to inspect dates, confidence, probability, and narrative.</div>', unsafe_allow_html=True)
    if windows_df.empty:
        st.warning("Forecast windows are not available.")
    else:
        detail = windows_df.copy()
        keep = [
            "start_date", "end_date", "taxonomy_v2", "duration_days",
            "average_confidence", "average_ml_probability", "average_astro_score",
            "average_risk_score", "v2_posture", "taxonomy_reason", "narrative_v2",
        ]
        keep = [c for c in keep if c in detail.columns]
        detail = detail[keep].copy()
        for c in ["average_confidence", "average_ml_probability", "average_risk_score"]:
            if c in detail.columns:
                detail[c] = pd.to_numeric(detail[c], errors="coerce")
        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True,
            column_config={
                "start_date": st.column_config.DateColumn("Start"),
                "end_date": st.column_config.DateColumn("End"),
                "taxonomy_v2": st.column_config.TextColumn("Taxonomy"),
                "duration_days": st.column_config.NumberColumn("Days"),
                "average_confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=1),
                "average_ml_probability": st.column_config.ProgressColumn("ML Probability", format="%.0f%%", min_value=0, max_value=1),
                "average_risk_score": st.column_config.ProgressColumn("Risk Score", format="%.0f%%", min_value=0, max_value=1),
                "v2_posture": st.column_config.TextColumn("Posture"),
                "taxonomy_reason": st.column_config.TextColumn("Evidence"),
                "narrative_v2": st.column_config.TextColumn("Narrative"),
            },
        )

with tab3:
    st.markdown('<div class="section-title">Turning Points</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Inflection dates where the model expects signal changes, momentum threshold breaks, or window transitions.</div>', unsafe_allow_html=True)
    if turning_df.empty:
        st.warning("Turning points are not available.")
    else:
        display = turning_df.copy()
        keep = ["turning_point_date", "turning_point_type", "old_signal", "new_signal", "confidence", "severity", "explanation"]
        keep = [c for c in keep if c in display.columns]
        display = display[keep]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "turning_point_date": st.column_config.DateColumn("Date"),
                "turning_point_type": st.column_config.TextColumn("Type"),
                "old_signal": st.column_config.TextColumn("Old"),
                "new_signal": st.column_config.TextColumn("New"),
                "confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=1),
                "severity": st.column_config.TextColumn("Severity"),
                "explanation": st.column_config.TextColumn("Explanation"),
            },
        )

with tab4:
    st.markdown('<div class="section-title">Research Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Optional raw data views for auditability. These are intentionally separated from the investor-facing dashboard.</div>', unsafe_allow_html=True)
    if show_detail_tables:
        with st.expander("Dashboard current state JSON", expanded=False):
            st.json(current_state)
        with st.expander("Dashboard summary JSON", expanded=False):
            st.json(summary_json)
        with st.expander("Price data sample", expanded=False):
            st.dataframe(price_df.tail(50), use_container_width=True)
    else:
        st.info("Enable 'Show detailed tables' in the sidebar to inspect raw research data.")

