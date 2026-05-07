import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import swisseph as swe
from datetime import datetime, timezone

st.set_page_config(page_title="Bitcoin Astro Indicator", layout="wide")

# -----------------------------
# CONFIG
# -----------------------------
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

PLANET_WEIGHTS = {
    "Jupiter": 2.0, "Saturn": 2.0, "Pluto": 2.0, "Uranus": 1.5,
    "Neptune": 1.2, "Mars": 1.0, "Mercury": 0.7, "Moon": 0.5,
    "Venus": 0.4, "Sun": 0.0,
}

ASPECTS = {
    "conjunction": (0, 1.00),
    "opposition": (180, 0.90),
    "square": (90, 0.85),
    "trine": (120, 0.75),
    "sextile": (60, 0.55),
}

TARGET_WEIGHTS = {
    "Asc": 1.20, "MC": 1.20, "Sun": 1.00, "Moon": 1.00,
    "Jupiter": 1.10, "Saturn": 1.10,
}

MAX_ORB_BY_PLANET = {
    "Jupiter": 3.0, "Saturn": 3.0, "Uranus": 3.0, "Neptune": 3.0,
    "Pluto": 3.0, "Mars": 2.0, "Mercury": 1.5, "Venus": 1.5,
    "Moon": 1.0, "Sun": 1.5,
}

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
.block-container { padding-top: 1rem; max-width: 1550px; }
.metric-card {
    background-color: #111827; padding: 16px 18px; border-radius: 14px;
    border: 1px solid #1f2937; min-height: 135px;
}
.metric-label { color: #9ca3af; font-size: 0.88rem; margin-bottom: 6px; }
.metric-value { color: white; font-size: 1.45rem; font-weight: 700; line-height: 1.2; }
.metric-sub { color: #60a5fa; font-size: 0.86rem; margin-top: 4px; }
.explain-box {
    background-color: #0f172a; border: 1px solid #1e293b;
    border-radius: 12px; padding: 14px 16px; margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPERS
# -----------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("data/bitcoin_astro_daily_score.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600)
def load_optimization():
    try:
        return pd.read_csv("data/model_optimization_results.csv")
    except Exception:
        return pd.DataFrame()

def julday(dt):
    return swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute / 60 + dt.second / 3600)

def norm360(x):
    return x % 360

def angle_diff(a, b):
    d = abs(norm360(a) - norm360(b))
    return min(d, 360 - d)

def get_planet_lon(jd, planet_id):
    xx, _ = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)
    return xx[0]

def get_houses(jd, lat, lon):
    cusps, ascmc = swe.houses_ex(jd, lat, lon, b'P', swe.FLG_SIDEREAL)
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

def classify_regime(score):
    if score >= 3.0:
        return "strong_bull"
    elif score >= 1.5:
        return "bull"
    elif score > -1.5:
        return "neutral"
    elif score > -3.0:
        return "bear"
    return "crash_risk"

def regime_color(regime):
    return {
        "strong_bull": "#22c55e",
        "bull": "#84cc16",
        "neutral": "#94a3b8",
        "bear": "#f59e0b",
        "crash_risk": "#ef4444",
    }.get(regime, "#94a3b8")

def signal_color(signal):
    return {
        "strong_buy": "#22c55e",
        "buy": "#84cc16",
        "neutral": "#94a3af",
        "sell": "#f59e0b",
        "strong_sell": "#ef4444",
    }.get(signal, "#94a3af")

def fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2%}"

def fmt_money(x):
    if pd.isna(x):
        return "N/A"
    return f"${x:,.0f}"

def fmt_num(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2f}"

def aspect_score(transit_planet_name, transit_lon, target_name, target_lon):
    base = PLANET_WEIGHTS.get(transit_planet_name, 0.0)
    max_orb = MAX_ORB_BY_PLANET.get(transit_planet_name, 2.0)
    target_weight = TARGET_WEIGHTS.get(target_name, 1.0)

    rows = []
    for aspect_name, (aspect_angle, aspect_weight) in ASPECTS.items():
        d = angle_diff(transit_lon, target_lon)
        orb = abs(d - aspect_angle)
        if orb <= max_orb:
            orb_factor = max(0.0, 1 - orb / max_orb)
            score = base * aspect_weight * target_weight * orb_factor
            rows.append({
                "transit_planet": transit_planet_name,
                "target": target_name,
                "aspect": aspect_name,
                "aspect_angle": aspect_angle,
                "orb": orb,
                "score": score,
            })
    return rows

def get_natal_chart():
    jd = julday(NATAL_DT)
    cusps, ascmc = get_houses(jd, LAT, LON)

    natal = {}
    for name, pid in PLANETS.items():
        natal[name] = get_planet_lon(jd, pid)
    natal["Asc"] = ascmc[0]
    natal["MC"] = ascmc[1]
    return natal, cusps, ascmc

def compute_daily_aspect_breakdown(date):
    dt = datetime(date.year, date.month, date.day, 12, 0, 0, tzinfo=timezone.utc)
    jd = julday(dt)

    natal, _, _ = get_natal_chart()
    transit = {name: get_planet_lon(jd, pid) for name, pid in PLANETS.items()}

    targets = ["Sun", "Moon", "Jupiter", "Saturn", "Asc", "MC"]
    all_rows = []

    for tname, tlon in transit.items():
        for target in targets:
            all_rows.extend(aspect_score(tname, tlon, target, natal[target]))

    df = pd.DataFrame(all_rows)

    if df.empty:
        return df

    def bucket(row):
        p = row["transit_planet"]
        s = row["score"]
        if p in ["Jupiter", "Uranus"]:
            return "expansion"
        if p == "Saturn":
            return "contraction"
        if p == "Pluto":
            return "mixed_regime"
        if p == "Neptune":
            return "narrative"
        if p in ["Mars", "Mercury", "Moon"]:
            return "trigger"
        return "minor"

    df["bucket"] = df.apply(bucket, axis=1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

def make_natal_wheel(natal):
    fig = go.Figure()

    theta = np.linspace(0, 360, 361)

    fig.add_trace(go.Scatterpolar(
        r=[1] * len(theta), theta=theta, mode="lines",
        line=dict(color="#64748b", width=2), showlegend=False
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.72] * len(theta), theta=theta, mode="lines",
        line=dict(color="#334155", width=1), showlegend=False
    ))

    # Zodiac divisions
    for i, (sign, sym) in enumerate(ZODIAC):
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

    # planets
    r_levels = {
        "Sun": 0.88, "Moon": 0.82, "Mercury": 0.76, "Venus": 0.70,
        "Mars": 0.64, "Jupiter": 0.58, "Saturn": 0.52,
        "Uranus": 0.46, "Neptune": 0.40, "Pluto": 0.34,
        "Asc": 0.96, "MC": 0.96
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
            name=name,
            hovertemplate=f"{name}<br>{fmt_deg(lon)}<extra></extra>"
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
        showlegend=False,
    )
    return fig

def next_turning_points(df_future, signal_col="astro_momentum_smooth", top_n=10, threshold=1.5):
    x = df_future[["date", signal_col]].copy().dropna().reset_index(drop=True)
    turning = []
    if len(x) < 3:
        return pd.DataFrame(columns=["date", signal_col, "type"])

    for i in range(1, len(x) - 1):
        prev_val = x.loc[i - 1, signal_col]
        curr_val = x.loc[i, signal_col]
        next_val = x.loc[i + 1, signal_col]

        if curr_val > prev_val and curr_val > next_val and abs(curr_val) >= threshold:
            turning.append({"date": x.loc[i, "date"], signal_col: curr_val, "type": "local_top"})
        if curr_val < prev_val and curr_val < next_val and abs(curr_val) >= threshold:
            turning.append({"date": x.loc[i, "date"], signal_col: curr_val, "type": "local_bottom"})

    if not turning:
        return pd.DataFrame(columns=["date", signal_col, "type"])

    tdf = pd.DataFrame(turning)
    tdf["abs_score"] = tdf[signal_col].abs()
    return tdf.sort_values("abs_score", ascending=False).head(top_n).sort_values("date")

# -----------------------------
# LOAD DATA
# -----------------------------
st.title("Bitcoin Astro Indicator")

df = load_data()
opt_df = load_optimization()

price_df = df.dropna(subset=["price"]).copy()
if price_df.empty:
    st.error("ยังไม่มีข้อมูลราคา BTC")
    st.stop()

last_price_date = price_df["date"].max()
latest = df[df["date"] == last_price_date].iloc[-1]
latest_price = price_df.iloc[-1]["price"]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Controls")

quick_range = st.sidebar.selectbox(
    "Historical Price Range",
    ["4Y", "3Y", "2Y", "1Y", "6M", "3M", "All"],
    index=0
)

indicator_options = {
    "Astro Momentum (Raw)": "astro_momentum",
    "Astro Momentum (Smooth)": "astro_momentum_smooth",
    "Expansion Score": "expansion_score",
    "Contraction Score": "contraction_score",
    "Narrative Score": "narrative_score",
    "Trigger Score": "trigger_score",
}
indicator_label = st.sidebar.selectbox("Astro Indicator", list(indicator_options.keys()), index=1)
indicator_col = indicator_options[indicator_label]

forecast_options = {
    "30 days": 30,
    "90 days": 90,
    "180 days": 180,
    "365 days": 365,
    "730 days": 730,
    "Max available": None,
}
forecast_label = st.sidebar.selectbox("Forecast Horizon", list(forecast_options.keys()), index=3)
forecast_days = forecast_options[forecast_label]

show_price_panel = st.sidebar.checkbox("Show price panel", True)
show_astro_panel = st.sidebar.checkbox("Show astro panel", True)
show_backtest_panel = st.sidebar.checkbox("Show backtest panel", True)
show_signal_markers = st.sidebar.checkbox("Show buy/sell markers", True)

st.sidebar.markdown("---")
st.sidebar.header("Natal / Aspect Viewer")
aspect_date = st.sidebar.date_input("Aspect breakdown date", value=last_price_date.date())

# -----------------------------
# RANGE
# -----------------------------
min_price_date = price_df["date"].min().date()
max_price_date = last_price_date.date()

if quick_range == "All":
    hist_start = min_price_date
elif quick_range == "4Y":
    hist_start = max(min_price_date, (last_price_date - pd.Timedelta(days=365 * 4)).date())
elif quick_range == "3Y":
    hist_start = max(min_price_date, (last_price_date - pd.Timedelta(days=365 * 3)).date())
elif quick_range == "2Y":
    hist_start = max(min_price_date, (last_price_date - pd.Timedelta(days=365 * 2)).date())
elif quick_range == "1Y":
    hist_start = max(min_price_date, (last_price_date - pd.Timedelta(days=365)).date())
elif quick_range == "6M":
    hist_start = max(min_price_date, (last_price_date - pd.Timedelta(days=183)).date())
else:
    hist_start = max(min_price_date, (last_price_date - pd.Timedelta(days=92)).date())

hist_view = df[(df["date"].dt.date >= hist_start) & (df["date"].dt.date <= max_price_date)].copy()
hist_price_view = hist_view.dropna(subset=["price"]).copy()

future_view = df[df["date"] > last_price_date].copy()
if forecast_days is not None:
    future_view = future_view[future_view["date"] <= last_price_date + pd.Timedelta(days=forecast_days)]

# -----------------------------
# OPTIMIZATION SUMMARY
# -----------------------------
best_model = None
if not opt_df.empty:
    best_model = opt_df.iloc[0]

# -----------------------------
# TOP CARDS
# -----------------------------
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest BTC Price</div>
        <div class="metric-value">{fmt_money(latest_price)}</div>
        <div class="metric-sub">Yahoo Finance BTC-USD</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Selected Astro</div>
        <div class="metric-value">{fmt_num(latest[indicator_col])}</div>
        <div class="metric-sub">{indicator_label}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    regime = classify_regime(latest["astro_momentum"])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Regime</div>
        <div class="metric-value" style="color:{regime_color(regime)};">{regime}</div>
        <div class="metric-sub">Raw astro regime</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    signal = latest["signal"] if "signal" in latest.index else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Signal</div>
        <div class="metric-value" style="color:{signal_color(signal)};">{signal}</div>
        <div class="metric-sub">Default signal</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    if best_model is not None:
        best_txt = f"{best_model['indicator']} / span {int(best_model['span'])}"
    else:
        best_txt = "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best Optimized Model</div>
        <div class="metric-value" style="font-size:1rem;">{best_txt}</div>
        <div class="metric-sub">Ranked by balanced score</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# OPTIMIZATION PANEL
# -----------------------------
if not opt_df.empty:
    with st.expander("Optimization Results", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Indicator", str(best_model["indicator"]))
        c2.metric("Best Span", int(best_model["span"]))
        c3.metric("Long / Short Threshold", f"{best_model['long_threshold']} / {best_model['short_threshold']}")
        c4.metric("Best Balanced Score", fmt_num(best_model["balanced_score"]))

        show_cols = [
            "indicator", "span", "long_threshold", "short_threshold",
            "total_return", "max_drawdown", "sharpe_like",
            "return_drawdown_ratio", "number_of_trades", "balanced_score"
        ]
        show_cols = [c for c in show_cols if c in opt_df.columns]
        st.dataframe(opt_df[show_cols].head(20), use_container_width=True)

# -----------------------------
# MAIN CHART
# -----------------------------
rows = sum([show_price_panel, show_astro_panel, show_backtest_panel])
if rows == 0:
    st.warning("เลือกอย่างน้อย 1 panel")
    st.stop()

titles = []
heights = []
if show_price_panel:
    titles.append("BTC Price")
    heights.append(0.45)
if show_astro_panel:
    titles.append(f"{indicator_label} — Historical + Future")
    heights.append(0.30)
if show_backtest_panel:
    titles.append("Strategy vs Buy & Hold")
    heights.append(0.25)

fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[h / sum(heights) for h in heights],
    subplot_titles=titles
)

row = 1

# Signal markers
buy_markers = pd.DataFrame()
sell_markers = pd.DataFrame()
if "signal" in hist_price_view.columns:
    sm = hist_price_view[["date", "price", "signal"]].copy()
    sm["prev_signal"] = sm["signal"].shift(1)
    sm = sm[sm["signal"] != sm["prev_signal"]]
    buy_markers = sm[sm["signal"].isin(["buy", "strong_buy"])]
    sell_markers = sm[sm["signal"].isin(["sell", "strong_sell"])]

if show_price_panel:
    fig.add_trace(go.Scatter(
        x=hist_price_view["date"], y=hist_price_view["price"],
        mode="lines", name="BTC Price",
        line=dict(color="#3b82f6", width=2)
    ), row=row, col=1)

    if show_signal_markers and not buy_markers.empty:
        fig.add_trace(go.Scatter(
            x=buy_markers["date"], y=buy_markers["price"],
            mode="markers", name="Buy Signal",
            marker=dict(color="#22c55e", size=9, symbol="triangle-up")
        ), row=row, col=1)

    if show_signal_markers and not sell_markers.empty:
        fig.add_trace(go.Scatter(
            x=sell_markers["date"], y=sell_markers["price"],
            mode="markers", name="Sell Signal",
            marker=dict(color="#ef4444", size=9, symbol="triangle-down")
        ), row=row, col=1)

    fig.add_vline(x=last_price_date, line_dash="dot", line_color="#e5e7eb", row=row, col=1)
    fig.update_yaxes(title_text="BTC Price", row=row, col=1)
    row += 1

if show_astro_panel:
    fig.add_trace(go.Scatter(
        x=hist_view["date"], y=hist_view[indicator_col],
        mode="lines", name=f"{indicator_label} history",
        line=dict(color="#93c5fd", width=2)
    ), row=row, col=1)

    if not future_view.empty:
        fig.add_trace(go.Scatter(
            x=future_view["date"], y=future_view[indicator_col],
            mode="lines", name=f"{indicator_label} future",
            line=dict(color="#f59e0b", width=2, dash="dash")
        ), row=row, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=row, col=1)
    fig.add_vline(x=last_price_date, line_dash="dot", line_color="#e5e7eb", row=row, col=1)
    fig.update_yaxes(title_text=indicator_label, row=row, col=1)
    row += 1

if show_backtest_panel and "strategy_equity" in hist_view.columns:
    bt = hist_view.dropna(subset=["strategy_equity", "buy_hold_equity"])
    fig.add_trace(go.Scatter(
        x=bt["date"], y=bt["strategy_equity"],
        mode="lines", name="Strategy Equity",
        line=dict(color="#22c55e", width=2)
    ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=bt["date"], y=bt["buy_hold_equity"],
        mode="lines", name="Buy & Hold Equity",
        line=dict(color="#e5e7eb", width=2, dash="dash")
    ), row=row, col=1)

    fig.update_yaxes(title_text="Equity", row=row, col=1)

fig.update_layout(
    height=1050,
    template="plotly_dark",
    legend=dict(orientation="h", y=1.02, x=0),
    margin=dict(l=40, r=40, t=60, b=40)
)
fig.update_xaxes(title_text="Date", row=rows, col=1)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# NATAL CHART + RAW ASPECTS
# -----------------------------
st.header("Bitcoin Natal Chart & Raw Astro Score Breakdown")

natal, natal_cusps, natal_ascmc = get_natal_chart()
left, right = st.columns([1.0, 1.2])

with left:
    st.subheader("Natal Chart Wheel")
    st.plotly_chart(make_natal_wheel(natal), use_container_width=True)

with right:
    st.subheader("Natal Positions")
    natal_rows = []
    for name, lon in natal.items():
        sign, sym, deg = degree_to_sign(lon)
        natal_rows.append({
            "body": name,
            "symbol": PLANET_SYMBOLS.get(name, ""),
            "longitude": round(lon, 4),
            "position": fmt_deg(lon),
        })
    st.dataframe(pd.DataFrame(natal_rows), use_container_width=True)

aspect_df = compute_daily_aspect_breakdown(pd.to_datetime(aspect_date))
st.subheader(f"Raw Daily Aspect Breakdown — {aspect_date}")

if aspect_df.empty:
    st.info("วันนี้ไม่มี aspect ที่เข้า orb ตามเงื่อนไข")
else:
    score_by_bucket = aspect_df.groupby("bucket")["score"].sum().reset_index()
    bar = go.Figure()
    bar.add_trace(go.Bar(
        x=score_by_bucket["bucket"],
        y=score_by_bucket["score"],
        marker_color="#60a5fa"
    ))
    bar.update_layout(
        template="plotly_dark",
        height=360,
        title="Raw Astro Score Contribution by Bucket",
        yaxis_title="Score Contribution",
        xaxis_title="Bucket"
    )
    st.plotly_chart(bar, use_container_width=True)

    show = aspect_df.copy()
    show["orb"] = show["orb"].round(3)
    show["score"] = show["score"].round(4)
    st.dataframe(show, use_container_width=True)

# -----------------------------
# TABLES
# -----------------------------
st.header("Tables")

t1, t2 = st.columns([1, 1])

with t1:
    st.subheader("Latest 30 Rows")
    cols = [
        "date", "price", "astro_momentum", "astro_momentum_smooth",
        "signal", "position", "strategy_equity", "buy_hold_equity"
    ]
    cols = [c for c in cols if c in df.columns]
    latest_table = df[cols].dropna(subset=["price"]).tail(30).copy()
    latest_table["date"] = latest_table["date"].dt.date
    st.dataframe(latest_table, use_container_width=True)

with t2:
    st.subheader("Regime Guide")
    guide = pd.DataFrame([
        {"score_range": ">= 3.0", "regime": "strong_bull", "meaning": "แรงขยายตัวสูงมาก"},
        {"score_range": "1.5 to 2.99", "regime": "bull", "meaning": "เอียงขึ้น"},
        {"score_range": "-1.49 to 1.49", "regime": "neutral", "meaning": "แกว่ง / ไม่มี edge ชัด"},
        {"score_range": "-3.0 to -1.5", "regime": "bear", "meaning": "อ่อนแรง"},
        {"score_range": "< -3.0", "regime": "crash_risk", "meaning": "เสี่ยงแรง"},
    ])
    st.dataframe(guide, use_container_width=True)

# =========================================================
# RAW ASTRO ASPECT EXPLORER
# =========================================================

st.header("Raw Astro Aspect Explorer")

@st.cache_data(ttl=3600)
def load_raw_aspects():
    raw = pd.read_csv("data/astro_aspects_raw.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    return raw

try:
    raw_aspects = load_raw_aspects()
except Exception as e:
    st.error("โหลดไฟล์ data/astro_aspects_raw.csv ไม่สำเร็จ")
    st.code(str(e))
    st.stop()

st.markdown("""
<div class="explain-box">
    <div class="explain-title">What this panel shows</div>
    <div class="explain-text">
    ตารางนี้แสดงว่าในแต่ละวัน ดาวจรดวงไหนไปกระทบดวงกำเนิด Bitcoin อย่างไร เช่น
    Jupiter trine Natal Sun, Saturn square Asc, Uranus opposition MC เป็นต้น<br><br>
    นี่คือชั้น “Raw Intelligence” ของระบบ ใช้ตรวจสอบว่า Astro Score มาจากดาวอะไร มุมอะไร และแรงแค่ไหน
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# FILTERS
# -----------------------------
f1, f2, f3, f4 = st.columns(4)

with f1:
    raw_start = st.date_input(
        "Raw aspect start date",
        value=last_price_date.date() - pd.Timedelta(days=30),
        key="raw_start"
    )

with f2:
    raw_end = st.date_input(
        "Raw aspect end date",
        value=last_price_date.date(),
        key="raw_end"
    )

with f3:
    planet_filter = st.multiselect(
        "Transit Planet",
        sorted(raw_aspects["transit_planet"].dropna().unique().tolist()),
        default=[]
    )

with f4:
    source_filter = st.multiselect(
        "Source",
        sorted(raw_aspects["source"].dropna().unique().tolist()),
        default=[]
    )

f5, f6, f7, f8 = st.columns(4)

with f5:
    aspect_filter = st.multiselect(
        "Aspect",
        sorted(raw_aspects["aspect"].dropna().unique().tolist()),
        default=[]
    )

with f6:
    target_filter = st.multiselect(
        "Target",
        sorted(raw_aspects["target"].dropna().unique().tolist()),
        default=[]
    )

with f7:
    score_type = st.selectbox(
        "Score Type",
        [
            "bullish",
            "bearish",
            "reversal",
            "volatility",
            "compression",
            "trend_start",
            "trend_end",
        ],
        index=0
    )

with f8:
    min_abs_score = st.number_input(
        "Min absolute score",
        min_value=0.0,
        max_value=10.0,
        value=0.1,
        step=0.1
    )

raw_view = raw_aspects[
    (raw_aspects["date"].dt.date >= raw_start) &
    (raw_aspects["date"].dt.date <= raw_end)
].copy()

if planet_filter:
    raw_view = raw_view[raw_view["transit_planet"].isin(planet_filter)]

if source_filter:
    raw_view = raw_view[raw_view["source"].isin(source_filter)]

if aspect_filter:
    raw_view = raw_view[raw_view["aspect"].isin(aspect_filter)]

if target_filter:
    raw_view = raw_view[raw_view["target"].isin(target_filter)]

raw_view["selected_score"] = raw_view[score_type].fillna(0)

raw_view = raw_view[raw_view["selected_score"].abs() >= min_abs_score].copy()

# -----------------------------
# SUMMARY CARDS
# -----------------------------
if raw_view.empty:
    st.info("ไม่พบ raw aspects ตาม filter ที่เลือก")
else:
    total_score = raw_view["selected_score"].sum()
    max_score = raw_view["selected_score"].max()
    aspect_count = len(raw_view)
    top_planet = (
        raw_view.groupby("transit_planet")["selected_score"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Aspect Count", f"{aspect_count:,}")
    c2.metric(f"Total {score_type} Score", f"{total_score:.2f}")
    c3.metric("Max Single Score", f"{max_score:.2f}")
    c4.metric("Top Contributing Planet", top_planet)

    # -----------------------------
    # PLANET IMPACT BAR CHART
    # -----------------------------
    st.subheader("Planet Impact Ranking")

    planet_impact = (
        raw_view.groupby("transit_planet")[score_type]
        .sum()
        .reset_index()
        .sort_values(score_type, ascending=False)
    )

    fig_planet = go.Figure()
    fig_planet.add_trace(
        go.Bar(
            x=planet_impact["transit_planet"],
            y=planet_impact[score_type],
            marker_color="#60a5fa",
            name=f"{score_type} score"
        )
    )

    fig_planet.update_layout(
        template="plotly_dark",
        height=420,
        title=f"Planet Contribution — {score_type}",
        xaxis_title="Transit Planet",
        yaxis_title=f"{score_type} score",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig_planet, use_container_width=True)

    # -----------------------------
    # DAILY HEATMAP STYLE BAR
    # -----------------------------
    st.subheader("Daily Raw Score Contribution")

    daily_impact = (
        raw_view.groupby("date")[score_type]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Bar(
            x=daily_impact["date"],
            y=daily_impact[score_type],
            marker_color="#f59e0b",
            name=f"Daily {score_type}"
        )
    )

    fig_daily.update_layout(
        template="plotly_dark",
        height=420,
        title=f"Daily {score_type} Contribution",
        xaxis_title="Date",
        yaxis_title=f"{score_type} score",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig_daily, use_container_width=True)

    # -----------------------------
    # ASPECT TABLE
    # -----------------------------
    st.subheader("Raw Aspect Table")

    show_cols = [
        "date",
        "source",
        "rule_name",
        "transit_planet",
        "target",
        "aspect",
        "orb",
        "orb_factor",
        "aspect_weight",
        "target_weight",
        "multiplier",
        "bullish",
        "bearish",
        "reversal",
        "volatility",
        "compression",
        "trend_start",
        "trend_end",
    ]

    show_cols = [c for c in show_cols if c in raw_view.columns]

    table = raw_view[show_cols].copy()
    table["date"] = table["date"].dt.date

    score_cols = [
        "orb",
        "orb_factor",
        "aspect_weight",
        "target_weight",
        "multiplier",
        "bullish",
        "bearish",
        "reversal",
        "volatility",
        "compression",
        "trend_start",
        "trend_end",
    ]

    for c in score_cols:
        if c in table.columns:
            table[c] = pd.to_numeric(table[c], errors="coerce").round(4)

    table = table.sort_values(
        by=[score_type],
        ascending=False
    ) if score_type in table.columns else table

    st.dataframe(table, use_container_width=True, height=520)
