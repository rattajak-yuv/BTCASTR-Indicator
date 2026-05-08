import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import swisseph as swe
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
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOADERS
# =========================================================
@st.cache_data(ttl=3600)
def load_daily():
    df = pd.read_csv("data/bitcoin_astro_daily_score.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600)
def load_raw_aspects():
    try:
        df = pd.read_csv("data/astro_aspects_raw.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_ml_predictions():
    try:
        df = pd.read_csv("data/ml_predictions.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_ml_summary():
    try:
        return pd.read_csv("data/ml_model_summary.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_ml_importance():
    try:
        return pd.read_csv("data/ml_feature_importance.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_optimization_v2():
    try:
        return pd.read_csv("data/model_optimization_v2_results.csv")
    except Exception:
        return pd.DataFrame()

# =========================================================
# HELPERS
# =========================================================
def fmt_money(x):
    return "N/A" if pd.isna(x) else f"${x:,.0f}"

def fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}"

def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2%}"

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

# =========================================================
# DATA
# =========================================================
df = load_daily()
raw_aspects = load_raw_aspects()
ml_pred = load_ml_predictions()
ml_summary = load_ml_summary()
ml_importance = load_ml_importance()
opt_v2 = load_optimization_v2()

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
st.sidebar.header("Global Controls")

range_option = st.sidebar.selectbox(
    "Historical Range",
    ["4Y", "3Y", "2Y", "1Y", "6M", "3M", "All"],
    index=0
)

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
indicator_label = st.sidebar.selectbox("Astro Indicator", list(indicator_options.keys()), index=1)
indicator_col = indicator_options[indicator_label]

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon",
    [30, 90, 180, 365, 730],
    index=3
)

show_signals = st.sidebar.checkbox("Show Rule-Based Buy/Sell Markers", value=True)

min_date = price_df["date"].min()
if range_option == "All":
    start_date = min_date
elif range_option == "4Y":
    start_date = last_price_date - pd.Timedelta(days=365 * 4)
elif range_option == "3Y":
    start_date = last_price_date - pd.Timedelta(days=365 * 3)
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
    fig.add_vline(x=last_price_date, line_dash="dot", line_color="#e5e7eb")
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
