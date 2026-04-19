import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Bitcoin Astro Indicator",
    layout="wide",
)

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
.metric-card {
    background-color: #111827;
    padding: 16px 18px;
    border-radius: 14px;
    border: 1px solid #1f2937;
}
.metric-label {
    color: #9ca3af;
    font-size: 0.9rem;
    margin-bottom: 6px;
}
.metric-value {
    color: white;
    font-size: 1.7rem;
    font-weight: 700;
}
.metric-sub {
    color: #60a5fa;
    font-size: 0.9rem;
    margin-top: 4px;
}
.small-note {
    color: #9ca3af;
    font-size: 0.85rem;
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
    mapping = {
        "strong_bull": "#22c55e",
        "bull": "#84cc16",
        "neutral": "#94a3b8",
        "bear": "#f59e0b",
        "crash_risk": "#ef4444",
    }
    return mapping.get(regime, "#94a3b8")

def next_turning_points(df_future, top_n=8):
    x = df_future[["date", "astro_momentum"]].copy().reset_index(drop=True)
    turning = []

    for i in range(1, len(x) - 1):
        prev_val = x.loc[i - 1, "astro_momentum"]
        curr_val = x.loc[i, "astro_momentum"]
        next_val = x.loc[i + 1, "astro_momentum"]

        # local top
        if curr_val > prev_val and curr_val > next_val:
            turning.append({
                "date": x.loc[i, "date"],
                "astro_momentum": curr_val,
                "type": "local_top"
            })

        # local bottom
        if curr_val < prev_val and curr_val < next_val:
            turning.append({
                "date": x.loc[i, "date"],
                "astro_momentum": curr_val,
                "type": "local_bottom"
            })

    if not turning:
        return pd.DataFrame(columns=["date", "astro_momentum", "type"])

    tdf = pd.DataFrame(turning)
    tdf["abs_score"] = tdf["astro_momentum"].abs()
    tdf = tdf.sort_values(["date"]).copy()

    # keep strongest points among upcoming windows
    tdf = tdf.sort_values("abs_score", ascending=False).head(top_n)
    tdf = tdf.sort_values("date").reset_index(drop=True)
    return tdf[["date", "astro_momentum", "type"]]

def add_regime_backgrounds(fig, dates, scores, row, col):
    regimes = [classify_regime(v) for v in scores]
    if len(dates) == 0:
        return fig

    start_idx = 0
    current_regime = regimes[0]

    for i in range(1, len(regimes)):
        if regimes[i] != current_regime:
            fig.add_vrect(
                x0=dates.iloc[start_idx],
                x1=dates.iloc[i - 1],
                fillcolor=regime_color(current_regime),
                opacity=0.08,
                line_width=0,
                row=row,
                col=col,
            )
            start_idx = i
            current_regime = regimes[i]

    fig.add_vrect(
        x0=dates.iloc[start_idx],
        x1=dates.iloc[len(dates) - 1],
        fillcolor=regime_color(current_regime),
        opacity=0.08,
        line_width=0,
        row=row,
        col=col,
    )
    return fig

# -----------------------------
# LOAD
# -----------------------------
st.title("Bitcoin Astro Indicator")

try:
    df = load_data()
except Exception as e:
    st.error("โหลดข้อมูล indicator ไม่สำเร็จ")
    st.code(str(e))
    st.stop()

required_cols = [
    "date",
    "price",
    "astro_momentum",
    "expansion_score",
    "contraction_score",
    "narrative_score",
    "trigger_score",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"ไฟล์ข้อมูลขาดคอลัมน์: {missing}")
    st.stop()

price_df = df.dropna(subset=["price"]).copy()
if price_df.empty:
    st.error("ยังไม่มีข้อมูลราคา BTC ที่ใช้งานได้")
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Controls")

min_date = price_df["date"].min().date()
max_date = price_df["date"].max().date()

default_start = max(min_date, (price_df["date"].max() - pd.Timedelta(days=365 * 4)).date())
default_end = max_date

quick_range = st.sidebar.selectbox(
    "Quick Range",
    ["4Y", "3Y", "2Y", "1Y", "6M", "3M", "All"],
    index=0
)

if quick_range == "All":
    start_date = min_date
elif quick_range == "4Y":
    start_date = max(min_date, (price_df["date"].max() - pd.Timedelta(days=365 * 4)).date())
elif quick_range == "3Y":
    start_date = max(min_date, (price_df["date"].max() - pd.Timedelta(days=365 * 3)).date())
elif quick_range == "2Y":
    start_date = max(min_date, (price_df["date"].max() - pd.Timedelta(days=365 * 2)).date())
elif quick_range == "1Y":
    start_date = max(min_date, (price_df["date"].max() - pd.Timedelta(days=365)).date())
elif quick_range == "6M":
    start_date = max(min_date, (price_df["date"].max() - pd.Timedelta(days=183)).date())
else:
    start_date = max(min_date, (price_df["date"].max() - pd.Timedelta(days=92)).date())

end_date = default_end

custom_dates = st.sidebar.checkbox("Use custom dates", value=False)

if custom_dates:
    start_date = st.sidebar.date_input("Start date", value=default_start, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", value=default_end, min_value=min_date, max_value=max_date)

show_turning_markers = st.sidebar.checkbox("Show turning markers", value=True)
show_regime_background = st.sidebar.checkbox("Show regime background", value=True)

view = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

chart_df = view.dropna(subset=["price"]).copy()

if chart_df.empty:
    st.warning("ช่วงวันที่ที่เลือกยังไม่มีข้อมูลราคา")
    st.stop()

latest = chart_df.iloc[-1]
latest_score = float(latest["astro_momentum"])
latest_regime = classify_regime(latest_score)

future_df = df[df["date"] > chart_df["date"].max()].copy()
turning_df = next_turning_points(future_df, top_n=8)

next_turning_text = "N/A"
if not turning_df.empty:
    next_turn = turning_df.iloc[0]
    next_turn_text = f"{next_turn['date'].date()} ({next_turn['type']})"

# -----------------------------
# TOP METRICS
# -----------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest BTC Price</div>
        <div class="metric-value">${latest["price"]:,.0f}</div>
        <div class="metric-sub">Yahoo Finance (BTC-USD)</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest Astro Momentum</div>
        <div class="metric-value">{latest_score:.2f}</div>
        <div class="metric-sub">Expansion - Contraction + Narrative/Trigger</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Regime</div>
        <div class="metric-value" style="color:{regime_color(latest_regime)};">{latest_regime}</div>
        <div class="metric-sub">Based on latest astro score</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Next Turning Window</div>
        <div class="metric-value" style="font-size:1.1rem;">{next_turn_text}</div>
        <div class="metric-sub">Future local top / bottom from astro series</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='small-note'>Price source: Yahoo Finance (BTC-USD) | Astro source: First Transaction natal chart + daily transit scoring</div>", unsafe_allow_html=True)

# -----------------------------
# MAIN CHART
# -----------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.65, 0.35],
    subplot_titles=("BTC Price", "Astro Momentum")
)

if show_regime_background:
    fig = add_regime_backgrounds(
        fig,
        chart_df["date"].reset_index(drop=True),
        chart_df["astro_momentum"].reset_index(drop=True),
        row=2,
        col=1
    )

fig.add_trace(
    go.Scatter(
        x=chart_df["date"],
        y=chart_df["price"],
        mode="lines",
        name="BTC Price",
        line=dict(color="#3b82f6", width=2),
        hovertemplate="Date=%{x}<br>Price=$%{y:,.0f}<extra></extra>",
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=chart_df["date"],
        y=chart_df["astro_momentum"],
        mode="lines",
        name="Astro Momentum",
        line=dict(color="#93c5fd", width=2),
        hovertemplate="Date=%{x}<br>Astro=%{y:.2f}<extra></extra>",
    ),
    row=2,
    col=1
)

# Baseline for momentum
fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="#94a3b8",
    line_width=1,
    row=2,
    col=1
)

# Turning markers
if show_turning_markers and not turning_df.empty:
    future_turning_in_range = turning_df[
        (turning_df["date"] >= chart_df["date"].min()) &
        (turning_df["date"] <= chart_df["date"].max())
    ].copy()

    if not future_turning_in_range.empty:
        tops = future_turning_in_range[future_turning_in_range["type"] == "local_top"]
        bottoms = future_turning_in_range[future_turning_in_range["type"] == "local_bottom"]

        if not tops.empty:
            fig.add_trace(
                go.Scatter(
                    x=tops["date"],
                    y=tops["astro_momentum"],
                    mode="markers",
                    name="Turning Top",
                    marker=dict(color="#ef4444", size=9, symbol="triangle-up"),
                    hovertemplate="Date=%{x}<br>Top=%{y:.2f}<extra></extra>",
                ),
                row=2,
                col=1
            )

        if not bottoms.empty:
            fig.add_trace(
                go.Scatter(
                    x=bottoms["date"],
                    y=bottoms["astro_momentum"],
                    mode="markers",
                    name="Turning Bottom",
                    marker=dict(color="#22c55e", size=9, symbol="triangle-down"),
                    hovertemplate="Date=%{x}<br>Bottom=%{y:.2f}<extra></extra>",
                ),
                row=2,
                col=1
            )

fig.update_layout(
    height=850,
    template="plotly_dark",
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis2=dict(title="Date"),
    yaxis=dict(title="BTC Price (USD)"),
    yaxis2=dict(title="Astro Momentum"),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# DETAIL PANELS
# -----------------------------
left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Latest 30 Days")
    latest_table = chart_df[[
        "date",
        "price",
        "astro_momentum",
        "expansion_score",
        "contraction_score",
        "narrative_score",
        "trigger_score",
    ]].tail(30).copy()

    latest_table["regime"] = latest_table["astro_momentum"].apply(classify_regime)
    latest_table["date"] = latest_table["date"].dt.date

    st.dataframe(latest_table, use_container_width=True)

with right:
    st.subheader("Upcoming Turning Dates")
    if turning_df.empty:
        st.info("ยังไม่พบ future turning points")
    else:
        show_turning = turning_df.copy()
        show_turning["date"] = show_turning["date"].dt.date
        show_turning["regime_hint"] = show_turning["astro_momentum"].apply(classify_regime)
        st.dataframe(show_turning, use_container_width=True)

# -----------------------------
# REGIME GUIDE
# -----------------------------
st.subheader("Regime Guide")
guide = pd.DataFrame([
    {"score_range": ">= 3.0", "regime": "strong_bull", "meaning": "แรงขยายตัวสูงมาก"},
    {"score_range": "1.5 to 2.99", "regime": "bull", "meaning": "เอียงขึ้น / buy-the-dip bias"},
    {"score_range": "-1.49 to 1.49", "regime": "neutral", "meaning": "แกว่ง / ไม่มี edge ชัด"},
    {"score_range": "-3.0 to -1.5", "regime": "bear", "meaning": "อ่อนแรง / defensive"},
    {"score_range": "< -3.0", "regime": "crash_risk", "meaning": "เสี่ยงแรง / regime ลบมาก"},
])
st.dataframe(guide, use_container_width=True)
