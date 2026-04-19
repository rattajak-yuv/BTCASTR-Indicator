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
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
.metric-card {
    background-color: #111827;
    padding: 16px 18px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    min-height: 145px;
}
.metric-label {
    color: #9ca3af;
    font-size: 0.88rem;
    margin-bottom: 6px;
}
.metric-value {
    color: white;
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.2;
    word-break: break-word;
}
.metric-sub {
    color: #60a5fa;
    font-size: 0.88rem;
    margin-top: 4px;
}
.small-note {
    color: #9ca3af;
    font-size: 0.84rem;
}
.explain-box {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 10px;
    margin-bottom: 10px;
}
.explain-title {
    color: white;
    font-weight: 700;
    margin-bottom: 8px;
}
.explain-text {
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.5;
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

def signal_color(signal):
    mapping = {
        "strong_buy": "#22c55e",
        "buy": "#84cc16",
        "neutral": "#94a3af",
        "sell": "#f59e0b",
        "strong_sell": "#ef4444",
    }
    return mapping.get(signal, "#94a3af")

def add_regime_backgrounds(fig, dates, scores, row, col):
    if len(dates) == 0:
        return fig

    regimes = [classify_regime(v) for v in scores]
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
            turning.append({
                "date": x.loc[i, "date"],
                signal_col: curr_val,
                "type": "local_top"
            })

        if curr_val < prev_val and curr_val < next_val and abs(curr_val) >= threshold:
            turning.append({
                "date": x.loc[i, "date"],
                signal_col: curr_val,
                "type": "local_bottom"
            })

    if not turning:
        return pd.DataFrame(columns=["date", signal_col, "type"])

    tdf = pd.DataFrame(turning)
    tdf["abs_score"] = tdf[signal_col].abs()
    tdf = tdf.sort_values("abs_score", ascending=False).head(top_n)
    tdf = tdf.sort_values("date").reset_index(drop=True)
    return tdf[["date", signal_col, "type"]]

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
    "astro_momentum_smooth",
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

last_price_date = price_df["date"].max()
latest_price_row = price_df.iloc[-1]

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
indicator_label = st.sidebar.selectbox(
    "Astro Indicator",
    list(indicator_options.keys()),
    index=1
)
indicator_col = indicator_options[indicator_label]

forecast_options = {
    "30 days (near)": 30,
    "90 days (near-medium)": 90,
    "180 days (medium)": 180,
    "365 days (long)": 365,
    "730 days (very long)": 730,
    "Max available": None,
}
forecast_label = st.sidebar.selectbox(
    "Forecast Horizon",
    list(forecast_options.keys()),
    index=3
)
forecast_days = forecast_options[forecast_label]

custom_dates = st.sidebar.checkbox("Use custom dates", value=False)
show_turning_markers = st.sidebar.checkbox("Show turning markers", value=True)
show_regime_background = st.sidebar.checkbox("Show regime background", value=True)
show_signal_markers = st.sidebar.checkbox("Show buy/sell markers", value=True)

st.sidebar.markdown("---")
show_price_panel = st.sidebar.checkbox("Show price panel", value=True)
show_astro_panel = st.sidebar.checkbox("Show astro panel", value=True)
show_backtest_panel = st.sidebar.checkbox("Show backtest panel", value=True)

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

hist_end = max_price_date

if custom_dates:
    hist_start = st.sidebar.date_input("Start date", value=hist_start, min_value=min_price_date, max_value=max_price_date)
    hist_end = st.sidebar.date_input("End date", value=hist_end, min_value=min_price_date, max_value=max_price_date)

hist_view = df[
    (df["date"].dt.date >= hist_start) &
    (df["date"].dt.date <= hist_end)
].copy()

hist_price_view = hist_view.dropna(subset=["price"]).copy()
if hist_price_view.empty:
    st.warning("ช่วงวันที่ที่เลือกยังไม่มีข้อมูลราคา")
    st.stop()

future_view = df[df["date"] > last_price_date].copy()
if forecast_days is not None:
    future_cutoff = last_price_date + pd.Timedelta(days=forecast_days)
    future_view = future_view[future_view["date"] <= future_cutoff].copy()

astro_hist = hist_view.copy()
astro_future = future_view.copy()
astro_combined = pd.concat([astro_hist, astro_future], ignore_index=True)

latest = df[df["date"] == last_price_date].iloc[-1]
latest_score = float(latest["astro_momentum"])
latest_regime = classify_regime(latest_score)
current_signal = latest["signal"] if "signal" in latest.index else "N/A"

turning_df = next_turning_points(future_view, signal_col=indicator_col, top_n=10, threshold=1.5)
next_turning_text = "N/A"
if not turning_df.empty:
    next_turn = turning_df.iloc[0]
    next_turning_text = f"{next_turn['date'].date()} ({next_turn['type']})"

strategy_total_return = latest["strategy_total_return"] if "strategy_total_return" in df.columns else None
strategy_max_dd = latest["strategy_max_drawdown"] if "strategy_max_drawdown" in df.columns else None
buy_hold_total_return = latest["buy_hold_total_return"] if "buy_hold_total_return" in df.columns else None
buy_hold_max_dd = latest["buy_hold_max_drawdown"] if "buy_hold_max_drawdown" in df.columns else None

strategy_beats = False
if pd.notna(strategy_total_return) and pd.notna(buy_hold_total_return):
    strategy_beats = strategy_total_return > buy_hold_total_return

# -----------------------------
# TOP CARDS
# -----------------------------
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest BTC Price</div>
        <div class="metric-value">{fmt_money(latest_price_row["price"])}</div>
        <div class="metric-sub">Yahoo Finance (BTC-USD)</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    current_indicator_value = latest[indicator_col] if indicator_col in latest.index else None
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Selected Astro Indicator</div>
        <div class="metric-value">{fmt_num(current_indicator_value)}</div>
        <div class="metric-sub">{indicator_label}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Regime</div>
        <div class="metric-value" style="color:{regime_color(latest_regime)};">{latest_regime}</div>
        <div class="metric-sub">Based on latest astro momentum</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Signal</div>
        <div class="metric-value" style="color:{signal_color(current_signal)};">{current_signal}</div>
        <div class="metric-sub">From smoothed astro momentum</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Next Turning Window</div>
        <div class="metric-value" style="font-size:1.05rem;">{next_turning_text}</div>
        <div class="metric-sub">Using {indicator_label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    "<div class='small-note'>Price source: Yahoo Finance (BTC-USD) | "
    "Astro source: First Transaction natal chart + daily transit scoring | "
    "Future chart uses precomputed astro data already stored in your CSV</div>",
    unsafe_allow_html=True
)

# -----------------------------
# BACKTEST SUMMARY
# -----------------------------
if "strategy_total_return" in df.columns:
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Strategy Total Return", fmt_pct(strategy_total_return))
    with b2:
        st.metric("Strategy Max Drawdown", fmt_pct(strategy_max_dd))
    with b3:
        st.metric("Buy & Hold Return", fmt_pct(buy_hold_total_return))
    with b4:
        st.metric("Buy & Hold Max Drawdown", fmt_pct(buy_hold_max_dd))

# explanation box
if "strategy_total_return" in df.columns:
    compare_text = "Strategy is currently outperforming Buy & Hold." if strategy_beats else "Strategy is currently underperforming Buy & Hold."
    st.markdown(f"""
    <div class="explain-box">
        <div class="explain-title">How to read the Backtest panel</div>
        <div class="explain-text">
        Green line = Strategy Equity (if you follow the astro signal).<br>
        White dashed line = Buy & Hold Equity (if you simply hold BTC).<br>
        If the green line stays above the white dashed line, the strategy is beating Buy & Hold.<br>
        If the green line stays below it, the strategy is lagging Buy & Hold.<br><br>
        <b>Current interpretation:</b> {compare_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# SIGNAL MARKERS
# -----------------------------
signal_markers = pd.DataFrame()
buy_markers = pd.DataFrame()
sell_markers = pd.DataFrame()

if "signal" in hist_price_view.columns:
    signal_markers = hist_price_view[["date", "price", "signal"]].copy()
    signal_markers["prev_signal"] = signal_markers["signal"].shift(1)
    signal_markers = signal_markers[signal_markers["signal"] != signal_markers["prev_signal"]].copy()

    buy_markers = signal_markers[signal_markers["signal"].isin(["buy", "strong_buy"])].copy()
    sell_markers = signal_markers[signal_markers["signal"].isin(["sell", "strong_sell"])].copy()

# -----------------------------
# CHART PANEL SELECTION
# -----------------------------
panel_titles = []
row_heights = []
rows = 0

if show_price_panel:
    rows += 1
    panel_titles.append("BTC Price")
    row_heights.append(0.45)

if show_astro_panel:
    rows += 1
    panel_titles.append(f"{indicator_label} (Historical + Future Forecast)")
    row_heights.append(0.30)

if show_backtest_panel:
    rows += 1
    panel_titles.append("Strategy vs Buy & Hold")
    row_heights.append(0.25)

if rows == 0:
    st.warning("กรุณาเลือกอย่างน้อย 1 panel ใน sidebar")
    st.stop()

# normalize row heights
total_h = sum(row_heights)
row_heights = [x / total_h for x in row_heights]

fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=row_heights,
    subplot_titles=tuple(panel_titles)
)

current_row = 1

# -----------------------------
# PRICE PANEL
# -----------------------------
if show_price_panel:
    fig.add_trace(
        go.Scatter(
            x=hist_price_view["date"],
            y=hist_price_view["price"],
            mode="lines",
            name="BTC Price",
            line=dict(color="#3b82f6", width=2),
            hovertemplate="Date=%{x}<br>Price=$%{y:,.0f}<extra></extra>",
        ),
        row=current_row,
        col=1
    )

    if show_signal_markers and not buy_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_markers["date"],
                y=buy_markers["price"],
                mode="markers",
                name="Buy Signal",
                marker=dict(color="#22c55e", size=9, symbol="triangle-up"),
                hovertemplate="Date=%{x}<br>Buy at $%{y:,.0f}<extra></extra>",
            ),
            row=current_row,
            col=1
        )

    if show_signal_markers and not sell_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_markers["date"],
                y=sell_markers["price"],
                mode="markers",
                name="Sell Signal",
                marker=dict(color="#ef4444", size=9, symbol="triangle-down"),
                hovertemplate="Date=%{x}<br>Sell at $%{y:,.0f}<extra></extra>",
            ),
            row=current_row,
            col=1
        )

    fig.add_vline(
        x=last_price_date,
        line_dash="dot",
        line_color="#e5e7eb",
        line_width=1,
        row=current_row,
        col=1
    )

    fig.update_yaxes(title_text="BTC Price (USD)", row=current_row, col=1)
    current_row += 1

# -----------------------------
# ASTRO PANEL
# -----------------------------
if show_astro_panel:
    if show_regime_background and not astro_combined.empty:
        astro_bg_df = astro_combined.dropna(subset=["astro_momentum"]).copy()
        fig = add_regime_backgrounds(
            fig,
            astro_bg_df["date"].reset_index(drop=True),
            astro_bg_df["astro_momentum"].reset_index(drop=True),
            row=current_row,
            col=1
        )

    astro_hist_valid = astro_hist.dropna(subset=[indicator_col]).copy()
    fig.add_trace(
        go.Scatter(
            x=astro_hist_valid["date"],
            y=astro_hist_valid[indicator_col],
            mode="lines",
            name=f"{indicator_label} (history)",
            line=dict(color="#93c5fd", width=2),
            hovertemplate="Date=%{x}<br>Value=%{y:.2f}<extra></extra>",
        ),
        row=current_row,
        col=1
    )

    astro_future_valid = astro_future.dropna(subset=[indicator_col]).copy()
    if not astro_future_valid.empty:
        fig.add_trace(
            go.Scatter(
                x=astro_future_valid["date"],
                y=astro_future_valid[indicator_col],
                mode="lines",
                name=f"{indicator_label} (future)",
                line=dict(color="#f59e0b", width=2, dash="dash"),
                hovertemplate="Date=%{x}<br>Future=%{y:.2f}<extra></extra>",
            ),
            row=current_row,
            col=1
        )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#94a3b8",
        line_width=1,
        row=current_row,
        col=1
    )

    fig.add_vline(
        x=last_price_date,
        line_dash="dot",
        line_color="#e5e7eb",
        line_width=1,
        row=current_row,
        col=1
    )

    if show_turning_markers and not turning_df.empty:
        tops = turning_df[turning_df["type"] == "local_top"]
        bottoms = turning_df[turning_df["type"] == "local_bottom"]

        if not tops.empty:
            fig.add_trace(
                go.Scatter(
                    x=tops["date"],
                    y=tops[indicator_col],
                    mode="markers",
                    name="Future Top",
                    marker=dict(color="#ef4444", size=9, symbol="triangle-up"),
                    hovertemplate="Date=%{x}<br>Top=%{y:.2f}<extra></extra>",
                ),
                row=current_row,
                col=1
            )

        if not bottoms.empty:
            fig.add_trace(
                go.Scatter(
                    x=bottoms["date"],
                    y=bottoms[indicator_col],
                    mode="markers",
                    name="Future Bottom",
                    marker=dict(color="#22c55e", size=9, symbol="triangle-down"),
                    hovertemplate="Date=%{x}<br>Bottom=%{y:.2f}<extra></extra>",
                ),
                row=current_row,
                col=1
            )

    fig.update_yaxes(title_text=indicator_label, row=current_row, col=1)
    current_row += 1

# -----------------------------
# BACKTEST PANEL
# -----------------------------
if show_backtest_panel and "strategy_equity" in hist_view.columns and "buy_hold_equity" in hist_view.columns:
    bt_df = hist_view.dropna(subset=["strategy_equity", "buy_hold_equity"]).copy()

    fig.add_trace(
        go.Scatter(
            x=bt_df["date"],
            y=bt_df["strategy_equity"],
            mode="lines",
            name="Strategy Equity",
            line=dict(color="#22c55e", width=2),
            hovertemplate="Date=%{x}<br>Strategy=%{y:.2f}<extra></extra>",
        ),
        row=current_row,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=bt_df["date"],
            y=bt_df["buy_hold_equity"],
            mode="lines",
            name="Buy & Hold Equity",
            line=dict(color="#e5e7eb", width=2, dash="dash"),
            hovertemplate="Date=%{x}<br>Buy & Hold=%{y:.2f}<extra></extra>",
        ),
        row=current_row,
        col=1
    )

    fig.update_yaxes(title_text="Equity", row=current_row, col=1)
    current_row += 1

fig.update_layout(
    height=1100,
    template="plotly_dark",
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="rgba(0,0,0,0)"
    ),
)

fig.update_xaxes(title_text="Date", row=rows, col=1)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TABLES
# -----------------------------
left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Latest 30 Rows")
    cols_to_show = [
        "date",
        "price",
        "astro_momentum",
        "astro_momentum_smooth",
        "signal",
        "position",
        "strategy_equity",
        "buy_hold_equity",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    latest_table = df[cols_to_show].dropna(subset=["price"]).tail(30).copy()
    latest_table["regime"] = df.loc[latest_table.index, "astro_momentum"].apply(classify_regime)
    latest_table["date"] = latest_table["date"].dt.date
    st.dataframe(latest_table, use_container_width=True)

with right:
    st.subheader("Upcoming Turning Dates")
    if turning_df.empty:
        st.info("ยังไม่พบ turning points ใน horizon ที่เลือก")
    else:
        show_turning = turning_df.copy()
        show_turning["date"] = show_turning["date"].dt.date
        show_turning["regime_hint"] = show_turning[indicator_col].apply(classify_regime)
        st.dataframe(show_turning, use_container_width=True)

st.subheader("Recent Signal Changes")
if signal_markers.empty:
    st.info("ยังไม่พบ signal changes ในช่วงที่เลือก")
else:
    signal_table = signal_markers[["date", "price", "signal"]].copy().tail(20)
    signal_table["date"] = pd.to_datetime(signal_table["date"]).dt.date
    st.dataframe(signal_table, use_container_width=True)

st.subheader("Regime Guide")
guide = pd.DataFrame([
    {"score_range": ">= 3.0", "regime": "strong_bull", "meaning": "แรงขยายตัวสูงมาก"},
    {"score_range": "1.5 to 2.99", "regime": "bull", "meaning": "เอียงขึ้น / buy-the-dip bias"},
    {"score_range": "-1.49 to 1.49", "regime": "neutral", "meaning": "แกว่ง / ไม่มี edge ชัด"},
    {"score_range": "-3.0 to -1.5", "regime": "bear", "meaning": "อ่อนแรง / defensive"},
    {"score_range": "< -3.0", "regime": "crash_risk", "meaning": "เสี่ยงแรง / regime ลบมาก"},
])
st.dataframe(guide, use_container_width=True)
