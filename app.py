import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Bitcoin Astro Indicator", layout="wide")


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("data/bitcoin_astro_daily_score.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


st.title("Bitcoin Astro Indicator")

try:
    df = load_data()
except Exception as e:
    st.error("โหลดข้อมูล indicator ไม่สำเร็จ")
    st.code(str(e))
    st.stop()

if "price" not in df.columns:
    st.error("ไม่พบคอลัมน์ price ในไฟล์ data/bitcoin_astro_daily_score.csv")
    st.stop()

price_df = df.dropna(subset=["price"]).copy()

if price_df.empty:
    st.error("ยังไม่มีข้อมูลราคา BTC ที่ใช้งานได้ในไฟล์ data/bitcoin_astro_daily_score.csv")
    st.stop()

st.caption("Price source: Yahoo Finance (BTC-USD)")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", price_df["date"].min().date())
with col2:
    end_date = st.date_input("End date", price_df["date"].max().date())

view = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

chart_df = view.dropna(subset=["price"]).copy()

if chart_df.empty:
    st.warning("ช่วงวันที่ที่เลือกยังไม่มีข้อมูลราคา")
    st.stop()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_df["date"],
        y=chart_df["price"],
        name="BTC Price",
        yaxis="y1",
        mode="lines"
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_df["date"],
        y=chart_df["astro_momentum"],
        name="Astro Momentum",
        yaxis="y2",
        mode="lines"
    )
)

fig.update_layout(
    height=700,
    xaxis=dict(title="Date"),
    yaxis=dict(title="BTC Price (USD)", side="left"),
    yaxis2=dict(
        title="Astro Momentum",
        overlaying="y",
        side="right"
    ),
    legend=dict(orientation="h"),
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Latest values")
st.dataframe(chart_df.tail(30), use_container_width=True)
