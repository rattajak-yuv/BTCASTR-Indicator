import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Bitcoin Astro Indicator", layout="wide")

@st.cache_data(ttl=3600)
def load_astro():
    df = pd.read_csv("data/bitcoin_astro_daily_score.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600)
def load_btc_price():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=max&interval=daily"
    data = requests.get(url, timeout=30).json()
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    return df[["date", "price"]]

astro = load_astro()
price = load_btc_price()

df = astro.merge(price, on="date", how="left")

st.title("Bitcoin Astro Indicator")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", df["date"].min().date())
with col2:
    end_date = st.date_input("End date", df["date"].max().date())

view = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=view["date"], y=view["price"], name="BTC Price", yaxis="y1"))
fig.add_trace(go.Scatter(x=view["date"], y=view["astro_momentum"], name="Astro Momentum", yaxis="y2"))

fig.update_layout(
    height=700,
    xaxis=dict(title="Date"),
    yaxis=dict(title="BTC Price (USD)", side="left"),
    yaxis2=dict(title="Astro Momentum", overlaying="y", side="right"),
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Latest values")
st.dataframe(view.tail(30), use_container_width=True)
