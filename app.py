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


def fetch_coingecko_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "max",
        "interval": "daily",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "prices" not in data:
        raise ValueError(f"CoinGecko response missing 'prices': {data}")

    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    return df[["date", "price"]]


def fetch_binance_prices():
    # Fallback source
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": 1000,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Binance response invalid: {data}")

    rows = []
    for row in data:
        rows.append({
            "date": pd.to_datetime(row[0], unit="ms").normalize(),
            "price": float(row[4]),  # close price
        })

    df = pd.DataFrame(rows)
    return df[["date", "price"]]


@st.cache_data(ttl=3600)
def load_btc_price():
    errors = []

    try:
        return fetch_coingecko_prices(), "CoinGecko"
    except Exception as e:
        errors.append(f"CoinGecko failed: {e}")

    try:
        return fetch_binance_prices(), "Binance"
    except Exception as e:
        errors.append(f"Binance failed: {e}")

    raise RuntimeError("Unable to load BTC price data.\n" + "\n".join(errors))


st.title("Bitcoin Astro Indicator")

astro = load_astro()

try:
    price, price_source = load_btc_price()
    st.caption(f"BTC price source: {price_source}")
except Exception as e:
    st.error("โหลดข้อมูลราคา Bitcoin ไม่สำเร็จ")
    st.code(str(e))
    st.stop()

df = astro.merge(price, on="date", how="left")

if df["price"].notna().sum() == 0:
    st.error("ไม่พบข้อมูลราคาที่ merge เข้ากับ Astro data")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", df["date"].min().date())
with col2:
    end_date = st.date_input("End date", df["date"].max().date())

view = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=view["date"],
        y=view["price"],
        name="BTC Price",
        yaxis="y1",
        mode="lines"
    )
)

fig.add_trace(
    go.Scatter(
        x=view["date"],
        y=view["astro_momentum"],
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
st.dataframe(view.tail(30), use_container_width=True)
