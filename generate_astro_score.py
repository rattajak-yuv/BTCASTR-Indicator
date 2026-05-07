import os
import json
import math
import swisseph as swe
import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime, timedelta, timezone

# =========================================================
# CONFIG
# =========================================================

CONFIG_PATH = "astro_model_config.json"

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

swe.set_sid_mode(swe.SIDM_LAHIRI)

# =========================================================
# BITCOIN NATAL CHART
# First Transaction Chart
# 2009-01-12 03:30:25 UTC
# Pasadena, California
# =========================================================

NATAL_DT = datetime(2009, 1, 12, 3, 30, 25, tzinfo=timezone.utc)

LAT = 34.1070
LON = -118.0570

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

# =========================================================
# HELPERS
# =========================================================

def julday(dt):
    return swe.julday(
        dt.year,
        dt.month,
        dt.day,
        dt.hour + dt.minute / 60 + dt.second / 3600,
    )

def norm360(x):
    return x % 360

def angle_diff(a, b):
    d = abs(norm360(a) - norm360(b))
    return min(d, 360 - d)

def get_planet_lon(jd, planet_id):
    xx, _ = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)
    return xx[0]

def get_houses(jd, lat, lon):
    cusps, ascmc = swe.houses_ex(
        jd,
        lat,
        lon,
        b'P',
        swe.FLG_SIDEREAL
    )
    return cusps, ascmc

def get_house_of_longitude(lon, house_cusps):
    for i in range(12):
        start = house_cusps[i]
        end = house_cusps[(i + 1) % 12]

        if start < end:
            if start <= lon < end:
                return i + 1
        else:
            if lon >= start or lon < end:
                return i + 1

    return 12

# =========================================================
# NATAL CHART
# =========================================================

natal_jd = julday(NATAL_DT)

natal_cusps, natal_ascmc = get_houses(
    natal_jd,
    LAT,
    LON
)

NATAL = {}

for name, pid in PLANETS.items():
    NATAL[name] = get_planet_lon(natal_jd, pid)

NATAL["Asc"] = natal_ascmc[0]
NATAL["MC"] = natal_ascmc[1]

# =========================================================
# SCORE ENGINE
# =========================================================

def empty_scores():
    return {
        "bullish": 0.0,
        "bearish": 0.0,
        "reversal": 0.0,
        "volatility": 0.0,
        "compression": 0.0,
        "trend_start": 0.0,
        "trend_end": 0.0,
    }

def apply_rule_scores(score_dict, score_add, multiplier):
    for k, v in score_add.items():
        score_dict[k] += v * multiplier

def evaluate_aspects(transit_name, transit_lon):

    scores = empty_scores()

    max_orb = CONFIG["max_orb_by_planet"].get(
        transit_name,
        2.0
    )

    for rule in CONFIG["rules"]:

        if rule["planet"] != transit_name:
            continue

        for target in rule["targets"]:

            target_lon = NATAL[target]

            d = angle_diff(transit_lon, target_lon)

            for aspect_name in rule["aspects"]:

                aspect_conf = CONFIG["aspects"][aspect_name]

                aspect_angle = aspect_conf["angle"]
                aspect_weight = aspect_conf["weight"]

                orb = abs(d - aspect_angle)

                if orb <= max_orb:

                    orb_factor = 1 - (orb / max_orb)

                    target_weight = CONFIG["target_weights"].get(
                        target,
                        1.0
                    )

                    multiplier = (
                        aspect_weight
                        * orb_factor
                        * target_weight
                    )

                    apply_rule_scores(
                        scores,
                        rule["scores"],
                        multiplier
                    )

    return scores

def evaluate_house_scores(
    transit_name,
    transit_lon,
    house_cusps
):

    scores = empty_scores()

    house_num = get_house_of_longitude(
        transit_lon,
        house_cusps
    )

    for rule in CONFIG["house_rules"]:

        if rule["planet"] != transit_name:
            continue

        if house_num in rule["houses"]:

            apply_rule_scores(
                scores,
                rule["scores"],
                1.0
            )

    return scores

# =========================================================
# MAIN DAILY ENGINE
# =========================================================

def calculate_day(dt):

    jd = julday(dt)

    transit_cusps, transit_ascmc = get_houses(
        jd,
        LAT,
        LON
    )

    total = empty_scores()

    for pname, pid in PLANETS.items():

        transit_lon = get_planet_lon(jd, pid)

        aspect_scores = evaluate_aspects(
            pname,
            transit_lon
        )

        house_scores = evaluate_house_scores(
            pname,
            transit_lon,
            transit_cusps
        )

        for k in total.keys():
            total[k] += aspect_scores[k]
            total[k] += house_scores[k]

    momentum = total["bullish"] - total["bearish"]

    if momentum >= CONFIG["regime_thresholds"]["strong_uptrend"]:
        regime = "strong_uptrend"

    elif momentum >= CONFIG["regime_thresholds"]["uptrend"]:
        regime = "uptrend"

    elif momentum <= CONFIG["regime_thresholds"]["crash_risk"]:
        regime = "crash_risk"

    elif momentum <= CONFIG["regime_thresholds"]["downtrend"]:
        regime = "downtrend"

    else:
        regime = "sideways"

    return {
        "date": dt.date(),

        "astro_bullish_score": total["bullish"],
        "astro_bearish_score": total["bearish"],
        "astro_reversal_score": total["reversal"],
        "astro_volatility_score": total["volatility"],
        "astro_compression_score": total["compression"],
        "astro_trend_start_score": total["trend_start"],
        "astro_trend_end_score": total["trend_end"],

        "astro_momentum_v2": momentum,
        "astro_regime_v2": regime,
    }

# =========================================================
# DATE RANGE
# =========================================================

start_date = datetime(2009, 1, 3, tzinfo=timezone.utc)
end_date = datetime.now(timezone.utc) + timedelta(days=365 * 2)

dates = pd.date_range(
    start_date,
    end_date,
    freq="D"
)

rows = []

print("Calculating astro scores...")

for dt in dates:
    rows.append(
        calculate_day(dt)
    )

df = pd.DataFrame(rows)

# =========================================================
# SMOOTHING
# =========================================================

df["astro_momentum_v2_smooth"] = (
    df["astro_momentum_v2"]
    .ewm(span=5, adjust=False)
    .mean()
)

# =========================================================
# SIGNAL ENGINE
# =========================================================

def classify_signal(x):

    if x >= 3:
        return "strong_buy"

    elif x >= 1.5:
        return "buy"

    elif x <= -3:
        return "strong_sell"

    elif x <= -1.5:
        return "sell"

    return "neutral"

df["signal"] = (
    df["astro_momentum_v2_smooth"]
    .apply(classify_signal)
)

# =========================================================
# POSITION ENGINE
# =========================================================

def signal_to_position(sig):

    if sig in ["buy", "strong_buy"]:
        return 1

    elif sig in ["sell", "strong_sell"]:
        return -1

    return 0

df["position"] = (
    df["signal"]
    .apply(signal_to_position)
)

# =========================================================
# PRICE DATA
# =========================================================

print("Downloading BTC price data...")

btc = yf.download(
    "BTC-USD",
    start="2009-01-03",
    progress=False,
    auto_adjust=True
)

if btc is None or btc.empty:
    raise ValueError("Yahoo Finance returned empty BTC data")

# Fix yfinance MultiIndex columns if present
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

btc = btc.reset_index()

if "Date" not in btc.columns:
    raise ValueError(f"Unexpected yfinance columns: {btc.columns.tolist()}")

if "Close" not in btc.columns:
    raise ValueError(f"Missing Close column from yfinance: {btc.columns.tolist()}")

btc["date"] = pd.to_datetime(btc["Date"]).dt.date
btc["price"] = pd.to_numeric(btc["Close"], errors="coerce")

btc = btc[["date", "price"]].dropna(subset=["price"])

df["date"] = pd.to_datetime(df["date"]).dt.date

df = df.merge(
    btc,
    on="date",
    how="left"
)

# =========================================================
# RETURNS
# =========================================================

df["returns"] = (
    df["price"]
    .pct_change()
)

df["strategy_returns"] = (
    df["returns"]
    * df["position"].shift(1)
)

df["strategy_equity"] = (
    1 + df["strategy_returns"].fillna(0)
).cumprod()

df["buy_hold_equity"] = (
    1 + df["returns"].fillna(0)
).cumprod()

# =========================================================
# EXPORT
# =========================================================

os.makedirs("data", exist_ok=True)

OUTPUT_PATH = "data/bitcoin_astro_daily_score.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Saved: {OUTPUT_PATH}")
print(df.tail())
