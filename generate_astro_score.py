import os
import json
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
# BITCOIN NATAL CHART — FIRST TRANSACTION
# 2009-01-12 03:30:25 UTC
# Temple City / Pasadena area, California
# =========================================================

NATAL_DT = datetime(2009, 1, 12, 3, 30, 25, tzinfo=timezone.utc)

LAT = 34.1070
LON = -118.0570

START_DATE = datetime(2009, 1, 3, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc) + timedelta(days=365 * 2)

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

def to_python_datetime(dt):
    if hasattr(dt, "to_pydatetime"):
        return dt.to_pydatetime()
    return dt


def julday(dt):
    dt = to_python_datetime(dt)
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
    cusps, ascmc = swe.houses_ex(jd, lat, lon, b"P", swe.FLG_SIDEREAL)
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
        if k in score_dict:
            score_dict[k] += v * multiplier


def classify_regime_v2(momentum, reversal, trend_end, compression):
    thresholds = CONFIG["regime_thresholds"]

    if reversal >= thresholds.get("reversal_zone", 2.5):
        return "reversal_zone"

    if trend_end >= thresholds.get("exhaustion_zone", 2.5):
        return "exhaustion_zone"

    if compression >= thresholds.get("compression_zone", 2.0):
        return "compression_zone"

    if momentum >= thresholds["strong_uptrend"]:
        return "strong_uptrend"

    if momentum >= thresholds["uptrend"]:
        return "uptrend"

    if momentum <= thresholds["crash_risk"]:
        return "crash_risk"

    if momentum <= thresholds["downtrend"]:
        return "downtrend"

    return "sideways"


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


def signal_to_position(sig):
    if sig in ["buy", "strong_buy"]:
        return 1
    elif sig in ["sell", "strong_sell"]:
        return -1
    return 0


def max_drawdown(equity):
    peak = equity.cummax()
    return ((equity / peak) - 1).min()

# =========================================================
# NATAL CHART
# =========================================================

natal_jd = julday(NATAL_DT)
natal_cusps, natal_ascmc = get_houses(natal_jd, LAT, LON)

NATAL = {}

for name, pid in PLANETS.items():
    NATAL[name] = get_planet_lon(natal_jd, pid)

NATAL["Asc"] = natal_ascmc[0]
NATAL["MC"] = natal_ascmc[1]

# =========================================================
# ASTRO SCORE ENGINE V2
# =========================================================

def evaluate_aspects(transit_name, transit_lon, date_value):
    scores = empty_scores()
    raw_rows = []

    max_orb = CONFIG["max_orb_by_planet"].get(transit_name, 2.0)

    for rule in CONFIG["rules"]:

        if rule["planet"] != transit_name:
            continue

        for target in rule["targets"]:

            if target not in NATAL:
                continue

            target_lon = NATAL[target]
            d = angle_diff(transit_lon, target_lon)

            for aspect_name in rule["aspects"]:

                aspect_conf = CONFIG["aspects"][aspect_name]

                aspect_angle = aspect_conf["angle"]
                aspect_weight = aspect_conf["weight"]

                orb = abs(d - aspect_angle)

                if orb <= max_orb:

                    orb_factor = 1 - (orb / max_orb)

                    target_weight = CONFIG["target_weights"].get(target, 1.0)

                    multiplier = aspect_weight * orb_factor * target_weight

                    aspect_scores = empty_scores()
                    apply_rule_scores(aspect_scores, rule["scores"], multiplier)
                    apply_rule_scores(scores, rule["scores"], multiplier)

                    raw_rows.append({
                        "date": date_value,
                        "source": "aspect",
                        "rule_name": rule["name"],
                        "transit_planet": transit_name,
                        "target": target,
                        "aspect": aspect_name,
                        "aspect_angle": aspect_angle,
                        "orb": round(orb, 4),
                        "orb_factor": round(orb_factor, 4),
                        "aspect_weight": aspect_weight,
                        "target_weight": target_weight,
                        "multiplier": round(multiplier, 4),
                        "bullish": aspect_scores["bullish"],
                        "bearish": aspect_scores["bearish"],
                        "reversal": aspect_scores["reversal"],
                        "volatility": aspect_scores["volatility"],
                        "compression": aspect_scores["compression"],
                        "trend_start": aspect_scores["trend_start"],
                        "trend_end": aspect_scores["trend_end"],
                    })

    return scores, raw_rows


def evaluate_house_scores(transit_name, transit_lon, house_cusps, date_value):
    scores = empty_scores()
    raw_rows = []

    house_num = get_house_of_longitude(transit_lon, house_cusps)

    for rule in CONFIG["house_rules"]:

        if rule["planet"] != transit_name:
            continue

        if house_num in rule["houses"]:

            house_scores = empty_scores()
            apply_rule_scores(house_scores, rule["scores"], 1.0)
            apply_rule_scores(scores, rule["scores"], 1.0)

            raw_rows.append({
                "date": date_value,
                "source": "house",
                "rule_name": f"{transit_name} in house {house_num}",
                "transit_planet": transit_name,
                "target": f"House {house_num}",
                "aspect": "house_position",
                "aspect_angle": np.nan,
                "orb": np.nan,
                "orb_factor": np.nan,
                "aspect_weight": np.nan,
                "target_weight": np.nan,
                "multiplier": 1.0,
                "bullish": house_scores["bullish"],
                "bearish": house_scores["bearish"],
                "reversal": house_scores["reversal"],
                "volatility": house_scores["volatility"],
                "compression": house_scores["compression"],
                "trend_start": house_scores["trend_start"],
                "trend_end": house_scores["trend_end"],
            })

    return scores, raw_rows


def calculate_day(dt):
    dt = to_python_datetime(dt)
    jd = julday(dt)

    transit_cusps, _ = get_houses(jd, LAT, LON)

    total = empty_scores()
    raw_rows = []

    date_value = dt.date()

    for pname, pid in PLANETS.items():

        transit_lon = get_planet_lon(jd, pid)

        aspect_scores, aspect_rows = evaluate_aspects(
            pname,
            transit_lon,
            date_value,
        )

        house_scores, house_rows = evaluate_house_scores(
            pname,
            transit_lon,
            transit_cusps,
            date_value,
        )

        for k in total.keys():
            total[k] += aspect_scores[k]
            total[k] += house_scores[k]

        raw_rows.extend(aspect_rows)
        raw_rows.extend(house_rows)

    momentum = total["bullish"] - total["bearish"]

    regime = classify_regime_v2(
        momentum=momentum,
        reversal=total["reversal"],
        trend_end=total["trend_end"],
        compression=total["compression"],
    )

    row = {
        "date": date_value,

        "astro_bullish_score": round(total["bullish"], 6),
        "astro_bearish_score": round(total["bearish"], 6),
        "astro_reversal_score": round(total["reversal"], 6),
        "astro_volatility_score": round(total["volatility"], 6),
        "astro_compression_score": round(total["compression"], 6),
        "astro_trend_start_score": round(total["trend_start"], 6),
        "astro_trend_end_score": round(total["trend_end"], 6),

        "astro_momentum_v2": round(momentum, 6),
        "astro_regime_v2": regime,
    }

    return row, raw_rows

# =========================================================
# CALCULATE DAILY ASTRO SCORES
# =========================================================

dates = pd.date_range(START_DATE, END_DATE, freq="D")

daily_rows = []
raw_aspect_rows = []

print("Calculating Astro Model v2 scores...")

for dt in dates:
    row, raw_rows = calculate_day(dt)
    daily_rows.append(row)
    raw_aspect_rows.extend(raw_rows)

df = pd.DataFrame(daily_rows)
raw_df = pd.DataFrame(raw_aspect_rows)

# =========================================================
# SMOOTHING + SIGNALS
# =========================================================

df["astro_momentum_v2_smooth"] = (
    df["astro_momentum_v2"]
    .ewm(span=5, adjust=False)
    .mean()
)

df["astro_bullish_score_smooth"] = (
    df["astro_bullish_score"]
    .ewm(span=5, adjust=False)
    .mean()
)

df["astro_bearish_score_smooth"] = (
    df["astro_bearish_score"]
    .ewm(span=5, adjust=False)
    .mean()
)

df["signal"] = df["astro_momentum_v2_smooth"].apply(classify_signal)
df["position"] = df["signal"].apply(signal_to_position)

# =========================================================
# BACKWARD COMPATIBILITY FOR UI / OLD OPTIMIZER
# =========================================================

df["astro_momentum"] = df["astro_momentum_v2"]
df["astro_momentum_smooth"] = df["astro_momentum_v2_smooth"]

df["expansion_score"] = df["astro_bullish_score"]
df["contraction_score"] = df["astro_bearish_score"]
df["narrative_score"] = df["astro_reversal_score"]
df["trigger_score"] = df["astro_volatility_score"]

df["regime"] = df["astro_regime_v2"]

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
# RETURNS + BACKTEST
# =========================================================

df["returns"] = df["price"].pct_change()

df["strategy_returns"] = (
    df["returns"] * df["position"].shift(1)
)

df["strategy_equity"] = (
    1 + df["strategy_returns"].fillna(0)
).cumprod()

df["buy_hold_equity"] = (
    1 + df["returns"].fillna(0)
).cumprod()

df["strategy_drawdown"] = (
    df["strategy_equity"] / df["strategy_equity"].cummax()
) - 1

df["buy_hold_drawdown"] = (
    df["buy_hold_equity"] / df["buy_hold_equity"].cummax()
) - 1

price_available = df.dropna(subset=["price"]).copy()

if not price_available.empty:
    last_idx = price_available.index[-1]

    strategy_total_return = df.loc[last_idx, "strategy_equity"] - 1
    buy_hold_total_return = df.loc[last_idx, "buy_hold_equity"] - 1

    strategy_max_drawdown = df.loc[:last_idx, "strategy_drawdown"].min()
    buy_hold_max_drawdown = df.loc[:last_idx, "buy_hold_drawdown"].min()
else:
    strategy_total_return = np.nan
    buy_hold_total_return = np.nan
    strategy_max_drawdown = np.nan
    buy_hold_max_drawdown = np.nan

df["strategy_total_return"] = strategy_total_return
df["buy_hold_total_return"] = buy_hold_total_return
df["strategy_max_drawdown"] = strategy_max_drawdown
df["buy_hold_max_drawdown"] = buy_hold_max_drawdown

# =========================================================
# EXPORT
# =========================================================

os.makedirs("data", exist_ok=True)

daily_output_path = "data/bitcoin_astro_daily_score.csv"
raw_output_path = "data/astro_aspects_raw.csv"

df.to_csv(daily_output_path, index=False)
raw_df.to_csv(raw_output_path, index=False)

print(f"Saved: {daily_output_path}")
print(f"Saved: {raw_output_path}")
print(df.tail())
