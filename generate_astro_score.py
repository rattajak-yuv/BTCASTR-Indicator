import argparse
import os
import json
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, timezone

try:
    import swisseph as swe
except ImportError:
    swe = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# =========================================================
# CONFIG
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate astro score outputs using a configurable astro model."
    )
    parser.add_argument(
        "--config",
        default="astro_model_config.json",
        help="Path to the astro model config JSON file.",
    )
    parser.add_argument(
        "--daily-output-path",
        default="data/bitcoin_astro_daily_score.csv",
        help="Path to write the daily astro score CSV.",
    )
    parser.add_argument(
        "--raw-output-path",
        default="data/astro_aspects_raw.csv",
        help="Path to write the raw aspects CSV.",
    )
    parser.add_argument(
        "--price-cache-path",
        default="data/bitcoin_astro_daily_score.csv",
        help="Optional daily score CSV to reuse cached BTC prices from before falling back to yfinance.",
    )
    parser.add_argument(
        "--reweight-existing-raw-path",
        default="",
        help="Optional raw aspects CSV to reweight from when only config weights change.",
    )
    return parser.parse_args()

ARGS = parse_args()
CONFIG_PATH = ARGS.config

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

if swe is not None:
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

if swe is not None:
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
else:
    PLANETS = {}

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


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def load_btc_price_data(price_cache_path):
    if price_cache_path and os.path.exists(price_cache_path):
        cached = pd.read_csv(price_cache_path)
        if "date" in cached.columns and "price" in cached.columns:
            cached["date"] = pd.to_datetime(cached["date"]).dt.date
            cached["price"] = pd.to_numeric(cached["price"], errors="coerce")
            cached = cached[["date", "price"]].dropna(subset=["price"]).drop_duplicates("date")
            if not cached.empty:
                print(f"Reusing cached BTC prices from {price_cache_path}")
                return cached

    print("Downloading BTC price data...")

    if yf is None:
        raise ImportError(
            "yfinance is not installed and no usable cached BTC price series was found."
        )

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

    return btc[["date", "price"]].dropna(subset=["price"])


PRICE_STATE_CACHE = None
PRICE_STATE_METADATA = None
PRICE_STATE_DATES = None


def has_dynamic_profiles():
    return bool(CONFIG.get("dynamic_weight_profiles"))


def build_price_regime_states(price_df):
    states = price_df.copy()
    states["date"] = pd.to_datetime(states["date"]).dt.date
    states["price"] = pd.to_numeric(states["price"], errors="coerce")
    states = states.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)

    states["btc_return_1d"] = states["price"].pct_change()
    states["btc_return_30d"] = states["price"].pct_change(30)
    states["btc_vol_30d"] = states["btc_return_1d"].rolling(30).std()
    states["price_sma_200"] = states["price"].rolling(200, min_periods=200).mean()
    states["trend_distance_200d"] = (states["price"] / states["price_sma_200"]) - 1.0

    bull_mask = (states["trend_distance_200d"] > 0.05) & (states["btc_return_30d"] > 0.05)
    bear_mask = (states["trend_distance_200d"] < -0.05) & (states["btc_return_30d"] < -0.05)

    states["market_regime"] = np.select(
        [bull_mask, bear_mask],
        ["Bull Market", "Bear Market"],
        default="Sideways",
    )

    valid_vol = states["btc_vol_30d"].dropna()
    if valid_vol.empty:
        low_threshold = 0.0
        high_threshold = 0.0
    else:
        low_threshold = float(valid_vol.quantile(0.30))
        high_threshold = float(valid_vol.quantile(0.70))

    def classify_volatility_state(volatility_value):
        if pd.isna(volatility_value):
            return "Low Volatility"
        if volatility_value <= low_threshold:
            return "Low Volatility"
        if volatility_value >= high_threshold:
            return "High Volatility"

        low_distance = abs(float(volatility_value) - low_threshold)
        high_distance = abs(float(volatility_value) - high_threshold)
        if high_distance < low_distance:
            return "High Volatility"
        return "Low Volatility"

    states["volatility_state"] = states["btc_vol_30d"].apply(classify_volatility_state)
    metadata = {
        "market_regime_rule": "price vs 200D SMA plus btc_return_30d",
        "bull_threshold": 0.05,
        "bear_threshold": -0.05,
        "volatility_window_days": 30,
        "volatility_low_quantile": 0.30,
        "volatility_high_quantile": 0.70,
        "volatility_low_threshold": low_threshold,
        "volatility_high_threshold": high_threshold,
    }
    return states[["date", "market_regime", "volatility_state"]], metadata


def get_price_state_table():
    global PRICE_STATE_CACHE, PRICE_STATE_METADATA, PRICE_STATE_DATES

    if PRICE_STATE_CACHE is None:
        price_df = load_btc_price_data(ARGS.price_cache_path)
        PRICE_STATE_CACHE, PRICE_STATE_METADATA = build_price_regime_states(price_df)
        PRICE_STATE_CACHE = PRICE_STATE_CACHE.sort_values("date").reset_index(drop=True)
        PRICE_STATE_DATES = pd.to_datetime(PRICE_STATE_CACHE["date"]).to_numpy()

    return PRICE_STATE_CACHE.copy(), dict(PRICE_STATE_METADATA)


def combine_dynamic_multiplier(component_type_key, component_name, market_regime, volatility_state):
    profiles = CONFIG.get("dynamic_weight_profiles", {})
    market_profiles = profiles.get("market_regimes", {})
    volatility_profiles = profiles.get("volatility_states", {})
    caps = profiles.get("caps", {})
    overlay_strength = float(profiles.get("volatility_overlay_strength", 0.60))

    component_caps = caps.get(component_type_key, [0.75, 1.35])
    lower_cap = float(component_caps[0])
    upper_cap = float(component_caps[1])

    market_profile = market_profiles.get(market_regime, {})
    volatility_profile = volatility_profiles.get(volatility_state, {})

    market_multiplier = float(
        market_profile.get(component_type_key, {}).get(component_name, 1.0)
    )
    volatility_multiplier = float(
        volatility_profile.get(component_type_key, {}).get(component_name, 1.0)
    )

    combined = market_multiplier * (1.0 + ((volatility_multiplier - 1.0) * overlay_strength))
    return clamp(float(combined), lower_cap, upper_cap)


def resolve_weight_profile(date_value):
    if not has_dynamic_profiles():
        return {
            "market_regime": "Sideways",
            "volatility_state": "Low Volatility",
            "applied_weight_profile": "global",
            "planets": {},
            "aspects": {},
            "natal_targets": {},
        }

    state_table, _ = get_price_state_table()
    query_date = pd.to_datetime(date_value).to_datetime64()

    if PRICE_STATE_DATES is None or len(PRICE_STATE_DATES) == 0:
        resolved = {"market_regime": "Sideways", "volatility_state": "Low Volatility"}
    else:
        position = int(np.searchsorted(PRICE_STATE_DATES, query_date, side="right") - 1)
        if position < 0:
            position = 0
        if position >= len(state_table):
            position = len(state_table) - 1
        resolved = state_table.iloc[position]

    market_regime = str(resolved["market_regime"])
    volatility_state = str(resolved["volatility_state"])

    return {
        "market_regime": market_regime,
        "volatility_state": volatility_state,
        "applied_weight_profile": f"{market_regime} | {volatility_state}",
    }


def finalize_daily_outputs(df, raw_df):
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

    # Backward compatibility for UI / old optimizer
    df["astro_momentum"] = df["astro_momentum_v2"]
    df["astro_momentum_smooth"] = df["astro_momentum_v2_smooth"]
    df["expansion_score"] = df["astro_bullish_score"]
    df["contraction_score"] = df["astro_bearish_score"]
    df["narrative_score"] = df["astro_reversal_score"]
    df["trigger_score"] = df["astro_volatility_score"]
    df["regime"] = df["astro_regime_v2"]

    btc = load_btc_price_data(ARGS.price_cache_path)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.merge(btc, on="date", how="left")

    df["returns"] = df["price"].pct_change()
    df["strategy_returns"] = df["returns"] * df["position"].shift(1)
    df["strategy_equity"] = (1 + df["strategy_returns"].fillna(0)).cumprod()
    df["buy_hold_equity"] = (1 + df["returns"].fillna(0)).cumprod()
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

    os.makedirs("data", exist_ok=True)
    df.to_csv(ARGS.daily_output_path, index=False)
    raw_df.to_csv(ARGS.raw_output_path, index=False)

    print(f"Saved: {ARGS.daily_output_path}")
    print(f"Saved: {ARGS.raw_output_path}")
    print(df.tail())


def build_daily_from_reweighted_raw(raw_df):
    score_columns = [
        "bullish",
        "bearish",
        "reversal",
        "volatility",
        "compression",
        "trend_start",
        "trend_end",
    ]

    raw_df = raw_df.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.date

    for column in score_columns + ["aspect_weight", "target_weight", "orb_factor", "multiplier"]:
        if column in raw_df.columns:
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    aspect_weight_map = {
        aspect_name: float(values["weight"])
        for aspect_name, values in CONFIG["aspects"].items()
    }
    target_weight_map = {
        target_name: float(weight)
        for target_name, weight in CONFIG.get("target_weights", {}).items()
    }

    if has_dynamic_profiles():
        state_table, _ = get_price_state_table()
        state_table["date"] = pd.to_datetime(state_table["date"]).dt.date
        for existing_column in [
            "market_regime",
            "volatility_state",
            "applied_weight_profile",
            "effective_planet_multiplier",
            "effective_aspect_multiplier",
            "effective_target_multiplier",
        ]:
            if existing_column in raw_df.columns:
                raw_df = raw_df.drop(columns=[existing_column])
        raw_df = raw_df.merge(state_table, on="date", how="left")
        raw_df = raw_df.sort_values("date").reset_index(drop=True)
        raw_df["market_regime"] = raw_df["market_regime"].ffill().bfill().fillna("Sideways")
        raw_df["volatility_state"] = raw_df["volatility_state"].ffill().bfill().fillna("Low Volatility")
    else:
        raw_df["market_regime"] = "Sideways"
        raw_df["volatility_state"] = "Low Volatility"

    for index, row in raw_df.iterrows():
        source = str(row.get("source", "")).strip().lower()
        aspect_name = str(row.get("aspect", "")).strip()
        transit_planet = str(row.get("transit_planet", "")).strip()
        target_name = str(row.get("target", "")).strip()
        market_regime = str(row.get("market_regime", "Sideways"))
        volatility_state = str(row.get("volatility_state", "Low Volatility"))

        planet_multiplier = 1.0
        aspect_multiplier = 1.0
        target_multiplier = 1.0

        if has_dynamic_profiles():
            planet_multiplier = combine_dynamic_multiplier(
                "planets",
                transit_planet,
                market_regime,
                volatility_state,
            )
            if source == "aspect" and aspect_name in aspect_weight_map:
                aspect_multiplier = combine_dynamic_multiplier(
                    "aspects",
                    aspect_name,
                    market_regime,
                    volatility_state,
                )
            if target_name in target_weight_map:
                target_multiplier = combine_dynamic_multiplier(
                    "natal_targets",
                    target_name,
                    market_regime,
                    volatility_state,
                )

        if source == "aspect" and aspect_name in aspect_weight_map:
            old_aspect_weight = row.get("aspect_weight", np.nan)
            base_aspect_weight = aspect_weight_map[aspect_name]
            new_aspect_weight = base_aspect_weight * aspect_multiplier

            if pd.isna(old_aspect_weight) or np.isclose(old_aspect_weight, 0.0):
                aspect_scale = 1.0
            else:
                aspect_scale = new_aspect_weight / float(old_aspect_weight)

            old_target_weight = row.get("target_weight", np.nan)
            base_target_weight = target_weight_map.get(target_name, 1.0)
            new_target_weight = base_target_weight * target_multiplier

            if pd.isna(old_target_weight) or np.isclose(old_target_weight, 0.0):
                target_scale = 1.0
            else:
                target_scale = new_target_weight / float(old_target_weight)

            scale = planet_multiplier * aspect_scale * target_scale

            for score_column in score_columns:
                raw_df.at[index, score_column] = float(row[score_column]) * scale

            raw_df.at[index, "aspect_weight"] = new_aspect_weight
            raw_df.at[index, "target_weight"] = new_target_weight

            orb_factor = row.get("orb_factor", np.nan)
            if pd.notna(orb_factor):
                raw_df.at[index, "multiplier"] = (
                    new_aspect_weight * float(orb_factor) * new_target_weight * planet_multiplier
                )
        elif source == "house":
            scale = planet_multiplier
            for score_column in score_columns:
                raw_df.at[index, score_column] = float(row[score_column]) * scale
            raw_df.at[index, "multiplier"] = float(row.get("multiplier", 1.0)) * scale

        raw_df.at[index, "effective_planet_multiplier"] = planet_multiplier
        raw_df.at[index, "effective_aspect_multiplier"] = aspect_multiplier
        raw_df.at[index, "effective_target_multiplier"] = target_multiplier
        raw_df.at[index, "applied_weight_profile"] = f"{market_regime} | {volatility_state}"

    daily = (
        raw_df.groupby("date")[score_columns]
        .sum()
        .reset_index()
    )

    daily["astro_bullish_score"] = daily["bullish"]
    daily["astro_bearish_score"] = daily["bearish"]
    daily["astro_reversal_score"] = daily["reversal"]
    daily["astro_volatility_score"] = daily["volatility"]
    daily["astro_compression_score"] = daily["compression"]
    daily["astro_trend_start_score"] = daily["trend_start"]
    daily["astro_trend_end_score"] = daily["trend_end"]

    daily["astro_momentum_v2"] = (
        daily["astro_bullish_score"] - daily["astro_bearish_score"]
    )
    daily["astro_regime_v2"] = daily.apply(
        lambda row: classify_regime_v2(
            momentum=row["astro_momentum_v2"],
            reversal=row["astro_reversal_score"],
            trend_end=row["astro_trend_end_score"],
            compression=row["astro_compression_score"],
        ),
        axis=1,
    )

    state_summary = (
        raw_df.groupby("date")[["market_regime", "volatility_state", "applied_weight_profile"]]
        .agg(lambda values: values.dropna().iloc[0] if len(values.dropna()) else "")
        .reset_index()
    )
    daily = daily.merge(state_summary, on="date", how="left")

    return daily[
        [
            "date",
            "astro_bullish_score",
            "astro_bearish_score",
            "astro_reversal_score",
            "astro_volatility_score",
            "astro_compression_score",
            "astro_trend_start_score",
            "astro_trend_end_score",
            "astro_momentum_v2",
            "astro_regime_v2",
            "market_regime",
            "volatility_state",
            "applied_weight_profile",
        ]
    ], raw_df


if ARGS.reweight_existing_raw_path:
    source_raw_path = ARGS.reweight_existing_raw_path
    if not os.path.exists(source_raw_path):
        raise FileNotFoundError(f"Missing raw input for reweight mode: {source_raw_path}")

    print(f"Reweighting existing raw aspects from {source_raw_path} using {CONFIG_PATH}")
    source_raw = pd.read_csv(source_raw_path)
    daily_df, adjusted_raw = build_daily_from_reweighted_raw(source_raw)
    finalize_daily_outputs(daily_df, adjusted_raw)
    raise SystemExit(0)

if swe is None:
    raise ImportError(
        "swisseph is not installed for full ephemeris generation. "
        "Use --reweight-existing-raw-path for aspect-only experimental rebuilds."
    )

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

def evaluate_aspects(transit_name, transit_lon, date_value, weight_profile):
    scores = empty_scores()
    raw_rows = []

    max_orb = CONFIG["max_orb_by_planet"].get(transit_name, 2.0)
    planet_multiplier = 1.0
    if has_dynamic_profiles():
        planet_multiplier = combine_dynamic_multiplier(
            "planets",
            transit_name,
            weight_profile["market_regime"],
            weight_profile["volatility_state"],
        )

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
                aspect_profile_multiplier = 1.0
                if has_dynamic_profiles():
                    aspect_profile_multiplier = combine_dynamic_multiplier(
                        "aspects",
                        aspect_name,
                        weight_profile["market_regime"],
                        weight_profile["volatility_state"],
                    )
                aspect_weight = aspect_conf["weight"] * aspect_profile_multiplier

                orb = abs(d - aspect_angle)

                if orb <= max_orb:

                    orb_factor = 1 - (orb / max_orb)

                    target_profile_multiplier = 1.0
                    if has_dynamic_profiles() and target in CONFIG["target_weights"]:
                        target_profile_multiplier = combine_dynamic_multiplier(
                            "natal_targets",
                            target,
                            weight_profile["market_regime"],
                            weight_profile["volatility_state"],
                        )
                    target_weight = CONFIG["target_weights"].get(target, 1.0) * target_profile_multiplier

                    multiplier = aspect_weight * orb_factor * target_weight * planet_multiplier

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
                        "market_regime": weight_profile["market_regime"],
                        "volatility_state": weight_profile["volatility_state"],
                        "applied_weight_profile": weight_profile["applied_weight_profile"],
                        "effective_planet_multiplier": planet_multiplier,
                        "effective_aspect_multiplier": aspect_profile_multiplier,
                        "effective_target_multiplier": target_profile_multiplier,
                        "bullish": aspect_scores["bullish"],
                        "bearish": aspect_scores["bearish"],
                        "reversal": aspect_scores["reversal"],
                        "volatility": aspect_scores["volatility"],
                        "compression": aspect_scores["compression"],
                        "trend_start": aspect_scores["trend_start"],
                        "trend_end": aspect_scores["trend_end"],
                    })

    return scores, raw_rows


def evaluate_house_scores(transit_name, transit_lon, house_cusps, date_value, weight_profile):
    scores = empty_scores()
    raw_rows = []

    house_num = get_house_of_longitude(transit_lon, house_cusps)
    planet_multiplier = 1.0
    if has_dynamic_profiles():
        planet_multiplier = combine_dynamic_multiplier(
            "planets",
            transit_name,
            weight_profile["market_regime"],
            weight_profile["volatility_state"],
        )

    for rule in CONFIG["house_rules"]:

        if rule["planet"] != transit_name:
            continue

        if house_num in rule["houses"]:

            house_scores = empty_scores()
            apply_rule_scores(house_scores, rule["scores"], planet_multiplier)
            apply_rule_scores(scores, rule["scores"], planet_multiplier)

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
                "multiplier": planet_multiplier,
                "market_regime": weight_profile["market_regime"],
                "volatility_state": weight_profile["volatility_state"],
                "applied_weight_profile": weight_profile["applied_weight_profile"],
                "effective_planet_multiplier": planet_multiplier,
                "effective_aspect_multiplier": 1.0,
                "effective_target_multiplier": 1.0,
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
    weight_profile = resolve_weight_profile(date_value)

    for pname, pid in PLANETS.items():

        transit_lon = get_planet_lon(jd, pid)

        aspect_scores, aspect_rows = evaluate_aspects(
            pname,
            transit_lon,
            date_value,
            weight_profile,
        )

        house_scores, house_rows = evaluate_house_scores(
            pname,
            transit_lon,
            transit_cusps,
            date_value,
            weight_profile,
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
        "market_regime": weight_profile["market_regime"],
        "volatility_state": weight_profile["volatility_state"],
        "applied_weight_profile": weight_profile["applied_weight_profile"],
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

finalize_daily_outputs(df, raw_df)
