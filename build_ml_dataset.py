import os
import numpy as np
import pandas as pd

DAILY_PATH = "data/bitcoin_astro_daily_score.csv"
RAW_PATH = "data/astro_aspects_raw.csv"
OUTPUT_PATH = "data/ml_dataset.csv"

RAW_SCORE_COLUMNS = [
    "bullish",
    "bearish",
    "reversal",
    "volatility",
    "compression",
    "trend_start",
    "trend_end",
]

PLANET_SIGNAL_COLUMNS = [
    "sun_signal",
    "moon_signal",
    "mercury_signal",
    "venus_signal",
    "mars_signal",
    "jupiter_signal",
    "saturn_signal",
    "uranus_signal",
    "neptune_signal",
    "pluto_signal",
]

ASPECT_STRENGTH_COLUMNS = [
    "conjunction_strength",
    "trine_strength",
    "sextile_strength",
    "square_strength",
    "opposition_strength",
]

NATAL_TARGET_STRENGTH_COLUMNS = [
    "sun_target_strength",
    "moon_target_strength",
    "asc_target_strength",
    "mc_target_strength",
]

PLANET_NAME_MAP = {
    "sun": "Sun",
    "moon": "Moon",
    "mercury": "Mercury",
    "venus": "Venus",
    "mars": "Mars",
    "jupiter": "Jupiter",
    "saturn": "Saturn",
    "uranus": "Uranus",
    "neptune": "Neptune",
    "pluto": "Pluto",
}

ASPECT_NAME_MAP = {
    "conjunction_strength": "conjunction",
    "trine_strength": "trine",
    "sextile_strength": "sextile",
    "square_strength": "square",
    "opposition_strength": "opposition",
}

TARGET_NAME_MAP = {
    "sun_target_strength": "Sun",
    "moon_target_strength": "Moon",
    "asc_target_strength": "Asc",
    "mc_target_strength": "MC",
}


def add_rolling_features(df, col):
    for span in [3, 5, 10, 21, 30, 60]:
        df[f"{col}_ema_{span}"] = df[col].ewm(span=span, adjust=False).mean()
        df[f"{col}_sma_{span}"] = df[col].rolling(span).mean()

    for window in [3, 5, 10, 21, 30]:
        df[f"{col}_chg_{window}"] = df[col].diff(window)
        df[f"{col}_roll_max_{window}"] = df[col].rolling(window).max()
        df[f"{col}_roll_min_{window}"] = df[col].rolling(window).min()

    return df


def build_raw_aspect_features(raw):
    raw["date"] = pd.to_datetime(raw["date"])
    raw["transit_planet"] = raw["transit_planet"].astype(str)
    raw["target"] = raw["target"].astype(str)
    raw["aspect"] = raw["aspect"].astype(str)

    for c in RAW_SCORE_COLUMNS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    raw["raw_directional_signal"] = (
        raw["bullish"]
        + raw["trend_start"]
        - raw["bearish"]
        - raw["trend_end"]
    )
    raw["raw_strength"] = raw[RAW_SCORE_COLUMNS].abs().sum(axis=1)

    # Aggregate by date
    daily_score = raw.groupby("date")[RAW_SCORE_COLUMNS].sum().reset_index()

    # Planet-level impact
    planet_pivot = raw.pivot_table(
        index="date",
        columns="transit_planet",
        values=RAW_SCORE_COLUMNS,
        aggfunc="sum",
        fill_value=0,
    )

    planet_pivot.columns = [
        f"planet_{score}_{planet}"
        for score, planet in planet_pivot.columns
    ]

    planet_pivot = planet_pivot.reset_index()

    # Aspect type count
    aspect_count = raw.pivot_table(
        index="date",
        columns="aspect",
        values="rule_name",
        aggfunc="count",
        fill_value=0,
    )

    aspect_count.columns = [f"aspect_count_{c}" for c in aspect_count.columns]
    aspect_count = aspect_count.reset_index()

    # Compact raw astro recovery aggregates.
    planet_signal = raw.pivot_table(
        index="date",
        columns="transit_planet",
        values="raw_directional_signal",
        aggfunc="sum",
        fill_value=0,
    )
    planet_signal.columns = [
        f"{str(planet).strip().lower()}_signal"
        for planet in planet_signal.columns
    ]
    planet_signal = planet_signal.reset_index()

    aspect_strength = raw[raw["aspect"].isin(ASPECT_NAME_MAP.values())].pivot_table(
        index="date",
        columns="aspect",
        values="raw_strength",
        aggfunc="sum",
        fill_value=0,
    )
    aspect_strength.columns = [
        f"{str(aspect).strip().lower()}_strength"
        for aspect in aspect_strength.columns
    ]
    aspect_strength = aspect_strength.reset_index()

    natal_target_strength = raw[raw["target"].isin(TARGET_NAME_MAP.values())].pivot_table(
        index="date",
        columns="target",
        values="raw_strength",
        aggfunc="sum",
        fill_value=0,
    )
    natal_target_strength.columns = [
        f"{str(target).strip().lower()}_target_strength"
        for target in natal_target_strength.columns
    ]
    natal_target_strength = natal_target_strength.reset_index()

    house_activation = (
        raw[raw["aspect"] == "house_position"]
        .groupby("date", as_index=False)["raw_strength"]
        .sum()
        .rename(columns={"raw_strength": "house_activation_strength"})
    )

    raw_totals = raw.groupby("date", as_index=False).agg(
        raw_astro_total_strength=("raw_strength", "sum"),
        raw_astro_directional_signal=("raw_directional_signal", "sum"),
        raw_astro_event_count=("rule_name", "count"),
    )

    out = daily_score.merge(planet_pivot, on="date", how="left")
    out = out.merge(aspect_count, on="date", how="left")
    out = out.merge(planet_signal, on="date", how="left")
    out = out.merge(aspect_strength, on="date", how="left")
    out = out.merge(natal_target_strength, on="date", how="left")
    out = out.merge(house_activation, on="date", how="left")
    out = out.merge(raw_totals, on="date", how="left")

    for column in PLANET_SIGNAL_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0

    for column in ASPECT_STRENGTH_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0

    for column in NATAL_TARGET_STRENGTH_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0

    for column in [
        "house_activation_strength",
        "raw_astro_total_strength",
        "raw_astro_directional_signal",
        "raw_astro_event_count",
    ]:
        if column not in out.columns:
            out[column] = 0.0

    return out


def main():
    print("Loading daily astro data...")
    df = pd.read_csv(DAILY_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print("Loading raw aspect data...")
    raw = pd.read_csv(RAW_PATH)
    raw_features = build_raw_aspect_features(raw)

    print("Merging daily + raw features...")
    df = df.merge(raw_features, on="date", how="left", suffixes=("", "_raw"))

    # Fill raw feature gaps
    raw_feature_cols = [
        c for c in df.columns
        if (
            c.startswith("planet_")
            or c.startswith("aspect_count_")
            or c in PLANET_SIGNAL_COLUMNS
            or c in ASPECT_STRENGTH_COLUMNS
            or c in NATAL_TARGET_STRENGTH_COLUMNS
            or c in {
                "house_activation_strength",
                "raw_astro_total_strength",
                "raw_astro_directional_signal",
                "raw_astro_event_count",
            }
        )
    ]
    df[raw_feature_cols] = df[raw_feature_cols].fillna(0)

    # Ensure numeric price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Core astro columns
    base_feature_cols = [
        "astro_bullish_score",
        "astro_bearish_score",
        "astro_reversal_score",
        "astro_volatility_score",
        "astro_compression_score",
        "astro_trend_start_score",
        "astro_trend_end_score",
        "astro_momentum_v2",
        "astro_momentum_v2_smooth",
        "astro_bullish_score_smooth",
        "astro_bearish_score_smooth",
    ]

    base_feature_cols = [c for c in base_feature_cols if c in df.columns]

    print("Creating rolling features...")
    for col in base_feature_cols:
        df = add_rolling_features(df, col)

    print("Creating price features...")
    df["btc_return_1d"] = df["price"].pct_change()
    df["btc_return_3d"] = df["price"].pct_change(3)
    df["btc_return_7d"] = df["price"].pct_change(7)
    df["btc_return_14d"] = df["price"].pct_change(14)
    df["btc_return_30d"] = df["price"].pct_change(30)

    df["btc_vol_7d"] = df["btc_return_1d"].rolling(7).std()
    df["btc_vol_14d"] = df["btc_return_1d"].rolling(14).std()
    df["btc_vol_30d"] = df["btc_return_1d"].rolling(30).std()

    # Future targets — do not use these as features
    print("Creating future targets...")
    for horizon in [3, 7, 14, 30, 60, 90]:
        df[f"future_return_{horizon}d"] = df["price"].shift(-horizon) / df["price"] - 1
        df[f"future_direction_{horizon}d"] = (df[f"future_return_{horizon}d"] > 0).astype(int)

    # Future drawdown risk
    for horizon in [7, 14, 30, 60]:
        future_min = (
            df["price"]
            .shift(-1)
            .rolling(window=horizon, min_periods=1)
            .min()
            .shift(-(horizon - 1))
        )
        df[f"future_drawdown_{horizon}d"] = future_min / df["price"] - 1

    # ML usable flag
    df["has_price"] = df["price"].notna().astype(int)

    # Remove rows without price for ML training
    ml_df = df[df["price"].notna()].copy()

    # Avoid infinite values
    ml_df = ml_df.replace([np.inf, -np.inf], np.nan)

    os.makedirs("data", exist_ok=True)
    ml_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(ml_df):,}")
    print(f"Columns: {len(ml_df.columns):,}")
    print(ml_df.tail())


if __name__ == "__main__":
    main()
