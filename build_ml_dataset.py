import os
import numpy as np
import pandas as pd

DAILY_PATH = "data/bitcoin_astro_daily_score.csv"
RAW_PATH = "data/astro_aspects_raw.csv"
OUTPUT_PATH = "data/ml_dataset.csv"


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
    score_cols = [
        "bullish",
        "bearish",
        "reversal",
        "volatility",
        "compression",
        "trend_start",
        "trend_end",
    ]

    raw["date"] = pd.to_datetime(raw["date"])

    for c in score_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    # Aggregate by date
    daily_score = raw.groupby("date")[score_cols].sum().reset_index()

    # Planet-level impact
    planet_pivot = raw.pivot_table(
        index="date",
        columns="transit_planet",
        values=score_cols,
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

    out = daily_score.merge(planet_pivot, on="date", how="left")
    out = out.merge(aspect_count, on="date", how="left")

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
    raw_feature_cols = [c for c in df.columns if c.startswith("planet_") or c.startswith("aspect_count_")]
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
