import os
import numpy as np
import pandas as pd

IMPORTANCE_PATH = "data/ml_feature_importance.csv"
OUTPUT_PATH = "data/feature_stability.csv"

TOP_N_PER_HORIZON = 50


def classify_feature(feature):
    f = feature.lower()

    if "jupiter" in f:
        return "jupiter"

    if "saturn" in f:
        return "saturn"

    if "uranus" in f:
        return "uranus"

    if "pluto" in f:
        return "pluto"

    if "neptune" in f:
        return "neptune"

    if "moon" in f:
        return "moon"

    if "mars" in f:
        return "mars"

    if "venus" in f:
        return "venus"

    if "mercury" in f:
        return "mercury"

    if "sun" in f:
        return "sun"

    if "reversal" in f:
        return "reversal"

    if "volatility" in f:
        return "volatility"

    if "compression" in f:
        return "compression"

    if "trend_start" in f:
        return "trend_start"

    if "trend_end" in f:
        return "trend_end"

    if "bullish" in f:
        return "bullish"

    if "bearish" in f:
        return "bearish"

    if "aspect_count" in f:
        return "aspect_count"

    return "other"


def classify_horizon_type(h):
    if h <= 7:
        return "timing"

    if h <= 30:
        return "swing"

    return "macro"


def main():

    print("Loading feature importance data...")

    df = pd.read_csv(IMPORTANCE_PATH)

    if df.empty:
        raise ValueError("ml_feature_importance.csv is empty")

    required_cols = [
        "horizon",
        "feature",
        "importance"
    ]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["importance"] = pd.to_numeric(
        df["importance"],
        errors="coerce"
    )

    df = df.dropna(subset=["importance"])

    # -----------------------------------------------------
    # KEEP TOP FEATURES ONLY
    # -----------------------------------------------------

    top_rows = []

    for h in sorted(df["horizon"].unique()):

        temp = (
            df[df["horizon"] == h]
            .sort_values("importance", ascending=False)
            .head(TOP_N_PER_HORIZON)
        )

        top_rows.append(temp)

    df = pd.concat(top_rows, ignore_index=True)

    # -----------------------------------------------------
    # FEATURE STATS
    # -----------------------------------------------------

    grouped = (
        df.groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            max_importance=("importance", "max"),
            min_importance=("importance", "min"),
            horizons=("horizon", "nunique"),
            total_observations=("importance", "count"),
        )
        .reset_index()
    )

    grouped["std_importance"] = grouped["std_importance"].fillna(0)

    # Stability score
    grouped["stability_score"] = (
        grouped["mean_importance"]
        / (grouped["std_importance"] + 1e-6)
    )

    # Horizon coverage ratio
    total_horizons = df["horizon"].nunique()

    grouped["horizon_coverage"] = (
        grouped["horizons"] / total_horizons
    )

    # Composite robustness score
    grouped["robustness_score"] = (
        grouped["mean_importance"] * 0.40
        + grouped["stability_score"] * 0.35
        + grouped["horizon_coverage"] * 0.25
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    grouped["feature_group"] = (
        grouped["feature"]
        .apply(classify_feature)
    )

    # Determine dominant horizon
    dominant = (
        df.groupby(["feature", "horizon"])["importance"]
        .mean()
        .reset_index()
    )

    idx = dominant.groupby("feature")["importance"].idxmax()

    dominant = dominant.loc[idx][
        ["feature", "horizon"]
    ]

    dominant["dominant_horizon_type"] = (
        dominant["horizon"]
        .apply(classify_horizon_type)
    )

    grouped = grouped.merge(
        dominant,
        on="feature",
        how="left"
    )

    # -----------------------------------------------------
    # RANKINGS
    # -----------------------------------------------------

    grouped = grouped.sort_values(
        "robustness_score",
        ascending=False
    )

    grouped["robust_rank"] = np.arange(
        1,
        len(grouped) + 1
    )

    # -----------------------------------------------------
    # FLAGS
    # -----------------------------------------------------

    grouped["stable_feature"] = (
        grouped["stability_score"] >= grouped["stability_score"].quantile(0.75)
    )

    grouped["high_importance_feature"] = (
        grouped["mean_importance"] >= grouped["mean_importance"].quantile(0.75)
    )

    grouped["cross_horizon_feature"] = (
        grouped["horizon_coverage"] >= 0.75
    )

    grouped["noisy_feature"] = (
        grouped["std_importance"] >= grouped["std_importance"].quantile(0.75)
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    os.makedirs("data", exist_ok=True)

    grouped.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Saved: {OUTPUT_PATH}")

    print("\nTop Stable Features:")
    print(
        grouped[
            grouped["stable_feature"] == True
        ][
            [
                "feature",
                "mean_importance",
                "stability_score",
                "robustness_score",
                "dominant_horizon_type"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nTop Noisy Features:")
    print(
        grouped[
            grouped["noisy_feature"] == True
        ][
            [
                "feature",
                "mean_importance",
                "std_importance",
                "robustness_score"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
