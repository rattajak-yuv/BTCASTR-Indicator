import os
import pandas as pd
import numpy as np

STABILITY_PATH = "data/feature_stability.csv"
OUTPUT_PATH = "data/selected_features.csv"

# Adjustable thresholds
MIN_MEAN_IMPORTANCE_QUANTILE = 0.50
MIN_ROBUSTNESS_QUANTILE = 0.50
MAX_NOISY_ALLOW = False

# Keep extra features that are useful across horizons
MIN_HORIZON_COVERAGE = 0.50


def main():
    print("Loading feature stability data...")

    df = pd.read_csv(STABILITY_PATH)

    if df.empty:
        raise ValueError("feature_stability.csv is empty")

    required_cols = [
        "feature",
        "mean_importance",
        "robustness_score",
        "horizon_coverage",
        "noisy_feature",
        "stable_feature",
        "high_importance_feature",
        "cross_horizon_feature",
        "dominant_horizon_type",
        "feature_group",
    ]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in feature_stability.csv: {c}")

    # Convert booleans safely
    for c in ["noisy_feature", "stable_feature", "high_importance_feature", "cross_horizon_feature"]:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])

    importance_cutoff = df["mean_importance"].quantile(MIN_MEAN_IMPORTANCE_QUANTILE)
    robustness_cutoff = df["robustness_score"].quantile(MIN_ROBUSTNESS_QUANTILE)

    print(f"Importance cutoff: {importance_cutoff:.6f}")
    print(f"Robustness cutoff: {robustness_cutoff:.6f}")

    # Core rule:
    # Keep features that are robust OR high importance OR cross-horizon,
    # but avoid noisy features unless they are cross-horizon and important.
    selected = df[
        (
            (df["mean_importance"] >= importance_cutoff)
            | (df["robustness_score"] >= robustness_cutoff)
            | (df["horizon_coverage"] >= MIN_HORIZON_COVERAGE)
            | (df["stable_feature"] == True)
            | (df["high_importance_feature"] == True)
            | (df["cross_horizon_feature"] == True)
        )
    ].copy()

    if not MAX_NOISY_ALLOW:
        selected = selected[
            (selected["noisy_feature"] == False)
            | (
                (selected["cross_horizon_feature"] == True)
                & (selected["high_importance_feature"] == True)
            )
        ].copy()

    # Always keep some core astro features if available
    always_keep_patterns = [
        "astro_momentum_v2",
        "astro_bullish_score",
        "astro_bearish_score",
        "astro_reversal_score",
        "astro_volatility_score",
        "astro_compression_score",
        "astro_trend_start_score",
        "astro_trend_end_score",
    ]

    always_keep = df[
        df["feature"].apply(
            lambda x: any(pattern in str(x) for pattern in always_keep_patterns)
        )
    ].copy()

    selected = pd.concat([selected, always_keep], ignore_index=True)
    selected = selected.drop_duplicates(subset=["feature"])

    selected = selected.sort_values(
        by=["robustness_score", "mean_importance", "horizon_coverage"],
        ascending=False,
    ).reset_index(drop=True)

    selected["selected"] = True

    os.makedirs("data", exist_ok=True)
    selected.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Total features in stability file: {len(df):,}")
    print(f"Selected features: {len(selected):,}")
    print("\nTop selected features:")
    print(
        selected[
            [
                "feature",
                "feature_group",
                "mean_importance",
                "robustness_score",
                "horizon_coverage",
                "dominant_horizon_type",
                "stable_feature",
                "high_importance_feature",
                "cross_horizon_feature",
                "noisy_feature",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
