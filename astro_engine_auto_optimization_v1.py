import os
import numpy as np
import pandas as pd

DATASET_PATH = "data/ml_dataset.csv"
SELECTED_FEATURES_PATH = "data/selected_features.csv"
IMPORTANCE_PATH = "data/ml_feature_importance.csv"

ASTRO_FEATURE_IMPORTANCE_PATH = "data/astro_feature_importance.csv"
PLANET_IMPORTANCE_PATH = "data/planet_importance.csv"
ASPECT_IMPORTANCE_PATH = "data/aspect_importance.csv"
NATAL_TARGET_IMPORTANCE_PATH = "data/natal_target_importance.csv"
SCORE_CATEGORY_IMPORTANCE_PATH = "data/score_category_importance.csv"
RECOMMENDATIONS_PATH = "data/astro_optimization_recommendations.csv"

PLANETS = [
    "jupiter",
    "saturn",
    "uranus",
    "pluto",
    "neptune",
    "moon",
    "mars",
    "venus",
    "mercury",
    "sun",
]

ASPECTS = [
    "conjunction",
    "opposition",
    "sextile",
    "square",
    "trine",
    "house_position",
]

NATAL_TARGETS = [
    "trend_start",
    "trend_end",
    "bullish",
    "bearish",
    "reversal",
    "volatility",
    "compression",
    "momentum",
]


def load_csv(path, required_columns):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    return df


def is_astro_feature(feature_name):
    feature = str(feature_name).lower()
    return (
        feature.startswith("astro_")
        or feature.startswith("planet_")
        or feature.startswith("aspect_count_")
    )


def classify_planet(feature_name):
    feature = str(feature_name).lower()
    for planet in PLANETS:
        if planet in feature:
            return planet
    return "composite"


def classify_aspect(feature_name):
    feature = str(feature_name).lower()
    for aspect in ASPECTS:
        if aspect in feature:
            return aspect
    return "none"


def classify_natal_target(feature_name):
    feature = str(feature_name).lower()

    if "trend_start" in feature:
        return "trend_start"
    if "trend_end" in feature:
        return "trend_end"
    if "bullish" in feature:
        return "bullish"
    if "bearish" in feature:
        return "bearish"
    if "reversal" in feature:
        return "reversal"
    if "volatility" in feature:
        return "volatility"
    if "compression" in feature:
        return "compression"
    if "momentum_v2" in feature or "momentum" in feature:
        return "momentum"

    if feature.startswith("aspect_count_"):
        return "aspect_count"

    return "other"


def classify_house(feature_name):
    feature = str(feature_name).lower()
    if "house_position" in feature:
        return "house_position"
    return "none"


def classify_score_category(feature_name):
    feature = str(feature_name).lower()

    if feature.startswith("planet_"):
        return "planet_component"
    if feature.startswith("aspect_count_"):
        return "aspect_count"
    if "_smooth_" in feature or feature.endswith("_smooth"):
        return "smoothed_score"
    if "_ema_" in feature:
        return "ema_transform"
    if "_sma_" in feature:
        return "sma_transform"
    if "_roll_max_" in feature:
        return "rolling_max"
    if "_roll_min_" in feature:
        return "rolling_min"
    if "_chg_" in feature:
        return "change_transform"
    if feature.startswith("astro_"):
        return "base_score"
    return "other"


def build_feature_table(importance_df, selected_features, dataset_columns):
    astro = importance_df[importance_df["feature"].apply(is_astro_feature)].copy()
    if astro.empty:
        raise ValueError("No astro features found in ml_feature_importance.csv")

    astro["feature"] = astro["feature"].astype(str)
    astro["selected_feature"] = astro["feature"].isin(selected_features)
    astro["exists_in_dataset"] = astro["feature"].isin(dataset_columns)
    astro["planet"] = astro["feature"].apply(classify_planet)
    astro["aspect"] = astro["feature"].apply(classify_aspect)
    astro["natal_target"] = astro["feature"].apply(classify_natal_target)
    astro["house"] = astro["feature"].apply(classify_house)
    astro["score_category"] = astro["feature"].apply(classify_score_category)

    feature_table = (
        astro.groupby(
            [
                "feature",
                "selected_feature",
                "exists_in_dataset",
                "planet",
                "aspect",
                "natal_target",
                "house",
                "score_category",
            ]
        )
        .agg(
            feature_set=("feature_set", "first"),
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            max_importance=("importance", "max"),
            min_importance=("importance", "min"),
            total_importance=("importance", "sum"),
            horizons=("horizon", "nunique"),
            observations=("importance", "count"),
        )
        .reset_index()
    )

    feature_table["std_importance"] = feature_table["std_importance"].fillna(0)
    feature_table["importance_share"] = (
        feature_table["total_importance"] / feature_table["total_importance"].sum()
    )
    feature_table = feature_table.sort_values(
        ["total_importance", "mean_importance", "horizons"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    feature_table["astro_feature_rank"] = np.arange(1, len(feature_table) + 1)

    return feature_table


def aggregate_component_importance(feature_table, group_column, expected_values=None):
    grouped = (
        feature_table.groupby(group_column)
        .agg(
            total_importance=("total_importance", "sum"),
            mean_importance=("mean_importance", "mean"),
            median_importance=("mean_importance", "median"),
            max_feature_importance=("max_importance", "max"),
            feature_count=("feature", "count"),
            selected_feature_count=("selected_feature", "sum"),
            total_horizon_coverage=("horizons", "sum"),
        )
        .reset_index()
    )

    if expected_values is not None:
        expected_df = pd.DataFrame({group_column: expected_values})
        grouped = expected_df.merge(grouped, on=group_column, how="left")
        numeric_columns = [
            "total_importance",
            "mean_importance",
            "median_importance",
            "max_feature_importance",
            "feature_count",
            "selected_feature_count",
            "total_horizon_coverage",
        ]
        grouped[numeric_columns] = grouped[numeric_columns].fillna(0)

    grouped["selected_feature_ratio"] = (
        grouped["selected_feature_count"] / grouped["feature_count"]
    ).fillna(0)
    grouped["importance_share"] = (
        grouped["total_importance"] / grouped["total_importance"].sum()
    ).fillna(0)
    grouped = grouped.sort_values(
        ["total_importance", "mean_importance", "feature_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped["feature_count"] = grouped["feature_count"].astype(int)
    grouped["selected_feature_count"] = grouped["selected_feature_count"].astype(int)
    grouped["total_horizon_coverage"] = grouped["total_horizon_coverage"].astype(int)

    return grouped


def recommend_component_weights(component_df, component_type, name_column):
    filtered = component_df[~component_df[name_column].isin(["none", "composite", "other"])].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "component_type",
                "component_name",
                "rank",
                "total_importance",
                "importance_share",
                "feature_count",
                "selected_feature_count",
                "selected_feature_ratio",
                "recommendation",
                "rationale",
            ]
        )

    filtered = filtered.reset_index(drop=True)
    filtered["peer_rank"] = np.arange(1, len(filtered) + 1)

    high_cutoff = filtered["total_importance"].quantile(0.75)
    low_cutoff = filtered["total_importance"].quantile(0.25)
    top_rank_cutoff = max(2, int(np.ceil(len(filtered) * 0.25)))

    recommendations = []

    for _, row in filtered.iterrows():
        component_name = row[name_column]
        selected_ratio = row["selected_feature_ratio"]
        importance_share = row["importance_share"]
        total_importance = row["total_importance"]
        rank = int(row["peer_rank"])

        if row["feature_count"] == 0:
            recommendation = "Remove"
            rationale = (
                f"{component_name} has no selected astro features contributing importance in the current model run."
            )
        elif total_importance >= high_cutoff or rank <= top_rank_cutoff:
            recommendation = "Increase weight"
            rationale = (
                f"{component_name} ranks #{rank} with {importance_share:.2%} importance share "
                f"and {row['selected_feature_count']}/{row['feature_count']} selected features."
            )
        elif total_importance <= low_cutoff and selected_ratio < 0.5:
            recommendation = "Remove"
            rationale = (
                f"{component_name} is in the lower-importance tier with only {importance_share:.2%} share "
                f"and low selected-feature support ({row['selected_feature_count']}/{row['feature_count']})."
            )
        else:
            recommendation = "Decrease weight"
            rationale = (
                f"{component_name} contributes some signal but sits below the top tier at rank #{rank} "
                f"with {importance_share:.2%} share."
            )

        recommendations.append(
            {
                "component_type": component_type,
                "component_name": component_name,
                "rank": rank,
                "total_importance": total_importance,
                "importance_share": importance_share,
                "feature_count": row["feature_count"],
                "selected_feature_count": row["selected_feature_count"],
                "selected_feature_ratio": selected_ratio,
                "recommendation": recommendation,
                "rationale": rationale,
            }
        )

    return pd.DataFrame(recommendations)


def main():
    print("Loading dataset columns, selected features, and model importance outputs...")

    dataset_preview = load_csv(DATASET_PATH, ["date"])
    dataset_columns = set(dataset_preview.columns)

    selected_df = load_csv(SELECTED_FEATURES_PATH, ["feature"])
    selected_features = set(selected_df["feature"].dropna().astype(str))

    importance_df = load_csv(IMPORTANCE_PATH, ["feature", "importance", "horizon"])
    importance_df["importance"] = pd.to_numeric(importance_df["importance"], errors="coerce")
    importance_df = importance_df.dropna(subset=["importance"])
    if importance_df.empty:
        raise ValueError("ml_feature_importance.csv has no usable numeric importance values")

    if "feature_set" not in importance_df.columns:
        importance_df["feature_set"] = "unknown"

    feature_table = build_feature_table(
        importance_df=importance_df,
        selected_features=selected_features,
        dataset_columns=dataset_columns,
    )

    planet_importance = aggregate_component_importance(
        feature_table,
        "planet",
        expected_values=["composite"] + PLANETS,
    )
    aspect_importance = aggregate_component_importance(
        feature_table,
        "aspect",
        expected_values=["none"] + ASPECTS,
    )
    natal_target_importance = aggregate_component_importance(
        feature_table,
        "natal_target",
        expected_values=NATAL_TARGETS + ["aspect_count", "other"],
    )
    score_category_importance = aggregate_component_importance(
        feature_table,
        "score_category",
    )

    recommendations = pd.concat(
        [
            recommend_component_weights(planet_importance, "planet", "planet"),
            recommend_component_weights(aspect_importance, "aspect", "aspect"),
        ],
        ignore_index=True,
    )

    os.makedirs("data", exist_ok=True)
    feature_table.to_csv(ASTRO_FEATURE_IMPORTANCE_PATH, index=False)
    planet_importance.to_csv(PLANET_IMPORTANCE_PATH, index=False)
    aspect_importance.to_csv(ASPECT_IMPORTANCE_PATH, index=False)
    natal_target_importance.to_csv(NATAL_TARGET_IMPORTANCE_PATH, index=False)
    score_category_importance.to_csv(SCORE_CATEGORY_IMPORTANCE_PATH, index=False)
    recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)

    print(f"Saved: {ASTRO_FEATURE_IMPORTANCE_PATH}")
    print(f"Saved: {PLANET_IMPORTANCE_PATH}")
    print(f"Saved: {ASPECT_IMPORTANCE_PATH}")
    print(f"Saved: {NATAL_TARGET_IMPORTANCE_PATH}")
    print(f"Saved: {SCORE_CATEGORY_IMPORTANCE_PATH}")
    print(f"Saved: {RECOMMENDATIONS_PATH}")

    print("\nTop astro features:")
    print(
        feature_table[
            [
                "astro_feature_rank",
                "feature",
                "planet",
                "aspect",
                "natal_target",
                "score_category",
                "mean_importance",
                "total_importance",
                "importance_share",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nPlanet importance:")
    print(planet_importance.head(12).to_string(index=False))

    print("\nAspect importance:")
    print(aspect_importance.head(12).to_string(index=False))

    print("\nOptimization recommendations:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
