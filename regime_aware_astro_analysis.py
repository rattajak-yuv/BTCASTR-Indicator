from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

DATASET_PATH = DATA_DIR / "ml_dataset.csv"
SELECTED_FEATURES_PATH = DATA_DIR / "selected_features.csv"
STABILITY_PATH = DATA_DIR / "feature_stability.csv"
IMPORTANCE_PATH = DATA_DIR / "ml_feature_importance.csv"
MODEL_SUMMARY_PATH = DATA_DIR / "ml_model_summary.csv"
RAW_ASPECTS_PATH = DATA_DIR / "astro_aspects_raw.csv"

PLANET_OUTPUT_PATH = DATA_DIR / "regime_planet_importance.csv"
ASPECT_OUTPUT_PATH = DATA_DIR / "regime_aspect_importance.csv"
NATAL_OUTPUT_PATH = DATA_DIR / "regime_natal_importance.csv"
SUMMARY_OUTPUT_PATH = DATA_DIR / "regime_astro_summary.md"

PLANETS = [
    "Sun",
    "Moon",
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
]

ASPECTS = [
    "conjunction",
    "trine",
    "sextile",
    "square",
    "opposition",
]

NATAL_TARGETS = [
    "Sun",
    "Moon",
    "Asc",
    "MC",
]

SCORE_CATEGORIES = [
    "bullish",
    "bearish",
    "reversal",
    "volatility",
]

REGIME_ORDER = [
    "Bull Market",
    "Bear Market",
    "Sideways",
    "High Volatility",
    "Low Volatility",
]

MIN_CORRELATION_SAMPLES = 120


def load_csv(path, required_columns=None, parse_dates=None):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path, parse_dates=parse_dates)
    if df.empty:
        raise ValueError(f"{path} is empty")

    required_columns = required_columns or []
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    return df


def normalize_name(value):
    return str(value).strip().lower()


def minmax_series(series):
    series = series.fillna(0.0).astype(float)
    if series.empty:
        return series

    min_value = series.min()
    max_value = series.max()
    if np.isclose(min_value, max_value):
        if np.isclose(max_value, 0.0):
            return pd.Series(0.0, index=series.index)
        return pd.Series(1.0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def safe_top_mean(values, top_n=3):
    filtered = sorted([float(value) for value in values if pd.notna(value)], reverse=True)
    if not filtered:
        return 0.0
    return float(np.mean(filtered[: min(top_n, len(filtered))]))


def format_list(values):
    cleaned = [str(value) for value in values if str(value).strip()]
    if not cleaned:
        return "none"
    return ", ".join(sorted(cleaned))


def render_markdown_table(df):
    headers = list(df.columns)
    rows = [headers] + [[str(row[column]) for column in headers] for _, row in df.iterrows()]
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]

    def render_row(values):
        return "| " + " | ".join(
            values[index].ljust(widths[index]) for index in range(len(values))
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    rendered = [render_row(rows[0]), separator]
    for row in rows[1:]:
        rendered.append(render_row(row))
    return "\n".join(rendered)


def parse_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def build_horizon_quality(summary_df):
    quality = summary_df.copy()
    quality["balanced_norm"] = minmax_series(quality["balanced_score"])
    quality["return_dd_norm"] = minmax_series(quality["return_drawdown_ratio"])
    quality["accuracy_norm"] = minmax_series(quality["direction_accuracy"])
    quality["horizon_quality"] = (
        quality["balanced_norm"] * 0.50
        + quality["return_dd_norm"] * 0.30
        + quality["accuracy_norm"] * 0.20
    )
    quality["horizon_quality"] = quality["horizon_quality"].clip(lower=0.05)
    return {
        int(row["horizon_days"]): float(row["horizon_quality"])
        for _, row in quality.iterrows()
    }


def component_features(dataset_columns, component_type, component_name):
    component_key = normalize_name(component_name)
    matched = []

    for column in dataset_columns:
        col = normalize_name(column)

        if component_type == "planet":
            if col.endswith(f"_{component_key}") and col.startswith("planet_"):
                matched.append(column)
            elif col == f"{component_key}_signal":
                matched.append(column)
        elif component_type == "aspect":
            if col == f"aspect_count_{component_key}" or col == f"{component_key}_strength":
                matched.append(column)
        elif component_type == "natal_target":
            if col == f"{component_key}_target_strength":
                matched.append(column)

    return sorted(set(matched))


def add_market_regimes(dataset_df):
    df = dataset_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["price_sma_200"] = pd.to_numeric(df["price"], errors="coerce").rolling(
        200, min_periods=200
    ).mean()
    df["trend_distance_200d"] = (pd.to_numeric(df["price"], errors="coerce") / df["price_sma_200"]) - 1.0

    trend_30d = pd.to_numeric(df["btc_return_30d"], errors="coerce")
    trend_distance = pd.to_numeric(df["trend_distance_200d"], errors="coerce")

    bull_mask = (trend_distance > 0.05) & (trend_30d > 0.05)
    bear_mask = (trend_distance < -0.05) & (trend_30d < -0.05)

    df["market_regime"] = np.select(
        [bull_mask, bear_mask],
        ["Bull Market", "Bear Market"],
        default="Sideways",
    )

    vol_30d = pd.to_numeric(df["btc_vol_30d"], errors="coerce")
    low_vol_threshold = float(vol_30d.quantile(0.30))
    high_vol_threshold = float(vol_30d.quantile(0.70))

    df["high_volatility_regime"] = np.where(
        vol_30d >= high_vol_threshold,
        "High Volatility",
        "",
    )
    df["low_volatility_regime"] = np.where(
        vol_30d <= low_vol_threshold,
        "Low Volatility",
        "",
    )

    metadata = {
        "bull_rule": "price > SMA200 by 5% and btc_return_30d > 5%",
        "bear_rule": "price < SMA200 by 5% and btc_return_30d < -5%",
        "sideways_rule": "all remaining rows",
        "high_vol_quantile": 0.70,
        "low_vol_quantile": 0.30,
        "high_vol_threshold": high_vol_threshold,
        "low_vol_threshold": low_vol_threshold,
    }
    return df, metadata


def regime_subsets(dataset_df):
    subsets = {
        "Bull Market": dataset_df[dataset_df["market_regime"] == "Bull Market"].copy(),
        "Bear Market": dataset_df[dataset_df["market_regime"] == "Bear Market"].copy(),
        "Sideways": dataset_df[dataset_df["market_regime"] == "Sideways"].copy(),
        "High Volatility": dataset_df[dataset_df["high_volatility_regime"] == "High Volatility"].copy(),
        "Low Volatility": dataset_df[dataset_df["low_volatility_regime"] == "Low Volatility"].copy(),
    }
    return subsets


def compute_feature_predictive_metrics(dataset_df, features, horizon_quality):
    metrics = {}
    horizon_columns = {
        horizon: f"future_return_{horizon}d"
        for horizon in horizon_quality
        if f"future_return_{horizon}d" in dataset_df.columns
    }

    for feature in features:
        if feature not in dataset_df.columns:
            metrics[feature] = {
                "predictive_score": 0.0,
                "predictive_consistency": 0.0,
                "valid_horizons": 0,
            }
            continue

        series = pd.to_numeric(dataset_df[feature], errors="coerce")
        if series.dropna().empty or np.isclose(series.dropna().std(), 0.0):
            metrics[feature] = {
                "predictive_score": 0.0,
                "predictive_consistency": 0.0,
                "valid_horizons": 0,
            }
            continue

        correlations = []
        weights = []
        for horizon, target_col in horizon_columns.items():
            target = pd.to_numeric(dataset_df[target_col], errors="coerce")
            valid = pd.concat([series, target], axis=1).dropna()
            if len(valid) < MIN_CORRELATION_SAMPLES:
                continue
            feature_std = valid.iloc[:, 0].std()
            target_std = valid.iloc[:, 1].std()
            if np.isclose(feature_std, 0.0) or np.isclose(target_std, 0.0):
                continue
            corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
            if pd.notna(corr):
                correlations.append(abs(float(corr)))
                weights.append(float(horizon_quality[horizon]))

        if not correlations:
            metrics[feature] = {
                "predictive_score": 0.0,
                "predictive_consistency": 0.0,
                "valid_horizons": 0,
            }
            continue

        metrics[feature] = {
            "predictive_score": float(np.average(correlations, weights=weights)),
            "predictive_consistency": float(1.0 / (1.0 + np.std(correlations) * 25.0)),
            "valid_horizons": int(len(correlations)),
        }

    return metrics


def raw_component_category_table(raw_df, component_type):
    if component_type == "planet":
        key_column = "transit_planet"
        components = PLANETS
    elif component_type == "aspect":
        key_column = "aspect"
        components = ASPECTS
    else:
        key_column = "target"
        components = NATAL_TARGETS

    normalized = raw_df.copy()
    normalized[key_column] = normalized[key_column].astype(str).str.strip()
    filtered = normalized[normalized[key_column].isin(components)].copy()

    if filtered.empty:
        rows = []
    else:
        rows = (
            filtered.groupby(key_column)[SCORE_CATEGORIES]
            .sum()
            .reset_index()
            .rename(columns={key_column: "component_name"})
            .to_dict("records")
        )

    present = {row["component_name"]: row for row in rows}
    normalized_rows = []
    for component in components:
        row = present.get(component, {"component_name": component})
        normalized_row = {"component_name": component}
        for category in SCORE_CATEGORIES:
            normalized_row[f"raw_{category}_total"] = float(row.get(category, 0.0))
        normalized_rows.append(normalized_row)

    return pd.DataFrame(normalized_rows)


def build_component_table(
    regime_name,
    regime_df,
    raw_regime_df,
    component_type,
    components,
    dataset_columns,
    selected_feature_names,
    stability_df,
    importance_df,
    horizon_quality,
):
    importance = importance_df.copy()
    importance["feature"] = importance["feature"].astype(str)
    importance["weighted_importance"] = importance.apply(
        lambda row: float(row["importance"]) * horizon_quality.get(int(row["horizon"]), 0.05),
        axis=1,
    )

    raw_categories = raw_component_category_table(raw_regime_df, component_type)
    feature_universe = []
    for component in components:
        feature_universe.extend(component_features(dataset_columns, component_type, component))
    feature_universe = sorted(set(feature_universe))
    predictive_metrics = compute_feature_predictive_metrics(regime_df, feature_universe, horizon_quality)

    rows = []
    for component in components:
        mapped_features = component_features(dataset_columns, component_type, component)
        selected_features = [feature for feature in mapped_features if feature in selected_feature_names]

        feature_importance_scores = []
        for feature in mapped_features:
            subset = importance[importance["feature"] == feature]
            if subset.empty:
                continue
            feature_importance_scores.append(subset["weighted_importance"].sum())

        stability_subset = stability_df[stability_df["feature"].astype(str).isin(mapped_features)].copy()
        stability_scores = pd.to_numeric(stability_subset["robustness_score"], errors="coerce").dropna().tolist()
        stability_support_scores = pd.to_numeric(
            stability_subset["stability_score"], errors="coerce"
        ).dropna().tolist()
        stable_share = (
            float(parse_bool(stability_subset["stable_feature"]).mean())
            if not stability_subset.empty
            else 0.0
        )

        predictive_scores = [
            predictive_metrics.get(feature, {}).get("predictive_score", 0.0)
            for feature in mapped_features
        ]
        predictive_consistency_scores = [
            predictive_metrics.get(feature, {}).get("predictive_consistency", 0.0)
            for feature in mapped_features
        ]
        valid_predictive_horizons = [
            predictive_metrics.get(feature, {}).get("valid_horizons", 0)
            for feature in mapped_features
        ]

        raw_row = raw_categories[raw_categories["component_name"] == component]
        raw_totals = {
            f"raw_{category}_total": float(raw_row.iloc[0][f"raw_{category}_total"])
            if not raw_row.empty
            else 0.0
            for category in SCORE_CATEGORIES
        }

        rows.append(
            {
                "regime": regime_name,
                "component_type": component_type,
                "component_name": component,
                "regime_row_count": int(len(regime_df)),
                "mapped_feature_count": int(len(mapped_features)),
                "selected_feature_count": int(len(selected_features)),
                "selected_support_ratio": float(
                    len(selected_features) / len(mapped_features)
                    if mapped_features
                    else 0.0
                ),
                "stable_feature_share": float(stable_share),
                "feature_importance_raw": float(safe_top_mean(feature_importance_scores)),
                "stability_score_raw": float(safe_top_mean(stability_scores)),
                "stability_support_raw": float(safe_top_mean(stability_support_scores)),
                "predictive_contribution_raw": float(safe_top_mean(predictive_scores)),
                "predictive_consistency_raw": float(safe_top_mean(predictive_consistency_scores)),
                "valid_predictive_horizon_count": int(max(valid_predictive_horizons) if valid_predictive_horizons else 0),
                "mapped_features": format_list(mapped_features),
                "selected_features": format_list(selected_features),
                **raw_totals,
            }
        )

    table = pd.DataFrame(rows)
    table["feature_importance_score"] = minmax_series(table["feature_importance_raw"])
    table["stability_score"] = minmax_series(table["stability_score_raw"])
    table["predictive_contribution_score"] = minmax_series(table["predictive_contribution_raw"])
    table["predictive_consistency_score"] = minmax_series(table["predictive_consistency_raw"])
    table["coverage_score"] = minmax_series(table["mapped_feature_count"])

    table["regime_importance_score"] = (
        table["predictive_contribution_score"] * 0.35
        + table["feature_importance_score"] * 0.25
        + table["stability_score"] * 0.20
        + table["selected_support_ratio"] * 0.10
        + table["coverage_score"] * 0.10
    )

    table["confidence_score"] = (
        (
            table["coverage_score"] * 0.20
            + table["selected_support_ratio"] * 0.25
            + table["stability_score"] * 0.20
            + table["predictive_contribution_score"] * 0.25
            + table["predictive_consistency_score"] * 0.10
        )
        * 100.0
    ).round(2)

    for category in SCORE_CATEGORIES:
        raw_norm = minmax_series(table[f"raw_{category}_total"])
        table[f"{category}_contribution_score"] = table["regime_importance_score"] * raw_norm

    table = table.sort_values(
        ["regime_importance_score", "confidence_score", "selected_feature_count", "mapped_feature_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def apply_dynamic_recommendations(table):
    updated = table.copy()
    component_mean = (
        updated.groupby("component_name")["regime_importance_score"]
        .mean()
        .replace(0.0, np.nan)
    )
    updated["component_global_mean_score"] = updated["component_name"].map(component_mean).fillna(0.0)
    updated["relative_regime_strength"] = updated.apply(
        lambda row: (
            row["regime_importance_score"] / row["component_global_mean_score"]
            if row["component_global_mean_score"] > 0
            else 0.0
        ),
        axis=1,
    )

    def recommend(row):
        if row["confidence_score"] < 25:
            return "insufficient evidence"
        if row["relative_regime_strength"] >= 1.20 and row["regime_importance_score"] >= 0.35:
            return "increase in this regime"
        if row["relative_regime_strength"] <= 0.80 and row["regime_importance_score"] <= row["component_global_mean_score"]:
            return "decrease in this regime"
        return "keep near global"

    updated["dynamic_weight_recommendation"] = updated.apply(recommend, axis=1)
    return updated


def save_component_table(path, table):
    table.to_csv(path, index=False)


def top_component_name(table, regime_name, score_column):
    subset = table[table["regime"] == regime_name].copy()
    subset = subset.sort_values(
        [score_column, "confidence_score", "regime_importance_score"],
        ascending=[False, False, False],
    )
    if subset.empty or np.isclose(float(subset.iloc[0][score_column]), 0.0):
        return "none"
    return str(subset.iloc[0]["component_name"])


def build_regime_counts_table(regime_dfs):
    rows = []
    for regime_name in REGIME_ORDER:
        subset = regime_dfs[regime_name]
        rows.append(
            {
                "regime": regime_name,
                "rows": int(len(subset)),
                "start": subset["date"].min().date() if not subset.empty else "",
                "end": subset["date"].max().date() if not subset.empty else "",
            }
        )
    return pd.DataFrame(rows)


def build_comparison_table(table, left_regime, right_regime, top_n=5):
    pivot = table.pivot_table(
        index="component_name",
        columns="regime",
        values="regime_importance_score",
        aggfunc="first",
        fill_value=0.0,
    ).reset_index()
    if left_regime not in pivot.columns:
        pivot[left_regime] = 0.0
    if right_regime not in pivot.columns:
        pivot[right_regime] = 0.0
    pivot["delta"] = pivot[left_regime] - pivot[right_regime]
    display = pivot.sort_values("delta", ascending=False).head(top_n).copy()
    display = display[["component_name", left_regime, right_regime, "delta"]]
    display[left_regime] = display[left_regime].map(lambda value: f"{value:.4f}")
    display[right_regime] = display[right_regime].map(lambda value: f"{value:.4f}")
    display["delta"] = display["delta"].map(lambda value: f"{value:.4f}")
    return display.rename(columns={"component_name": "component"})


def build_recommendation_lines(*tables):
    combined = pd.concat(tables, ignore_index=True)
    lines = []
    for regime_name in REGIME_ORDER:
        subset = combined[
            (combined["regime"] == regime_name)
            & (combined["dynamic_weight_recommendation"] == "increase in this regime")
        ].copy()
        subset = subset.sort_values(
            ["regime_importance_score", "confidence_score"],
            ascending=[False, False],
        ).head(5)

        if subset.empty:
            lines.append(f"- {regime_name}: keep weights near global until stronger evidence appears.")
            continue

        formatted = ", ".join(
            f"{row.component_type}:{row.component_name}"
            for row in subset.itertuples()
        )
        lines.append(f"- {regime_name}: increase emphasis on {formatted}.")
    return lines


def write_summary(
    metadata,
    regime_dfs,
    planet_table,
    aspect_table,
    natal_table,
):
    counts_table = build_regime_counts_table(regime_dfs)
    counts_display = counts_table.copy()
    counts_display["rows"] = counts_display["rows"].astype(int)

    strongest_rows = []
    for regime_name in REGIME_ORDER:
        strongest_rows.append(
            {
                "regime": regime_name,
                "strongest_bullish_planet": top_component_name(planet_table, regime_name, "bullish_contribution_score"),
                "strongest_bearish_planet": top_component_name(planet_table, regime_name, "bearish_contribution_score"),
                "strongest_reversal_planet": top_component_name(planet_table, regime_name, "reversal_contribution_score"),
                "strongest_volatility_planet": top_component_name(planet_table, regime_name, "volatility_contribution_score"),
            }
        )
    strongest_table = pd.DataFrame(strongest_rows)

    bull_bear_planets = build_comparison_table(planet_table, "Bull Market", "Bear Market")
    bull_bear_aspects = build_comparison_table(aspect_table, "Bull Market", "Bear Market")
    bull_bear_natal = build_comparison_table(natal_table, "Bull Market", "Bear Market")

    high_low_planets = build_comparison_table(planet_table, "High Volatility", "Low Volatility")
    high_low_aspects = build_comparison_table(aspect_table, "High Volatility", "Low Volatility")
    high_low_natal = build_comparison_table(natal_table, "High Volatility", "Low Volatility")

    summary_lines = [
        "# Regime-Aware Astro Summary",
        "",
        "Analysis only. No production config or engine weights were changed.",
        "",
        "## Regime Methodology",
        f"- Bull Market: {metadata['bull_rule']}",
        f"- Bear Market: {metadata['bear_rule']}",
        f"- Sideways: {metadata['sideways_rule']}",
        f"- High Volatility: btc_vol_30d >= {metadata['high_vol_threshold']:.6f} (70th percentile)",
        f"- Low Volatility: btc_vol_30d <= {metadata['low_vol_threshold']:.6f} (30th percentile)",
        "",
        "## Regime Coverage",
        render_markdown_table(counts_display),
        "",
        "## Strongest Planet Signals By Regime",
        render_markdown_table(strongest_table),
        "",
        "## Bull vs Bear",
        "### Planets",
        render_markdown_table(bull_bear_planets),
        "",
        "### Aspects",
        render_markdown_table(bull_bear_aspects),
        "",
        "### Natal Targets",
        render_markdown_table(bull_bear_natal),
        "",
        "## High Volatility vs Low Volatility",
        "### Planets",
        render_markdown_table(high_low_planets),
        "",
        "### Aspects",
        render_markdown_table(high_low_aspects),
        "",
        "### Natal Targets",
        render_markdown_table(high_low_natal),
        "",
        "## Dynamic Weight Recommendations",
        *build_recommendation_lines(planet_table, aspect_table, natal_table),
        "",
        "## Interpretation",
        "- The regime scores blend current ML feature importance, feature stability, selected-feature support, and regime-specific predictive contribution from future-return correlations.",
        "- Use these results to design regime-specific weights in Astro Engine v4 instead of applying one global weight profile to every market environment.",
    ]

    SUMMARY_OUTPUT_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main():
    dataset_df = load_csv(DATASET_PATH, required_columns=["date", "price", "btc_return_30d", "btc_vol_30d"], parse_dates=["date"])
    selected_df = load_csv(SELECTED_FEATURES_PATH, required_columns=["feature"])
    stability_df = load_csv(
        STABILITY_PATH,
        required_columns=["feature", "robustness_score", "stability_score", "stable_feature"],
    )
    importance_df = load_csv(
        IMPORTANCE_PATH,
        required_columns=["horizon", "feature", "importance"],
    )
    summary_df = load_csv(
        MODEL_SUMMARY_PATH,
        required_columns=["horizon_days", "balanced_score", "return_drawdown_ratio", "direction_accuracy"],
    )
    raw_df = load_csv(
        RAW_ASPECTS_PATH,
        required_columns=["date", "transit_planet", "target", "aspect", *SCORE_CATEGORIES],
        parse_dates=["date"],
    )

    dataset_df, metadata = add_market_regimes(dataset_df)
    regime_dfs = regime_subsets(dataset_df)

    raw_df["date"] = pd.to_datetime(raw_df["date"])
    dataset_columns = list(dataset_df.columns)
    selected_feature_names = set(selected_df["feature"].dropna().astype(str).tolist())
    horizon_quality = build_horizon_quality(summary_df)

    planet_tables = []
    aspect_tables = []
    natal_tables = []

    for regime_name in REGIME_ORDER:
        regime_df = regime_dfs[regime_name]
        regime_dates = regime_df["date"].dt.normalize().dropna().unique()
        raw_regime_df = raw_df[raw_df["date"].dt.normalize().isin(regime_dates)].copy()

        planet_tables.append(
            build_component_table(
                regime_name,
                regime_df,
                raw_regime_df,
                "planet",
                PLANETS,
                dataset_columns,
                selected_feature_names,
                stability_df,
                importance_df,
                horizon_quality,
            )
        )
        aspect_tables.append(
            build_component_table(
                regime_name,
                regime_df,
                raw_regime_df,
                "aspect",
                ASPECTS,
                dataset_columns,
                selected_feature_names,
                stability_df,
                importance_df,
                horizon_quality,
            )
        )
        natal_tables.append(
            build_component_table(
                regime_name,
                regime_df,
                raw_regime_df,
                "natal_target",
                NATAL_TARGETS,
                dataset_columns,
                selected_feature_names,
                stability_df,
                importance_df,
                horizon_quality,
            )
        )

    planet_table = apply_dynamic_recommendations(pd.concat(planet_tables, ignore_index=True))
    aspect_table = apply_dynamic_recommendations(pd.concat(aspect_tables, ignore_index=True))
    natal_table = apply_dynamic_recommendations(pd.concat(natal_tables, ignore_index=True))

    save_component_table(PLANET_OUTPUT_PATH, planet_table)
    save_component_table(ASPECT_OUTPUT_PATH, aspect_table)
    save_component_table(NATAL_OUTPUT_PATH, natal_table)
    write_summary(metadata, regime_dfs, planet_table, aspect_table, natal_table)

    print(f"Wrote {PLANET_OUTPUT_PATH}")
    print(f"Wrote {ASPECT_OUTPUT_PATH}")
    print(f"Wrote {NATAL_OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
