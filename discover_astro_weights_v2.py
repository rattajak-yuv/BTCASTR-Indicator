import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
CONFIG_PATH = Path("astro_model_config.json")

DATASET_PATH = DATA_DIR / "ml_dataset.csv"
RAW_ASPECTS_PATH = DATA_DIR / "astro_aspects_raw.csv"
SELECTED_FEATURES_PATH = DATA_DIR / "selected_features.csv"
FEATURE_STABILITY_PATH = DATA_DIR / "feature_stability.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "ml_feature_importance.csv"
MODEL_SUMMARY_PATH = DATA_DIR / "ml_model_summary.csv"
RAW_RECOVERY_SUMMARY_PATH = DATA_DIR / "raw_astro_recovery_summary.csv"

PLANET_OUTPUT_PATH = DATA_DIR / "discovered_planet_weights.csv"
ASPECT_OUTPUT_PATH = DATA_DIR / "discovered_aspect_weights.csv"
NATAL_OUTPUT_PATH = DATA_DIR / "discovered_natal_target_weights.csv"
SUMMARY_OUTPUT_PATH = DATA_DIR / "discovered_astro_weights_summary.md"

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


def load_csv(path, required_columns=None):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
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
    if np.isclose(max_value, min_value):
        if np.isclose(max_value, 0.0):
            return pd.Series(0.0, index=series.index)
        return pd.Series(1.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def safe_mean(values):
    values = [float(value) for value in values if pd.notna(value)]
    if not values:
        return 0.0
    return float(np.mean(values))


def top_mean(values, top_n=3):
    filtered = sorted([float(value) for value in values if pd.notna(value)], reverse=True)
    if not filtered:
        return 0.0
    return float(np.mean(filtered[: min(top_n, len(filtered))]))


def format_list(values):
    if not values:
        return "none"
    return ", ".join(values)


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


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def build_current_planet_weights(config):
    aspect_weights = {
        name: float(values["weight"])
        for name, values in config["aspects"].items()
    }
    target_weights = {
        name: float(value)
        for name, value in config["target_weights"].items()
    }

    weights = {}
    for planet in PLANETS:
        total = 0.0
        for rule in config["rules"]:
            if rule["planet"] != planet:
                continue

            rule_score_mass = sum(abs(float(score)) for score in rule["scores"].values())
            aspect_mass = safe_mean(
                aspect_weights.get(aspect_name, 0.0)
                for aspect_name in rule["aspects"]
            )
            target_mass = safe_mean(
                target_weights.get(target_name, 1.0)
                for target_name in rule["targets"]
            )
            total += (
                rule_score_mass
                * max(len(rule["aspects"]), 1)
                * max(len(rule["targets"]), 1)
                * aspect_mass
                * target_mass
            )

        weights[planet] = total

    return weights


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

    filtered = raw_df[raw_df[key_column].astype(str).isin(components)].copy()
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
        normalized = {"component_name": component}
        for category in SCORE_CATEGORIES:
            normalized[category] = float(row.get(category, 0.0))
        normalized_rows.append(normalized)

    return pd.DataFrame(normalized_rows)


def compute_feature_predictive_metrics(dataset_df, features, horizon_quality):
    metrics = {}
    horizon_columns = {
        horizon: f"future_return_{horizon}d"
        for horizon in horizon_quality
        if f"future_return_{horizon}d" in dataset_df.columns
    }

    for feature in features:
        if feature not in dataset_df.columns:
            metrics[feature] = {"predictive_score": 0.0, "predictive_consistency": 0.0}
            continue

        series = pd.to_numeric(dataset_df[feature], errors="coerce")
        if series.dropna().empty or np.isclose(series.dropna().std(), 0.0):
            metrics[feature] = {"predictive_score": 0.0, "predictive_consistency": 0.0}
            continue

        correlations = []
        weights = []
        for horizon, target_col in horizon_columns.items():
            target = pd.to_numeric(dataset_df[target_col], errors="coerce")
            corr = series.corr(target)
            if pd.notna(corr):
                correlations.append(abs(float(corr)))
                weights.append(float(horizon_quality[horizon]))

        if not correlations:
            metrics[feature] = {"predictive_score": 0.0, "predictive_consistency": 0.0}
            continue

        predictive_score = float(np.average(correlations, weights=weights))
        predictive_consistency = float(1.0 / (1.0 + np.std(correlations) * 25.0))
        metrics[feature] = {
            "predictive_score": predictive_score,
            "predictive_consistency": predictive_consistency,
        }

    return metrics


def build_component_table(
    component_type,
    components,
    current_weights,
    current_weight_basis,
    dataset_df,
    selected_feature_names,
    stability_df,
    importance_df,
    summary_df,
    raw_df,
):
    dataset_columns = list(dataset_df.columns)
    horizon_quality = build_horizon_quality(summary_df)
    importance = importance_df.copy()
    importance["feature"] = importance["feature"].astype(str)
    importance["weighted_importance"] = importance.apply(
        lambda row: float(row["importance"]) * horizon_quality.get(int(row["horizon"]), 0.05),
        axis=1,
    )

    raw_categories = raw_component_category_table(raw_df, component_type)
    feature_universe = []
    for component in components:
        feature_universe.extend(component_features(dataset_columns, component_type, component))
    feature_universe = sorted(set(feature_universe))
    predictive_metrics = compute_feature_predictive_metrics(dataset_df, feature_universe, horizon_quality)

    rows = []
    for component in components:
        mapped_features = component_features(dataset_columns, component_type, component)
        selected_features = [
            feature for feature in mapped_features if feature in selected_feature_names
        ]

        feature_importance_scores = []
        for feature in mapped_features:
            subset = importance[importance["feature"] == feature]
            if subset.empty:
                continue
            feature_importance_scores.append(subset["weighted_importance"].sum())

        stability_subset = stability_df[stability_df["feature"].astype(str).isin(mapped_features)].copy()
        stability_scores = stability_subset["robustness_score"].tolist()
        stability_support_scores = stability_subset["stability_score"].tolist()
        selected_support_ratio = (
            len(selected_features) / len(mapped_features)
            if mapped_features
            else 0.0
        )
        stable_share = (
            float(stability_subset["stable_feature"].astype(bool).mean())
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

        raw_category_row = raw_categories[raw_categories["component_name"] == component]
        category_totals = {
            category: float(raw_category_row.iloc[0][category]) if not raw_category_row.empty else 0.0
            for category in SCORE_CATEGORIES
        }

        rows.append(
            {
                "component_type": component_type,
                "component_name": component,
                "current_weight": float(current_weights.get(component, 0.0)),
                "current_weight_basis": current_weight_basis,
                "mapped_feature_count": int(len(mapped_features)),
                "selected_feature_count": int(len(selected_features)),
                "selected_support_ratio": float(selected_support_ratio),
                "stable_feature_share": float(stable_share),
                "importance_score_raw": float(top_mean(feature_importance_scores)),
                "stability_score_raw": float(top_mean(stability_scores)),
                "stability_support_raw": float(top_mean(stability_support_scores)),
                "predictive_contribution_raw": float(top_mean(predictive_scores)),
                "predictive_consistency_raw": float(top_mean(predictive_consistency_scores)),
                "mapped_features": format_list(mapped_features),
                "selected_features": format_list(selected_features),
                **category_totals,
            }
        )

    table = pd.DataFrame(rows)
    table["importance_score"] = minmax_series(table["importance_score_raw"])
    table["stability_score"] = minmax_series(table["stability_score_raw"])
    table["predictive_contribution_score"] = minmax_series(table["predictive_contribution_raw"])
    table["coverage_score"] = minmax_series(table["mapped_feature_count"])
    table["predictive_consistency_score"] = minmax_series(table["predictive_consistency_raw"])

    table["evidence_score"] = (
        table["importance_score"] * 0.45
        + table["stability_score"] * 0.25
        + table["predictive_contribution_score"] * 0.30
    )

    evidence_total = float(table["evidence_score"].sum())
    current_total = float(table["current_weight"].sum())

    if evidence_total <= 0:
        if len(table) == 0:
            table["discovered_weight"] = []
        else:
            table["discovered_weight"] = current_total / len(table)
    else:
        table["discovered_weight"] = (
            table["evidence_score"] / evidence_total * current_total
        )

    table["confidence_score"] = (
        (
            table["coverage_score"] * 0.25
            + table["selected_support_ratio"] * 0.25
            + table["stability_score"] * 0.20
            + table["predictive_contribution_score"] * 0.20
            + table["predictive_consistency_score"] * 0.10
        )
        * 100.0
    ).round(2)

    table["discovered_weight_delta"] = table["discovered_weight"] - table["current_weight"]
    table["discovered_weight_ratio"] = table.apply(
        lambda row: (
            row["discovered_weight"] / row["current_weight"]
            if row["current_weight"] > 0
            else np.nan
        ),
        axis=1,
    )

    for category in SCORE_CATEGORIES:
        norm = minmax_series(table[category])
        table[f"{category}_contribution_score"] = table["evidence_score"] * norm

    def recommend(row):
        if row["current_weight"] <= 0 and row["discovered_weight"] > 0.05 and row["confidence_score"] >= 35:
            return "add weight"
        if row["confidence_score"] < 25 and row["discovered_weight"] <= 0.05:
            return "remove"
        if row["current_weight"] > 0:
            ratio = row["discovered_weight"] / row["current_weight"]
            if ratio <= 0.40:
                return "remove"
            if ratio <= 0.75:
                return "decrease weight"
            if ratio >= 1.25:
                return "increase weight"
        return "keep near current"

    table["recommendation"] = table.apply(recommend, axis=1)
    table = table.sort_values(
        ["discovered_weight", "confidence_score", "evidence_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)

    ordered_columns = [
        "component_type",
        "component_name",
        "current_weight",
        "current_weight_basis",
        "discovered_weight",
        "discovered_weight_delta",
        "discovered_weight_ratio",
        "confidence_score",
        "recommendation",
        "rank",
        "mapped_feature_count",
        "selected_feature_count",
        "selected_support_ratio",
        "importance_score",
        "stability_score",
        "predictive_contribution_score",
        "predictive_consistency_score",
        "evidence_score",
        "bullish_contribution_score",
        "bearish_contribution_score",
        "reversal_contribution_score",
        "volatility_contribution_score",
        "mapped_features",
        "selected_features",
    ]
    return table[ordered_columns]


def summarize_rankings(planet_df, aspect_df, natal_df):
    combined = pd.concat([planet_df, aspect_df, natal_df], ignore_index=True)
    rankings = {}
    for category in SCORE_CATEGORIES:
        score_column = f"{category}_contribution_score"
        top = (
            combined.sort_values([score_column, "confidence_score"], ascending=[False, False])
            .head(5)[["component_type", "component_name", score_column, "recommendation"]]
            .copy()
        )
        top[score_column] = top[score_column].map(lambda value: f"{value:.4f}")
        rankings[category] = top
    return rankings


def build_summary_markdown(
    planet_df,
    aspect_df,
    natal_df,
    rankings,
    raw_recovery_summary,
):
    def compact_view(df):
        out = df[[
            "component_name",
            "current_weight",
            "discovered_weight",
            "confidence_score",
            "recommendation",
        ]].copy()
        out["current_weight"] = out["current_weight"].map(lambda value: f"{value:.4f}")
        out["discovered_weight"] = out["discovered_weight"].map(lambda value: f"{value:.4f}")
        out["confidence_score"] = out["confidence_score"].map(lambda value: f"{value:.2f}")
        return out

    recovery_after = raw_recovery_summary[raw_recovery_summary["stage"] == "after_recovery"]
    if recovery_after.empty:
        recovery_line = "Raw recovery summary unavailable."
    else:
        row = recovery_after.iloc[0]
        recovery_line = (
            f"Post-recovery selected raw aspect features: {int(row['selected_raw_aspect_features'])}, "
            f"selected planet features: {int(row['selected_planet_features'])}, "
            f"selected natal-target features: {int(row['selected_natal_target_features'])}."
        )

    lines = [
        "# Astro Auto-Optimization v2",
        "",
        "Analysis only. No engine weights were changed.",
        "",
        "## Method",
        "- Discovered weights combine selected-feature ML importance, cross-horizon stability, and direct predictive contribution from `ml_dataset.csv` future-return correlations.",
        "- Aspect current weights come from active `astro_model_config.json` aspect weights.",
        "- Natal-target current weights come from active `astro_model_config.json` target weights.",
        "- Planet current weights use an active rule-mass proxy derived from the live config's rule scores, target coverage, and aspect coverage because the engine does not currently expose a single explicit planet multiplier.",
        f"- {recovery_line}",
        "",
        "## Discovered Planet Weights",
        render_markdown_table(compact_view(planet_df)),
        "",
        "## Discovered Aspect Weights",
        render_markdown_table(compact_view(aspect_df)),
        "",
        "## Discovered Natal-Target Weights",
        render_markdown_table(compact_view(natal_df)),
        "",
        "## Strongest Bullish Contributors",
        render_markdown_table(rankings["bullish"]),
        "",
        "## Strongest Bearish Contributors",
        render_markdown_table(rankings["bearish"]),
        "",
        "## Strongest Reversal Contributors",
        render_markdown_table(rankings["reversal"]),
        "",
        "## Strongest Volatility Contributors",
        render_markdown_table(rankings["volatility"]),
        "",
        "## Notes",
        "- Components with low confidence but non-zero discovered weights should be treated as candidates for manual review rather than immediate automation.",
        "- Sun and Venus currently have low-confidence evidence because the live engine config defines orbital settings for them but no active rules, so their compact recovery features mainly act as placeholders until explicit rule coverage exists.",
    ]
    return "\n".join(lines) + "\n"


def main():
    config = load_config()
    dataset = load_csv(DATASET_PATH)
    raw_aspects = load_csv(RAW_ASPECTS_PATH)
    selected = load_csv(SELECTED_FEATURES_PATH, ["feature"])
    feature_stability = load_csv(FEATURE_STABILITY_PATH, ["feature", "robustness_score", "stability_score", "stable_feature"])
    feature_importance = load_csv(FEATURE_IMPORTANCE_PATH, ["horizon", "feature", "importance"])
    model_summary = load_csv(MODEL_SUMMARY_PATH, ["horizon_days", "balanced_score", "return_drawdown_ratio", "direction_accuracy"])
    raw_recovery_summary = load_csv(RAW_RECOVERY_SUMMARY_PATH, ["stage"])

    selected_feature_names = set(selected["feature"].dropna().astype(str))

    aspect_current_weights = {
        aspect: float(config["aspects"][aspect]["weight"])
        for aspect in ASPECTS
    }
    natal_current_weights = {
        target: float(config["target_weights"].get(target, 1.0))
        for target in NATAL_TARGETS
    }
    planet_current_weights = build_current_planet_weights(config)

    planet_df = build_component_table(
        component_type="planet",
        components=PLANETS,
        current_weights=planet_current_weights,
        current_weight_basis="active_rule_mass_proxy",
        dataset_df=dataset,
        selected_feature_names=selected_feature_names,
        stability_df=feature_stability,
        importance_df=feature_importance,
        summary_df=model_summary,
        raw_df=raw_aspects,
    )
    aspect_df = build_component_table(
        component_type="aspect",
        components=ASPECTS,
        current_weights=aspect_current_weights,
        current_weight_basis="active_config_weight",
        dataset_df=dataset,
        selected_feature_names=selected_feature_names,
        stability_df=feature_stability,
        importance_df=feature_importance,
        summary_df=model_summary,
        raw_df=raw_aspects,
    )
    natal_df = build_component_table(
        component_type="natal_target",
        components=NATAL_TARGETS,
        current_weights=natal_current_weights,
        current_weight_basis="active_config_weight",
        dataset_df=dataset,
        selected_feature_names=selected_feature_names,
        stability_df=feature_stability,
        importance_df=feature_importance,
        summary_df=model_summary,
        raw_df=raw_aspects,
    )

    rankings = summarize_rankings(planet_df, aspect_df, natal_df)
    markdown = build_summary_markdown(
        planet_df=planet_df,
        aspect_df=aspect_df,
        natal_df=natal_df,
        rankings=rankings,
        raw_recovery_summary=raw_recovery_summary,
    )

    planet_df.to_csv(PLANET_OUTPUT_PATH, index=False)
    aspect_df.to_csv(ASPECT_OUTPUT_PATH, index=False)
    natal_df.to_csv(NATAL_OUTPUT_PATH, index=False)
    SUMMARY_OUTPUT_PATH.write_text(markdown, encoding="utf-8")

    print(f"Wrote {PLANET_OUTPUT_PATH}")
    print(f"Wrote {ASPECT_OUTPUT_PATH}")
    print(f"Wrote {NATAL_OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
