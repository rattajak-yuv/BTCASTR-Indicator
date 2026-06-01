import os
from pathlib import Path

import pandas as pd

DATASET_PATH = Path("data/ml_dataset.csv")
SELECTED_FEATURES_PATH = Path("data/selected_features.csv")
COVERAGE_OUTPUT_PATH = Path("data/astro_feature_coverage.csv")
AUDIT_OUTPUT_PATH = Path("data/astro_feature_audit.md")

NON_FEATURE_COLUMNS = {
    "date",
    "astro_regime_v2",
    "signal",
    "regime",
    "price",
    "strategy_total_return",
    "buy_hold_total_return",
    "strategy_max_drawdown",
    "buy_hold_max_drawdown",
}

PLANETS = [
    "jupiter",
    "mars",
    "mercury",
    "moon",
    "neptune",
    "pluto",
    "saturn",
    "sun",
    "uranus",
    "venus",
]

ASPECTS = [
    "conjunction",
    "opposition",
    "sextile",
    "square",
    "trine",
]

NATAL_TARGETS = [
    "bullish",
    "bearish",
    "reversal",
    "volatility",
    "compression",
    "trend_start",
    "trend_end",
    "momentum",
]

STANDALONE_NATAL_TARGET_FEATURES = {
    "bullish",
    "bearish",
    "reversal",
    "volatility",
    "compression",
    "trend_start",
    "trend_end",
}

NON_PREFIX_ASTRO_COMPOSITES = {
    "astro_momentum",
    "astro_momentum_smooth",
    "expansion_score",
    "contraction_score",
    "narrative_score",
    "trigger_score",
}

RAW_ASPECT_PREFIX = "aspect_count_"
PLANET_PREFIX = "planet_"
ASTRO_PREFIX = "astro_"
HOUSE_TOKEN = "house_position"


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")

    return df


def load_selected_features(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    selected_df = pd.read_csv(path)
    if "feature" not in selected_df.columns:
        raise ValueError(f"{path} must contain a 'feature' column")

    features = selected_df["feature"].dropna().astype(str).tolist()
    if not features:
        raise ValueError(f"{path} does not contain any selected features")

    return selected_df, features


def normalize(feature_name):
    return str(feature_name).strip().lower()


def current_importance_mapping_covers(feature_name):
    feature = normalize(feature_name)
    return (
        feature.startswith(ASTRO_PREFIX)
        or feature.startswith(PLANET_PREFIX)
        or feature.startswith(RAW_ASPECT_PREFIX)
    )


def is_numeric_feature(df, feature_name):
    return feature_name in df.columns and pd.api.types.is_numeric_dtype(df[feature_name])


def is_valid_model_feature(df, feature_name):
    if feature_name not in df.columns:
        return False

    if feature_name in NON_FEATURE_COLUMNS:
        return False

    if feature_name.startswith("future_"):
        return False

    if not pd.api.types.is_numeric_dtype(df[feature_name]):
        return False

    if df[feature_name].isna().all():
        return False

    return True


def parse_planet(feature_name):
    feature = normalize(feature_name)
    for planet in PLANETS:
        if planet in feature:
            return planet
    return ""


def parse_aspect(feature_name):
    feature = normalize(feature_name)
    for aspect in ASPECTS:
        if aspect in feature:
            return aspect
    return ""


def parse_natal_target(feature_name):
    feature = normalize(feature_name)
    for target in NATAL_TARGETS:
        if target in feature:
            return target
    return ""


def is_house_feature(feature_name):
    return HOUSE_TOKEN in normalize(feature_name)


def is_raw_aspect_feature(feature_name):
    feature = normalize(feature_name)
    return feature.startswith(RAW_ASPECT_PREFIX) and not is_house_feature(feature_name)


def is_planet_feature(feature_name):
    return normalize(feature_name).startswith(PLANET_PREFIX)


def is_standalone_natal_target_feature(feature_name):
    return normalize(feature_name) in STANDALONE_NATAL_TARGET_FEATURES


def is_non_prefix_astro_composite(feature_name):
    return normalize(feature_name) in NON_PREFIX_ASTRO_COMPOSITES


def is_broad_astro_feature(feature_name):
    feature = normalize(feature_name)
    return (
        current_importance_mapping_covers(feature)
        or is_standalone_natal_target_feature(feature)
        or is_non_prefix_astro_composite(feature)
    )


def classify_primary_category(feature_name, valid_model_feature):
    if not valid_model_feature:
        return "non_model_or_metadata"

    if is_house_feature(feature_name):
        return "house_feature"

    if is_raw_aspect_feature(feature_name):
        return "raw_aspect_feature"

    if is_planet_feature(feature_name):
        return "planet_feature"

    if is_standalone_natal_target_feature(feature_name):
        return "natal_target_feature"

    if normalize(feature_name).startswith(ASTRO_PREFIX) or is_non_prefix_astro_composite(feature_name):
        return "composite_astro_score"

    return "ml_derived_feature"


def build_feature_coverage(df, selected_features):
    dataset_columns = list(df.columns)
    universe = dataset_columns + [feature for feature in selected_features if feature not in df.columns]
    rows = []

    for feature in universe:
        in_dataset = feature in df.columns
        numeric_feature = is_numeric_feature(df, feature)
        valid_model_feature = is_valid_model_feature(df, feature)
        broad_astro_feature = is_broad_astro_feature(feature)
        current_mapping = current_importance_mapping_covers(feature)
        primary_category = classify_primary_category(feature, valid_model_feature)

        if not in_dataset:
            exclusion_reason = "missing_from_dataset"
        elif feature in NON_FEATURE_COLUMNS:
            exclusion_reason = "explicit_non_feature_column"
        elif feature.startswith("future_"):
            exclusion_reason = "future_leakage_prefix"
        elif not numeric_feature:
            exclusion_reason = "non_numeric"
        elif df[feature].isna().all():
            exclusion_reason = "all_nan"
        else:
            exclusion_reason = ""

        mapping_gap = broad_astro_feature and not current_mapping
        if mapping_gap:
            mapping_fix = "expand_astro_feature_detector"
        elif broad_astro_feature:
            mapping_fix = "covered_by_current_mapping"
        else:
            mapping_fix = ""

        rows.append(
            {
                "feature": feature,
                "in_ml_dataset": in_dataset,
                "in_selected_features": feature in selected_features,
                "is_numeric": numeric_feature,
                "is_valid_model_feature": valid_model_feature,
                "is_broad_astro_feature": broad_astro_feature,
                "covered_by_current_importance_mapping": current_mapping,
                "mapping_gap": mapping_gap,
                "primary_category": primary_category,
                "planet": parse_planet(feature),
                "aspect": parse_aspect(feature),
                "natal_target": parse_natal_target(feature),
                "is_house_feature": is_house_feature(feature),
                "exclusion_reason": exclusion_reason,
                "recommended_mapping_fix": mapping_fix,
            }
        )

    coverage = pd.DataFrame(rows)
    coverage = coverage.sort_values(
        [
            "is_valid_model_feature",
            "in_selected_features",
            "is_broad_astro_feature",
            "primary_category",
            "feature",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    return coverage


def category_counts(coverage):
    feature_rows = coverage[coverage["is_valid_model_feature"]].copy()
    categories = [
        "raw_aspect_feature",
        "planet_feature",
        "natal_target_feature",
        "house_feature",
        "composite_astro_score",
        "ml_derived_feature",
    ]

    rows = []
    for category in categories:
        group = feature_rows[feature_rows["primary_category"] == category]
        rows.append(
            {
                "category": category,
                "total_features": int(len(group)),
                "selected_features": int(group["in_selected_features"].sum()),
                "removed_features": int(len(group) - group["in_selected_features"].sum()),
            }
        )

    return pd.DataFrame(rows)


def list_present(values):
    cleaned = sorted({value for value in values if value})
    return cleaned


def format_list(values):
    if not values:
        return "none"
    return ", ".join(values)


def dataframe_to_markdown(df):
    headers = list(df.columns)
    rows = [headers]
    for _, row in df.iterrows():
        rows.append([str(row[column]) for column in headers])

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]

    def render_row(values):
        cells = [values[index].ljust(widths[index]) for index in range(len(values))]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    rendered = [render_row(rows[0]), separator]
    for row in rows[1:]:
        rendered.append(render_row(row))

    return "\n".join(rendered)


def build_markdown_report(coverage):
    valid_features = coverage[coverage["is_valid_model_feature"]].copy()
    selected_features = valid_features[valid_features["in_selected_features"]].copy()

    raw_aspects = valid_features[valid_features["primary_category"] == "raw_aspect_feature"]
    selected_raw_aspects = raw_aspects[raw_aspects["in_selected_features"]]

    dataset_planets = list_present(valid_features["planet"])
    selected_planets = list_present(selected_features["planet"])

    dataset_aspects = list_present(raw_aspects["aspect"])
    selected_aspects = list_present(selected_raw_aspects["aspect"])

    dataset_natal_targets = list_present(
        valid_features.loc[
            valid_features["is_broad_astro_feature"],
            "natal_target",
        ]
    )
    selected_natal_targets = list_present(
        selected_features.loc[
            selected_features["is_broad_astro_feature"],
            "natal_target",
        ]
    )

    astro_valid_features = valid_features[valid_features["is_broad_astro_feature"]]
    astro_selected_features = selected_features[selected_features["is_broad_astro_feature"]]
    mapping_gaps = astro_valid_features[astro_valid_features["mapping_gap"]]
    selected_mapping_gaps = astro_selected_features[astro_selected_features["mapping_gap"]]

    coverage_counts = category_counts(coverage)
    coverage_table = dataframe_to_markdown(coverage_counts)

    missing_mapping_features = mapping_gaps["feature"].tolist()
    selected_missing_mapping_features = selected_mapping_gaps["feature"].tolist()

    lines = [
        "# Astro Feature Coverage Audit",
        "",
        "## Overview",
        f"- Dataset columns inspected: {int(len(coverage[coverage['in_ml_dataset']]))}",
        f"- Valid model features: {int(len(valid_features))}",
        f"- Selected features: {int(len(selected_features))}",
        f"- Broad astro-related valid features: {int(len(astro_valid_features))}",
        f"- Astro-related selected features: {int(len(astro_selected_features))}",
        "",
        "## Category Counts",
        coverage_table,
        "",
        "## Retention Summary",
        f"- Raw aspect features retained: {int(len(selected_raw_aspects))}",
        f"- Raw aspect features removed: {int(len(raw_aspects) - len(selected_raw_aspects))}",
        f"- Planets represented in valid features: {format_list(dataset_planets)}",
        f"- Planets represented in selected features: {format_list(selected_planets)}",
        f"- Aspects represented in valid features: {format_list(dataset_aspects)}",
        f"- Aspects represented in selected features: {format_list(selected_aspects)}",
        f"- Natal targets represented in valid features: {format_list(dataset_natal_targets)}",
        f"- Natal targets represented in selected features: {format_list(selected_natal_targets)}",
        "",
        "## Mapping Verification",
        f"- Astro valid features covered by current importance mapping: {int((astro_valid_features['covered_by_current_importance_mapping']).sum())}",
        f"- Astro valid features missing from current importance mapping: {int(len(mapping_gaps))}",
        f"- Astro selected features missing from current importance mapping: {int(len(selected_mapping_gaps))}",
        "",
        "Current explainability mapping is correct for prefix-based `astro_*`, `planet_*`, and `aspect_count_*` features, but it is incomplete for astro-related standalone and non-prefix composite columns.",
        "",
        "### Missing Mapping Features",
        format_list(missing_mapping_features),
        "",
        "### Missing Mapping Features That Are Currently Selected",
        format_list(selected_missing_mapping_features),
        "",
        "## Recommended Fixes Before Astro Auto-Optimization v2",
        "- Expand the astro feature detector to include standalone astro components such as `trend_start`, `trend_end`, `bullish`, `bearish`, `reversal`, `volatility`, and `compression`.",
        "- Treat non-prefix astro composites such as `astro_momentum`, `astro_momentum_smooth`, `expansion_score`, `contraction_score`, `narrative_score`, and `trigger_score` as astro features in the explainability layer.",
        "- Split `house_position` out from raw aspect counts so house coverage is reported separately instead of being mixed into aspect buckets.",
        "- Add a mapping audit step to the auto-optimization workflow so it fails loudly when selected astro-related features are not covered by the explainability taxonomy.",
        "- Interpret near-zero aspect importance carefully: the current selected feature set retains zero raw aspect-count features, so weak aspect results are partly a real feature-coverage outcome, not only a grouping bug.",
    ]

    return "\n".join(lines) + "\n"


def main():
    df = load_dataset(DATASET_PATH)
    _, selected_features = load_selected_features(SELECTED_FEATURES_PATH)

    coverage = build_feature_coverage(df, selected_features)
    report = build_markdown_report(coverage)

    COVERAGE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(COVERAGE_OUTPUT_PATH, index=False)
    AUDIT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print(f"Wrote {COVERAGE_OUTPUT_PATH}")
    print(f"Wrote {AUDIT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
