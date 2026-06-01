from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

DATASET_PATH = DATA_DIR / "ml_dataset.csv"
SELECTED_BEFORE_PATH = DATA_DIR / "selected_features_before_raw_recovery_v1.csv"
SUMMARY_BEFORE_PATH = DATA_DIR / "ml_model_summary_before_raw_recovery_v1.csv"
SELECTED_AFTER_PATH = DATA_DIR / "selected_features.csv"
SUMMARY_AFTER_PATH = DATA_DIR / "ml_model_summary.csv"

SUMMARY_OUTPUT_PATH = DATA_DIR / "raw_astro_recovery_summary.csv"
REPORT_OUTPUT_PATH = DATA_DIR / "raw_astro_recovery_report.md"

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


def select_features(df):
    return df["feature"].dropna().astype(str).tolist()


def best_model_row(summary_df):
    ranked = summary_df.sort_values(
        ["balanced_score", "return_drawdown_ratio", "direction_accuracy"],
        ascending=[False, False, False],
    )
    return ranked.iloc[0]


def count_selected_planet_features(features):
    return sorted(
        feature
        for feature in features
        if feature.startswith("planet_") or feature in PLANET_SIGNAL_COLUMNS
    )


def count_selected_raw_aspect_features(features):
    return sorted(
        feature
        for feature in features
        if feature.startswith("aspect_count_") or feature in ASPECT_STRENGTH_COLUMNS
    )


def count_selected_natal_target_features(features):
    return sorted(
        feature
        for feature in features
        if feature in NATAL_TARGET_STRENGTH_COLUMNS
    )


def format_list(values):
    if not values:
        return "none"
    return ", ".join(values)


def render_markdown_table(rows, columns):
    table_rows = [columns]
    for row in rows:
        table_rows.append([str(row[column]) for column in columns])

    widths = [max(len(row[index]) for row in table_rows) for index in range(len(columns))]

    def render(values):
        return "| " + " | ".join(
            values[index].ljust(widths[index]) for index in range(len(values))
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [render(table_rows[0]), separator]
    for row in table_rows[1:]:
        lines.append(render(row))
    return "\n".join(lines)


def build_metric_row(stage, selected_df, summary_df):
    features = select_features(selected_df)
    best = best_model_row(summary_df)

    raw_aspects = count_selected_raw_aspect_features(features)
    planet_features = count_selected_planet_features(features)
    natal_target_features = count_selected_natal_target_features(features)

    return {
        "stage": stage,
        "best_horizon_days": int(best["horizon_days"]),
        "selected_feature_count": int(len(set(features))),
        "selected_raw_aspect_features": int(len(raw_aspects)),
        "selected_planet_features": int(len(planet_features)),
        "selected_natal_target_features": int(len(natal_target_features)),
        "balanced_score": float(best["balanced_score"]),
        "return_drawdown_ratio": float(best["return_drawdown_ratio"]),
        "accuracy": float(best["direction_accuracy"]),
        "raw_aspect_feature_list": format_list(raw_aspects),
        "planet_feature_list": format_list(planet_features),
        "natal_target_feature_list": format_list(natal_target_features),
    }


def main():
    dataset = load_csv(DATASET_PATH)
    selected_before = load_csv(SELECTED_BEFORE_PATH, ["feature"])
    summary_before = load_csv(SUMMARY_BEFORE_PATH, ["horizon_days", "balanced_score", "return_drawdown_ratio", "direction_accuracy"])
    selected_after = load_csv(SELECTED_AFTER_PATH, ["feature"])
    summary_after = load_csv(SUMMARY_AFTER_PATH, ["horizon_days", "balanced_score", "return_drawdown_ratio", "direction_accuracy"])

    before_row = build_metric_row("before_recovery", selected_before, summary_before)
    after_row = build_metric_row("after_recovery", selected_after, summary_after)

    summary = pd.DataFrame([before_row, after_row])
    summary["delta_selected_raw_aspect_features"] = summary["selected_raw_aspect_features"].diff()
    summary["delta_selected_planet_features"] = summary["selected_planet_features"].diff()
    summary["delta_balanced_score"] = summary["balanced_score"].diff()
    summary["delta_return_drawdown_ratio"] = summary["return_drawdown_ratio"].diff()
    summary["delta_accuracy"] = summary["accuracy"].diff()

    dataset_columns = set(dataset.columns)
    recovered_columns_present = [
        column for column in (
            PLANET_SIGNAL_COLUMNS
            + ASPECT_STRENGTH_COLUMNS
            + NATAL_TARGET_STRENGTH_COLUMNS
        )
        if column in dataset_columns
    ]

    report_lines = [
        "# Raw Astro Feature Recovery v1",
        "",
        "## Recovered Feature Coverage",
        f"- `ml_dataset.csv` columns: {len(dataset.columns)}",
        f"- Recovered aggregate columns present: {len(recovered_columns_present)} / {len(PLANET_SIGNAL_COLUMNS) + len(ASPECT_STRENGTH_COLUMNS) + len(NATAL_TARGET_STRENGTH_COLUMNS)}",
        f"- Recovered columns: {format_list(recovered_columns_present)}",
        "",
        "## Before vs After",
        render_markdown_table(
            [
                {
                    "stage": row["stage"],
                    "best_horizon_days": row["best_horizon_days"],
                    "selected_raw_aspect_features": row["selected_raw_aspect_features"],
                    "selected_planet_features": row["selected_planet_features"],
                    "selected_natal_target_features": row["selected_natal_target_features"],
                    "balanced_score": f"{row['balanced_score']:.4f}",
                    "return_drawdown_ratio": f"{row['return_drawdown_ratio']:.4f}",
                    "accuracy": f"{row['accuracy']:.4f}",
                }
                for row in [before_row, after_row]
            ],
            [
                "stage",
                "best_horizon_days",
                "selected_raw_aspect_features",
                "selected_planet_features",
                "selected_natal_target_features",
                "balanced_score",
                "return_drawdown_ratio",
                "accuracy",
            ],
        ),
        "",
        "## Selection Detail",
        f"- Before selected raw aspect features: {before_row['raw_aspect_feature_list']}",
        f"- After selected raw aspect features: {after_row['raw_aspect_feature_list']}",
        f"- Before selected planet features: {before_row['planet_feature_list']}",
        f"- After selected planet features: {after_row['planet_feature_list']}",
        f"- Before selected natal-target features: {before_row['natal_target_feature_list']}",
        f"- After selected natal-target features: {after_row['natal_target_feature_list']}",
        "",
        "## Interpretation",
        "- The recovery is successful if compact raw aspect and planet aggregates are present in `ml_dataset.csv` and survive into `selected_features.csv`.",
        "- Balanced-score, return/drawdown, and accuracy changes should be interpreted together with the raw-feature retention counts, because the goal is not just performance but preserving raw astro structure for downstream explainability and optimization.",
    ]

    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    REPORT_OUTPUT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote {SUMMARY_OUTPUT_PATH}")
    print(f"Wrote {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
