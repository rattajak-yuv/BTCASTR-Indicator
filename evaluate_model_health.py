import os
import numpy as np
import pandas as pd

MODEL_SUMMARY_PATH = "data/ml_model_summary.csv"
THRESHOLD_RESULTS_PATH = "data/ml_threshold_tuning_results.csv"
SELECTED_FEATURES_PATH = "data/selected_features.csv"
FEATURE_STABILITY_PATH = "data/feature_stability.csv"

REPORT_PATH = "data/model_health_report.csv"
SUMMARY_PATH = "data/model_health_summary.md"

TOP_FEATURE_COUNT = 20


def load_csv(path, required_columns):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{path} is empty")

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    return df


def parse_bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def format_number(value, decimals=4):
    if pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def markdown_table(df, columns):
    table = df.loc[:, columns].copy()
    headers = list(table.columns)
    rows = [headers, ["---"] * len(headers)]

    for _, row in table.iterrows():
        formatted = []
        for value in row.tolist():
            if pd.isna(value):
                formatted.append("n/a")
            elif isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        rows.append(formatted)

    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def best_thresholds_by_horizon(threshold_df):
    ranked = threshold_df.sort_values(
        ["horizon", "balanced_score", "return_drawdown_ratio", "total_return"],
        ascending=[True, False, False, False],
    )
    return ranked.groupby("horizon", as_index=False).head(1).reset_index(drop=True)


def top_robust_features(stability_df):
    ranked = stability_df.sort_values(
        ["robustness_score", "mean_importance", "horizon_coverage"],
        ascending=[False, False, False],
    )
    return ranked.head(TOP_FEATURE_COUNT).reset_index(drop=True)


def top_noisy_features(stability_df):
    noisy = stability_df[stability_df["noisy_feature"]].copy()
    ranked = noisy.sort_values(
        ["std_importance", "mean_importance", "robustness_score"],
        ascending=[False, False, True],
    )
    return ranked.head(TOP_FEATURE_COUNT).reset_index(drop=True)


def recommend_next_action(summary_df, threshold_best_df, selected_df):
    baseline = summary_df[
        [
            "horizon_days",
            "balanced_score",
            "return_drawdown_ratio",
            "direction_accuracy",
        ]
    ].rename(
        columns={
            "horizon_days": "horizon",
            "balanced_score": "baseline_balanced_score",
            "return_drawdown_ratio": "baseline_return_drawdown_ratio",
            "direction_accuracy": "baseline_direction_accuracy",
        }
    )

    tuned = threshold_best_df[
        ["horizon", "balanced_score", "return_drawdown_ratio"]
    ].rename(
        columns={
            "balanced_score": "tuned_balanced_score",
            "return_drawdown_ratio": "tuned_return_drawdown_ratio",
        }
    )

    combined = baseline.merge(tuned, on="horizon", how="inner")
    combined["balanced_score_improvement"] = (
        combined["tuned_balanced_score"] - combined["baseline_balanced_score"]
    )
    combined["balanced_score_improvement_pct"] = (
        combined["balanced_score_improvement"]
        / combined["baseline_balanced_score"].abs().replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    selected_count = len(selected_df)
    noisy_selected_count = int(selected_df["noisy_feature"].sum())
    noisy_ratio = noisy_selected_count / selected_count if selected_count else 0.0

    median_threshold_gain = combined["balanced_score_improvement_pct"].median(skipna=True)
    max_threshold_gain = combined["balanced_score_improvement_pct"].max(skipna=True)
    best_accuracy = summary_df["direction_accuracy"].max()

    if (
        pd.notna(median_threshold_gain)
        and median_threshold_gain >= 0.15
    ) or (
        pd.notna(max_threshold_gain)
        and max_threshold_gain >= 0.30
    ):
        action = "continue threshold tuning"
        rationale = (
            "Threshold tuning still shows meaningful headroom across horizons, "
            f"with median balanced-score improvement of {format_number(median_threshold_gain, 2)}x "
            f"and max improvement of {format_number(max_threshold_gain, 2)}x over the baseline thresholds."
        )
    elif noisy_ratio >= 0.25 or best_accuracy < 0.55:
        action = "improve signal filter"
        rationale = (
            "Signal quality looks unstable relative to the current feature set, "
            f"with {noisy_selected_count}/{selected_count} selected features flagged as noisy "
            f"and best direction accuracy at {format_number(best_accuracy, 3)}."
        )
    else:
        action = "proceed to Astro Engine Auto-Optimization"
        rationale = (
            "Threshold headroom looks limited and the selected feature set appears stable enough "
            "to move into broader engine-level optimization."
        )

    return action, rationale, combined


def build_report_rows(
    best_balanced_row,
    best_ratio_row,
    threshold_best_df,
    selected_feature_count,
    robust_features_df,
    noisy_features_df,
    action,
    rationale,
):
    rows = []

    rows.append(
        {
            "section": "best_horizon_balanced_score",
            "rank": 1,
            "label": "Best horizon by balanced score",
            "horizon_days": int(best_balanced_row["horizon_days"]),
            "balanced_score": best_balanced_row["balanced_score"],
            "return_drawdown_ratio": best_balanced_row["return_drawdown_ratio"],
            "direction_accuracy": best_balanced_row["direction_accuracy"],
            "number_of_trades": best_balanced_row["number_of_trades"],
        }
    )

    rows.append(
        {
            "section": "best_horizon_return_drawdown_ratio",
            "rank": 1,
            "label": "Best horizon by return/drawdown ratio",
            "horizon_days": int(best_ratio_row["horizon_days"]),
            "balanced_score": best_ratio_row["balanced_score"],
            "return_drawdown_ratio": best_ratio_row["return_drawdown_ratio"],
            "direction_accuracy": best_ratio_row["direction_accuracy"],
            "number_of_trades": best_ratio_row["number_of_trades"],
        }
    )

    for rank, (_, row) in enumerate(threshold_best_df.iterrows(), start=1):
        rows.append(
            {
                "section": "best_threshold_per_horizon",
                "rank": rank,
                "label": f"Best threshold for {int(row['horizon'])}D",
                "horizon_days": int(row["horizon"]),
                "recommended_long_threshold": row["long_threshold"],
                "recommended_short_threshold": row["short_threshold"],
                "balanced_score": row["balanced_score"],
                "return_drawdown_ratio": row["return_drawdown_ratio"],
                "number_of_trades": row["number_of_trades"],
            }
        )

    rows.append(
        {
            "section": "selected_feature_count",
            "rank": 1,
            "label": "Selected feature count",
            "selected_feature_count": selected_feature_count,
        }
    )

    for rank, (_, row) in enumerate(robust_features_df.iterrows(), start=1):
        rows.append(
            {
                "section": "top_robust_feature",
                "rank": rank,
                "label": "Top robust feature",
                "feature": row["feature"],
                "mean_importance": row["mean_importance"],
                "std_importance": row["std_importance"],
                "robustness_score": row["robustness_score"],
                "horizon_coverage": row["horizon_coverage"],
                "dominant_horizon_type": row["dominant_horizon_type"],
                "stable_feature": row["stable_feature"],
                "noisy_feature": row["noisy_feature"],
            }
        )

    for rank, (_, row) in enumerate(noisy_features_df.iterrows(), start=1):
        rows.append(
            {
                "section": "top_noisy_feature",
                "rank": rank,
                "label": "Top noisy feature",
                "feature": row["feature"],
                "mean_importance": row["mean_importance"],
                "std_importance": row["std_importance"],
                "robustness_score": row["robustness_score"],
                "horizon_coverage": row["horizon_coverage"],
                "dominant_horizon_type": row["dominant_horizon_type"],
                "stable_feature": row["stable_feature"],
                "noisy_feature": row["noisy_feature"],
            }
        )

    rows.append(
        {
            "section": "recommended_next_action",
            "rank": 1,
            "label": "Recommended next action",
            "recommendation": action,
            "rationale": rationale,
        }
    )

    return pd.DataFrame(rows)


def write_summary_markdown(
    best_balanced_row,
    best_ratio_row,
    threshold_summary_df,
    selected_feature_count,
    noisy_selected_count,
    robust_features_df,
    noisy_features_df,
    action,
    rationale,
):
    threshold_table = threshold_summary_df.rename(
        columns={
            "horizon": "Horizon",
            "current_long_threshold": "Current Long",
            "current_short_threshold": "Current Short",
            "long_threshold": "Best Long",
            "short_threshold": "Best Short",
            "baseline_balanced_score": "Baseline Balanced Score",
            "balanced_score": "Best Balanced Score",
            "balanced_score_gain_pct": "Gain vs Baseline",
            "return_drawdown_ratio": "Return/DD Ratio",
        }
    )

    robust_table = robust_features_df.rename(
        columns={
            "feature": "Feature",
            "robustness_score": "Robustness Score",
            "mean_importance": "Mean Importance",
            "horizon_coverage": "Horizon Coverage",
            "dominant_horizon_type": "Dominant Horizon",
        }
    )

    noisy_table = noisy_features_df.rename(
        columns={
            "feature": "Feature",
            "std_importance": "Std Importance",
            "mean_importance": "Mean Importance",
            "robustness_score": "Robustness Score",
            "dominant_horizon_type": "Dominant Horizon",
        }
    )

    lines = [
        "# Model Health Summary",
        "",
        "## Horizon Summary",
        (
            f"- Best horizon by balanced score: "
            f"{int(best_balanced_row['horizon_days'])}D "
            f"(balanced_score={format_number(best_balanced_row['balanced_score'])}, "
            f"return/drawdown={format_number(best_balanced_row['return_drawdown_ratio'])})"
        ),
        (
            f"- Best horizon by return/drawdown ratio: "
            f"{int(best_ratio_row['horizon_days'])}D "
            f"(return/drawdown={format_number(best_ratio_row['return_drawdown_ratio'])}, "
            f"balanced_score={format_number(best_ratio_row['balanced_score'])})"
        ),
        "",
        "## Best Threshold Per Horizon",
        markdown_table(
            threshold_table,
            [
                "Horizon",
                "Current Long",
                "Current Short",
                "Best Long",
                "Best Short",
                "Baseline Balanced Score",
                "Best Balanced Score",
                "Gain vs Baseline",
                "Return/DD Ratio",
            ],
        ),
        "",
        "## Feature Selection Health",
        f"- Selected features: {selected_feature_count}",
        f"- Selected features flagged as noisy: {noisy_selected_count}",
        "",
        "### Top 20 Robust Features",
        markdown_table(
            robust_table,
            [
                "Feature",
                "Robustness Score",
                "Mean Importance",
                "Horizon Coverage",
                "Dominant Horizon",
            ],
        ),
        "",
        "### Top 20 Noisy Features",
        markdown_table(
            noisy_table,
            [
                "Feature",
                "Std Importance",
                "Mean Importance",
                "Robustness Score",
                "Dominant Horizon",
            ],
        ),
        "",
        "## Recommended Next Action",
        f"- {action}",
        f"- Reason: {rationale}",
    ]

    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    model_summary = load_csv(
        MODEL_SUMMARY_PATH,
        [
            "horizon_days",
            "long_probability_threshold",
            "short_probability_threshold",
            "balanced_score",
            "return_drawdown_ratio",
            "direction_accuracy",
            "number_of_trades",
        ],
    )
    threshold_results = load_csv(
        THRESHOLD_RESULTS_PATH,
        [
            "horizon",
            "long_threshold",
            "short_threshold",
            "balanced_score",
            "return_drawdown_ratio",
            "number_of_trades",
        ],
    )
    selected_features = load_csv(
        SELECTED_FEATURES_PATH,
        ["feature", "noisy_feature"],
    )
    feature_stability = load_csv(
        FEATURE_STABILITY_PATH,
        [
            "feature",
            "mean_importance",
            "std_importance",
            "robustness_score",
            "horizon_coverage",
            "dominant_horizon_type",
            "stable_feature",
            "noisy_feature",
        ],
    )

    selected_features["noisy_feature"] = parse_bool_series(selected_features["noisy_feature"])
    feature_stability["stable_feature"] = parse_bool_series(feature_stability["stable_feature"])
    feature_stability["noisy_feature"] = parse_bool_series(feature_stability["noisy_feature"])

    best_balanced_row = model_summary.sort_values(
        ["balanced_score", "return_drawdown_ratio"],
        ascending=[False, False],
    ).iloc[0]
    best_ratio_row = model_summary.sort_values(
        ["return_drawdown_ratio", "balanced_score"],
        ascending=[False, False],
    ).iloc[0]

    threshold_best = best_thresholds_by_horizon(threshold_results)
    threshold_summary = threshold_best.merge(
        model_summary[
            [
                "horizon_days",
                "long_probability_threshold",
                "short_probability_threshold",
                "balanced_score",
            ]
        ].rename(
            columns={
                "horizon_days": "horizon",
                "long_probability_threshold": "current_long_threshold",
                "short_probability_threshold": "current_short_threshold",
                "balanced_score": "baseline_balanced_score",
            }
        ),
        on="horizon",
        how="left",
    )
    threshold_summary["balanced_score_gain_pct"] = (
        (threshold_summary["balanced_score"] - threshold_summary["baseline_balanced_score"])
        / threshold_summary["baseline_balanced_score"].abs().replace(0, np.nan)
    )

    robust_features = top_robust_features(feature_stability)
    noisy_features = top_noisy_features(feature_stability)
    selected_feature_count = int(selected_features["feature"].nunique())
    noisy_selected_count = int(selected_features["noisy_feature"].sum())

    action, rationale, _ = recommend_next_action(
        summary_df=model_summary,
        threshold_best_df=threshold_best,
        selected_df=selected_features,
    )

    report_df = build_report_rows(
        best_balanced_row=best_balanced_row,
        best_ratio_row=best_ratio_row,
        threshold_best_df=threshold_summary,
        selected_feature_count=selected_feature_count,
        robust_features_df=robust_features,
        noisy_features_df=noisy_features,
        action=action,
        rationale=rationale,
    )

    os.makedirs("data", exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    write_summary_markdown(
        best_balanced_row=best_balanced_row,
        best_ratio_row=best_ratio_row,
        threshold_summary_df=threshold_summary,
        selected_feature_count=selected_feature_count,
        noisy_selected_count=noisy_selected_count,
        robust_features_df=robust_features,
        noisy_features_df=noisy_features,
        action=action,
        rationale=rationale,
    )

    print(f"Saved: {REPORT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Recommended next action: {action}")


if __name__ == "__main__":
    main()
