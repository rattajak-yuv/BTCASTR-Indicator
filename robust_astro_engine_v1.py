import argparse
import numpy as np
import pandas as pd

from overfit_attribution_analysis import (
    DATA_PATH,
    build_fold_schedule,
    compute_stack_summary,
    dataframe_to_markdown,
    load_selected_features,
    load_threshold_configs,
    pick_target_horizon,
    run_fold_for_stack,
)

RESULTS_PATH = "data/robust_engine_results.csv"
REPORT_PATH = "data/robust_engine_report.md"

ROBUST_STACK = {
    "stack_id": "R",
    "stack_name": "Robust Astro Engine v1",
    "feature_set": "selected",
    "use_tuned_thresholds": False,
    "signal_layer_mode": "none",
    "portfolio_mapping": "spot",
    "description": (
        "Astro Engine + Raw Astro Recovery substrate + Feature Selection + ML, "
        "with default thresholds and spot-style mapping."
    ),
}

PRODUCTION_STACK = {
    "stack_id": "P",
    "stack_name": "Current Production Stack",
    "feature_set": "selected",
    "use_tuned_thresholds": True,
    "signal_layer_mode": "signal_plus_regime",
    "portfolio_mapping": "long_short",
    "description": (
        "Current production candidate using tuned thresholds, signal layer, "
        "regime layer, and long/short mapping."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Robust Astro Engine v1 against the current production stack."
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=5,
        help="Number of calendar years in each rolling training window.",
    )
    return parser.parse_args()


def run_stack_validation(df, horizon, default_thresholds, tuned_thresholds, train_years, stack):
    data = df.dropna(subset=["date", "price", f"future_direction_{horizon}d"]).copy()
    data = data.replace([float("inf"), float("-inf")], np.nan)
    data = data.sort_values("date").reset_index(drop=True)
    data["btc_return_1d"] = data["price"].pct_change().fillna(0.0)

    feature_cols = load_selected_features(data)
    threshold_config = tuned_thresholds if stack["use_tuned_thresholds"] else default_thresholds
    folds = build_fold_schedule(data, train_years=train_years)

    rows = []

    print(
        f"Running {stack['stack_name']} | "
        f"features={len(feature_cols)} | "
        f"thresholds={'tuned' if stack['use_tuned_thresholds'] else 'default'} | "
        f"signal={stack['signal_layer_mode']} | mapping={stack['portfolio_mapping']}"
    )

    for fold_index, (train_start_year, train_end_year, test_year) in enumerate(folds, start=1):
        train_mask = (
            (data["date"].dt.year >= train_start_year)
            & (data["date"].dt.year <= train_end_year)
        )
        test_mask = data["date"].dt.year == test_year

        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()

        train = train.dropna(subset=feature_cols + [f"future_direction_{horizon}d"])
        test = test.dropna(subset=feature_cols + [f"future_direction_{horizon}d"])

        if len(train) < 300 or len(test) == 0:
            continue

        train_metrics, test_metrics = run_fold_for_stack(
            train=train,
            test=test,
            feature_cols=feature_cols,
            horizon=horizon,
            threshold_config=threshold_config,
            stack=stack,
        )

        rows.append(
            {
                "engine_variant": stack["stack_name"],
                "stack_id": stack["stack_id"],
                "stack_description": stack["description"],
                "feature_set": stack["feature_set"],
                "feature_count": len(feature_cols),
                "threshold_mode": "tuned" if stack["use_tuned_thresholds"] else "default",
                "signal_layer_mode": stack["signal_layer_mode"],
                "portfolio_mapping": stack["portfolio_mapping"],
                "fold_id": fold_index,
                "train_start": train["date"].min().date().isoformat(),
                "train_end": train["date"].max().date().isoformat(),
                "test_start": test["date"].min().date().isoformat(),
                "test_end": test["date"].max().date().isoformat(),
                "test_year": int(test_year),
                "train_score": train_metrics["balanced_score"],
                "train_return_drawdown_ratio": train_metrics["return_drawdown_ratio"],
                "train_total_return": train_metrics["total_return"],
                "train_max_drawdown": train_metrics["max_drawdown"],
                "train_accuracy": train_metrics["accuracy"],
                "test_score": test_metrics["balanced_score"],
                "test_return_drawdown_ratio": test_metrics["return_drawdown_ratio"],
                "test_total_return": test_metrics["total_return"],
                "test_max_drawdown": test_metrics["max_drawdown"],
                "test_accuracy": test_metrics["accuracy"],
                "train_test_score_delta": test_metrics["balanced_score"] - train_metrics["balanced_score"],
            }
        )

        print(
            f"  Fold {fold_index} | test {test_year} | "
            f"test_score={test_metrics['balanced_score']:.4f} | "
            f"test_return_dd={test_metrics['return_drawdown_ratio']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

    return pd.DataFrame(rows)


def build_report(summary_df, horizon, train_years):
    robust_row = summary_df[summary_df["stack_id"] == "R"].iloc[0]
    production_row = summary_df[summary_df["stack_id"] == "P"].iloc[0]

    comparison_df = summary_df[
        [
            "stack_name",
            "avg_test_score",
            "stability_score",
            "overfit_ratio",
            "avg_test_accuracy",
            "avg_test_return_drawdown_ratio",
            "best_test_score",
            "worst_test_score",
        ]
    ].copy()

    replace_candidate = (
        robust_row["avg_test_score"] > production_row["avg_test_score"]
        and robust_row["avg_test_return_drawdown_ratio"] >= production_row["avg_test_return_drawdown_ratio"]
        and robust_row["overfit_ratio"] >= production_row["overfit_ratio"]
    )

    if replace_candidate:
        recommendation = (
            "Robust Astro Engine v1 should replace the current production candidate for further research, "
            "because it delivers stronger average out-of-sample score, better average return/drawdown, "
            "and a healthier overfit ratio with less stack complexity."
        )
    else:
        recommendation = (
            "Robust Astro Engine v1 should not replace the current production candidate yet, "
            "because its simpler architecture does not improve the key out-of-sample comparison metrics enough."
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Robust Astro Engine v1\n\n")
        handle.write("## Scope\n\n")
        handle.write(f"- Target horizon: `{horizon}D`\n")
        handle.write(f"- Rolling training window: `{train_years}` calendar years\n")
        handle.write(
            "- Assumption: the current live `ml_dataset.csv` already includes the Raw Astro Recovery substrate, "
            "so this engine isolates the simplest current model-side architecture: Astro -> Feature Selection -> ML.\n\n"
        )

        handle.write("## Comparison\n\n")
        handle.write(dataframe_to_markdown(comparison_df))
        handle.write("\n\n## Robust Engine Summary\n\n")
        handle.write(f"- Average test score: `{robust_row['avg_test_score']:.4f}`\n")
        handle.write(f"- Stability score: `{robust_row['stability_score']:.2f}`\n")
        handle.write(f"- Overfit ratio: `{robust_row['overfit_ratio']:.6f}`\n")
        handle.write(f"- Accuracy: `{robust_row['avg_test_accuracy']:.4f}`\n")
        handle.write(f"- Return/drawdown ratio: `{robust_row['avg_test_return_drawdown_ratio']:.4f}`\n\n")

        handle.write("## Production Delta\n\n")
        handle.write(
            f"- Test score delta vs production: `{robust_row['avg_test_score'] - production_row['avg_test_score']:.4f}`\n"
        )
        handle.write(
            f"- Stability score delta vs production: `{robust_row['stability_score'] - production_row['stability_score']:.2f}`\n"
        )
        handle.write(
            f"- Overfit ratio delta vs production: `{robust_row['overfit_ratio'] - production_row['overfit_ratio']:.6f}`\n"
        )
        handle.write(
            f"- Accuracy delta vs production: `{robust_row['avg_test_accuracy'] - production_row['avg_test_accuracy']:.4f}`\n"
        )
        handle.write(
            f"- Return/drawdown delta vs production: `{robust_row['avg_test_return_drawdown_ratio'] - production_row['avg_test_return_drawdown_ratio']:.4f}`\n\n"
        )

        handle.write("## Recommendation\n\n")
        handle.write(f"- {recommendation}\n")


def main():
    args = parse_args()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    horizon = pick_target_horizon()
    default_thresholds, tuned_thresholds = load_threshold_configs()

    print(
        f"Robust Astro Engine v1 | horizon={horizon}D | train_years={args.train_years}"
    )

    robust_results = run_stack_validation(
        df=df,
        horizon=horizon,
        default_thresholds=default_thresholds,
        tuned_thresholds=tuned_thresholds,
        train_years=args.train_years,
        stack=ROBUST_STACK,
    )
    production_results = run_stack_validation(
        df=df,
        horizon=horizon,
        default_thresholds=default_thresholds,
        tuned_thresholds=tuned_thresholds,
        train_years=args.train_years,
        stack=PRODUCTION_STACK,
    )

    results_df = pd.concat([robust_results, production_results], ignore_index=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    summary_input = results_df.rename(columns={"engine_variant": "stack_name"})
    summary_df = compute_stack_summary(summary_input)
    summary_df = summary_df[summary_df["stack_id"].isin(["R", "P"])].reset_index(drop=True)
    build_report(summary_df, horizon=horizon, train_years=args.train_years)

    robust_row = summary_df[summary_df["stack_id"] == "R"].iloc[0]
    print(
        f"Saved {RESULTS_PATH} and {REPORT_PATH}. "
        f"Robust avg_test_score={robust_row['avg_test_score']:.4f}"
    )


if __name__ == "__main__":
    main()
