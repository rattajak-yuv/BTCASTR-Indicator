import argparse
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "data/ml_dataset.csv"
SELECTED_FEATURES_PATH = "data/selected_features.csv"
THRESHOLD_RESULTS_PATH = "data/ml_threshold_tuning_results.csv"
REGIME_WEIGHTED_RESULTS_PATH = "data/regime_weighted_results.csv"
MODEL_SUMMARY_PATH = "data/ml_model_summary.csv"

RESULTS_PATH = "data/walk_forward_results.csv"
REPORT_PATH = "data/walk_forward_report.md"

PROBA_THRESHOLDS = {
    3: {"long": 0.56, "short": 0.44},
    7: {"long": 0.57, "short": 0.43},
    14: {"long": 0.58, "short": 0.42},
    30: {"long": 0.60, "short": 0.40},
    60: {"long": 0.62, "short": 0.38},
    90: {"long": 0.63, "short": 0.37},
}

NON_FEATURE_COLUMNS = {
    "date",
    "astro_regime_v2",
    "signal",
    "regime",
    "market_regime",
    "volatility_state",
    "applied_weight_profile",
    "price",
    "strategy_total_return",
    "buy_hold_total_return",
    "strategy_max_drawdown",
    "buy_hold_max_drawdown",
}

warnings.filterwarnings(
    "ignore",
    message=(
        "`sklearn.utils.parallel.delayed` should be used with "
        "`sklearn.utils.parallel.Parallel`"
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run yearly rolling walk-forward validation for Astro Engine V4."
    )
    parser.add_argument(
        "--feature-set",
        choices=["all", "selected"],
        default="selected",
        help="Feature source to use during refits.",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=5,
        help="Number of calendar years to include in each rolling training window.",
    )
    return parser.parse_args()


def sharpe_like(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std() == 0:
        return np.nan
    return float((clean.mean() / clean.std()) * np.sqrt(365))


def format_markdown_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []

    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_markdown_value(row[col]) for col in df.columns) + " |")

    return "\n".join([header_row, separator_row] + rows)


def validate_thresholds(horizon: int, long_th: float, short_th: float):
    if pd.isna(long_th) or pd.isna(short_th):
        raise ValueError(f"Thresholds for horizon {horizon}D must not be NaN")
    if short_th >= long_th:
        raise ValueError(
            f"Invalid thresholds for horizon {horizon}D: short threshold must be below long threshold"
        )


def load_threshold_config() -> Dict[int, Dict[str, float]]:
    thresholds = {
        horizon: {
            "long": values["long"],
            "short": values["short"],
            "source": "default",
        }
        for horizon, values in PROBA_THRESHOLDS.items()
    }

    if not os.path.exists(THRESHOLD_RESULTS_PATH):
        return thresholds

    tuning = pd.read_csv(THRESHOLD_RESULTS_PATH)
    required_columns = ["horizon", "long_threshold", "short_threshold", "balanced_score"]
    missing = [column for column in required_columns if column not in tuning.columns]
    if missing:
        raise ValueError(f"{THRESHOLD_RESULTS_PATH} is missing required columns: {missing}")

    ranking_columns = ["horizon", "balanced_score"]
    ranking_order = [True, False]

    if "return_drawdown_ratio" in tuning.columns:
        ranking_columns.append("return_drawdown_ratio")
        ranking_order.append(False)
    if "total_return" in tuning.columns:
        ranking_columns.append("total_return")
        ranking_order.append(False)
    if "number_of_trades" in tuning.columns:
        ranking_columns.append("number_of_trades")
        ranking_order.append(True)

    ranking_columns.extend(["long_threshold", "short_threshold"])
    ranking_order.extend([True, True])

    ranked = tuning.sort_values(ranking_columns, ascending=ranking_order)
    best_by_horizon = ranked.groupby("horizon", as_index=False).head(1)

    for _, row in best_by_horizon.iterrows():
        horizon = int(row["horizon"])
        if horizon not in thresholds:
            continue

        long_th = float(row["long_threshold"])
        short_th = float(row["short_threshold"])
        validate_thresholds(horizon, long_th, short_th)

        thresholds[horizon] = {
            "long": long_th,
            "short": short_th,
            "source": "tuned",
        }

    return thresholds


def is_valid_feature_column(df: pd.DataFrame, col: str) -> bool:
    if col in NON_FEATURE_COLUMNS:
        return False
    if col.startswith("future_"):
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    if df[col].isna().all():
        return False
    return True


def load_all_features(df: pd.DataFrame) -> List[str]:
    features = sorted(col for col in df.columns if is_valid_feature_column(df, col))
    if not features:
        raise ValueError("No valid numeric features found in ml_dataset.csv")
    return features


def load_selected_features(df: pd.DataFrame) -> List[str]:
    sf = pd.read_csv(SELECTED_FEATURES_PATH)
    if "feature" not in sf.columns:
        raise ValueError("selected_features.csv must contain a 'feature' column")

    selected = sf["feature"].dropna().astype(str).unique().tolist()
    selected = [
        feature for feature in selected
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])
    ]
    if not selected:
        raise ValueError("No selected features found in ml_dataset.csv")
    return selected


def resolve_feature_columns(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], str]:
    if feature_set == "all":
        return load_all_features(df), "all_features"
    return load_selected_features(df), "selected_features"


def pick_target_horizon() -> int:
    regime_results = pd.read_csv(REGIME_WEIGHTED_RESULTS_PATH)
    row = regime_results[regime_results["stage"] == "regime_aware_v4"].iloc[0]
    return int(row["best_horizon_days"])


def create_signal(prob_up: float, horizon: int, threshold_config: Dict[int, Dict[str, float]]) -> int:
    long_th = threshold_config[horizon]["long"]
    short_th = threshold_config[horizon]["short"]
    if prob_up >= long_th:
        return 1
    if prob_up <= short_th:
        return -1
    return 0


def compute_strategy_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    g = frame.sort_values("date").reset_index(drop=True).copy()
    g["ml_position"] = g["ml_position_raw"].shift(1).fillna(0)
    g["ml_strategy_return"] = g["btc_return_1d"] * g["ml_position"]
    g["ml_strategy_equity"] = (1 + g["ml_strategy_return"]).cumprod()
    g["ml_strategy_drawdown"] = (
        g["ml_strategy_equity"] / g["ml_strategy_equity"].cummax()
    ) - 1

    total_return = float(g["ml_strategy_equity"].iloc[-1] - 1)
    max_dd = float(g["ml_strategy_drawdown"].min())
    accuracy = float(
        accuracy_score(
            g["actual_direction"].astype(int),
            g["ml_pred_direction"].astype(int),
        )
    )
    trades = int((g["ml_position_raw"].diff().fillna(0) != 0).sum())
    sharpe = sharpe_like(g["ml_strategy_return"])
    dd_abs = abs(max_dd) if pd.notna(max_dd) else np.nan
    return_dd_ratio = (
        total_return / dd_abs
        if pd.notna(dd_abs) and dd_abs != 0
        else np.nan
    )
    balanced_score = (
        total_return * 0.30
        + (sharpe if pd.notna(sharpe) else 0.0) * 0.35
        + (return_dd_ratio if pd.notna(return_dd_ratio) else 0.0) * 0.20
        - (dd_abs if pd.notna(dd_abs) else 0.0) * 1.25
        - trades * 0.002
    )

    return {
        "balanced_score": float(balanced_score),
        "return_drawdown_ratio": float(return_dd_ratio) if pd.notna(return_dd_ratio) else np.nan,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "accuracy": accuracy,
        "trades": trades,
    }


def build_fold_schedule(df: pd.DataFrame, train_years: int) -> List[Tuple[int, int, int]]:
    years = sorted(df["date"].dt.year.dropna().unique().tolist())
    folds = []

    for idx in range(train_years, len(years)):
        train_start_year = years[idx - train_years]
        train_end_year = years[idx - 1]
        test_year = years[idx]
        folds.append((train_start_year, train_end_year, test_year))

    return folds


def run_walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    feature_set_name: str,
    horizon: int,
    threshold_config: Dict[int, Dict[str, float]],
    train_years: int,
) -> pd.DataFrame:
    target_col = f"future_direction_{horizon}d"
    if target_col not in df.columns:
        raise ValueError(f"Missing target column {target_col}")

    data = df.dropna(subset=["date", "price", target_col]).copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.sort_values("date").reset_index(drop=True)
    data["btc_return_1d"] = data["price"].pct_change().fillna(0.0)

    folds = build_fold_schedule(data, train_years=train_years)
    results = []

    for fold_index, (train_start_year, train_end_year, test_year) in enumerate(folds, start=1):
        train_mask = (
            (data["date"].dt.year >= train_start_year)
            & (data["date"].dt.year <= train_end_year)
        )
        test_mask = data["date"].dt.year == test_year

        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()

        train = train.dropna(subset=feature_cols + [target_col])
        test = test.dropna(subset=feature_cols + [target_col])

        if len(train) < 300 or len(test) == 0:
            continue

        x_train = train[feature_cols]
        y_train = train[target_col].astype(int)
        x_test = test[feature_cols]
        y_test = test[target_col].astype(int)

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42 + horizon,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)

        train_prob_up = model.predict_proba(x_train)[:, 1]
        train_pred = (train_prob_up >= 0.5).astype(int)
        test_prob_up = model.predict_proba(x_test)[:, 1]
        test_pred = (test_prob_up >= 0.5).astype(int)

        train_eval = train[["date", "price", "btc_return_1d"]].copy()
        train_eval["ml_prob_up"] = train_prob_up
        train_eval["ml_pred_direction"] = train_pred
        train_eval["ml_position_raw"] = [
            create_signal(prob, horizon, threshold_config) for prob in train_prob_up
        ]
        train_eval["actual_direction"] = y_train.values

        test_eval = test[["date", "price", "btc_return_1d"]].copy()
        test_eval["ml_prob_up"] = test_prob_up
        test_eval["ml_pred_direction"] = test_pred
        test_eval["ml_position_raw"] = [
            create_signal(prob, horizon, threshold_config) for prob in test_prob_up
        ]
        test_eval["actual_direction"] = y_test.values

        train_metrics = compute_strategy_metrics(train_eval)
        test_metrics = compute_strategy_metrics(test_eval)

        results.append(
            {
                "fold_id": fold_index,
                "feature_set": feature_set_name,
                "horizon_days": horizon,
                "threshold_long": threshold_config[horizon]["long"],
                "threshold_short": threshold_config[horizon]["short"],
                "threshold_source": threshold_config[horizon]["source"],
                "train_start": train["date"].min().date().isoformat(),
                "train_end": train["date"].max().date().isoformat(),
                "test_start": test["date"].min().date().isoformat(),
                "test_end": test["date"].max().date().isoformat(),
                "test_year": int(test_year),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_balanced_score": train_metrics["balanced_score"],
                "train_return_drawdown_ratio": train_metrics["return_drawdown_ratio"],
                "train_total_return": train_metrics["total_return"],
                "train_max_drawdown": train_metrics["max_drawdown"],
                "train_accuracy": train_metrics["accuracy"],
                "train_trades": train_metrics["trades"],
                "test_balanced_score": test_metrics["balanced_score"],
                "test_return_drawdown_ratio": test_metrics["return_drawdown_ratio"],
                "test_total_return": test_metrics["total_return"],
                "test_max_drawdown": test_metrics["max_drawdown"],
                "test_accuracy": test_metrics["accuracy"],
                "test_trades": test_metrics["trades"],
                "balanced_score_decay": test_metrics["balanced_score"] - train_metrics["balanced_score"],
                "accuracy_decay": test_metrics["accuracy"] - train_metrics["accuracy"],
                "return_dd_decay": (
                    test_metrics["return_drawdown_ratio"] - train_metrics["return_drawdown_ratio"]
                    if pd.notna(test_metrics["return_drawdown_ratio"]) and pd.notna(train_metrics["return_drawdown_ratio"])
                    else np.nan
                ),
            }
        )

        print(
            f"Fold {fold_index} | train {train_start_year}-{train_end_year} | "
            f"test {test_year} | test_balanced={test_metrics['balanced_score']:.4f} "
            f"test_return_dd={test_metrics['return_drawdown_ratio']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

    if not results:
        raise ValueError("No walk-forward folds were produced")

    return pd.DataFrame(results)


def compute_stability_score(results_df: pd.DataFrame) -> float:
    mean_balanced = float(results_df["test_balanced_score"].mean())
    std_balanced = float(results_df["test_balanced_score"].std(ddof=0))
    positive_balanced_rate = float((results_df["test_balanced_score"] > 0).mean())
    positive_return_rate = float((results_df["test_total_return"] > 0).mean())
    avg_accuracy = float(results_df["test_accuracy"].mean())
    avg_abs_dd = float(results_df["test_max_drawdown"].abs().mean())

    consistency = 1.0 / (1.0 + (std_balanced / max(abs(mean_balanced), 1.0)))
    accuracy_edge = np.clip((avg_accuracy - 0.50) / 0.10, 0.0, 1.0)
    drawdown_resilience = max(0.0, 1.0 - min(avg_abs_dd, 1.0))

    stability_score = 100 * (
        0.30 * positive_balanced_rate
        + 0.20 * positive_return_rate
        + 0.20 * consistency
        + 0.15 * accuracy_edge
        + 0.15 * drawdown_resilience
    )
    return float(stability_score)


def create_report(results_df: pd.DataFrame, in_sample_row: pd.Series, horizon: int, train_years: int):
    avg_row = {
        "balanced_score": float(results_df["test_balanced_score"].mean()),
        "return_drawdown_ratio": float(results_df["test_return_drawdown_ratio"].mean()),
        "total_return": float(results_df["test_total_return"].mean()),
        "max_drawdown": float(results_df["test_max_drawdown"].mean()),
        "accuracy": float(results_df["test_accuracy"].mean()),
    }
    best_row = results_df.sort_values(
        ["test_balanced_score", "test_return_drawdown_ratio", "test_total_return"],
        ascending=[False, False, False],
    ).iloc[0]
    worst_row = results_df.sort_values(
        ["test_balanced_score", "test_return_drawdown_ratio", "test_total_return"],
        ascending=[True, True, True],
    ).iloc[0]
    stability_score = compute_stability_score(results_df)

    avg_train_balanced = float(results_df["train_balanced_score"].mean())
    avg_test_balanced = float(results_df["test_balanced_score"].mean())
    avg_train_accuracy = float(results_df["train_accuracy"].mean())
    avg_test_accuracy = float(results_df["test_accuracy"].mean())
    avg_train_return_dd = float(results_df["train_return_drawdown_ratio"].mean())
    avg_test_return_dd = float(results_df["test_return_drawdown_ratio"].mean())
    median_test_balanced = float(results_df["test_balanced_score"].median())
    positive_test_windows = int((results_df["test_balanced_score"] > 0).sum())
    positive_return_windows = int((results_df["test_total_return"] > 0).sum())

    in_sample_balanced = float(in_sample_row["balanced_score"])
    in_sample_return_dd = float(in_sample_row["return_drawdown_ratio"])
    in_sample_accuracy = float(in_sample_row["direction_accuracy"])

    overfit_signals = 0
    if avg_test_balanced < avg_train_balanced * 0.35:
        overfit_signals += 1
    if avg_test_balanced < in_sample_balanced * 0.35:
        overfit_signals += 1
    if float((results_df["test_balanced_score"] > 0).mean()) < 0.50:
        overfit_signals += 1
    if stability_score < 45:
        overfit_signals += 1

    overfit_assessment = (
        "likely overfit"
        if overfit_signals >= 3
        else "mixed / moderate overfit risk"
        if overfit_signals == 2
        else "not obviously overfit"
    )

    report_table = results_df[
        [
            "fold_id",
            "train_start",
            "train_end",
            "test_year",
            "test_balanced_score",
            "test_return_drawdown_ratio",
            "test_total_return",
            "test_max_drawdown",
            "test_accuracy",
        ]
    ].copy()

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Walk-Forward Validation Framework v1\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- Target horizon: `{horizon}D`\n")
        handle.write(f"- Rolling training window: `{train_years}` calendar years\n")
        handle.write(f"- Folds evaluated: `{len(results_df)}`\n\n")

        handle.write("## Out-of-Sample Results By Fold\n\n")
        handle.write(dataframe_to_markdown(report_table))
        handle.write("\n\n## Summary\n\n")
        handle.write(f"- Average out-of-sample balanced score: `{avg_row['balanced_score']:.4f}`\n")
        handle.write(f"- Average out-of-sample return/drawdown ratio: `{avg_row['return_drawdown_ratio']:.4f}`\n")
        handle.write(f"- Average out-of-sample total return: `{avg_row['total_return']:.4f}`\n")
        handle.write(f"- Average out-of-sample max drawdown: `{avg_row['max_drawdown']:.4f}`\n")
        handle.write(f"- Average out-of-sample accuracy: `{avg_row['accuracy']:.4f}`\n")
        handle.write(f"- Median out-of-sample balanced score: `{median_test_balanced:.4f}`\n")
        handle.write(f"- Positive balanced-score windows: `{positive_test_windows}` / `{len(results_df)}`\n")
        handle.write(f"- Positive return windows: `{positive_return_windows}` / `{len(results_df)}`\n")
        handle.write(
            f"- Best test period: `{best_row['test_year']}` with balanced score `{best_row['test_balanced_score']:.4f}`\n"
        )
        handle.write(
            f"- Worst test period: `{worst_row['test_year']}` with balanced score `{worst_row['test_balanced_score']:.4f}`\n"
        )
        handle.write(f"- Stability score: `{stability_score:.2f}` / 100\n\n")

        handle.write("## In-Sample vs Out-of-Sample\n\n")
        handle.write(f"- Current full-period in-sample balanced score: `{in_sample_balanced:.4f}`\n")
        handle.write(f"- Average rolling train balanced score: `{avg_train_balanced:.4f}`\n")
        handle.write(f"- Average rolling test balanced score: `{avg_test_balanced:.4f}`\n")
        handle.write(f"- Current full-period in-sample return/drawdown ratio: `{in_sample_return_dd:.4f}`\n")
        handle.write(f"- Average rolling train return/drawdown ratio: `{avg_train_return_dd:.4f}`\n")
        handle.write(f"- Average rolling test return/drawdown ratio: `{avg_test_return_dd:.4f}`\n")
        handle.write(f"- Current full-period in-sample accuracy: `{in_sample_accuracy:.4f}`\n")
        handle.write(f"- Average rolling train accuracy: `{avg_train_accuracy:.4f}`\n")
        handle.write(f"- Average rolling test accuracy: `{avg_test_accuracy:.4f}`\n\n")
        handle.write(
            "- Rolling train metrics are optimistic by design because each fold is evaluated on the same data used to fit that fold's model. "
            "The more important robustness signal is the yearly unseen test-window behavior.\n\n"
        )

        handle.write("## Overfit Assessment\n\n")
        handle.write(f"- Astro Engine V4 is `{overfit_assessment}` based on the current walk-forward evidence.\n")
        handle.write(
            f"- Average test balanced score is `{avg_test_balanced / max(avg_train_balanced, 1e-6):.4f}`x the rolling train average and "
            f"`{avg_test_balanced / max(in_sample_balanced, 1e-6):.4f}`x the current full-period in-sample result.\n"
        )


def main():
    args = parse_args()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feature_cols, feature_set_name = resolve_feature_columns(df, args.feature_set)
    horizon = pick_target_horizon()
    threshold_config = load_threshold_config()

    print(
        f"Walk-Forward Validation v1 | horizon={horizon}D | "
        f"feature_set={feature_set_name} | features={len(feature_cols)} | "
        f"train_years={args.train_years}"
    )

    results_df = run_walk_forward_validation(
        df=df,
        feature_cols=feature_cols,
        feature_set_name=feature_set_name,
        horizon=horizon,
        threshold_config=threshold_config,
        train_years=args.train_years,
    )
    results_df.to_csv(RESULTS_PATH, index=False)

    in_sample_summary = pd.read_csv(MODEL_SUMMARY_PATH)
    in_sample_row = in_sample_summary[
        in_sample_summary["horizon_days"].astype(int) == horizon
    ].iloc[0]

    create_report(results_df, in_sample_row, horizon=horizon, train_years=args.train_years)
    print(
        f"Saved {RESULTS_PATH} and {REPORT_PATH}. "
        f"Avg OOS balanced score={results_df['test_balanced_score'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
