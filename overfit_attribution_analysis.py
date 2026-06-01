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

RESULTS_PATH = "data/ablation_results.csv"
REPORT_PATH = "data/ablation_report.md"

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

LONG_ALLOWED_REGIMES = {"uptrend", "strong_uptrend"}
SHORT_ALLOWED_REGIMES = {"downtrend", "crash_risk"}

STACK_DEFINITIONS = [
    {
        "stack_id": "A",
        "stack_name": "Astro Engine only",
        "feature_set": "all",
        "use_tuned_thresholds": False,
        "signal_layer_mode": "none",
        "portfolio_mapping": "spot",
        "description": "Base classifier on the live Astro Engine dataset with default thresholds and spot-style capital mapping.",
    },
    {
        "stack_id": "B",
        "stack_name": "Astro + Feature Selection",
        "feature_set": "selected",
        "use_tuned_thresholds": False,
        "signal_layer_mode": "none",
        "portfolio_mapping": "spot",
        "description": "Adds the selected-features pipeline while keeping default thresholds and spot-style mapping.",
    },
    {
        "stack_id": "C",
        "stack_name": "Astro + Feature Selection + Threshold Tuning",
        "feature_set": "selected",
        "use_tuned_thresholds": True,
        "signal_layer_mode": "none",
        "portfolio_mapping": "spot",
        "description": "Adds tuned long/short thresholds on top of selected features.",
    },
    {
        "stack_id": "D",
        "stack_name": "Astro + Feature Selection + Threshold Tuning + Signal Layer",
        "feature_set": "selected",
        "use_tuned_thresholds": True,
        "signal_layer_mode": "signal",
        "portfolio_mapping": "spot",
        "description": "Adds non-regime signal confirmation from ML probability, astro momentum, and trend alignment.",
    },
    {
        "stack_id": "E",
        "stack_name": "Astro + Feature Selection + Threshold Tuning + Signal Layer + Regime Layer",
        "feature_set": "selected",
        "use_tuned_thresholds": True,
        "signal_layer_mode": "signal_plus_regime",
        "portfolio_mapping": "spot",
        "description": "Adds regime-aware confirmation to the same signal-layer stack while still using spot-style mapping.",
    },
    {
        "stack_id": "F",
        "stack_name": "Full Production Stack",
        "feature_set": "selected",
        "use_tuned_thresholds": True,
        "signal_layer_mode": "signal_plus_regime",
        "portfolio_mapping": "long_short",
        "description": "Adds true long/short capital mapping to the regime-aware signal-layer stack.",
    },
]

warnings.filterwarnings(
    "ignore",
    message=(
        "`sklearn.utils.parallel.delayed` should be used with "
        "`sklearn.utils.parallel.Parallel`"
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stack ablation walk-forward analysis for Astro Engine V4."
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=5,
        help="Number of calendar years in each rolling training window.",
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


def normalize_regime_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def validate_thresholds(horizon: int, long_th: float, short_th: float):
    if pd.isna(long_th) or pd.isna(short_th):
        raise ValueError(f"Thresholds for horizon {horizon}D must not be NaN")
    if short_th >= long_th:
        raise ValueError(
            f"Invalid thresholds for horizon {horizon}D: short threshold must be below long threshold"
        )


def load_threshold_configs() -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    default_thresholds = {
        horizon: {
            "long": values["long"],
            "short": values["short"],
            "source": "default",
        }
        for horizon, values in PROBA_THRESHOLDS.items()
    }
    tuned_thresholds = {
        horizon: values.copy()
        for horizon, values in default_thresholds.items()
    }

    if not os.path.exists(THRESHOLD_RESULTS_PATH):
        return default_thresholds, tuned_thresholds

    tuning = pd.read_csv(THRESHOLD_RESULTS_PATH)
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
        if horizon not in tuned_thresholds:
            continue

        long_th = float(row["long_threshold"])
        short_th = float(row["short_threshold"])
        validate_thresholds(horizon, long_th, short_th)

        tuned_thresholds[horizon] = {
            "long": long_th,
            "short": short_th,
            "source": "tuned",
        }

    return default_thresholds, tuned_thresholds


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


def resolve_feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    if feature_set == "all":
        return load_all_features(df)
    return load_selected_features(df)


def pick_target_horizon() -> int:
    regime_results = pd.read_csv(REGIME_WEIGHTED_RESULTS_PATH)
    row = regime_results[regime_results["stage"] == "regime_aware_v4"].iloc[0]
    return int(row["best_horizon_days"])


def build_fold_schedule(df: pd.DataFrame, train_years: int) -> List[Tuple[int, int, int]]:
    years = sorted(df["date"].dt.year.dropna().unique().tolist())
    return [
        (years[idx - train_years], years[idx - 1], years[idx])
        for idx in range(train_years, len(years))
    ]


def create_raw_signal(prob_up: float, horizon: int, threshold_config: Dict[int, Dict[str, float]]) -> int:
    long_th = threshold_config[horizon]["long"]
    short_th = threshold_config[horizon]["short"]
    if prob_up >= long_th:
        return 1
    if prob_up <= short_th:
        return -1
    return 0


def create_ml_vote(prob_up: float, horizon: int, threshold_config: Dict[int, Dict[str, float]]) -> int:
    long_th = threshold_config[horizon]["long"]
    short_th = threshold_config[horizon]["short"]
    if prob_up >= long_th:
        return 2
    if prob_up <= short_th:
        return -2
    if prob_up > 0.55:
        return 1
    if prob_up < 0.45:
        return -1
    return 0


def create_momentum_vote(momentum_value: float) -> int:
    if pd.isna(momentum_value):
        return 0
    if momentum_value > 0:
        return 1
    if momentum_value < 0:
        return -1
    return 0


def create_trend_vote(trend_start_score: float, trend_end_score: float) -> int:
    if pd.isna(trend_start_score) or pd.isna(trend_end_score):
        return 0
    if trend_start_score > trend_end_score:
        return 1
    if trend_end_score > trend_start_score:
        return -1
    return 0


def create_regime_vote(regime_value) -> int:
    regime_label = normalize_regime_label(regime_value)
    if regime_label in LONG_ALLOWED_REGIMES:
        return 1
    if regime_label in SHORT_ALLOWED_REGIMES:
        return -1
    return 0


def apply_signal_layer(frame: pd.DataFrame, horizon: int, threshold_config: Dict[int, Dict[str, float]], mode: str) -> pd.DataFrame:
    out = frame.copy()
    out["ml_vote_component"] = out["ml_prob_up"].apply(lambda value: create_ml_vote(value, horizon, threshold_config))
    out["momentum_vote_component"] = out["astro_momentum_v2_smooth"].apply(create_momentum_vote)
    out["trend_vote_component"] = out.apply(
        lambda row: create_trend_vote(row["astro_trend_start_score"], row["astro_trend_end_score"]),
        axis=1,
    )
    out["regime_vote_component"] = out["astro_regime_v2"].apply(create_regime_vote)

    out["signal_vote_score_core"] = (
        out["ml_vote_component"]
        + out["momentum_vote_component"]
        + out["trend_vote_component"]
    )
    out["signal_vote_score_regime"] = out["signal_vote_score_core"] + out["regime_vote_component"]

    if mode == "signal":
        vote_score = out["signal_vote_score_core"]
    elif mode == "signal_plus_regime":
        vote_score = out["signal_vote_score_regime"]
    else:
        vote_score = pd.Series(np.zeros(len(out)), index=out.index)

    out["signal_vote_score"] = vote_score
    out["signal_position_raw"] = 0
    out.loc[vote_score >= 2, "signal_position_raw"] = 1
    out.loc[vote_score <= -2, "signal_position_raw"] = -1
    return out


def map_exposure(position_raw: pd.Series, portfolio_mapping: str) -> pd.Series:
    if portfolio_mapping == "spot":
        return position_raw.map({1: 1.0, 0: 0.5, -1: 0.0}).fillna(0.5)
    if portfolio_mapping == "long_short":
        return position_raw.map({1: 1.0, 0: 0.0, -1: -1.0}).fillna(0.0)
    raise ValueError(f"Unsupported portfolio_mapping: {portfolio_mapping}")


def compute_stack_metrics(frame: pd.DataFrame, position_column: str, portfolio_mapping: str) -> Dict[str, float]:
    g = frame.sort_values("date").reset_index(drop=True).copy()
    g["model_exposure_raw"] = map_exposure(g[position_column].astype(int), portfolio_mapping)
    g["model_exposure"] = g["model_exposure_raw"].shift(1).fillna(
        0.5 if portfolio_mapping == "spot" else 0.0
    )
    g["strategy_return"] = g["btc_return_1d"] * g["model_exposure"]
    g["strategy_equity"] = (1 + g["strategy_return"]).cumprod()
    g["strategy_drawdown"] = (g["strategy_equity"] / g["strategy_equity"].cummax()) - 1

    total_return = float(g["strategy_equity"].iloc[-1] - 1)
    max_dd = float(g["strategy_drawdown"].min())
    accuracy = float(
        accuracy_score(g["actual_direction"].astype(int), g["ml_pred_direction"].astype(int))
    )
    trades = int((g["model_exposure_raw"].diff().fillna(0) != 0).sum())
    sharpe = sharpe_like(g["strategy_return"])
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
        "sharpe_like": sharpe,
    }


def run_fold_for_stack(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    threshold_config: Dict[int, Dict[str, float]],
    stack: Dict[str, object],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    target_col = f"future_direction_{horizon}d"
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

    base_columns = [
        "date",
        "price",
        "btc_return_1d",
        "astro_regime_v2",
        "astro_momentum_v2_smooth",
        "astro_trend_start_score",
        "astro_trend_end_score",
        "actual_direction",
        "ml_prob_up",
        "ml_pred_direction",
        "ml_position_raw",
    ]

    train_eval = train[
        [
            "date",
            "price",
            "btc_return_1d",
            "astro_regime_v2",
            "astro_momentum_v2_smooth",
            "astro_trend_start_score",
            "astro_trend_end_score",
        ]
    ].copy()
    train_eval["actual_direction"] = y_train.values
    train_eval["ml_prob_up"] = train_prob_up
    train_eval["ml_pred_direction"] = train_pred
    train_eval["ml_position_raw"] = [create_raw_signal(prob, horizon, threshold_config) for prob in train_prob_up]

    test_eval = test[
        [
            "date",
            "price",
            "btc_return_1d",
            "astro_regime_v2",
            "astro_momentum_v2_smooth",
            "astro_trend_start_score",
            "astro_trend_end_score",
        ]
    ].copy()
    test_eval["actual_direction"] = y_test.values
    test_eval["ml_prob_up"] = test_prob_up
    test_eval["ml_pred_direction"] = test_pred
    test_eval["ml_position_raw"] = [create_raw_signal(prob, horizon, threshold_config) for prob in test_prob_up]

    if stack["signal_layer_mode"] != "none":
        train_eval = apply_signal_layer(train_eval, horizon, threshold_config, stack["signal_layer_mode"])
        test_eval = apply_signal_layer(test_eval, horizon, threshold_config, stack["signal_layer_mode"])
        position_column = "signal_position_raw"
    else:
        position_column = "ml_position_raw"

    train_metrics = compute_stack_metrics(train_eval[base_columns + (["signal_position_raw"] if "signal_position_raw" in train_eval.columns else [])], position_column, stack["portfolio_mapping"])
    test_metrics = compute_stack_metrics(test_eval[base_columns + (["signal_position_raw"] if "signal_position_raw" in test_eval.columns else [])], position_column, stack["portfolio_mapping"])
    return train_metrics, test_metrics


def run_ablation(df: pd.DataFrame, horizon: int, default_thresholds, tuned_thresholds, train_years: int) -> pd.DataFrame:
    data = df.dropna(subset=["date", "price", f"future_direction_{horizon}d"]).copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.sort_values("date").reset_index(drop=True)
    data["btc_return_1d"] = data["price"].pct_change().fillna(0.0)

    all_features = load_all_features(data)
    selected_features = load_selected_features(data)
    feature_map = {
        "all": all_features,
        "selected": selected_features,
    }

    folds = build_fold_schedule(data, train_years=train_years)
    rows = []

    for stack in STACK_DEFINITIONS:
        feature_cols = feature_map[stack["feature_set"]]
        threshold_config = tuned_thresholds if stack["use_tuned_thresholds"] else default_thresholds

        print(
            f"Evaluating stack {stack['stack_id']} | {stack['stack_name']} | "
            f"features={stack['feature_set']} ({len(feature_cols)}) | "
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
                    "stack_id": stack["stack_id"],
                    "stack_name": stack["stack_name"],
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
                f"train_score={train_metrics['balanced_score']:.4f} | "
                f"test_score={test_metrics['balanced_score']:.4f}"
            )

    if not rows:
        raise ValueError("No ablation folds were produced")

    return pd.DataFrame(rows)


def compute_stack_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for stack_id, group in results_df.groupby("stack_id"):
        g = group.sort_values("fold_id").reset_index(drop=True)
        avg_train = float(g["train_score"].mean())
        avg_test = float(g["test_score"].mean())
        std_test = float(g["test_score"].std(ddof=0))
        positive_test_rate = float((g["test_score"] > 0).mean())
        positive_return_rate = float((g["test_total_return"] > 0).mean())
        avg_test_accuracy = float(g["test_accuracy"].mean())
        avg_abs_dd = float(g["test_max_drawdown"].abs().mean())

        consistency = 1.0 / (1.0 + (std_test / max(abs(avg_test), 1.0)))
        accuracy_edge = np.clip((avg_test_accuracy - 0.50) / 0.10, 0.0, 1.0)
        drawdown_resilience = max(0.0, 1.0 - min(avg_abs_dd, 1.0))
        stability_score = 100 * (
            0.30 * positive_test_rate
            + 0.20 * positive_return_rate
            + 0.20 * consistency
            + 0.15 * accuracy_edge
            + 0.15 * drawdown_resilience
        )

        overfit_ratio = avg_test / avg_train if avg_train not in (0, np.nan) else np.nan
        overfit_gap = avg_test - avg_train
        alpha_score = avg_test * max(stability_score, 1.0) / 100.0

        summary_rows.append(
            {
                "stack_id": stack_id,
                "stack_name": g["stack_name"].iloc[0],
                "feature_set": g["feature_set"].iloc[0],
                "feature_count": int(g["feature_count"].iloc[0]),
                "threshold_mode": g["threshold_mode"].iloc[0],
                "signal_layer_mode": g["signal_layer_mode"].iloc[0],
                "portfolio_mapping": g["portfolio_mapping"].iloc[0],
                "folds": int(len(g)),
                "avg_train_score": avg_train,
                "avg_test_score": avg_test,
                "best_test_score": float(g["test_score"].max()),
                "worst_test_score": float(g["test_score"].min()),
                "avg_test_return_drawdown_ratio": float(g["test_return_drawdown_ratio"].mean()),
                "avg_test_total_return": float(g["test_total_return"].mean()),
                "avg_test_max_drawdown": float(g["test_max_drawdown"].mean()),
                "avg_test_accuracy": avg_test_accuracy,
                "stability_score": float(stability_score),
                "overfit_ratio": float(overfit_ratio) if pd.notna(overfit_ratio) else np.nan,
                "overfit_gap": float(overfit_gap),
                "positive_test_rate": positive_test_rate,
                "positive_return_rate": positive_return_rate,
                "alpha_score": float(alpha_score),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["avg_test_score", "stability_score", "avg_test_return_drawdown_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_df["delta_test_score_vs_prev_stack"] = summary_df["avg_test_score"].diff()
    summary_df["delta_stability_vs_prev_stack"] = summary_df["stability_score"].diff()
    summary_df["delta_overfit_ratio_vs_prev_stack"] = summary_df["overfit_ratio"].diff()
    return summary_df


def create_report(results_df: pd.DataFrame, summary_df: pd.DataFrame, horizon: int, train_years: int):
    best_alpha = summary_df.sort_values(
        ["avg_test_score", "stability_score", "avg_test_return_drawdown_ratio"],
        ascending=[False, False, False],
    ).iloc[0]
    most_overfit = summary_df.sort_values(
        ["overfit_ratio", "avg_train_score"],
        ascending=[True, False],
    ).iloc[0]
    most_stable = summary_df.sort_values(
        ["stability_score", "avg_test_score"],
        ascending=[False, False],
    ).iloc[0]

    robust_candidates = summary_df[
        (summary_df["avg_test_score"] > 0)
        & (summary_df["stability_score"] >= 45)
    ].copy()
    if robust_candidates.empty:
        minimal_robust = summary_df.sort_values(
            ["stability_score", "avg_test_score"],
            ascending=[False, False],
        ).iloc[0]
    else:
        minimal_robust = robust_candidates.sort_values(
            ["stack_id", "avg_test_score"],
            ascending=[True, False],
        ).iloc[0]

    summary_table = summary_df[
        [
            "stack_id",
            "stack_name",
            "avg_train_score",
            "avg_test_score",
            "stability_score",
            "overfit_ratio",
            "avg_test_return_drawdown_ratio",
            "avg_test_accuracy",
        ]
    ].copy()

    incremental_rows = []
    stack_order = {stack["stack_id"]: idx for idx, stack in enumerate(STACK_DEFINITIONS)}
    ordered_summary = summary_df.copy()
    ordered_summary["stack_order"] = ordered_summary["stack_id"].map(stack_order)
    ordered_summary = ordered_summary.sort_values("stack_order").reset_index(drop=True)

    for idx in range(1, len(ordered_summary)):
        prev_row = ordered_summary.iloc[idx - 1]
        curr_row = ordered_summary.iloc[idx]
        incremental_rows.append(
            {
                "component_added": curr_row["stack_name"].replace(prev_row["stack_name"] + " + ", ""),
                "from_stack": prev_row["stack_id"],
                "to_stack": curr_row["stack_id"],
                "delta_avg_test_score": curr_row["avg_test_score"] - prev_row["avg_test_score"],
                "delta_stability_score": curr_row["stability_score"] - prev_row["stability_score"],
                "delta_overfit_ratio": curr_row["overfit_ratio"] - prev_row["overfit_ratio"],
            }
        )

    incremental_df = pd.DataFrame(incremental_rows)
    real_alpha_components = incremental_df[
        (incremental_df["delta_avg_test_score"] > 0)
        & (incremental_df["delta_stability_score"] >= 0)
    ]
    overfit_components = incremental_df[
        (incremental_df["delta_avg_test_score"] < 0)
        & (incremental_df["delta_overfit_ratio"] < 0)
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Overfit Attribution Analysis\n\n")
        handle.write("## Scope\n\n")
        handle.write(f"- Target horizon: `{horizon}D`\n")
        handle.write(f"- Rolling training window: `{train_years}` calendar years\n")
        handle.write(
            "- Assumption: the current live `ml_dataset.csv` already embeds Raw Astro Recovery and Regime-Aware V4 score generation. "
            "This ablation therefore isolates the model-side layers we can truly toggle from the current repo state.\n\n"
        )

        handle.write("## Stack Summary\n\n")
        handle.write(dataframe_to_markdown(summary_table))
        handle.write("\n\n## Incremental Attribution\n\n")
        handle.write(dataframe_to_markdown(incremental_df))
        handle.write("\n\n## Findings\n\n")
        handle.write(
            f"- Component adding the strongest real out-of-sample alpha: `{best_alpha['stack_name']}` "
            f"with average test score `{best_alpha['avg_test_score']:.4f}` and stability `{best_alpha['stability_score']:.2f}`.\n"
        )
        handle.write(
            f"- Most overfit stack: `{most_overfit['stack_name']}` "
            f"with overfit ratio `{most_overfit['overfit_ratio']:.6f}`.\n"
        )
        handle.write(
            f"- Most stable stack: `{most_stable['stack_name']}` "
            f"with stability score `{most_stable['stability_score']:.2f}`.\n"
        )

        if not real_alpha_components.empty:
            alpha_labels = ", ".join(
                f"{row['from_stack']}->{row['to_stack']}"
                for _, row in real_alpha_components.iterrows()
            )
            handle.write(f"- Components adding real OOS alpha: `{alpha_labels}`.\n")
        else:
            handle.write("- No incremental component met the strict real-OOS-alpha test of improving both average test score and stability.\n")

        if not overfit_components.empty:
            overfit_labels = ", ".join(
                f"{row['from_stack']}->{row['to_stack']}"
                for _, row in overfit_components.iterrows()
            )
            handle.write(f"- Components increasing overfit pressure: `{overfit_labels}`.\n")
        else:
            handle.write("- No component showed the strict pattern of both lower test score and worse overfit ratio.\n")

        handle.write(
            f"- Recommended minimal robust architecture: `{minimal_robust['stack_id']} - {minimal_robust['stack_name']}` "
            f"with average test score `{minimal_robust['avg_test_score']:.4f}`, stability `{minimal_robust['stability_score']:.2f}`, "
            f"and overfit ratio `{minimal_robust['overfit_ratio']:.6f}`.\n"
        )


def main():
    args = parse_args()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    horizon = pick_target_horizon()
    default_thresholds, tuned_thresholds = load_threshold_configs()

    print(
        f"Overfit Attribution Analysis | horizon={horizon}D | train_years={args.train_years}"
    )

    results_df = run_ablation(
        df=df,
        horizon=horizon,
        default_thresholds=default_thresholds,
        tuned_thresholds=tuned_thresholds,
        train_years=args.train_years,
    )
    results_df.to_csv(RESULTS_PATH, index=False)

    summary_df = compute_stack_summary(results_df)
    create_report(results_df, summary_df, horizon=horizon, train_years=args.train_years)

    best = summary_df.iloc[0]
    print(
        f"Saved {RESULTS_PATH} and {REPORT_PATH}. "
        f"Best avg OOS stack: {best['stack_id']} ({best['stack_name']}) | "
        f"avg_test_score={best['avg_test_score']:.4f}"
    )


if __name__ == "__main__":
    main()
