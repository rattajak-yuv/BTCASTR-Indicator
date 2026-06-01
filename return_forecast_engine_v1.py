import argparse
import os
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/ml_dataset.csv"
SELECTED_FEATURES_PATH = "data/selected_features.csv"
CURRENT_SUMMARY_PATH = "data/ml_model_summary.csv"
PRODUCTION_SUMMARY_PATH = "data/ml_model_summary_before_regime_weighted_v4.csv"
REGIME_WEIGHTED_RESULTS_PATH = "data/regime_weighted_results.csv"

RESULTS_PATH = "data/return_forecast_results.csv"
SUMMARY_PATH = "data/return_forecast_summary.csv"
REPORT_PATH = "data/return_forecast_report.md"

HORIZONS = [7, 14, 30, 60]
TRAIN_WINDOW = 730
TEST_WINDOW = 90
STEP_SIZE = 90

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
    message="X has feature names, but DecisionTreeRegressor was fitted without feature names",
)
warnings.filterwarnings(
    "ignore",
    message=(
        "`sklearn.utils.parallel.delayed` should be used with "
        "`sklearn.utils.parallel.Parallel`"
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train deterministic walk-forward BTC return-forecast models."
    )
    parser.add_argument(
        "--feature-set",
        choices=["all", "selected"],
        default="selected",
        help="Feature source to use for training.",
    )
    return parser.parse_args()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = (equity / peak) - 1
    return float(drawdown.min()) if not drawdown.empty else np.nan


def sharpe_like(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std() == 0:
        return np.nan
    return float((clean.mean() / clean.std()) * np.sqrt(365))


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
    feature_cols = sorted(
        col for col in df.columns
        if is_valid_feature_column(df, col)
    )
    if not feature_cols:
        raise ValueError("No valid numeric features found in ml_dataset.csv")
    return feature_cols


def load_selected_features(df: pd.DataFrame) -> List[str]:
    if not os.path.exists(SELECTED_FEATURES_PATH):
        raise FileNotFoundError(f"Missing {SELECTED_FEATURES_PATH}")

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


def robust_scale(series: pd.Series, fallback: float = 0.01, quantile: float = 0.75) -> float:
    clean = (
        pd.Series(series)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .abs()
    )
    if clean.empty:
        return fallback

    scale = clean.quantile(quantile)
    if pd.isna(scale) or scale <= 1e-6:
        return fallback
    return float(scale)


def forecast_direction_from_values(predicted_return: pd.Series, neutral_band: float) -> pd.Series:
    direction = np.where(
        predicted_return >= neutral_band,
        "LONG",
        np.where(predicted_return <= -neutral_band, "SHORT", "FLAT"),
    )
    return pd.Series(direction, index=predicted_return.index)


def confidence_from_forest(
    model: RandomForestRegressor,
    x_test: pd.DataFrame,
    y_train: pd.Series,
) -> np.ndarray:
    x_values = x_test.to_numpy()
    tree_predictions = np.vstack(
        [tree.predict(x_values) for tree in model.estimators_]
    )
    prediction_std = tree_predictions.std(axis=0)
    scale = max(robust_scale(y_train, fallback=0.01), float(np.nanstd(y_train)), 0.01)
    confidence = 1.0 / (1.0 + (prediction_std / scale))
    return np.clip(confidence, 0.0, 1.0)


def exposure_from_forecast(
    predicted_return: pd.Series,
    confidence_score: pd.Series,
    y_train: pd.Series,
) -> pd.Series:
    scale = robust_scale(y_train, fallback=0.01)
    raw_exposure = predicted_return / scale
    scaled_exposure = raw_exposure.clip(-1.0, 1.0) * confidence_score

    small_signal_band = scale * 0.10
    scaled_exposure = scaled_exposure.where(predicted_return.abs() >= small_signal_band, 0.0)
    return scaled_exposure.clip(-1.0, 1.0)


def compute_balanced_score(
    total_return: float,
    max_dd: float,
    strategy_returns: pd.Series,
    trades: int,
) -> Tuple[float, float]:
    sharpe = sharpe_like(strategy_returns)
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
    return float(balanced_score), float(return_dd_ratio) if pd.notna(return_dd_ratio) else np.nan


def run_return_forecast_for_horizon(
    df: pd.DataFrame,
    feature_cols: List[str],
    feature_set_name: str,
    horizon: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    target_col = f"future_return_{horizon}d"
    data = df.dropna(subset=["price", target_col]).copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.sort_values("date").reset_index(drop=True)

    results = []
    start = TRAIN_WINDOW
    model_counter = 0

    while start + TEST_WINDOW <= len(data):
        train = data.iloc[start - TRAIN_WINDOW:start].copy()
        test = data.iloc[start:start + TEST_WINDOW].copy()

        train = train.dropna(subset=feature_cols + [target_col])
        test = test.dropna(subset=feature_cols + [target_col])

        if len(train) < 300 or len(test) == 0:
            start += STEP_SIZE
            continue

        x_train = train[feature_cols]
        y_train = train[target_col].astype(float)
        x_test = test[feature_cols]
        y_test = test[target_col].astype(float)

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42 + horizon + model_counter,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        predicted_return = pd.Series(model.predict(x_test), index=test.index)
        confidence_score = pd.Series(
            confidence_from_forest(model, x_test, y_train),
            index=test.index,
        )
        neutral_band = robust_scale(y_train, fallback=0.01, quantile=0.25) * 0.25
        forecast_direction = forecast_direction_from_values(
            predicted_return,
            neutral_band=max(neutral_band, 0.001),
        )

        out = test[["date", "price"]].copy()
        out["horizon_days"] = horizon
        out["feature_set"] = feature_set_name
        out["predicted_return"] = predicted_return.values
        out["actual_return"] = y_test.values
        out["forecast_error"] = out["predicted_return"] - out["actual_return"]
        out["confidence_score"] = confidence_score.values
        out["forecast_direction"] = forecast_direction.values
        out["allocation_signal_raw"] = exposure_from_forecast(
            predicted_return,
            confidence_score,
            y_train,
        ).values
        out["train_start"] = train["date"].iloc[0]
        out["train_end"] = train["date"].iloc[-1]
        out["test_start"] = test["date"].iloc[0]
        out["test_end"] = test["date"].iloc[-1]
        results.append(out)

        model_counter += 1
        start += STEP_SIZE

    if not results:
        raise ValueError(f"No walk-forward forecast windows produced for {horizon}D")

    forecast_df = pd.concat(results, ignore_index=True)
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)
    forecast_df["btc_return_1d"] = forecast_df["price"].pct_change().fillna(0.0)
    forecast_df["allocation_exposure"] = forecast_df["allocation_signal_raw"].shift(1).fillna(0.0)
    forecast_df["allocation_direction"] = np.sign(forecast_df["allocation_exposure"])
    forecast_df["strategy_return"] = (
        forecast_df["btc_return_1d"] * forecast_df["allocation_exposure"]
    )
    forecast_df["buy_hold_return_same_period"] = forecast_df["btc_return_1d"]
    forecast_df["strategy_equity"] = (1 + forecast_df["strategy_return"]).cumprod()
    forecast_df["buy_hold_equity"] = (1 + forecast_df["buy_hold_return_same_period"]).cumprod()
    forecast_df["strategy_drawdown"] = (
        forecast_df["strategy_equity"] / forecast_df["strategy_equity"].cummax()
    ) - 1
    forecast_df["buy_hold_drawdown"] = (
        forecast_df["buy_hold_equity"] / forecast_df["buy_hold_equity"].cummax()
    ) - 1

    total_return = float(forecast_df["strategy_equity"].iloc[-1] - 1)
    max_dd = float(forecast_df["strategy_drawdown"].min())
    trades = int(
        (
            np.sign(forecast_df["allocation_signal_raw"]).diff().fillna(0) != 0
        ).sum()
    )
    mae = float(forecast_df["forecast_error"].abs().mean())
    rmse = float(np.sqrt(np.mean(np.square(forecast_df["forecast_error"]))))

    actual_direction = np.sign(forecast_df["actual_return"])
    predicted_direction = np.sign(forecast_df["predicted_return"])
    directional_accuracy = float((actual_direction == predicted_direction).mean())

    balanced_score, return_dd_ratio = compute_balanced_score(
        total_return=total_return,
        max_dd=max_dd,
        strategy_returns=forecast_df["strategy_return"],
        trades=trades,
    )

    buy_hold_total_return = float(forecast_df["buy_hold_equity"].iloc[-1] - 1)
    buy_hold_max_dd = float(forecast_df["buy_hold_drawdown"].min())
    buy_hold_balanced_score, buy_hold_return_dd_ratio = compute_balanced_score(
        total_return=buy_hold_total_return,
        max_dd=buy_hold_max_dd,
        strategy_returns=forecast_df["buy_hold_return_same_period"],
        trades=0,
    )

    summary = {
        "strategy_type": "return_forecast",
        "strategy_name": f"Return Forecast {horizon}D",
        "model": "RandomForestRegressor",
        "feature_set": feature_set_name,
        "horizon_days": horizon,
        "train_window_days": TRAIN_WINDOW,
        "test_window_days": TEST_WINDOW,
        "prediction_start": forecast_df["date"].iloc[0],
        "prediction_end": forecast_df["date"].iloc[-1],
        "selected_features": len(feature_cols),
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "accuracy": directional_accuracy,
        "average_confidence_score": float(forecast_df["confidence_score"].mean()),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "return_drawdown_ratio": return_dd_ratio,
        "balanced_score": balanced_score,
        "trades": trades,
        "buy_hold_return_same_period": buy_hold_total_return,
        "buy_hold_max_drawdown_same_period": buy_hold_max_dd,
        "buy_hold_return_drawdown_ratio_same_period": buy_hold_return_dd_ratio,
        "buy_hold_balanced_score_same_period": buy_hold_balanced_score,
    }

    return forecast_df, summary


def load_production_baseline() -> Dict[str, object]:
    prod = pd.read_csv(PRODUCTION_SUMMARY_PATH)
    best = prod.sort_values(
        ["balanced_score", "return_drawdown_ratio", "ml_total_return", "horizon_days"],
        ascending=[False, False, False, True],
    ).iloc[0]

    return {
        "strategy_type": "benchmark",
        "strategy_name": "Production",
        "model": best["model"],
        "feature_set": best["feature_set"],
        "horizon_days": int(best["horizon_days"]),
        "train_window_days": int(best["train_window_days"]),
        "test_window_days": int(best["test_window_days"]),
        "prediction_start": best["prediction_start"],
        "prediction_end": best["prediction_end"],
        "selected_features": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "directional_accuracy": float(best["direction_accuracy"]),
        "accuracy": float(best["direction_accuracy"]),
        "average_confidence_score": np.nan,
        "total_return": float(best["ml_total_return"]),
        "max_drawdown": float(best["ml_max_drawdown"]),
        "return_drawdown_ratio": float(best["return_drawdown_ratio"]),
        "balanced_score": float(best["balanced_score"]),
        "trades": int(best["number_of_trades"]),
        "buy_hold_return_same_period": float(best["buy_hold_return_same_period"]),
        "buy_hold_max_drawdown_same_period": float(best["buy_hold_max_drawdown_same_period"]),
        "buy_hold_return_drawdown_ratio_same_period": (
            float(best["buy_hold_return_same_period"]) / abs(float(best["buy_hold_max_drawdown_same_period"]))
            if float(best["buy_hold_max_drawdown_same_period"]) != 0
            else np.nan
        ),
        "buy_hold_balanced_score_same_period": np.nan,
    }


def load_regime_aware_v4_baseline() -> Dict[str, object]:
    current = pd.read_csv(CURRENT_SUMMARY_PATH)
    best = current.sort_values(
        ["balanced_score", "return_drawdown_ratio", "ml_total_return", "horizon_days"],
        ascending=[False, False, False, True],
    ).iloc[0]

    return {
        "strategy_type": "benchmark",
        "strategy_name": "Regime-Aware V4",
        "model": best["model"],
        "feature_set": best["feature_set"],
        "horizon_days": int(best["horizon_days"]),
        "train_window_days": int(best["train_window_days"]),
        "test_window_days": int(best["test_window_days"]),
        "prediction_start": best["prediction_start"],
        "prediction_end": best["prediction_end"],
        "selected_features": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "directional_accuracy": float(best["direction_accuracy"]),
        "accuracy": float(best["direction_accuracy"]),
        "average_confidence_score": np.nan,
        "total_return": float(best["ml_total_return"]),
        "max_drawdown": float(best["ml_max_drawdown"]),
        "return_drawdown_ratio": float(best["return_drawdown_ratio"]),
        "balanced_score": float(best["balanced_score"]),
        "trades": int(best["number_of_trades"]),
        "buy_hold_return_same_period": float(best["buy_hold_return_same_period"]),
        "buy_hold_max_drawdown_same_period": float(best["buy_hold_max_drawdown_same_period"]),
        "buy_hold_return_drawdown_ratio_same_period": (
            float(best["buy_hold_return_same_period"]) / abs(float(best["buy_hold_max_drawdown_same_period"]))
            if float(best["buy_hold_max_drawdown_same_period"]) != 0
            else np.nan
        ),
        "buy_hold_balanced_score_same_period": np.nan,
    }


def create_buy_and_hold_row(best_forecast_summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "strategy_type": "benchmark",
        "strategy_name": "Buy & Hold",
        "model": "Spot BTC",
        "feature_set": best_forecast_summary["feature_set"],
        "horizon_days": int(best_forecast_summary["horizon_days"]),
        "train_window_days": best_forecast_summary["train_window_days"],
        "test_window_days": best_forecast_summary["test_window_days"],
        "prediction_start": best_forecast_summary["prediction_start"],
        "prediction_end": best_forecast_summary["prediction_end"],
        "selected_features": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "directional_accuracy": np.nan,
        "accuracy": np.nan,
        "average_confidence_score": np.nan,
        "total_return": float(best_forecast_summary["buy_hold_return_same_period"]),
        "max_drawdown": float(best_forecast_summary["buy_hold_max_drawdown_same_period"]),
        "return_drawdown_ratio": float(best_forecast_summary["buy_hold_return_drawdown_ratio_same_period"]),
        "balanced_score": float(best_forecast_summary["buy_hold_balanced_score_same_period"]),
        "trades": 0,
        "buy_hold_return_same_period": float(best_forecast_summary["buy_hold_return_same_period"]),
        "buy_hold_max_drawdown_same_period": float(best_forecast_summary["buy_hold_max_drawdown_same_period"]),
        "buy_hold_return_drawdown_ratio_same_period": float(
            best_forecast_summary["buy_hold_return_drawdown_ratio_same_period"]
        ),
        "buy_hold_balanced_score_same_period": float(
            best_forecast_summary["buy_hold_balanced_score_same_period"]
        ),
    }


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

    body_rows = []
    for _, row in df.iterrows():
        body_rows.append(
            "| " + " | ".join(format_markdown_value(row[col]) for col in df.columns) + " |"
        )

    return "\n".join([header_row, separator_row] + body_rows)


def create_report(summary_df: pd.DataFrame, best_forecast: pd.Series):
    forecast_rows = summary_df[summary_df["strategy_type"] == "return_forecast"].copy()
    benchmark_rows = summary_df[summary_df["strategy_type"] == "benchmark"].copy()

    comparison_columns = [
        "strategy_name",
        "horizon_days",
        "balanced_score",
        "return_drawdown_ratio",
        "total_return",
        "max_drawdown",
        "trades",
        "accuracy",
    ]

    comparison_rows = pd.concat(
        [
            best_forecast.to_frame().T,
            benchmark_rows,
        ],
        ignore_index=True,
    )[comparison_columns]

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Return Forecast Engine v1\n\n")
        handle.write("## Best Forecast Horizon\n\n")
        handle.write(
            f"- Best forecast horizon by balanced score: `{int(best_forecast['horizon_days'])}D`\n"
        )
        handle.write(f"- Balanced score: `{best_forecast['balanced_score']:.4f}`\n")
        handle.write(f"- Return/drawdown ratio: `{best_forecast['return_drawdown_ratio']:.4f}`\n")
        handle.write(f"- Total return: `{best_forecast['total_return']:.4f}`\n")
        handle.write(f"- Max drawdown: `{best_forecast['max_drawdown']:.4f}`\n")
        handle.write(f"- Trades: `{int(best_forecast['trades'])}`\n")
        handle.write(f"- MAE: `{best_forecast['mae']:.4f}`\n")
        handle.write(f"- RMSE: `{best_forecast['rmse']:.4f}`\n")
        handle.write(f"- Directional accuracy: `{best_forecast['directional_accuracy']:.4f}`\n")
        handle.write(f"- Average confidence score: `{best_forecast['average_confidence_score']:.4f}`\n\n")

        handle.write("## Forecast Horizon Ranking\n\n")
        handle.write(
            dataframe_to_markdown(
                forecast_rows[
                    [
                        "strategy_name",
                        "balanced_score",
                        "return_drawdown_ratio",
                        "total_return",
                        "max_drawdown",
                        "trades",
                        "mae",
                        "rmse",
                        "directional_accuracy",
                    ]
                ]
            )
        )
        handle.write("\n\n## Portfolio Comparison\n\n")
        handle.write(dataframe_to_markdown(comparison_rows))
        handle.write("\n\n## Conclusion\n\n")

        v4_row = benchmark_rows[benchmark_rows["strategy_name"] == "Regime-Aware V4"].iloc[0]
        production_row = benchmark_rows[benchmark_rows["strategy_name"] == "Production"].iloc[0]
        buy_hold_row = benchmark_rows[benchmark_rows["strategy_name"] == "Buy & Hold"].iloc[0]

        handle.write(
            f"- Versus Regime-Aware V4: balanced score delta = "
            f"`{best_forecast['balanced_score'] - v4_row['balanced_score']:.4f}`\n"
        )
        handle.write(
            f"- Versus Production: balanced score delta = "
            f"`{best_forecast['balanced_score'] - production_row['balanced_score']:.4f}`\n"
        )
        handle.write(
            f"- Versus Buy & Hold on the same period: balanced score delta = "
            f"`{best_forecast['balanced_score'] - buy_hold_row['balanced_score']:.4f}`\n"
        )


def main():
    args = parse_args()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feature_cols, feature_set_name = resolve_feature_columns(df, args.feature_set)

    print(
        f"Return Forecast Engine v1 | feature set: {feature_set_name} | "
        f"selected features: {len(feature_cols)}"
    )

    forecast_frames = []
    forecast_summaries = []

    for horizon in HORIZONS:
        print(
            f"Training walk-forward return regressor for {horizon}D | "
            f"feature set={feature_set_name}"
        )
        forecast_df, summary = run_return_forecast_for_horizon(
            df=df,
            feature_cols=feature_cols,
            feature_set_name=feature_set_name,
            horizon=horizon,
        )
        forecast_frames.append(forecast_df)
        forecast_summaries.append(summary)

    results_df = pd.concat(forecast_frames, ignore_index=True)
    results_df = results_df.sort_values(["horizon_days", "date"]).reset_index(drop=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    forecast_summary_df = pd.DataFrame(forecast_summaries).sort_values(
        ["balanced_score", "return_drawdown_ratio", "total_return", "trades", "horizon_days"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    best_forecast = forecast_summary_df.iloc[0].to_dict()
    benchmark_rows = [
        load_production_baseline(),
        load_regime_aware_v4_baseline(),
        create_buy_and_hold_row(best_forecast),
    ]

    summary_df = pd.concat(
        [
            forecast_summary_df,
            pd.DataFrame(benchmark_rows),
        ],
        ignore_index=True,
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)

    create_report(summary_df, forecast_summary_df.iloc[0])
    print(
        f"Saved {RESULTS_PATH}, {SUMMARY_PATH}, and {REPORT_PATH}. "
        f"Best forecast horizon: {int(forecast_summary_df.iloc[0]['horizon_days'])}D"
    )


if __name__ == "__main__":
    main()
