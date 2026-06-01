from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

PREDICTIONS_PATH = DATA_DIR / "ml_predictions.csv"
MODEL_SUMMARY_PATH = DATA_DIR / "ml_model_summary.csv"
REGIME_WEIGHTED_RESULTS_PATH = DATA_DIR / "regime_weighted_results.csv"
PRODUCTION_BASELINE_SUMMARY_PATH = DATA_DIR / "ml_model_summary_before_regime_weighted_v4.csv"

ENSEMBLE_SIGNALS_PATH = DATA_DIR / "ensemble_signals_v1.csv"
ENSEMBLE_RESULTS_PATH = DATA_DIR / "ensemble_results.csv"
ENSEMBLE_REPORT_PATH = DATA_DIR / "ensemble_report.md"

ENSEMBLE_HORIZONS = [7, 14, 30, 60]
ENSEMBLE_METHODS = {
    "equal_weight_voting": "Equal Weight Voting",
    "balanced_score_weighted_voting": "Balanced Score Weighted Voting",
    "return_dd_weighted_voting": "Return/DD Weighted Voting",
}
MIN_SIGNAL_THRESHOLD = 0.15


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


def sharpe_like(returns):
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(365)


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


def select_best_row(summary_df):
    ranked = summary_df.sort_values(
        ["balanced_score", "return_drawdown_ratio", "direction_accuracy"],
        ascending=[False, False, False],
    )
    return ranked.iloc[0]


def build_horizon_weights(summary_df):
    subset = summary_df[summary_df["horizon_days"].isin(ENSEMBLE_HORIZONS)].copy()
    if subset["horizon_days"].nunique() != len(ENSEMBLE_HORIZONS):
        missing = sorted(set(ENSEMBLE_HORIZONS) - set(subset["horizon_days"].astype(int)))
        raise ValueError(f"Missing horizon summaries for ensemble construction: {missing}")

    subset["balanced_weight_raw"] = subset["balanced_score"].clip(lower=0)
    subset["return_dd_weight_raw"] = subset["return_drawdown_ratio"].clip(lower=0)

    balanced_total = float(subset["balanced_weight_raw"].sum())
    return_dd_total = float(subset["return_dd_weight_raw"].sum())

    subset["equal_weight"] = 1.0 / len(subset)
    subset["balanced_score_weight"] = (
        subset["balanced_weight_raw"] / balanced_total
        if balanced_total > 0
        else 1.0 / len(subset)
    )
    subset["return_dd_weight"] = (
        subset["return_dd_weight_raw"] / return_dd_total
        if return_dd_total > 0
        else 1.0 / len(subset)
    )

    return subset.sort_values("horizon_days").reset_index(drop=True)


def build_prediction_frame(predictions_df):
    subset = predictions_df[predictions_df["horizon"].isin(ENSEMBLE_HORIZONS)].copy()
    if subset.empty:
        raise ValueError("No ensemble horizons found in ml_predictions.csv")

    subset["date"] = pd.to_datetime(subset["date"])
    duplicates = subset.duplicated(subset=["date", "horizon"]).sum()
    if duplicates:
        raise ValueError(f"Duplicate date/horizon rows found in ml_predictions.csv: {duplicates}")

    base_columns = ["date", "price", "btc_return_1d"]
    base = subset[base_columns].drop_duplicates("date").sort_values("date").reset_index(drop=True)

    for column in ["ml_prob_up", "ml_position_raw", "actual_direction"]:
        pivot = subset.pivot(index="date", columns="horizon", values=column)
        pivot.columns = [f"{column}_{int(horizon)}d" for horizon in pivot.columns]
        pivot = pivot.reset_index()
        base = base.merge(pivot, on="date", how="left")

    base = base.sort_values("date").reset_index(drop=True)
    return base


def ensemble_signal_from_vote(vote_score):
    if vote_score > MIN_SIGNAL_THRESHOLD:
        return 1
    if vote_score < -MIN_SIGNAL_THRESHOLD:
        return -1
    return 0


def compute_ensemble_method(frame_df, weights_df, method_key, method_label):
    if method_key == "equal_weight_voting":
        weight_column = "equal_weight"
    elif method_key == "balanced_score_weighted_voting":
        weight_column = "balanced_score_weight"
    else:
        weight_column = "return_dd_weight"

    horizon_weights = {
        int(row["horizon_days"]): float(row[weight_column])
        for _, row in weights_df.iterrows()
    }

    rows = frame_df.copy()
    rows["weighting_method"] = method_key
    rows["weighting_label"] = method_label

    probability_sum = 0.0
    signal_sum = 0.0
    actual_sum = 0.0
    for horizon, weight in horizon_weights.items():
        probability_sum += rows[f"ml_prob_up_{horizon}d"] * weight
        signal_sum += rows[f"ml_position_raw_{horizon}d"] * weight
        actual_vote = rows[f"actual_direction_{horizon}d"].map({1: 1, 0: -1}).fillna(0)
        actual_sum += actual_vote * weight

    rows["ensemble_probability"] = probability_sum
    rows["ensemble_vote_score"] = signal_sum
    rows["ensemble_actual_vote"] = actual_sum
    rows["ensemble_signal"] = rows["ensemble_vote_score"].apply(ensemble_signal_from_vote)
    rows["ensemble_actual_signal"] = rows["ensemble_actual_vote"].apply(ensemble_signal_from_vote)
    rows["ensemble_confidence"] = (
        0.5 * rows["ensemble_vote_score"].abs()
        + 0.5 * ((rows["ensemble_probability"] - 0.5).abs() * 2.0)
    ).clip(0.0, 1.0)

    rows["ensemble_position"] = rows["ensemble_signal"].shift(1).fillna(0)
    rows["ensemble_strategy_return"] = rows["btc_return_1d"].fillna(0.0) * rows["ensemble_position"]
    rows["buy_hold_return"] = rows["btc_return_1d"].fillna(0.0)
    rows["ensemble_strategy_equity"] = (1 + rows["ensemble_strategy_return"]).cumprod()
    rows["buy_hold_equity"] = (1 + rows["buy_hold_return"]).cumprod()
    rows["ensemble_strategy_drawdown"] = (
        rows["ensemble_strategy_equity"] / rows["ensemble_strategy_equity"].cummax()
    ) - 1

    return rows


def summarize_ensemble(method_df):
    total_return = float(method_df["ensemble_strategy_equity"].iloc[-1] - 1)
    max_drawdown = float(method_df["ensemble_strategy_drawdown"].min())
    drawdown_abs = abs(max_drawdown)
    return_dd_ratio = total_return / drawdown_abs if drawdown_abs > 0 else np.nan
    trades = int((method_df["ensemble_signal"].diff().fillna(0) != 0).sum())
    active_mask = method_df["ensemble_signal"] != 0
    if active_mask.any():
        accuracy = float(
            (method_df.loc[active_mask, "ensemble_signal"] == method_df.loc[active_mask, "ensemble_actual_signal"]).mean()
        )
    else:
        accuracy = np.nan
    sharpe = sharpe_like(method_df["ensemble_strategy_return"])

    balanced_score = (
        total_return * 0.30
        + (sharpe if pd.notna(sharpe) else 0.0) * 0.35
        + (return_dd_ratio if pd.notna(return_dd_ratio) else 0.0) * 0.20
        - (drawdown_abs if pd.notna(drawdown_abs) else 0.0) * 1.25
        - trades * 0.002
    )

    return {
        "strategy_name": method_df["weighting_label"].iloc[0],
        "strategy_type": "ensemble_v1",
        "weighting_method": method_df["weighting_method"].iloc[0],
        "source_horizons": ",".join(str(h) for h in ENSEMBLE_HORIZONS),
        "balanced_score": float(balanced_score),
        "return_drawdown_ratio": float(return_dd_ratio) if pd.notna(return_dd_ratio) else np.nan,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "trades": trades,
        "accuracy": accuracy,
        "prediction_start": method_df["date"].min(),
        "prediction_end": method_df["date"].max(),
    }


def strategy_row_from_summary(strategy_name, strategy_type, summary_row):
    return {
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "weighting_method": "n/a",
        "source_horizons": str(int(summary_row["horizon_days"])),
        "balanced_score": float(summary_row["balanced_score"]),
        "return_drawdown_ratio": float(summary_row["return_drawdown_ratio"]),
        "max_drawdown": float(summary_row["ml_max_drawdown"]),
        "total_return": float(summary_row["ml_total_return"]),
        "trades": int(summary_row["number_of_trades"]),
        "accuracy": float(summary_row["direction_accuracy"]),
        "prediction_start": summary_row["prediction_start"],
        "prediction_end": summary_row["prediction_end"],
    }


def annotate_rankings(results_df):
    ranked = results_df.copy()
    ranked["trading_rank"] = ranked["balanced_score"].rank(ascending=False, method="dense")
    ranked["long_term_rank"] = ranked["total_return"].rank(ascending=False, method="dense")
    ranked["return_dd_rank"] = ranked["return_drawdown_ratio"].rank(ascending=False, method="dense")
    ranked["drawdown_rank"] = ranked["max_drawdown"].rank(ascending=False, method="dense")
    ranked["accuracy_rank"] = ranked["accuracy"].rank(ascending=False, method="dense")
    ranked["combined_rank_score"] = (
        ranked["trading_rank"]
        + ranked["long_term_rank"]
        + ranked["return_dd_rank"]
        + ranked["drawdown_rank"]
        + ranked["accuracy_rank"]
    )
    return ranked.sort_values(
        ["balanced_score", "return_drawdown_ratio", "total_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def write_report(results_df, weights_df):
    trading_best = results_df.sort_values(
        ["balanced_score", "return_drawdown_ratio", "total_return"],
        ascending=[False, False, False],
    ).iloc[0]
    long_term_best = results_df.sort_values(
        ["total_return", "balanced_score", "return_drawdown_ratio"],
        ascending=[False, False, False],
    ).iloc[0]
    combined_best = results_df.sort_values(
        ["combined_rank_score", "balanced_score", "return_drawdown_ratio"],
        ascending=[True, False, False],
    ).iloc[0]

    weights_display = weights_df[
        [
            "horizon_days",
            "balanced_score",
            "return_drawdown_ratio",
            "equal_weight",
            "balanced_score_weight",
            "return_dd_weight",
        ]
    ].copy()
    for column in [
        "balanced_score",
        "return_drawdown_ratio",
        "equal_weight",
        "balanced_score_weight",
        "return_dd_weight",
    ]:
        weights_display[column] = weights_display[column].map(lambda value: f"{float(value):.4f}")

    display = results_df[
        [
            "strategy_name",
            "strategy_type",
            "balanced_score",
            "return_drawdown_ratio",
            "max_drawdown",
            "total_return",
            "trades",
            "accuracy",
        ]
    ].copy()
    for column in ["balanced_score", "return_drawdown_ratio", "max_drawdown", "total_return", "accuracy"]:
        display[column] = display[column].map(lambda value: f"{float(value):.4f}")

    report_lines = [
        "# Ensemble Engine V1 Report",
        "",
        "Analysis only. `app.py` was not modified.",
        "",
        "## Ensemble Horizons",
        "- Horizons used: 7D, 14D, 30D, 60D",
        "- Ensemble signal source: `ml_position_raw` from `ml_predictions.csv`",
        "- Ensemble probability source: weighted average of `ml_prob_up`",
        "- Ensemble confidence: blend of vote strength and probability distance from 0.5",
        "",
        "## Horizon Weights",
        render_markdown_table(weights_display),
        "",
        "## Strategy Comparison",
        render_markdown_table(display),
        "",
        "## Portfolio Winners",
        f"- Best Trading Portfolio: {trading_best['strategy_name']} ({trading_best['balanced_score']:.4f})",
        f"- Best Long-Term Portfolio: {long_term_best['strategy_name']} ({long_term_best['total_return']:.4f})",
        f"- Best Combined Portfolio: {combined_best['strategy_name']} (combined rank score {combined_best['combined_rank_score']:.2f})",
    ]

    ENSEMBLE_REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    predictions_df = load_csv(
        PREDICTIONS_PATH,
        ["date", "price", "horizon", "ml_prob_up", "ml_position_raw", "actual_direction", "btc_return_1d"],
    )
    current_summary_df = load_csv(
        MODEL_SUMMARY_PATH,
        ["horizon_days", "balanced_score", "return_drawdown_ratio", "ml_max_drawdown", "ml_total_return", "number_of_trades", "direction_accuracy", "prediction_start", "prediction_end"],
    )
    regime_weighted_df = load_csv(
        REGIME_WEIGHTED_RESULTS_PATH,
        ["stage", "best_horizon_days", "balanced_score", "return_drawdown_ratio", "accuracy", "trades"],
    )
    production_summary_df = load_csv(
        PRODUCTION_BASELINE_SUMMARY_PATH,
        ["horizon_days", "balanced_score", "return_drawdown_ratio", "ml_max_drawdown", "ml_total_return", "number_of_trades", "direction_accuracy", "prediction_start", "prediction_end"],
    )

    weights_df = build_horizon_weights(current_summary_df)
    prediction_frame = build_prediction_frame(predictions_df)

    ensemble_frames = []
    ensemble_rows = []
    for method_key, method_label in ENSEMBLE_METHODS.items():
        method_frame = compute_ensemble_method(prediction_frame, weights_df, method_key, method_label)
        ensemble_frames.append(method_frame)
        ensemble_rows.append(summarize_ensemble(method_frame))

    ensemble_signals_df = pd.concat(ensemble_frames, ignore_index=True)
    ensemble_signals_df.to_csv(ENSEMBLE_SIGNALS_PATH, index=False)

    production_best = select_best_row(production_summary_df)
    v4_best = select_best_row(current_summary_df)

    results_rows = [
        strategy_row_from_summary("Production", "production", production_best),
        strategy_row_from_summary("Regime-Aware V4", "regime_aware_v4", v4_best),
        *ensemble_rows,
    ]

    results_df = pd.DataFrame(results_rows)
    results_df = annotate_rankings(results_df)
    results_df.to_csv(ENSEMBLE_RESULTS_PATH, index=False)
    write_report(results_df, weights_df)

    # Cross-check that the named regime-aware row lines up with the saved comparison file.
    saved_v4 = regime_weighted_df[regime_weighted_df["stage"] == "regime_aware_v4"]
    if not saved_v4.empty:
        saved_v4 = saved_v4.iloc[0]
        print(
            "Regime-aware V4 reference | "
            f"balanced_score={float(saved_v4['balanced_score']):.4f} "
            f"return_dd={float(saved_v4['return_drawdown_ratio']):.4f}"
        )

    print(f"Wrote {ENSEMBLE_SIGNALS_PATH}")
    print(f"Wrote {ENSEMBLE_RESULTS_PATH}")
    print(f"Wrote {ENSEMBLE_REPORT_PATH}")


if __name__ == "__main__":
    main()
