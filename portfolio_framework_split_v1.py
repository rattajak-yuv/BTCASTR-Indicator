import os
from typing import Dict, List

import numpy as np
import pandas as pd

PREDICTIONS_PATH = "data/ml_predictions.csv"
SUMMARY_PATH = "data/ml_model_summary.csv"
REGIME_RESULTS_PATH = "data/regime_weighted_results.csv"

SPOT_RESULTS_PATH = "data/spot_portfolio_results.csv"
LONG_SHORT_RESULTS_PATH = "data/long_short_portfolio_results.csv"
REPORT_PATH = "data/portfolio_comparison_report.md"

SPOT_STATE_MAP = {
    1: ("BTC_100", 1.0),
    0: ("BTC_50", 0.5),
    -1: ("CASH_100", 0.0),
}

LONG_SHORT_STATE_MAP = {
    1: ("LONG_100", 1.0),
    0: ("CASH", 0.0),
    -1: ("SHORT_100", -1.0),
}


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity / peak) - 1).min()) if not equity.empty else np.nan


def sharpe_like(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std() == 0:
        return np.nan
    return float((clean.mean() / clean.std()) * np.sqrt(365))


def compute_balanced_score(total_return: float, max_dd: float, returns: pd.Series, trades: int) -> Dict[str, float]:
    sharpe = sharpe_like(returns)
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
        "sharpe_like": sharpe,
        "return_drawdown_ratio": float(return_dd_ratio) if pd.notna(return_dd_ratio) else np.nan,
        "balanced_score": float(balanced_score),
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
    rows = []

    for _, row in df.iterrows():
        rows.append(
            "| " + " | ".join(format_markdown_value(row[col]) for col in df.columns) + " |"
        )

    return "\n".join([header_row, separator_row] + rows)


def load_inputs():
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    summary = pd.read_csv(SUMMARY_PATH)
    regime_results = pd.read_csv(REGIME_RESULTS_PATH)
    return predictions, summary, regime_results


def map_state_series(raw_signal: pd.Series, state_map: Dict[int, tuple]) -> pd.DataFrame:
    normalized = raw_signal.fillna(0).astype(int).clip(-1, 1)
    state_name = normalized.map(lambda value: state_map[int(value)][0])
    exposure_raw = normalized.map(lambda value: state_map[int(value)][1]).astype(float)
    return pd.DataFrame(
        {
            "state_name_raw": state_name,
            "exposure_raw": exposure_raw,
        },
        index=raw_signal.index,
    )


def evaluate_portfolio(
    group: pd.DataFrame,
    summary_row: pd.Series,
    portfolio_name: str,
    state_map: Dict[int, tuple],
) -> Dict[str, object]:
    g = group.sort_values("date").reset_index(drop=True).copy()
    mapped = map_state_series(g["ml_position_raw"], state_map)

    g["portfolio_state_raw"] = mapped["state_name_raw"]
    g["portfolio_exposure_raw"] = mapped["exposure_raw"]
    g["portfolio_exposure"] = g["portfolio_exposure_raw"].shift(1).fillna(0.0)
    g["portfolio_state"] = g["portfolio_exposure"].map(
        lambda exposure: next(
            name for name, value in [item for item in state_map.values()]
            if value == exposure
        ) if exposure in [item[1] for item in state_map.values()] else "INIT_CASH"
    )

    g["portfolio_return"] = g["btc_return_1d"] * g["portfolio_exposure"]
    g["portfolio_equity"] = (1 + g["portfolio_return"]).cumprod()
    g["portfolio_drawdown"] = (g["portfolio_equity"] / g["portfolio_equity"].cummax()) - 1

    total_return = float(g["portfolio_equity"].iloc[-1] - 1)
    max_dd = float(g["portfolio_drawdown"].min())
    trades = int((g["portfolio_exposure_raw"].diff().fillna(0) != 0).sum())
    accuracy = float((g["ml_pred_direction"].astype(int) == g["actual_direction"].astype(int)).mean())

    score_metrics = compute_balanced_score(
        total_return=total_return,
        max_dd=max_dd,
        returns=g["portfolio_return"],
        trades=trades,
    )

    state_counts = g["portfolio_state_raw"].value_counts()
    state_mix = ", ".join(
        f"{state}:{int(count)}" for state, count in state_counts.items()
    )

    result = {
        "portfolio_type": portfolio_name,
        "signal_source": "ml_position_raw",
        "horizon_days": int(g["horizon"].iloc[0]),
        "prediction_start": g["date"].iloc[0].date().isoformat(),
        "prediction_end": g["date"].iloc[-1].date().isoformat(),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "return_drawdown_ratio": score_metrics["return_drawdown_ratio"],
        "balanced_score": score_metrics["balanced_score"],
        "accuracy": accuracy,
        "trades": trades,
        "sharpe_like": score_metrics["sharpe_like"],
        "state_mix": state_mix,
        "buy_hold_return_same_period": float(g["buy_hold_equity_ml_period"].iloc[-1] - 1),
        "buy_hold_max_drawdown_same_period": float(g["buy_hold_drawdown_ml_period"].min()),
        "v4_total_return_same_horizon": float(summary_row["ml_total_return"]),
        "v4_max_drawdown_same_horizon": float(summary_row["ml_max_drawdown"]),
        "v4_balanced_score_same_horizon": float(summary_row["balanced_score"]),
        "v4_return_drawdown_ratio_same_horizon": float(summary_row["return_drawdown_ratio"]),
        "v4_accuracy_same_horizon": float(summary_row["direction_accuracy"]),
        "v4_trades_same_horizon": int(summary_row["number_of_trades"]),
    }

    return result


def build_results(predictions: pd.DataFrame, summary: pd.DataFrame):
    horizons = sorted(predictions["horizon"].dropna().astype(int).unique().tolist())
    summary_index = summary.set_index("horizon_days")

    spot_results: List[Dict[str, object]] = []
    long_short_results: List[Dict[str, object]] = []

    for horizon in horizons:
        group = predictions[predictions["horizon"].astype(int) == horizon].copy()
        if group.empty or horizon not in summary_index.index:
            continue

        summary_row = summary_index.loc[horizon]
        spot_results.append(
            evaluate_portfolio(
                group=group,
                summary_row=summary_row,
                portfolio_name="spot_portfolio",
                state_map=SPOT_STATE_MAP,
            )
        )
        long_short_results.append(
            evaluate_portfolio(
                group=group,
                summary_row=summary_row,
                portfolio_name="long_short_portfolio",
                state_map=LONG_SHORT_STATE_MAP,
            )
        )

    spot_df = pd.DataFrame(spot_results).sort_values(
        ["balanced_score", "return_drawdown_ratio", "total_return", "horizon_days"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    long_short_df = pd.DataFrame(long_short_results).sort_values(
        ["balanced_score", "return_drawdown_ratio", "total_return", "horizon_days"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return spot_df, long_short_df


def recommend_portfolio_type(best_spot: pd.Series, best_long_short: pd.Series) -> str:
    if best_long_short["balanced_score"] > best_spot["balanced_score"] * 1.25:
        return (
            "Long/Short Portfolio is better suited for the current Astro Engine V4 "
            "because the v4 signal stream converts bearish calls into materially stronger "
            "return/risk performance than the no-shorting spot mapping."
        )

    if best_spot["max_drawdown"] > best_long_short["max_drawdown"]:
        return (
            "Spot Portfolio is better suited for the current Astro Engine V4 because it "
            "delivers a safer drawdown profile without giving up enough balanced score to "
            "justify short exposure."
        )

    return (
        "Long/Short Portfolio is slightly better suited for the current Astro Engine V4, "
        "but the gap is narrow enough that spot may still be preferable if operational "
        "simplicity or shorting constraints matter."
    )


def create_report(spot_df: pd.DataFrame, long_short_df: pd.DataFrame, regime_results: pd.DataFrame):
    best_spot = spot_df.iloc[0]
    best_long_short = long_short_df.iloc[0]
    recommendation = recommend_portfolio_type(best_spot, best_long_short)
    v4_stage = regime_results[regime_results["stage"] == "regime_aware_v4"].iloc[0]

    comparison_df = pd.DataFrame(
        [
            {
                "portfolio_type": "Best Spot Strategy",
                "horizon_days": int(best_spot["horizon_days"]),
                "balanced_score": best_spot["balanced_score"],
                "return_drawdown_ratio": best_spot["return_drawdown_ratio"],
                "total_return": best_spot["total_return"],
                "max_drawdown": best_spot["max_drawdown"],
                "accuracy": best_spot["accuracy"],
                "trades": int(best_spot["trades"]),
            },
            {
                "portfolio_type": "Best Long/Short Strategy",
                "horizon_days": int(best_long_short["horizon_days"]),
                "balanced_score": best_long_short["balanced_score"],
                "return_drawdown_ratio": best_long_short["return_drawdown_ratio"],
                "total_return": best_long_short["total_return"],
                "max_drawdown": best_long_short["max_drawdown"],
                "accuracy": best_long_short["accuracy"],
                "trades": int(best_long_short["trades"]),
            },
            {
                "portfolio_type": "Regime-Aware V4 Baseline",
                "horizon_days": int(v4_stage["best_horizon_days"]),
                "balanced_score": float(v4_stage["balanced_score"]),
                "return_drawdown_ratio": float(v4_stage["return_drawdown_ratio"]),
                "total_return": np.nan,
                "max_drawdown": np.nan,
                "accuracy": float(v4_stage["accuracy"]),
                "trades": int(v4_stage["trades"]),
            },
        ]
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Portfolio Framework Split v1\n\n")
        handle.write("## Best Spot Strategy\n\n")
        handle.write(f"- Horizon: `{int(best_spot['horizon_days'])}D`\n")
        handle.write(f"- Total return: `{best_spot['total_return']:.4f}`\n")
        handle.write(f"- Max drawdown: `{best_spot['max_drawdown']:.4f}`\n")
        handle.write(f"- Return/drawdown ratio: `{best_spot['return_drawdown_ratio']:.4f}`\n")
        handle.write(f"- Balanced score: `{best_spot['balanced_score']:.4f}`\n")
        handle.write(f"- Accuracy: `{best_spot['accuracy']:.4f}`\n")
        handle.write(f"- Trades: `{int(best_spot['trades'])}`\n")
        handle.write(f"- State mix: `{best_spot['state_mix']}`\n\n")

        handle.write("## Best Long/Short Strategy\n\n")
        handle.write(f"- Horizon: `{int(best_long_short['horizon_days'])}D`\n")
        handle.write(f"- Total return: `{best_long_short['total_return']:.4f}`\n")
        handle.write(f"- Max drawdown: `{best_long_short['max_drawdown']:.4f}`\n")
        handle.write(f"- Return/drawdown ratio: `{best_long_short['return_drawdown_ratio']:.4f}`\n")
        handle.write(f"- Balanced score: `{best_long_short['balanced_score']:.4f}`\n")
        handle.write(f"- Accuracy: `{best_long_short['accuracy']:.4f}`\n")
        handle.write(f"- Trades: `{int(best_long_short['trades'])}`\n")
        handle.write(f"- State mix: `{best_long_short['state_mix']}`\n\n")

        handle.write("## Portfolio Comparison\n\n")
        handle.write(dataframe_to_markdown(comparison_df))
        handle.write("\n\n## Recommendation\n\n")
        handle.write(f"- {recommendation}\n")


def main():
    predictions, summary, regime_results = load_inputs()
    spot_df, long_short_df = build_results(predictions, summary)

    spot_df.to_csv(SPOT_RESULTS_PATH, index=False)
    long_short_df.to_csv(LONG_SHORT_RESULTS_PATH, index=False)
    create_report(spot_df, long_short_df, regime_results)

    print(
        f"Saved {SPOT_RESULTS_PATH}, {LONG_SHORT_RESULTS_PATH}, and {REPORT_PATH}. "
        f"Best spot horizon: {int(spot_df.iloc[0]['horizon_days'])}D | "
        f"Best long/short horizon: {int(long_short_df.iloc[0]['horizon_days'])}D"
    )


if __name__ == "__main__":
    main()
