import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PREDICTIONS_PATH = "data/ml_predictions.csv"
ENSEMBLE_SIGNALS_PATH = "data/ensemble_signals_v1.csv"
ENSEMBLE_RESULTS_PATH = "data/ensemble_results.csv"
REGIME_WEIGHTED_RESULTS_PATH = "data/regime_weighted_results.csv"
LONG_SHORT_RESULTS_PATH = "data/long_short_portfolio_results.csv"

RESULTS_PATH = "data/position_sizing_results.csv"
REPORT_PATH = "data/position_sizing_report.md"

SIZE_BUCKETS = [0.0, 0.25, 0.50, 0.75, 1.00]

REGIME_SUPPORT_LONG = {
    "strong_uptrend": 1.00,
    "uptrend": 0.85,
    "sideways": 0.55,
    "compression_zone": 0.40,
    "reversal_zone": 0.30,
    "downtrend": 0.15,
    "crash_risk": 0.05,
}

REGIME_SUPPORT_SHORT = {
    "crash_risk": 1.00,
    "downtrend": 0.85,
    "reversal_zone": 0.60,
    "compression_zone": 0.45,
    "sideways": 0.35,
    "uptrend": 0.15,
    "strong_uptrend": 0.05,
}


def sharpe_like(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std() == 0:
        return np.nan
    return float((clean.mean() / clean.std()) * np.sqrt(365))


def robust_scale(series: pd.Series, fallback: float = 1.0, quantile: float = 0.75) -> float:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().abs()
    if clean.empty:
        return fallback

    scale = clean.quantile(quantile)
    if pd.isna(scale) or scale <= 1e-6:
        return fallback

    return float(scale)


def compute_balanced_score(total_return: float, max_dd: float, returns: pd.Series, trades: int) -> Tuple[float, float]:
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

    return float(balanced_score), float(return_dd_ratio) if pd.notna(return_dd_ratio) else np.nan


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
    body = []

    for _, row in df.iterrows():
        body.append("| " + " | ".join(format_markdown_value(row[col]) for col in df.columns) + " |")

    return "\n".join([header_row, separator_row] + body)


def normalize_regime_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def pick_best_baseline_horizon() -> int:
    baseline = pd.read_csv(LONG_SHORT_RESULTS_PATH)
    best = baseline.sort_values(
        ["balanced_score", "return_drawdown_ratio", "total_return", "horizon_days"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return int(best["horizon_days"])


def pick_best_ensemble_method() -> str:
    ensemble = pd.read_csv(ENSEMBLE_RESULTS_PATH)
    ensemble = ensemble[ensemble["strategy_type"] == "ensemble_v1"].copy()
    best = ensemble.sort_values(
        ["balanced_score", "return_drawdown_ratio", "accuracy", "weighting_method"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return str(best["weighting_method"])


def load_base_frame() -> Tuple[pd.DataFrame, pd.Series, str]:
    selected_horizon = pick_best_baseline_horizon()
    best_ensemble_method = pick_best_ensemble_method()

    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    predictions = predictions[predictions["horizon"].astype(int) == selected_horizon].copy()
    predictions = predictions.sort_values("date").reset_index(drop=True)

    ensemble = pd.read_csv(ENSEMBLE_SIGNALS_PATH, parse_dates=["date"])
    ensemble = ensemble[ensemble["weighting_method"] == best_ensemble_method].copy()
    ensemble = ensemble.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    merged = predictions.merge(
        ensemble[
            [
                "date",
                "ensemble_probability",
                "ensemble_signal",
                "ensemble_confidence",
                "ensemble_vote_score",
                "ensemble_actual_signal",
            ]
        ],
        on="date",
        how="left",
    )

    if merged["ensemble_confidence"].isna().all():
        merged["ensemble_confidence"] = 0.5
        merged["ensemble_probability"] = 0.5
        merged["ensemble_signal"] = 0
        merged["ensemble_vote_score"] = 0.0

    baseline_results = pd.read_csv(LONG_SHORT_RESULTS_PATH)
    baseline_row = baseline_results[baseline_results["horizon_days"].astype(int) == selected_horizon].iloc[0]
    return merged, baseline_row, best_ensemble_method


def compute_ml_probability_strength(row: pd.Series) -> float:
    signal = int(row["ml_position_raw"])
    prob_up = float(row["ml_prob_up"])
    long_th = float(row["long_probability_threshold"])
    short_th = float(row["short_probability_threshold"])

    if signal > 0:
        scale = max(1.0 - long_th, 1e-6)
        return float(np.clip((prob_up - long_th) / scale, 0.0, 1.0))

    if signal < 0:
        scale = max(short_th, 1e-6)
        return float(np.clip((short_th - prob_up) / scale, 0.0, 1.0))

    return 0.0


def classify_volatility_state(series: pd.Series) -> pd.Series:
    abs_values = series.abs()
    low_cutoff = abs_values.quantile(0.33)
    high_cutoff = abs_values.quantile(0.67)

    return pd.Series(
        np.where(
            abs_values >= high_cutoff,
            "HighVol",
            np.where(abs_values <= low_cutoff, "LowVol", "MidVol"),
        ),
        index=series.index,
    )


def regime_support_strength(regime_label: str, signal: int) -> float:
    if signal > 0:
        return REGIME_SUPPORT_LONG.get(regime_label, 0.35)
    if signal < 0:
        return REGIME_SUPPORT_SHORT.get(regime_label, 0.35)
    return 0.0


def bucketize_size(score: float) -> float:
    if score < 0.20:
        return 0.0
    if score < 0.40:
        return 0.25
    if score < 0.60:
        return 0.50
    if score < 0.80:
        return 0.75
    return 1.0


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["btc_return_1d"] = out["btc_return_1d"].fillna(0.0)
    out["signal_direction"] = np.sign(out["ml_position_raw"]).astype(int)
    out["ensemble_signal_direction"] = np.sign(out["ensemble_signal"]).fillna(0).astype(int)
    out["regime_label_normalized"] = out["astro_regime_v2"].apply(normalize_regime_label)

    momentum_scale = robust_scale(out["astro_momentum_v2_smooth"], fallback=1.0)
    vote_scale = max(robust_scale(out["signal_vote_score"], fallback=1.0), 1.0)

    out["ml_probability_strength"] = out.apply(compute_ml_probability_strength, axis=1)
    out["ensemble_confidence_strength"] = out["ensemble_confidence"].fillna(0.5).clip(0.0, 1.0)

    signed_momentum = out["signal_direction"] * out["astro_momentum_v2_smooth"].fillna(0.0)
    out["momentum_alignment_strength"] = (
        (np.tanh(signed_momentum / momentum_scale) + 1.0) / 2.0
    ).clip(0.0, 1.0)

    out["regime_strength"] = out.apply(
        lambda row: regime_support_strength(row["regime_label_normalized"], int(row["signal_direction"])),
        axis=1,
    )

    out["volatility_state"] = classify_volatility_state(out["astro_volatility_score"].fillna(0.0))
    out["volatility_multiplier"] = out["volatility_state"].map(
        {"LowVol": 1.00, "MidVol": 0.75, "HighVol": 0.50}
    ).fillna(0.75)
    out["hybrid_volatility_multiplier"] = out["volatility_state"].map(
        {"LowVol": 1.00, "MidVol": 0.85, "HighVol": 0.65}
    ).fillna(0.85)

    out["signal_vote_strength"] = (out["signal_vote_score"].abs() / vote_scale).clip(0.0, 1.0)
    out["ensemble_alignment_strength"] = np.where(
        out["signal_direction"] == 0,
        0.0,
        np.where(
            out["ensemble_signal_direction"] == out["signal_direction"],
            1.0,
            np.where(out["ensemble_signal_direction"] == 0, 0.50, 0.15),
        ),
    )
    return out


def add_sizing_methods(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    base = df.copy()

    base_confidence = (
        base["ml_probability_strength"] * 0.40
        + base["ensemble_confidence_strength"] * 0.25
        + base["momentum_alignment_strength"] * 0.15
        + base["regime_strength"] * 0.10
        + base["signal_vote_strength"] * 0.10
    ).clip(0.0, 1.0)

    confidence_score = (
        base["ml_probability_strength"] * 0.35
        + base["ensemble_confidence_strength"] * 0.30
        + base["momentum_alignment_strength"] * 0.15
        + base["regime_strength"] * 0.10
        + base["ensemble_alignment_strength"] * 0.10
    ).clip(0.0, 1.0)

    volatility_adjusted_score = (base_confidence * base["volatility_multiplier"]).clip(0.0, 1.0)

    hybrid_score = (
        base["ml_probability_strength"] * 0.28
        + base["ensemble_confidence_strength"] * 0.22
        + base["momentum_alignment_strength"] * 0.15
        + base["regime_strength"] * 0.12
        + base["signal_vote_strength"] * 0.13
        + base["ensemble_alignment_strength"] * 0.10
    )
    hybrid_score = (hybrid_score * base["hybrid_volatility_multiplier"]).clip(0.0, 1.0)

    method_configs = {
        "Fixed 100%": pd.Series(np.where(base["signal_direction"] != 0, 1.0, 0.0), index=base.index),
        "Confidence-Based Sizing": confidence_score.apply(bucketize_size),
        "Volatility-Adjusted Sizing": volatility_adjusted_score.apply(bucketize_size),
        "Hybrid Sizing": hybrid_score.apply(bucketize_size),
    }

    for method_name, position_sizes in method_configs.items():
        frame = base.copy()
        frame["sizing_method"] = method_name
        frame["position_size"] = np.where(frame["signal_direction"] == 0, 0.0, position_sizes.astype(float))
        frame["net_exposure_raw"] = frame["position_size"] * frame["signal_direction"]
        frame["net_exposure"] = frame["net_exposure_raw"].shift(1).fillna(0.0)
        frame["strategy_return"] = frame["btc_return_1d"] * frame["net_exposure"]
        frame["strategy_equity"] = (1 + frame["strategy_return"]).cumprod()
        frame["strategy_drawdown"] = (
            frame["strategy_equity"] / frame["strategy_equity"].cummax()
        ) - 1
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def summarize_methods(detail_df: pd.DataFrame, baseline_row: pd.Series, selected_horizon: int, ensemble_method: str) -> pd.DataFrame:
    summaries: List[Dict[str, object]] = []

    for method_name, group in detail_df.groupby("sizing_method"):
        g = group.sort_values("date").reset_index(drop=True)
        total_return = float(g["strategy_equity"].iloc[-1] - 1)
        max_dd = float(g["strategy_drawdown"].min())
        trades = int((g["net_exposure_raw"].diff().fillna(0) != 0).sum())
        balanced_score, return_dd_ratio = compute_balanced_score(
            total_return=total_return,
            max_dd=max_dd,
            returns=g["strategy_return"],
            trades=trades,
        )
        volatility = float(g["strategy_return"].std() * np.sqrt(365))

        active_mask = g["position_size"] > 0
        if active_mask.any():
            predicted_binary = np.where(g.loc[active_mask, "net_exposure_raw"] > 0, 1, 0)
            actual_binary = g.loc[active_mask, "actual_direction"].astype(int).to_numpy()
            accuracy = float((predicted_binary == actual_binary).mean())
        else:
            accuracy = np.nan

        long_days = int((g["net_exposure_raw"] > 0).sum())
        short_days = int((g["net_exposure_raw"] < 0).sum())
        flat_days = int((g["net_exposure_raw"] == 0).sum())

        size_distribution = (
            g["position_size"]
            .value_counts()
            .sort_index()
            .reindex(SIZE_BUCKETS, fill_value=0)
        )
        size_mix = ", ".join(f"{bucket:.2f}:{int(count)}" for bucket, count in size_distribution.items())

        summaries.append(
            {
                "sizing_method": method_name,
                "horizon_days": selected_horizon,
                "ensemble_method": ensemble_method,
                "prediction_start": g["date"].iloc[0].date().isoformat(),
                "prediction_end": g["date"].iloc[-1].date().isoformat(),
                "total_return": total_return,
                "max_drawdown": max_dd,
                "return_drawdown_ratio": return_dd_ratio,
                "balanced_score": balanced_score,
                "volatility": volatility,
                "accuracy": accuracy,
                "trades": trades,
                "avg_position_size": float(g["position_size"].mean()),
                "avg_abs_net_exposure": float(g["net_exposure"].abs().mean()),
                "long_days": long_days,
                "short_days": short_days,
                "flat_days": flat_days,
                "size_mix": size_mix,
                "baseline_total_return": float(baseline_row["total_return"]),
                "baseline_max_drawdown": float(baseline_row["max_drawdown"]),
                "baseline_return_drawdown_ratio": float(baseline_row["return_drawdown_ratio"]),
                "baseline_balanced_score": float(baseline_row["balanced_score"]),
                "baseline_volatility": np.nan,
                "baseline_accuracy": float(baseline_row["accuracy"]),
                "baseline_trades": int(baseline_row["trades"]),
                "delta_total_return_vs_baseline": total_return - float(baseline_row["total_return"]),
                "delta_max_drawdown_vs_baseline": max_dd - float(baseline_row["max_drawdown"]),
                "delta_return_drawdown_ratio_vs_baseline": (
                    return_dd_ratio - float(baseline_row["return_drawdown_ratio"])
                    if pd.notna(return_dd_ratio)
                    else np.nan
                ),
                "delta_balanced_score_vs_baseline": balanced_score - float(baseline_row["balanced_score"]),
                "delta_accuracy_vs_baseline": (
                    accuracy - float(baseline_row["accuracy"])
                    if pd.notna(accuracy)
                    else np.nan
                ),
                "delta_trades_vs_baseline": trades - int(baseline_row["trades"]),
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values(
        ["balanced_score", "return_drawdown_ratio", "total_return", "volatility"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return summary_df


def create_report(summary_df: pd.DataFrame, baseline_row: pd.Series, selected_horizon: int, ensemble_method: str):
    aggressive = summary_df.sort_values(
        ["total_return", "balanced_score", "return_drawdown_ratio"],
        ascending=[False, False, False],
    ).iloc[0]
    balanced = summary_df.sort_values(
        ["balanced_score", "return_drawdown_ratio", "volatility"],
        ascending=[False, False, True],
    ).iloc[0]
    conservative = summary_df[summary_df["total_return"] > 0].sort_values(
        ["volatility", "max_drawdown", "balanced_score"],
        ascending=[True, False, False],
    ).iloc[0]

    comparison_df = summary_df[
        [
            "sizing_method",
            "balanced_score",
            "return_drawdown_ratio",
            "total_return",
            "max_drawdown",
            "volatility",
            "accuracy",
            "trades",
        ]
    ].copy()

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Position Sizing Engine v2\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- Base horizon: `{selected_horizon}D`\n")
        handle.write(f"- Ensemble confidence source: `{ensemble_method}`\n")
        handle.write(
            f"- Current Long/Short baseline balanced score: `{float(baseline_row['balanced_score']):.4f}`\n\n"
        )

        handle.write("## Method Comparison\n\n")
        handle.write(dataframe_to_markdown(comparison_df))
        handle.write("\n\n## Winners\n\n")
        handle.write(f"- Best Aggressive Strategy: `{aggressive['sizing_method']}`\n")
        handle.write(f"  Total return: `{aggressive['total_return']:.4f}`\n")
        handle.write(f"  Balanced score: `{aggressive['balanced_score']:.4f}`\n")
        handle.write(f"  Volatility: `{aggressive['volatility']:.4f}`\n")
        handle.write(f"- Best Balanced Strategy: `{balanced['sizing_method']}`\n")
        handle.write(f"  Total return: `{balanced['total_return']:.4f}`\n")
        handle.write(f"  Balanced score: `{balanced['balanced_score']:.4f}`\n")
        handle.write(f"  Return/drawdown ratio: `{balanced['return_drawdown_ratio']:.4f}`\n")
        handle.write(f"- Best Conservative Strategy: `{conservative['sizing_method']}`\n")
        handle.write(f"  Total return: `{conservative['total_return']:.4f}`\n")
        handle.write(f"  Return/drawdown ratio: `{conservative['return_drawdown_ratio']:.4f}`\n")
        handle.write(f"  Volatility: `{conservative['volatility']:.4f}`\n\n")

        best_overall = balanced
        handle.write("## Baseline Comparison\n\n")
        handle.write(
            f"- Best balanced sizing method vs current long/short baseline: "
            f"`{best_overall['delta_balanced_score_vs_baseline']:.4f}` balanced-score delta, "
            f"`{best_overall['delta_return_drawdown_ratio_vs_baseline']:.4f}` return/drawdown delta, "
            f"`{best_overall['delta_total_return_vs_baseline']:.4f}` total-return delta.\n"
        )

        if best_overall["balanced_score"] > float(baseline_row["balanced_score"]):
            handle.write(
                "- Position sizing improved the current Long/Short portfolio on the primary balanced-score objective.\n"
            )
        else:
            handle.write(
                "- Position sizing did not beat the current Long/Short baseline on balanced score, even if it improved one or more secondary risk metrics.\n"
            )


def main():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"Missing {PREDICTIONS_PATH}")

    base_df, baseline_row, ensemble_method = load_base_frame()
    selected_horizon = int(base_df["horizon"].iloc[0])

    prepared = prepare_features(base_df)
    detail_df = add_sizing_methods(prepared)
    detail_df = detail_df.sort_values(["sizing_method", "date"]).reset_index(drop=True)
    detail_df.to_csv(RESULTS_PATH, index=False)

    summary_df = summarize_methods(
        detail_df=detail_df,
        baseline_row=baseline_row,
        selected_horizon=selected_horizon,
        ensemble_method=ensemble_method,
    )
    create_report(summary_df, baseline_row, selected_horizon, ensemble_method)

    best = summary_df.iloc[0]
    print(
        f"Saved {RESULTS_PATH} and {REPORT_PATH}. "
        f"Best balanced method: {best['sizing_method']} | "
        f"balanced_score={best['balanced_score']:.4f}"
    )


if __name__ == "__main__":
    main()
