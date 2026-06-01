from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from build_ml_dataset import (
    ASPECT_STRENGTH_COLUMNS,
    DAILY_PATH,
    NATAL_TARGET_STRENGTH_COLUMNS,
    PLANET_SIGNAL_COLUMNS,
    RAW_PATH,
    add_rolling_features,
    build_raw_aspect_features,
)
from forecast_intelligence_v1 import classify_windows
from forecast_system_v1 import (
    TARGET_HORIZON_DAYS,
    build_forecast_windows,
    compute_astro_score,
    detect_turning_points,
    load_selected_features,
    probability_to_signal,
    risk_level_from_features,
    run_historical_walk_forward_predictions,
)
from forecast_taxonomy_v2 import (
    POSTURE_MAP,
    build_class_evidence,
    classify_taxonomy_v2,
    taxonomy_reason,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

FORECAST_CALIBRATION_PATH = DATA_DIR / "forecast_calibration.csv"

MOMENTUM_AUDIT_PATH = DATA_DIR / "astro_momentum_alpha_audit.csv"
TAXONOMY_AUDIT_PATH = DATA_DIR / "taxonomy_alpha_audit.csv"
TURNING_POINT_AUDIT_PATH = DATA_DIR / "turning_point_alpha_audit.csv"
BACKTEST_RESULTS_PATH = DATA_DIR / "institutional_backtest_results.csv"
BACKTEST_ANNUAL_PATH = DATA_DIR / "institutional_backtest_annual.csv"
REPORT_PATH = DATA_DIR / "astro_alpha_audit_report.md"

HORIZONS = [7, 14, 30]
MIN_ANALYSIS_YEAR = 2014


def format_markdown_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    headers = [str(col) for col in df.columns]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_markdown_value(row[col]) for col in df.columns) + " |")
    return "\n".join([header_row, separator_row] + rows)


def resolve_git_conflicts(text: str, preferred_side: str = "top") -> str:
    if "<<<<<<<" not in text:
        return text

    keep_top = preferred_side == "top"
    resolved: List[str] = []
    state = "normal"
    top_lines: List[str] = []
    bottom_lines: List[str] = []

    for line in text.splitlines(keepends=True):
        if line.startswith("<<<<<<<"):
            state = "top"
            top_lines = []
            bottom_lines = []
            continue
        if state == "top" and line.startswith("======="):
            state = "bottom"
            continue
        if state == "bottom" and line.startswith(">>>>>>>"):
            resolved.extend(top_lines if keep_top else bottom_lines)
            state = "normal"
            continue

        if state == "normal":
            resolved.append(line)
        elif state == "top":
            top_lines.append(line)
        else:
            bottom_lines.append(line)

    if state != "normal":
        raise ValueError("Unterminated git conflict markers detected while loading CSV data")
    return "".join(resolved)


def read_csv_conflict_safe(path: Path | str, **kwargs) -> pd.DataFrame:
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    resolved = resolve_git_conflicts(text, preferred_side="top")
    return pd.read_csv(io.StringIO(resolved), **kwargs)


def build_full_feature_frame_safe() -> pd.DataFrame:
    daily = read_csv_conflict_safe(ROOT / DAILY_PATH, parse_dates=["date"])
    raw = read_csv_conflict_safe(ROOT / RAW_PATH)
    raw_features = build_raw_aspect_features(raw)

    df = daily.merge(raw_features, on="date", how="left", suffixes=("", "_raw"))

    raw_feature_cols = [
        c for c in df.columns
        if (
            c.startswith("planet_")
            or c.startswith("aspect_count_")
            or c in PLANET_SIGNAL_COLUMNS
            or c in ASPECT_STRENGTH_COLUMNS
            or c in NATAL_TARGET_STRENGTH_COLUMNS
            or c in {
                "house_activation_strength",
                "raw_astro_total_strength",
                "raw_astro_directional_signal",
                "raw_astro_event_count",
            }
        )
    ]
    df[raw_feature_cols] = df[raw_feature_cols].fillna(0)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    base_feature_cols = [
        "astro_bullish_score",
        "astro_bearish_score",
        "astro_reversal_score",
        "astro_volatility_score",
        "astro_compression_score",
        "astro_trend_start_score",
        "astro_trend_end_score",
        "astro_momentum_v2",
        "astro_momentum_v2_smooth",
        "astro_bullish_score_smooth",
        "astro_bearish_score_smooth",
    ]
    base_feature_cols = [col for col in base_feature_cols if col in df.columns]

    for col in base_feature_cols:
        df = add_rolling_features(df, col)

    df["btc_return_1d"] = df["price"].pct_change()
    df["btc_return_3d"] = df["price"].pct_change(3)
    df["btc_return_7d"] = df["price"].pct_change(7)
    df["btc_return_14d"] = df["price"].pct_change(14)
    df["btc_return_30d"] = df["price"].pct_change(30)
    df["btc_vol_7d"] = df["btc_return_1d"].rolling(7).std()
    df["btc_vol_14d"] = df["btc_return_1d"].rolling(14).std()
    df["btc_vol_30d"] = df["btc_return_1d"].rolling(30).std()
    df["future_return_1d"] = df["price"].shift(-1) / df["price"] - 1

    for horizon in [3, 7, 14, 30, 60, 90]:
        df[f"future_return_{horizon}d"] = df["price"].shift(-horizon) / df["price"] - 1
        df[f"future_direction_{horizon}d"] = (df[f"future_return_{horizon}d"] > 0).astype(int)

    return df.sort_values("date").reset_index(drop=True)


def build_historical_engine_frame() -> pd.DataFrame:
    full_df = build_full_feature_frame_safe()
    feature_cols = load_selected_features(full_df)

    historical_df = full_df[full_df["price"].notna()].copy()
    historical_df = historical_df.replace([np.inf, -np.inf], np.nan)
    historical_df = historical_df.dropna(subset=feature_cols + [f"future_direction_{TARGET_HORIZON_DAYS}d"])
    historical_df = historical_df.sort_values("date").reset_index(drop=True)

    historical_predictions = run_historical_walk_forward_predictions(historical_df, feature_cols)

    merged = historical_predictions.merge(
        full_df[
            [
                "date",
                "future_return_1d",
                "future_return_7d",
                "future_return_14d",
                "future_return_30d",
                "btc_return_1d",
                "btc_vol_30d",
                "astro_regime_v2",
                "astro_momentum_v2",
                "astro_momentum_v2_smooth",
            ]
        ],
        on="date",
        how="left",
    )
    rename_map = {
        "future_return_7d_x": "future_return_7d_model",
        "future_return_7d_y": "future_return_7d",
        "astro_momentum_v2_smooth_x": "astro_momentum_v2_smooth",
    }
    merged = merged.rename(columns=rename_map)
    for redundant_col in ["astro_momentum_v2_smooth_y"]:
        if redundant_col in merged.columns:
            merged = merged.drop(columns=redundant_col)
    if "future_return_7d" not in merged.columns and "future_return_7d_model" in merged.columns:
        merged["future_return_7d"] = merged["future_return_7d_model"]

    historical_vol_reference = historical_df["astro_volatility_score"].dropna()
    merged["risk_level"] = risk_level_from_features(
        merged,
        historical_volatility_reference=historical_vol_reference,
        confidence_score=merged["confidence_score"],
    )
    merged["forecast_strength"] = np.where(
        merged["confidence_score"] >= 0.75,
        "Very Strong",
        np.where(merged["confidence_score"] >= 0.55, "Strong", np.where(merged["confidence_score"] >= 0.35, "Moderate", "Weak")),
    )
    merged["future_direction_1d"] = (merged["future_return_1d"] > 0).astype(float)
    merged = merged[merged["date"].dt.year >= MIN_ANALYSIS_YEAR].copy()
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def build_historical_taxonomy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timeline = frame.copy()
    timeline["astro_score"] = compute_astro_score(timeline)
    for col in ["within_30d", "within_90d", "within_180d", "within_365d"]:
        timeline[col] = True

    turning_points = detect_turning_points(
        timeline[["date", "signal", "confidence_score", "ml_probability", "astro_momentum_v2_smooth"]].copy(),
        historical_momentum=timeline["astro_momentum_v2_smooth"].dropna(),
    )
    if turning_points.empty:
        turning_points["turning_point_date"] = pd.Series(dtype="datetime64[ns]")
    else:
        turning_points["turning_point_date"] = pd.to_datetime(turning_points["turning_point_date"])

    forecast_windows = build_forecast_windows(timeline)
    forecast_windows["start_date"] = pd.to_datetime(forecast_windows["start_date"])
    forecast_windows["end_date"] = pd.to_datetime(forecast_windows["end_date"])

    classified_windows = classify_windows(forecast_windows, timeline, turning_points)

    calibration_df = pd.read_csv(FORECAST_CALIBRATION_PATH)
    evidence_df = build_class_evidence(calibration_df)
    evidence_df["taxonomy_v2"] = evidence_df.apply(classify_taxonomy_v2, axis=1)
    evidence_df["taxonomy_reason"] = evidence_df.apply(taxonomy_reason, axis=1)
    evidence_df["v2_posture"] = evidence_df["taxonomy_v2"].map(POSTURE_MAP)

    classified_windows = classified_windows.merge(
        evidence_df[
            [
                "window_class",
                "taxonomy_v2",
                "v2_posture",
                "taxonomy_reason",
                "avg_forward_return",
                "avg_win_rate",
                "avg_volatility",
            ]
        ],
        on="window_class",
        how="left",
    )

    classified_windows["start_date"] = pd.to_datetime(classified_windows["start_date"])
    classified_windows["end_date"] = pd.to_datetime(classified_windows["end_date"])

    dated_rows = []
    for row in classified_windows.to_dict("records"):
        mask = (timeline["date"] >= row["start_date"]) & (timeline["date"] <= row["end_date"])
        if not mask.any():
            continue
        payload = {key: row[key] for key in row.keys()}
        slice_df = timeline.loc[mask, ["date"]].copy()
        for key, value in payload.items():
            if key not in {"start_date", "end_date"}:
                slice_df[key] = value
        dated_rows.append(slice_df)

    taxonomy_daily = pd.concat(dated_rows, ignore_index=True) if dated_rows else pd.DataFrame(columns=["date"])
    enriched = timeline.merge(taxonomy_daily, on="date", how="left")
    return enriched, classified_windows, turning_points


def compute_horizon_metrics(series: pd.Series) -> Dict[str, float]:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "sample_count": 0,
            "average_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "volatility": np.nan,
            "max_gain": np.nan,
            "max_loss": np.nan,
        }

    return {
        "sample_count": int(clean.shape[0]),
        "average_return": float(clean.mean()),
        "median_return": float(clean.median()),
        "win_rate": float((clean > 0).mean()),
        "volatility": float(clean.std(ddof=0)),
        "max_gain": float(clean.max()),
        "max_loss": float(clean.min()),
    }


def build_wide_audit_row(label: str, mask: pd.Series, frame: pd.DataFrame, group: str, meta: Dict[str, object] | None = None) -> Dict[str, object]:
    row: Dict[str, object] = {
        "audit_group": group,
        "label": label,
        "event_count": int(mask.fillna(False).sum()),
    }
    if meta:
        row.update(meta)
    for horizon in HORIZONS:
        metrics = compute_horizon_metrics(frame.loc[mask.fillna(False), f"future_return_{horizon}d"])
        for key, value in metrics.items():
            row[f"{key}_{horizon}d"] = value
    return row


def build_momentum_audit(frame: pd.DataFrame) -> pd.DataFrame:
    momentum = frame["astro_momentum_v2_smooth"]
    slope = momentum.diff()
    prev_slope = slope.shift(1)
    q25 = float(momentum.quantile(0.25))
    q75 = float(momentum.quantile(0.75))

    conditions = {
        "momentum_gt_2_5": momentum > 2.5,
        "momentum_gt_2_0": momentum > 2.0,
        "momentum_crosses_above_zero": (momentum.shift(1) <= 0) & (momentum > 0),
        "momentum_crosses_below_zero": (momentum.shift(1) >= 0) & (momentum < 0),
        "momentum_slope_turns_positive": (prev_slope <= 0) & (slope > 0),
        "momentum_slope_turns_negative": (prev_slope >= 0) & (slope < 0),
        "momentum_rolling_over_from_high_zone": (momentum.shift(1) >= q75) & (prev_slope >= 0) & (slope < 0),
        "momentum_recovering_from_low_zone": (momentum.shift(1) <= q25) & (prev_slope <= 0) & (slope > 0),
    }

    rows = [
        build_wide_audit_row(
            label,
            mask,
            frame,
            group="astro_momentum",
            meta={
                "high_zone_threshold": q75,
                "low_zone_threshold": q25,
            },
        )
        for label, mask in conditions.items()
    ]
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def build_taxonomy_audit(frame: pd.DataFrame) -> pd.DataFrame:
    classes = [
        "Constructive / Positive Drift",
        "Neutral / Tactical",
        "False Bull / Exhaustion Risk",
        "Bearish",
        "High Risk",
    ]
    rows = [
        build_wide_audit_row(label, frame["taxonomy_v2"] == label, frame, group="taxonomy")
        for label in classes
    ]
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def build_turning_point_audit(frame: pd.DataFrame, turning_points: pd.DataFrame) -> pd.DataFrame:
    joined = turning_points.merge(
        frame[["date", "future_return_7d", "future_return_14d", "future_return_30d"]],
        left_on="turning_point_date",
        right_on="date",
        how="left",
    )

    rows: List[Dict[str, object]] = []
    point_types = [
        "signal_flip",
        "bullish_window_break",
        "bearish_window_relief",
        "momentum_breakout_up",
        "momentum_breakdown_down",
        "momentum_neutral_cross",
    ]
    for point_type in point_types:
        mask = joined["turning_point_type"] == point_type
        rows.append(build_wide_audit_row(point_type, mask, joined, group="turning_point_type"))

    transitions = [
        ("Neutral", "Bullish"),
        ("Bullish", "Neutral"),
        ("Neutral", "Bearish"),
        ("Bearish", "Neutral"),
    ]
    for old_signal, new_signal in transitions:
        label = f"{old_signal} -> {new_signal}"
        mask = (joined["old_signal"] == old_signal) & (joined["new_signal"] == new_signal)
        rows.append(
            build_wide_audit_row(
                label,
                mask,
                joined,
                group="signal_transition",
                meta={"old_signal": old_signal, "new_signal": new_signal},
            )
        )

    return pd.DataFrame(rows).sort_values(["audit_group", "label"]).reset_index(drop=True)


def compute_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1


def sharpe_ratio(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or clean.std(ddof=0) == 0:
        return np.nan
    return float((clean.mean() / clean.std(ddof=0)) * np.sqrt(365))


def sortino_ratio(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    downside = clean[clean < 0]
    if clean.empty or downside.empty or downside.std(ddof=0) == 0:
        return np.nan
    return float((clean.mean() / downside.std(ddof=0)) * np.sqrt(365))


def compute_trade_returns(exposure: pd.Series, next_day_returns: pd.Series) -> List[float]:
    trades: List[float] = []
    current_trade: List[float] = []
    current_exposure = 0.0

    for exposure_value, next_ret in zip(exposure.fillna(0.0), next_day_returns.fillna(0.0)):
        exposure_value = float(exposure_value)
        if exposure_value == 0:
            if current_trade:
                trades.append(float(np.prod([1 + ret for ret in current_trade]) - 1))
                current_trade = []
            current_exposure = 0.0
            continue

        if current_trade and exposure_value != current_exposure:
            trades.append(float(np.prod([1 + ret for ret in current_trade]) - 1))
            current_trade = []

        current_exposure = exposure_value
        current_trade.append(exposure_value * float(next_ret))

    if current_trade:
        trades.append(float(np.prod([1 + ret for ret in current_trade]) - 1))
    return trades


def summarize_strategy(name: str, exposure: pd.Series, frame: pd.DataFrame) -> Dict[str, object]:
    returns = frame["future_return_1d"].fillna(0.0)
    strategy_returns = exposure.fillna(0.0) * returns
    equity = (1.0 + strategy_returns).cumprod()
    drawdown = compute_drawdown(equity)
    total_return = float(equity.iloc[-1] - 1.0)
    days = max((frame["date"].iloc[-1] - frame["date"].iloc[0]).days, 1)
    cagr = float((equity.iloc[-1] ** (365 / days)) - 1.0) if equity.iloc[-1] > 0 else np.nan
    max_drawdown = float(drawdown.min())
    trades = compute_trade_returns(exposure, returns)
    active_returns = strategy_returns[exposure.fillna(0.0) != 0.0]
    trade_changes = exposure.fillna(0.0).ne(exposure.fillna(0.0).shift(1)).sum()

    return {
        "strategy": name,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "sortino_ratio": sortino_ratio(strategy_returns),
        "volatility": float(strategy_returns.std(ddof=0) * np.sqrt(365)) if len(strategy_returns) else np.nan,
        "win_rate": float((active_returns > 0).mean()) if not active_returns.empty else np.nan,
        "number_of_trades": int(len(trades)),
        "position_change_events": int(trade_changes),
        "average_trade_return": float(np.mean(trades)) if trades else np.nan,
        "best_trade": float(np.max(trades)) if trades else np.nan,
        "worst_trade": float(np.min(trades)) if trades else np.nan,
        "exposure_ratio": float(np.abs(exposure.fillna(0.0)).mean()),
        "return_drawdown_ratio": float(total_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan,
    }


def summarize_annual(name: str, exposure: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
    returns = frame["future_return_1d"].fillna(0.0)
    strategy_returns = exposure.fillna(0.0) * returns
    out = frame[["date"]].copy()
    out["strategy"] = name
    out["exposure"] = exposure.fillna(0.0)
    out["strategy_return"] = strategy_returns
    out["year"] = out["date"].dt.year

    rows = []
    for year, year_df in out.groupby("year", sort=True):
        equity = (1.0 + year_df["strategy_return"]).cumprod()
        drawdown = compute_drawdown(equity)
        active = year_df.loc[year_df["exposure"] != 0.0, "strategy_return"]
        annual_return = float(equity.iloc[-1] - 1.0)
        annual_dd = float(drawdown.min())
        rows.append(
            {
                "strategy": name,
                "year": int(year),
                "annual_return": annual_return,
                "annual_max_drawdown": annual_dd,
                "annual_win_rate": float((active > 0).mean()) if not active.empty else np.nan,
                "annual_return_drawdown_ratio": float(annual_return / abs(annual_dd)) if annual_dd < 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_strategy_exposures(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    momentum = frame["astro_momentum_v2_smooth"]
    slope = momentum.diff()
    q75 = float(momentum.quantile(0.75))
    rolling_over_high = (momentum.shift(1) >= q75) & (slope < 0)

    taxonomy_spot_map = {
        "Constructive / Positive Drift": 1.0,
        "Neutral / Tactical": 0.5,
        "False Bull / Exhaustion Risk": 0.25,
        "Bearish": 0.0,
        "High Risk": 0.0,
    }
    taxonomy_ls_map = {
        "Constructive / Positive Drift": 1.0,
        "Neutral / Tactical": 0.0,
        "False Bull / Exhaustion Risk": -0.25,
        "Bearish": -1.0,
        "High Risk": -0.5,
    }

    exposures = {
        "Buy & Hold": pd.Series(1.0, index=frame.index),
        "Astro Momentum Spot": pd.Series(
            np.where(momentum > 0, np.where(rolling_over_high, 0.5, 1.0), 0.0),
            index=frame.index,
        ),
        "Taxonomy Spot": frame["taxonomy_v2"].map(taxonomy_spot_map).fillna(0.0),
        "Taxonomy Long/Short": frame["taxonomy_v2"].map(taxonomy_ls_map).fillna(0.0),
    }

    hybrid = []
    for _, row in frame.iterrows():
        taxonomy = row["taxonomy_v2"]
        momentum_value = row["astro_momentum_v2_smooth"]
        high_roll = bool(row["rolling_over_high_zone"])
        if taxonomy == "Constructive / Positive Drift":
            exposure = 1.0 if momentum_value > 0 else 0.25
        elif taxonomy == "Neutral / Tactical":
            exposure = 0.5 if momentum_value > 0 else 0.0
        elif taxonomy == "False Bull / Exhaustion Risk":
            exposure = -0.25 if momentum_value < 0 else 0.0
        elif taxonomy == "Bearish":
            exposure = -1.0 if momentum_value < 0 else 0.0
        elif taxonomy == "High Risk":
            exposure = -0.5 if momentum_value < 0 else 0.0
        else:
            exposure = 0.0

        if high_roll and exposure > 0:
            exposure = min(exposure, 0.25)
        hybrid.append(exposure)

    exposures["Hybrid Taxonomy + Momentum"] = pd.Series(hybrid, index=frame.index, dtype=float)
    return exposures


def run_backtests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["rolling_over_high_zone"] = (
        (frame["astro_momentum_v2_smooth"].shift(1) >= frame["astro_momentum_v2_smooth"].quantile(0.75))
        & (frame["astro_momentum_v2_smooth"].diff() < 0)
    )

    exposures = build_strategy_exposures(frame)
    result_rows = [summarize_strategy(name, exposure, frame) for name, exposure in exposures.items()]
    results = pd.DataFrame(result_rows)

    buy_hold_return = float(results.loc[results["strategy"] == "Buy & Hold", "total_return"].iloc[0])
    buy_hold_ratio = float(results.loc[results["strategy"] == "Buy & Hold", "return_drawdown_ratio"].iloc[0])
    results["beats_buy_hold_total_return"] = results["total_return"] > buy_hold_return
    results["total_return_delta_vs_buy_hold"] = results["total_return"] - buy_hold_return
    results["return_drawdown_ratio_delta_vs_buy_hold"] = results["return_drawdown_ratio"] - buy_hold_ratio
    results = results.sort_values(["return_drawdown_ratio", "total_return"], ascending=[False, False]).reset_index(drop=True)

    annual_frames = [summarize_annual(name, exposure, frame) for name, exposure in exposures.items()]
    annual = pd.concat(annual_frames, ignore_index=True)
    buy_hold_by_year = annual[annual["strategy"] == "Buy & Hold"][["year", "annual_return"]].rename(
        columns={"annual_return": "buy_hold_annual_return"}
    )
    annual = annual.merge(buy_hold_by_year, on="year", how="left")
    annual["beats_buy_hold"] = annual["annual_return"] > annual["buy_hold_annual_return"]
    annual = annual.sort_values(["strategy", "year"]).reset_index(drop=True)
    return results, annual


def choose_best_row(df: pd.DataFrame, metric: str, ascending: bool = False) -> pd.Series:
    candidate = df.dropna(subset=[metric])
    if candidate.empty:
        return pd.Series(dtype=object)
    return candidate.sort_values(metric, ascending=ascending).iloc[0]


def summarize_alpha_findings(
    momentum_audit: pd.DataFrame,
    taxonomy_audit: pd.DataFrame,
    turning_audit: pd.DataFrame,
    backtest_results: pd.DataFrame,
    annual_results: pd.DataFrame,
) -> str:
    best_momentum = choose_best_row(momentum_audit, "average_return_30d")
    best_taxonomy = choose_best_row(taxonomy_audit, "average_return_30d")
    best_turning = choose_best_row(turning_audit, "average_return_30d")
    best_strategy = choose_best_row(backtest_results, "return_drawdown_ratio")
    best_return_strategy = choose_best_row(backtest_results, "total_return")
    taxonomy_lookup = taxonomy_audit.set_index("label")

    buy_hold = backtest_results.loc[backtest_results["strategy"] == "Buy & Hold"].iloc[0]
    non_baseline = backtest_results[backtest_results["strategy"] != "Buy & Hold"].copy()
    beat_buy_hold = non_baseline[non_baseline["beats_buy_hold_total_return"]].copy()
    taxonomy_semantic_conflict = False
    for risk_label in ["False Bull / Exhaustion Risk", "Bearish", "High Risk"]:
        if risk_label in taxonomy_lookup.index and float(taxonomy_lookup.loc[risk_label, "average_return_30d"]) > 0.02:
            taxonomy_semantic_conflict = True
            break

    if not beat_buy_hold.empty:
        alpha_call = (
            f"The strongest strategy was `{best_strategy['strategy']}` with return/drawdown "
            f"`{best_strategy['return_drawdown_ratio']:.2f}` and total return `{best_strategy['total_return']:.2%}`."
        )
    else:
        alpha_call = (
            "No tested strategy exceeded Buy & Hold on total return, so the current stack looks "
            "more useful for timing and risk framing than as a standalone alpha engine."
        )

    if best_strategy.get("strategy", "") == "Taxonomy Long/Short":
        primary_use = "Long/short trader"
    elif best_strategy.get("strategy", "") in {"Taxonomy Spot", "Hybrid Taxonomy + Momentum", "Astro Momentum Spot"}:
        primary_use = "Spot investor"
    else:
        primary_use = "Risk timing only"

    momentum_answer = (
        f"Astro Momentum v2 Smooth showed its best 30D average return in `{best_momentum['label']}` "
        f"at `{best_momentum['average_return_30d']:.2%}` across `{int(best_momentum['sample_count_30d'])}` samples."
        if not best_momentum.empty
        else "Momentum alpha could not be established from the reconstructed out-of-sample history."
    )

    if best_taxonomy.empty:
        taxonomy_answer = "Taxonomy labels did not show clear separation in the reconstructed history."
    elif taxonomy_semantic_conflict:
        taxonomy_answer = (
            "Taxonomy does separate regimes enough to support portfolio mapping, but the label semantics are not "
            "fully trustworthy yet because at least one defensive label still delivered materially positive forward returns."
        )
    else:
        taxonomy_answer = (
            f"Taxonomy added useful interpretation through `{best_taxonomy['label']}` with "
            f"`{best_taxonomy['average_return_30d']:.2%}` average 30D forward return."
        )

    turning_answer = (
        f"The best timing event was `{best_turning['label']}` with `{best_turning['average_return_30d']:.2%}` "
        f"average 30D forward return."
        if not best_turning.empty
        else "Turning points were not clearly useful for timing in the reconstructed history."
    )

    best_year = choose_best_row(annual_results, "annual_return")
    worst_year = choose_best_row(annual_results, "annual_return", ascending=True)
    positive_years = int((annual_results["annual_return"] > 0).sum())
    years_beating_buy_hold = int(annual_results["beats_buy_hold"].sum())

    if taxonomy_semantic_conflict:
        recommendation = "revise taxonomy"
    elif beat_buy_hold.empty:
        recommendation = "revise taxonomy"
    elif best_strategy["strategy"] == "Astro Momentum Spot":
        recommendation = "promote Astro Momentum as primary signal"
    elif best_strategy["strategy"] in {"Taxonomy Spot", "Taxonomy Long/Short", "Hybrid Taxonomy + Momentum"}:
        recommendation = "create paper trading monitor"
    else:
        recommendation = "keep current dashboard"

    lines = [
        "# Astro Alpha Audit & Institutional Backtest Framework v1",
        "",
        "## Context",
        "- Historical analysis was rebuilt from the current real repo's engine code, not from the legacy blueprint.",
        "- The saved `bitcoin_astro_daily_score.csv`, `astro_aspects_raw.csv`, and `ml_dataset.csv` snapshots contain git conflict markers in the current workspace, so this audit resolved those conflicts in-memory without editing the source files.",
        "- Results are based on the out-of-sample historical prediction path produced by `Robust Astro Engine v1` logic.",
        "",
        "## Core Answers",
        f"- A. Astro Momentum v2 Smooth standalone alpha: {momentum_answer}",
        f"- B. Taxonomy v2 interpretation value: {taxonomy_answer}",
        f"- C. Turning Point timing value: {turning_answer}",
        f"- D. Strategy beating Buy & Hold: {alpha_call}",
        f"- E. Best risk-adjusted strategy: `{best_strategy.get('strategy', 'N/A')}` with return/drawdown `{best_strategy.get('return_drawdown_ratio', np.nan):.2f}`.",
        f"- F. System is currently most useful for: `{primary_use}`.",
        f"- G. Recommended next step: `{recommendation}`.",
        "",
        "## Strategy Scorecard",
        dataframe_to_markdown(
            backtest_results[
                [
                    "strategy",
                    "total_return",
                    "cagr",
                    "max_drawdown",
                    "sharpe_ratio",
                    "sortino_ratio",
                    "volatility",
                    "win_rate",
                    "number_of_trades",
                    "average_trade_return",
                    "exposure_ratio",
                    "return_drawdown_ratio",
                    "beats_buy_hold_total_return",
                ]
            ]
        ),
        "",
        "## Momentum Audit Highlights",
        dataframe_to_markdown(
            momentum_audit[
                [
                    "label",
                    "event_count",
                    "average_return_7d",
                    "average_return_14d",
                    "average_return_30d",
                    "win_rate_30d",
                    "sample_count_30d",
                ]
            ].sort_values("average_return_30d", ascending=False).head(8)
        ),
        "",
        "## Taxonomy Audit Highlights",
        dataframe_to_markdown(
            taxonomy_audit[
                [
                    "label",
                    "event_count",
                    "average_return_7d",
                    "average_return_14d",
                    "average_return_30d",
                    "win_rate_30d",
                    "sample_count_30d",
                ]
            ].sort_values("average_return_30d", ascending=False)
        ),
        "",
        "## Turning Point Audit Highlights",
        dataframe_to_markdown(
            turning_audit[
                [
                    "audit_group",
                    "label",
                    "event_count",
                    "average_return_7d",
                    "average_return_14d",
                    "average_return_30d",
                    "win_rate_30d",
                    "sample_count_30d",
                ]
            ].sort_values("average_return_30d", ascending=False).head(10)
        ),
        "",
        "## Annual / Out-of-Sample Breakdown",
        f"- Best strategy by total return: `{best_return_strategy.get('strategy', 'N/A')}` at `{best_return_strategy.get('total_return', np.nan):.2%}`.",
        f"- Best annual row: `{best_year.get('strategy', 'N/A')}` in `{int(best_year.get('year', 0))}` with `{best_year.get('annual_return', np.nan):.2%}` return.",
        f"- Worst annual row: `{worst_year.get('strategy', 'N/A')}` in `{int(worst_year.get('year', 0))}` with `{worst_year.get('annual_return', np.nan):.2%}` return.",
        f"- Positive strategy-years: `{positive_years}`.",
        f"- Strategy-years beating Buy & Hold: `{years_beating_buy_hold}`.",
        "",
        "## Caveats",
        "- Backtests are frictionless and do not include fees, slippage, funding, or borrow costs.",
        "- Daily strategy returns apply each day's forecast exposure to the next day's BTC return, which is a conservative and explicit timing assumption.",
        "- Taxonomy labels were reconstructed from historical forecast windows using the current calibrated taxonomy mapping, so this audit measures the current dashboard interpretation layer rather than inventing a new one.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    historical_frame = build_historical_engine_frame()
    historical_frame, classified_windows, turning_points = build_historical_taxonomy(historical_frame)

    momentum_audit = build_momentum_audit(historical_frame)
    taxonomy_audit = build_taxonomy_audit(historical_frame)
    turning_audit = build_turning_point_audit(historical_frame, turning_points)
    backtest_results, annual_results = run_backtests(historical_frame)

    momentum_audit.to_csv(MOMENTUM_AUDIT_PATH, index=False)
    taxonomy_audit.to_csv(TAXONOMY_AUDIT_PATH, index=False)
    turning_audit.to_csv(TURNING_POINT_AUDIT_PATH, index=False)
    backtest_results.to_csv(BACKTEST_RESULTS_PATH, index=False)
    annual_results.to_csv(BACKTEST_ANNUAL_PATH, index=False)

    report_text = summarize_alpha_findings(
        momentum_audit=momentum_audit,
        taxonomy_audit=taxonomy_audit,
        turning_audit=turning_audit,
        backtest_results=backtest_results,
        annual_results=annual_results,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"Saved {MOMENTUM_AUDIT_PATH}")
    print(f"Saved {TAXONOMY_AUDIT_PATH}")
    print(f"Saved {TURNING_POINT_AUDIT_PATH}")
    print(f"Saved {BACKTEST_RESULTS_PATH}")
    print(f"Saved {BACKTEST_ANNUAL_PATH}")
    print(f"Saved {REPORT_PATH}")
    print(
        "Best strategy:",
        backtest_results.iloc[0]["strategy"],
        "| Return/DD:",
        f"{backtest_results.iloc[0]['return_drawdown_ratio']:.2f}",
    )


if __name__ == "__main__":
    main()
