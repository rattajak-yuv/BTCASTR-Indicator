from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from forecast_taxonomy_v3 import dataframe_to_markdown
from taxonomy_performance_validation_v1 import build_historical_taxonomy_daily, read_resilient_csv


ROOT = Path(".")
MAPPING_PATH = ROOT / "data" / "forecast_taxonomy_v4_mapping.csv"
INTELLIGENCE_V4_PATH = ROOT / "data" / "forecast_intelligence_v4.csv"
DASHBOARD_CURRENT_PATH = ROOT / "data" / "dashboard_current_state.json"
DASHBOARD_SUMMARY_PATH = ROOT / "data" / "dashboard_summary.json"
DASHBOARD_TIMELINE_PATH = ROOT / "data" / "dashboard_timeline.json"
DASHBOARD_RISK_CALENDAR_PATH = ROOT / "data" / "dashboard_risk_calendar.json"
VALIDATION_PATH = ROOT / "data" / "taxonomy_performance_validation.csv"
YEARLY_PATH = ROOT / "data" / "taxonomy_performance_by_year.csv"
REGIME_AUDIT_PATH = ROOT / "data" / "taxonomy_regime_audit.csv"
TRANSITION_MATRIX_PATH = ROOT / "data" / "taxonomy_transition_matrix.csv"
BTC_ASTRO_PATH = ROOT / "data" / "bitcoin_astro_daily_score.csv"
FUTURE_TIMELINE_PATH = ROOT / "data" / "future_forecast_timeline.csv"

CURRENT_ALLOCATION_OUTPUT = ROOT / "data" / "current_allocation.json"
ALLOCATION_TIMELINE_OUTPUT = ROOT / "data" / "allocation_timeline.csv"
BACKTEST_RESULTS_OUTPUT = ROOT / "data" / "allocation_backtest_results.csv"
BACKTEST_ANNUAL_OUTPUT = ROOT / "data" / "allocation_backtest_annual.csv"
STRESS_TEST_OUTPUT = ROOT / "data" / "allocation_stress_test.csv"
REPORT_OUTPUT = ROOT / "data" / "portfolio_allocation_report.md"

V4_BASE_RULES = {
    "High Conviction Expansion": {
        "btc_min": 90.0,
        "btc_max": 100.0,
        "base_btc": 95.0,
        "allocation_posture": "Aggressive risk-on",
        "rationale": "High-probability expansion window, but require confirmation due to sample fragility.",
    },
    "Constructive Drift": {
        "btc_min": 60.0,
        "btc_max": 80.0,
        "base_btc": 70.0,
        "allocation_posture": "Measured risk-on",
        "rationale": "Stable positive drift with measured long bias.",
    },
    "Recovery / Reversal Setup": {
        "btc_min": 50.0,
        "btc_max": 80.0,
        "base_btc": 65.0,
        "allocation_posture": "Opportunistic accumulation",
        "rationale": "Strong historical recovery edge, but confirm with price action because the setup appears after stress.",
    },
    "Transition / Low Conviction": {
        "btc_min": 25.0,
        "btc_max": 50.0,
        "base_btc": 40.0,
        "allocation_posture": "Selective exposure",
        "rationale": "Low-conviction transition state with selective exposure only.",
    },
    "Volatility Caution": {
        "btc_min": 0.0,
        "btc_max": 25.0,
        "base_btc": 15.0,
        "allocation_posture": "Capital preservation",
        "rationale": "Volatility dominates directional edge and capital preservation takes priority.",
    },
}

V4_PRIORITY = {
    "High Conviction Expansion": 6,
    "Constructive Drift": 5,
    "Recovery / Reversal Setup": 4,
    "Transition / Low Conviction": 3,
    "Volatility Caution": 2,
}

POSITIVE_TAXONOMIES = {
    "High Conviction Expansion",
    "Constructive Drift",
    "Recovery / Reversal Setup",
}
NEUTRAL_TAXONOMIES = {"Transition / Low Conviction"}
CAUTION_TAXONOMIES = {"Volatility Caution"}
V3_TO_V4 = {
    "Constructive Drift": "Constructive Drift",
    "High Momentum Expansion": "High Conviction Expansion",
    "Tactical Neutral": "Transition / Low Conviction",
    "Defensive / Weak Trend": "Recovery / Reversal Setup",
    "High Volatility Risk": "Volatility Caution",
}


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else {}


def clean_value(value):
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def directional_group(label: str) -> str:
    if label in POSITIVE_TAXONOMIES:
        return "positive"
    if label in CAUTION_TAXONOMIES:
        return "caution"
    return "neutral"


def choose_dominant_taxonomy(labels: pd.Series) -> str:
    labels = labels.dropna().astype(str)
    if labels.empty:
        return "Transition / Low Conviction"
    counts = labels.value_counts()
    best_count = counts.max()
    tied = counts[counts == best_count].index.tolist()
    tied = sorted(tied, key=lambda x: V4_PRIORITY.get(x, 0), reverse=True)
    return tied[0]


def compute_horizon_taxonomy(frame: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    frame = frame.sort_values("date").reset_index(drop=True).copy()
    for horizon in horizons:
        end_offsets = frame["date"] + pd.to_timedelta(horizon - 1, unit="D")
        dominant_labels: List[str] = []
        for idx, end_date in enumerate(end_offsets):
            subset = frame.loc[(frame["date"] >= frame.loc[idx, "date"]) & (frame["date"] <= end_date), "taxonomy_v4"]
            dominant_labels.append(choose_dominant_taxonomy(subset))
        frame[f"taxonomy_{horizon}d"] = dominant_labels
    return frame


def momentum_adjustment(momentum_value: float) -> Tuple[float, str]:
    if pd.isna(momentum_value):
        return 0.0, "Momentum unavailable, so no momentum adjustment was applied."
    if momentum_value >= 1.5:
        return 5.0, f"Astro momentum is supportive at {momentum_value:.2f}, adding 5 percentage points."
    if momentum_value <= 0:
        return -5.0, f"Astro momentum is weak at {momentum_value:.2f}, subtracting 5 percentage points."
    return 0.0, f"Astro momentum is neutral-to-positive at {momentum_value:.2f}, so no momentum adjustment was applied."


def allocation_posture_from_pct(btc_pct: float) -> str:
    if btc_pct >= 85:
        return "Aggressive risk-on"
    if btc_pct >= 65:
        return "Measured risk-on"
    if btc_pct >= 50:
        return "Constructive accumulation"
    if btc_pct >= 25:
        return "Selective exposure"
    if btc_pct > 0:
        return "Defensive exposure"
    return "Cash preservation"


def apply_dynamic_rules(
    taxonomy_v4: str,
    ml_probability: float,
    confidence: float,
    risk_level: str,
    astro_momentum: float,
    outlook_30d: str,
    outlook_90d: str,
    outlook_365d: str,
) -> Dict[str, object]:
    base_rule = V4_BASE_RULES[taxonomy_v4]
    base_btc = float(base_rule["base_btc"])
    adjusted_btc = base_btc
    notes = [f"Base BTC allocation starts at {base_btc:.1f}% from the `{taxonomy_v4}` rule."]

    if confidence < 0.30:
        adjusted_btc -= 15.0
        notes.append(f"Confidence is low at {confidence:.2%}, so BTC allocation is reduced by 15 percentage points.")
    elif confidence > 0.45:
        adjusted_btc += 10.0
        notes.append(f"Confidence is strong at {confidence:.2%}, so BTC allocation is increased by 10 percentage points.")
    else:
        notes.append(f"Confidence at {confidence:.2%} stays inside the neutral adjustment band.")

    if ml_probability > 0.60:
        adjusted_btc += 10.0
        notes.append(f"ML probability is bullish at {ml_probability:.2%}, so BTC allocation is increased by 10 percentage points.")
    elif ml_probability < 0.45:
        adjusted_btc -= 10.0
        notes.append(f"ML probability is weak at {ml_probability:.2%}, so BTC allocation is reduced by 10 percentage points.")
    else:
        notes.append(f"ML probability at {ml_probability:.2%} does not trigger an additional change.")

    momentum_delta, momentum_note = momentum_adjustment(astro_momentum)
    adjusted_btc += momentum_delta
    notes.append(momentum_note)

    agreement_groups = {
        directional_group(outlook_30d),
        directional_group(outlook_90d),
        directional_group(outlook_365d),
    }
    if agreement_groups == {"positive"}:
        adjusted_btc += 10.0
        notes.append("The 30D, 90D, and 365D outlooks all align positively, so BTC allocation gets a 10-point alignment bonus.")
    else:
        notes.append("The 30D, 90D, and 365D outlooks do not all align positively, so no alignment bonus is applied.")

    caps = []
    if risk_level == "High":
        caps.append(25.0)
        notes.append("Risk level is High, so BTC allocation is capped at 25%.")
    elif risk_level == "Medium":
        caps.append(60.0)
        notes.append("Risk level is Medium, so BTC allocation is capped at 60%.")

    if outlook_30d == "Volatility Caution":
        caps.append(25.0)
        notes.append("The 30D outlook is Volatility Caution, so BTC allocation is capped at 25%.")

    if outlook_30d == "Transition / Low Conviction" and outlook_365d == "Constructive Drift":
        caps.append(65.0)
        notes.append("The 30D outlook is Transition / Low Conviction while the 365D outlook is Constructive Drift, so allocation is kept measured with a 65% cap.")

    adjusted_btc = max(0.0, min(100.0, adjusted_btc))
    if caps:
        adjusted_btc = min(adjusted_btc, min(caps))

    cash_allocation = 100.0 - adjusted_btc
    return {
        "base_btc_allocation": base_btc,
        "adjusted_btc_allocation": adjusted_btc,
        "cash_allocation": cash_allocation,
        "allocation_posture": allocation_posture_from_pct(adjusted_btc),
        "explanation_notes": notes,
    }


def compute_metrics(returns: pd.Series, exposure: pd.Series | None = None) -> Dict[str, float]:
    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(returns)
    if n == 0:
        return {k: np.nan for k in [
            "total_return", "CAGR", "max_drawdown", "Sharpe ratio", "Sortino ratio",
            "annual_volatility", "win_rate", "exposure_ratio", "turnover",
            "number_of_allocation_changes", "return_max_drawdown_ratio"
        ]}
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(n / 365.0, 1 / 365.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if equity.iloc[-1] > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    vol = float(returns.std(ddof=0) * np.sqrt(365))
    mean_daily = float(returns.mean())
    sharpe = float((mean_daily / returns.std(ddof=0)) * np.sqrt(365)) if returns.std(ddof=0) > 1e-12 else np.nan
    downside = returns[returns < 0]
    sortino = float((mean_daily / downside.std(ddof=0)) * np.sqrt(365)) if len(downside) > 0 and downside.std(ddof=0) > 1e-12 else np.nan
    exposure_series = pd.Series(exposure).fillna(0.0) if exposure is not None else pd.Series(np.ones(n))
    turnover = float(exposure_series.diff().abs().fillna(0.0).sum())
    allocation_changes = int((exposure_series.diff().abs().fillna(0.0) > 1e-12).sum())
    return {
        "total_return": total_return,
        "CAGR": cagr,
        "max_drawdown": max_drawdown,
        "Sharpe ratio": sharpe,
        "Sortino ratio": sortino,
        "annual_volatility": vol,
        "win_rate": float((returns > 0).mean()),
        "exposure_ratio": float(exposure_series.mean()),
        "turnover": turnover,
        "number_of_allocation_changes": allocation_changes,
        "return_max_drawdown_ratio": float(total_return / abs(max_drawdown)) if max_drawdown < -1e-12 else np.nan,
    }


def build_historical_allocation_frame() -> pd.DataFrame:
    historical = build_historical_taxonomy_daily().copy()
    historical["taxonomy_v4"] = historical["taxonomy_v3"].map(V3_TO_V4)
    historical = compute_horizon_taxonomy(historical, [30, 90, 365])
    return historical


def determine_current_date(future_timeline: pd.DataFrame) -> pd.Timestamp:
    today = pd.Timestamp(datetime.now().date())
    dates = pd.to_datetime(future_timeline["date"])
    eligible = dates[dates <= today]
    if not eligible.empty:
        return eligible.max()
    return dates.min()


def load_current_momentum(current_date: pd.Timestamp) -> float:
    astro = read_resilient_csv(BTC_ASTRO_PATH)
    match = astro.loc[astro["date"] == current_date]
    if not match.empty and "astro_momentum_v2_smooth" in match.columns:
        return float(match.iloc[0]["astro_momentum_v2_smooth"])
    return np.nan


def make_current_allocation(
    current_date: pd.Timestamp,
    dashboard_current: Dict[str, object],
    dashboard_summary: Dict[str, object],
    future_timeline: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> Dict[str, object]:
    current_row = future_timeline.loc[future_timeline["date"] == current_date].copy()
    if current_row.empty:
        current_row = future_timeline.iloc[[0]].copy()
    current_row = current_row.iloc[0]

    current_taxonomy = str(dashboard_current.get("current_taxonomy", "Transition / Low Conviction"))
    current_signal = str(current_row.get("signal", dashboard_current.get("current_signal", "Neutral")))
    current_probability = float(current_row.get("ml_probability", dashboard_current.get("current_probability", np.nan)))
    current_confidence = float(current_row.get("confidence_score", dashboard_current.get("current_confidence", np.nan)))
    current_risk_level = str(current_row.get("risk_level", dashboard_current.get("risk_level", "Medium")))
    current_momentum = load_current_momentum(current_date)

    outlook_30d = str(dashboard_summary.get("30D Outlook", {}).get("dominant_taxonomy", current_taxonomy))
    outlook_90d = str(dashboard_summary.get("90D Outlook", {}).get("dominant_taxonomy", current_taxonomy))
    outlook_365d = str(dashboard_summary.get("365D Outlook", {}).get("dominant_taxonomy", current_taxonomy))

    allocation = apply_dynamic_rules(
        current_taxonomy,
        current_probability,
        current_confidence,
        current_risk_level,
        current_momentum,
        outlook_30d,
        outlook_90d,
        outlook_365d,
    )

    mapping_row = mapping_df.loc[mapping_df["taxonomy_v4"] == current_taxonomy].iloc[0]
    next_review_date = dashboard_current.get("next_turning_point", {}).get("turning_point_date")
    if not next_review_date:
        next_review_date = dashboard_current.get("current_window", {}).get("end_date")

    key_risks = [
        str(mapping_row.get("caveat", "")),
        f"30D outlook: {outlook_30d}",
        f"90D outlook: {outlook_90d}",
        f"365D outlook: {outlook_365d}",
    ]

    payload = {
        "current_date": current_date.date().isoformat(),
        "current_taxonomy": current_taxonomy,
        "current_signal": current_signal,
        "current_ml_probability": current_probability,
        "current_confidence": current_confidence,
        "current_risk_level": current_risk_level,
        "current_astro_momentum": clean_value(current_momentum),
        "base_btc_allocation": allocation["base_btc_allocation"],
        "adjusted_btc_allocation": allocation["adjusted_btc_allocation"],
        "cash_allocation": allocation["cash_allocation"],
        "allocation_posture": allocation["allocation_posture"],
        "explanation": " ".join(allocation["explanation_notes"]),
        "key_risks": key_risks,
        "next_review_date": next_review_date,
    }
    return payload


def build_allocation_timeline(
    intelligence_v4: pd.DataFrame,
    dashboard_summary: Dict[str, object],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    outlook_30d = str(dashboard_summary.get("30D Outlook", {}).get("dominant_taxonomy", "Transition / Low Conviction"))
    outlook_90d = str(dashboard_summary.get("90D Outlook", {}).get("dominant_taxonomy", "Transition / Low Conviction"))
    outlook_365d = str(dashboard_summary.get("365D Outlook", {}).get("dominant_taxonomy", "Transition / Low Conviction"))

    for _, row in intelligence_v4.iterrows():
        allocation = apply_dynamic_rules(
            str(row["taxonomy_v4"]),
            float(row["average_ml_probability"]),
            float(row["average_confidence"]),
            "High" if float(row.get("average_risk_score", 0.0)) >= 0.75 else "Medium" if float(row.get("average_risk_score", 0.0)) >= 0.35 else "Low",
            float(row.get("average_astro_score", np.nan)),
            outlook_30d,
            outlook_90d,
            outlook_365d,
        )
        rows.append(
            {
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "taxonomy_v4": row["taxonomy_v4"],
                "btc_allocation": allocation["adjusted_btc_allocation"],
                "cash_allocation": allocation["cash_allocation"],
                "allocation_posture": allocation["allocation_posture"],
                "confidence": row["average_confidence"],
                "ml_probability": row["average_ml_probability"],
                "risk_score": row.get("average_risk_score", np.nan),
                "explanation": " ".join(allocation["explanation_notes"]),
            }
        )
    return pd.DataFrame(rows)


def build_historical_backtest(
    historical: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    allocations: List[float] = []
    for _, row in historical.iterrows():
        allocation = apply_dynamic_rules(
            str(row["taxonomy_v4"]),
            float(row["ml_probability"]),
            float(row["confidence_score"]),
            str(row["risk_level"]),
            float(row["astro_momentum_v2_smooth"]),
            str(row["taxonomy_30d"]),
            str(row["taxonomy_90d"]),
            str(row["taxonomy_365d"]),
        )
        allocations.append(allocation["adjusted_btc_allocation"])

    historical = historical.sort_values("date").reset_index(drop=True).copy()
    historical["btc_allocation_pct"] = allocations
    historical["cash_allocation_pct"] = 100.0 - historical["btc_allocation_pct"]
    historical["btc_exposure"] = historical["btc_allocation_pct"] / 100.0
    historical["btc_daily_return"] = historical["price"].pct_change().fillna(0.0)
    historical["strategy_return"] = historical["btc_exposure"].shift(1).fillna(historical["btc_exposure"].iloc[0]) * historical["btc_daily_return"]
    historical["buy_hold_return"] = historical["btc_daily_return"]
    historical["strategy_equity"] = (1.0 + historical["strategy_return"]).cumprod()
    historical["buy_hold_equity"] = (1.0 + historical["buy_hold_return"]).cumprod()
    return historical, pd.DataFrame()


def build_backtest_summary(historical: pd.DataFrame) -> pd.DataFrame:
    strategy_metrics = compute_metrics(historical["strategy_return"], historical["btc_exposure"])
    buy_hold_metrics = compute_metrics(historical["buy_hold_return"], pd.Series(np.ones(len(historical))))
    rows = []
    rows.append(
        {
            "strategy": "Allocation Strategy",
            **strategy_metrics,
            "comparison_vs_buy_hold_total_return": strategy_metrics["total_return"] - buy_hold_metrics["total_return"],
            "comparison_vs_buy_hold_CAGR": strategy_metrics["CAGR"] - buy_hold_metrics["CAGR"],
            "comparison_vs_buy_hold_max_drawdown": strategy_metrics["max_drawdown"] - buy_hold_metrics["max_drawdown"],
            "comparison_vs_buy_hold_sharpe": strategy_metrics["Sharpe ratio"] - buy_hold_metrics["Sharpe ratio"],
            "comparison_vs_buy_hold_sortino": strategy_metrics["Sortino ratio"] - buy_hold_metrics["Sortino ratio"],
        }
    )
    rows.append(
        {
            "strategy": "Buy & Hold",
            **buy_hold_metrics,
            "comparison_vs_buy_hold_total_return": 0.0,
            "comparison_vs_buy_hold_CAGR": 0.0,
            "comparison_vs_buy_hold_max_drawdown": 0.0,
            "comparison_vs_buy_hold_sharpe": 0.0,
            "comparison_vs_buy_hold_sortino": 0.0,
        }
    )
    return pd.DataFrame(rows)


def build_annual_summary(historical: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    historical["year"] = historical["date"].dt.year
    for year, group in historical.groupby("year"):
        strategy_metrics = compute_metrics(group["strategy_return"], group["btc_exposure"])
        buy_hold_metrics = compute_metrics(group["buy_hold_return"], pd.Series(np.ones(len(group))))
        rows.append({"year": int(year), "strategy": "Allocation Strategy", **strategy_metrics})
        rows.append({"year": int(year), "strategy": "Buy & Hold", **buy_hold_metrics})
    return pd.DataFrame(rows)


def build_stress_test(historical: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("2020 bull market", "2020-01-01", "2020-12-31"),
        ("2021 peak / drawdown", "2021-01-01", "2021-12-31"),
        ("2022 bear market", "2022-01-01", "2022-12-31"),
        ("2023 recovery", "2023-01-01", "2023-12-31"),
        ("2024 bull / ETF cycle", "2024-01-01", "2024-12-31"),
        ("2025-2026 available period", "2025-01-01", str(historical["date"].max().date())),
    ]
    rows: List[Dict[str, object]] = []
    for label, start, end in periods:
        subset = historical[(historical["date"] >= pd.Timestamp(start)) & (historical["date"] <= pd.Timestamp(end))].copy()
        if subset.empty:
            continue
        strategy_metrics = compute_metrics(subset["strategy_return"], subset["btc_exposure"])
        buy_hold_metrics = compute_metrics(subset["buy_hold_return"], pd.Series(np.ones(len(subset))))
        rows.append(
            {
                "period": label,
                "start_date": start,
                "end_date": end,
                "strategy_total_return": strategy_metrics["total_return"],
                "buy_hold_total_return": buy_hold_metrics["total_return"],
                "strategy_max_drawdown": strategy_metrics["max_drawdown"],
                "buy_hold_max_drawdown": buy_hold_metrics["max_drawdown"],
                "strategy_sharpe": strategy_metrics["Sharpe ratio"],
                "buy_hold_sharpe": buy_hold_metrics["Sharpe ratio"],
                "strategy_sortino": strategy_metrics["Sortino ratio"],
                "buy_hold_sortino": buy_hold_metrics["Sortino ratio"],
                "strategy_exposure_ratio": strategy_metrics["exposure_ratio"],
                "strategy_turnover": strategy_metrics["turnover"],
                "allocation_changes": strategy_metrics["number_of_allocation_changes"],
            }
        )
    return pd.DataFrame(rows)


def write_current_allocation(payload: Dict[str, object]) -> None:
    CURRENT_ALLOCATION_OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(
    current_allocation: Dict[str, object],
    mapping_df: pd.DataFrame,
    results_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
) -> None:
    allocation_row = results_df.loc[results_df["strategy"] == "Allocation Strategy"].iloc[0]
    buy_hold_row = results_df.loc[results_df["strategy"] == "Buy & Hold"].iloc[0]
    top_driver_states = mapping_df.sort_values("forward_return_30d", ascending=False)[["taxonomy_v4", "forward_return_30d", "forward_return_60d", "stability_assessment"]]
    failures = []
    if allocation_row["total_return"] <= buy_hold_row["total_return"]:
        failures.append("Raw total return does not exceed Buy & Hold.")
    if allocation_row["Sharpe ratio"] <= buy_hold_row["Sharpe ratio"]:
        failures.append("Sharpe ratio does not improve versus Buy & Hold.")
    if allocation_row["return_max_drawdown_ratio"] <= buy_hold_row["return_max_drawdown_ratio"]:
        failures.append("Return / max drawdown ratio does not improve versus Buy & Hold.")
    if failures:
        next_step = "revise rules"
    elif allocation_row["Sharpe ratio"] > buy_hold_row["Sharpe ratio"] and allocation_row["max_drawdown"] > buy_hold_row["max_drawdown"]:
        next_step = "add transaction costs"
    else:
        next_step = "create paper trading monitor"

    lines = [
        "# Portfolio Allocation Engine v1",
        "",
        "## Objective",
        "Translate Forecast Taxonomy v4 into research-based BTC / Cash allocation guidance and test whether it improves risk-adjusted outcomes versus Buy & Hold.",
        "",
        "## Current Allocation",
        f"- Current date: `{current_allocation['current_date']}`",
        f"- Current BTC / Cash allocation: `{current_allocation['adjusted_btc_allocation']:.1f}% / {current_allocation['cash_allocation']:.1f}%`",
        f"- Current taxonomy: `{current_allocation['current_taxonomy']}`",
        f"- Current signal: `{current_allocation['current_signal']}`",
        f"- Current confidence: `{current_allocation['current_confidence']:.2%}`",
        f"- Current ML probability: `{current_allocation['current_ml_probability']:.2%}`",
        f"- Current risk level: `{current_allocation['current_risk_level']}`",
        "",
        "## Validation Answers",
        f"A. What is the current BTC / Cash allocation? `{current_allocation['adjusted_btc_allocation']:.1f}% BTC / {current_allocation['cash_allocation']:.1f}% cash`.",
        f"B. Why is this allocation recommended? {current_allocation['explanation']}",
        f"C. Which taxonomy states drive allocation most? `{top_driver_states.iloc[0]['taxonomy_v4']}`, `{top_driver_states.iloc[1]['taxonomy_v4']}`, and `{top_driver_states.iloc[2]['taxonomy_v4']}` carry the strongest 30D edge in the current mapping table.",
        f"D. Does allocation strategy beat Buy & Hold? {'Yes' if allocation_row['total_return'] > buy_hold_row['total_return'] else 'No'} on total return.",
        f"E. Does allocation reduce drawdown? {'Yes' if allocation_row['max_drawdown'] > buy_hold_row['max_drawdown'] else 'No'} based on max drawdown comparison.",
        f"F. Does allocation improve Sharpe / Sortino? Sharpe improvement = `{allocation_row['Sharpe ratio'] - buy_hold_row['Sharpe ratio']:.4f}`, Sortino improvement = `{allocation_row['Sortino ratio'] - buy_hold_row['Sortino ratio']:.4f}`.",
        f"G. Where does allocation fail? {'; '.join(failures) if failures else 'The biggest remaining gap is missing transaction-cost realism and live monitoring.'}",
        f"H. Is the system ready for paper trading? {'Yes, for monitored paper testing only.' if not failures else 'Not yet; the rule set still needs refinement before paper trading.'}",
        f"I. Recommended next step: `{next_step}`",
        "",
        "## Backtest Results",
        dataframe_to_markdown(results_df),
        "",
        "## Allocation Driver States",
        dataframe_to_markdown(top_driver_states),
        "",
        "## Stress Test",
        dataframe_to_markdown(stress_df),
        "",
        "## Annual Snapshot",
        dataframe_to_markdown(yearly_df.head(16)),
        "",
        "## Key Risks",
        *[f"- {risk}" for risk in current_allocation["key_risks"]],
        "",
        "## Research Limitation",
        "This v1 backtest assumes zero transaction costs and next-day application of daily allocation decisions. It should be treated as a research allocation layer, not execution-ready portfolio logic.",
    ]
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    mapping_df = pd.read_csv(MAPPING_PATH)
    intelligence_v4 = pd.read_csv(INTELLIGENCE_V4_PATH, parse_dates=["start_date", "end_date"])
    dashboard_current = load_json(DASHBOARD_CURRENT_PATH)
    dashboard_summary = load_json(DASHBOARD_SUMMARY_PATH)
    _ = load_json(DASHBOARD_TIMELINE_PATH)
    _ = load_json(DASHBOARD_RISK_CALENDAR_PATH)
    _ = pd.read_csv(VALIDATION_PATH)
    _ = pd.read_csv(YEARLY_PATH)
    _ = pd.read_csv(REGIME_AUDIT_PATH)
    _ = pd.read_csv(TRANSITION_MATRIX_PATH)
    future_timeline = pd.read_csv(FUTURE_TIMELINE_PATH, parse_dates=["date"])

    current_date = determine_current_date(future_timeline)
    current_allocation = make_current_allocation(
        current_date,
        dashboard_current,
        dashboard_summary,
        future_timeline,
        mapping_df,
    )
    allocation_timeline = build_allocation_timeline(intelligence_v4, dashboard_summary)
    historical = build_historical_allocation_frame()
    historical_backtest, _ = build_historical_backtest(historical)
    results_df = build_backtest_summary(historical_backtest)
    annual_df = build_annual_summary(historical_backtest)
    stress_df = build_stress_test(historical_backtest)

    write_current_allocation(current_allocation)
    allocation_timeline.to_csv(ALLOCATION_TIMELINE_OUTPUT, index=False)
    results_df.to_csv(BACKTEST_RESULTS_OUTPUT, index=False)
    annual_df.to_csv(BACKTEST_ANNUAL_OUTPUT, index=False)
    stress_df.to_csv(STRESS_TEST_OUTPUT, index=False)
    write_report(current_allocation, mapping_df, results_df, stress_df, annual_df)

    print(f"Wrote {CURRENT_ALLOCATION_OUTPUT}")
    print(f"Wrote {ALLOCATION_TIMELINE_OUTPUT}")
    print(f"Wrote {BACKTEST_RESULTS_OUTPUT}")
    print(f"Wrote {BACKTEST_ANNUAL_OUTPUT}")
    print(f"Wrote {STRESS_TEST_OUTPUT}")
    print(f"Wrote {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
