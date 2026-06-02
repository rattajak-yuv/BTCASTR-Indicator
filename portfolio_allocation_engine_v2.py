from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from forecast_taxonomy_v3 import dataframe_to_markdown
from portfolio_allocation_engine_v1 import (
    build_historical_allocation_frame,
    clean_value,
    compute_metrics,
    determine_current_date,
    load_current_momentum,
    load_json,
)


ROOT = Path(".")
MAPPING_PATH = ROOT / "data" / "forecast_taxonomy_v4_mapping.csv"
INTELLIGENCE_V4_PATH = ROOT / "data" / "forecast_intelligence_v4.csv"
DASHBOARD_CURRENT_PATH = ROOT / "data" / "dashboard_current_state.json"
DASHBOARD_SUMMARY_PATH = ROOT / "data" / "dashboard_summary.json"
DASHBOARD_TIMELINE_PATH = ROOT / "data" / "dashboard_timeline.json"
DASHBOARD_RISK_CALENDAR_PATH = ROOT / "data" / "dashboard_risk_calendar.json"
FUTURE_TIMELINE_PATH = ROOT / "data" / "future_forecast_timeline.csv"

CURRENT_OUTPUT = ROOT / "data" / "current_allocation_v2.json"
TIMELINE_OUTPUT = ROOT / "data" / "allocation_timeline_v2.csv"
RESULTS_OUTPUT = ROOT / "data" / "allocation_variant_results.csv"
ANNUAL_OUTPUT = ROOT / "data" / "allocation_variant_annual.csv"
STRESS_OUTPUT = ROOT / "data" / "allocation_variant_stress_test.csv"
GRID_OUTPUT = ROOT / "data" / "allocation_grid_search.csv"
REPORT_OUTPUT = ROOT / "data" / "portfolio_allocation_v2_report.md"

POSITIVE_TAXONOMIES = {
    "High Conviction Expansion",
    "Constructive Drift",
    "Recovery / Reversal Setup",
}
NEUTRAL_TAXONOMIES = {"Transition / Low Conviction"}
CAUTION_TAXONOMIES = {"Volatility Caution"}

COMMON_DYNAMIC = {
    "confidence_low_threshold": 0.30,
    "confidence_high_threshold": 0.45,
    "confidence_low_delta": -5.0,
    "confidence_high_delta": 5.0,
    "ml_high_threshold": 0.60,
    "ml_low_threshold": 0.45,
    "ml_high_delta": 5.0,
    "ml_low_delta": -5.0,
    "momentum_positive_threshold": 1.5,
    "momentum_negative_threshold": 0.0,
    "momentum_positive_delta": 5.0,
    "momentum_negative_delta": -5.0,
    "alignment_bonus": 10.0,
    "cap_high_risk": 35.0,
    "cap_medium_risk": 70.0,
    "cap_30d_volatility_caution": 25.0,
    "cap_transition_constructive": 65.0,
}

VARIANT_CONFIGS: Dict[str, Dict[str, object]] = {
    "A_Conservative_v1_Baseline": {
        "kind": "v1_baseline",
        "label": "Conservative v1 Baseline",
    },
    "B_Upside_Capture_v2": {
        "kind": "rule_variant",
        "label": "Upside Capture v2",
        "base_allocations": {
            "High Conviction Expansion": 100.0,
            "Constructive Drift": 85.0,
            "Recovery / Reversal Setup": 85.0,
            "Transition / Low Conviction": 50.0,
            "Volatility Caution": 20.0,
        },
        "dynamic": {**COMMON_DYNAMIC},
    },
    "C_Trend_Preserving_v2": {
        "kind": "rule_variant",
        "label": "Trend-Preserving v2",
        "base_allocations": {
            "High Conviction Expansion": 100.0,
            "Constructive Drift": 90.0,
            "Recovery / Reversal Setup": 75.0,
            "Transition / Low Conviction": 60.0,
            "Volatility Caution": 10.0,
        },
        "dynamic": {
            **COMMON_DYNAMIC,
            "constructive_drift_floor_non_high_risk": 70.0,
        },
    },
    "D_Drawdown_Guard_v2": {
        "kind": "rule_variant",
        "label": "Drawdown Guard v2",
        "base_allocations": {
            "High Conviction Expansion": 100.0,
            "Constructive Drift": 85.0,
            "Recovery / Reversal Setup": 80.0,
            "Transition / Low Conviction": 55.0,
            "Volatility Caution": 0.0,
        },
        "dynamic": {
            **COMMON_DYNAMIC,
            "cap_high_risk": 25.0,
            "cap_medium_risk": 65.0,
            "cap_30d_volatility_caution": 0.0,
        },
    },
    "E_Recovery_Aggressive_v2": {
        "kind": "rule_variant",
        "label": "Recovery Aggressive v2",
        "base_allocations": {
            "High Conviction Expansion": 95.0,
            "Constructive Drift": 80.0,
            "Recovery / Reversal Setup": 100.0,
            "Transition / Low Conviction": 45.0,
            "Volatility Caution": 10.0,
        },
        "dynamic": {**COMMON_DYNAMIC},
    },
    "F_Balanced_v2": {
        "kind": "rule_variant",
        "label": "Balanced v2",
        "base_allocations": {
            "High Conviction Expansion": 95.0,
            "Constructive Drift": 80.0,
            "Recovery / Reversal Setup": 85.0,
            "Transition / Low Conviction": 50.0,
            "Volatility Caution": 15.0,
        },
        "dynamic": {**COMMON_DYNAMIC},
    },
}


def directional_group(label: str) -> str:
    if label in POSITIVE_TAXONOMIES:
        return "positive"
    if label in CAUTION_TAXONOMIES:
        return "caution"
    return "neutral"


def allocation_posture_from_pct(btc_pct: float) -> str:
    if btc_pct >= 90:
        return "Full risk-on"
    if btc_pct >= 75:
        return "Aggressive risk-on"
    if btc_pct >= 60:
        return "Measured risk-on"
    if btc_pct >= 45:
        return "Constructive accumulation"
    if btc_pct >= 25:
        return "Selective exposure"
    if btc_pct > 0:
        return "Defensive exposure"
    return "Cash preservation"


def v1_baseline_allocation(
    taxonomy_v4: str,
    ml_probability: float,
    confidence: float,
    risk_level: str,
    astro_momentum: float,
    outlook_30d: str,
    outlook_90d: str,
    outlook_365d: str,
) -> Dict[str, object]:
    base_map = {
        "High Conviction Expansion": 95.0,
        "Constructive Drift": 70.0,
        "Recovery / Reversal Setup": 65.0,
        "Transition / Low Conviction": 40.0,
        "Volatility Caution": 15.0,
    }
    base_btc = base_map[taxonomy_v4]
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

    if not pd.isna(astro_momentum):
        if astro_momentum >= 1.5:
            adjusted_btc += 5.0
            notes.append(f"Astro momentum is supportive at {astro_momentum:.2f}, adding 5 percentage points.")
        elif astro_momentum <= 0:
            adjusted_btc -= 5.0
            notes.append(f"Astro momentum is weak at {astro_momentum:.2f}, subtracting 5 percentage points.")
        else:
            notes.append(f"Astro momentum is neutral-to-positive at {astro_momentum:.2f}, so no momentum adjustment was applied.")
    else:
        notes.append("Momentum unavailable, so no momentum adjustment was applied.")

    if {directional_group(outlook_30d), directional_group(outlook_90d), directional_group(outlook_365d)} == {"positive"}:
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
    return {
        "base_btc_allocation": base_btc,
        "adjusted_btc_allocation": adjusted_btc,
        "cash_allocation": 100.0 - adjusted_btc,
        "allocation_posture": allocation_posture_from_pct(adjusted_btc),
        "explanation_notes": notes,
    }


def rule_variant_allocation(
    config: Dict[str, object],
    taxonomy_v4: str,
    ml_probability: float,
    confidence: float,
    risk_level: str,
    astro_momentum: float,
    outlook_30d: str,
    outlook_90d: str,
    outlook_365d: str,
) -> Dict[str, object]:
    base_btc = float(config["base_allocations"][taxonomy_v4])
    dynamic = dict(config["dynamic"])
    adjusted_btc = base_btc
    notes = [f"Base BTC allocation starts at {base_btc:.1f}% from the `{taxonomy_v4}` rule."]

    if confidence < dynamic["confidence_low_threshold"]:
        adjusted_btc += dynamic["confidence_low_delta"]
        notes.append(f"Confidence is low at {confidence:.2%}, applying a {dynamic['confidence_low_delta']:+.1f} percentage point confidence adjustment.")
    elif confidence > dynamic["confidence_high_threshold"]:
        adjusted_btc += dynamic["confidence_high_delta"]
        notes.append(f"Confidence is strong at {confidence:.2%}, applying a {dynamic['confidence_high_delta']:+.1f} percentage point confidence adjustment.")
    else:
        notes.append(f"Confidence at {confidence:.2%} sits in the neutral adjustment band.")

    if ml_probability > dynamic["ml_high_threshold"]:
        adjusted_btc += dynamic["ml_high_delta"]
        notes.append(f"ML probability is bullish at {ml_probability:.2%}, applying a {dynamic['ml_high_delta']:+.1f} percentage point ML adjustment.")
    elif ml_probability < dynamic["ml_low_threshold"]:
        adjusted_btc += dynamic["ml_low_delta"]
        notes.append(f"ML probability is weak at {ml_probability:.2%}, applying a {dynamic['ml_low_delta']:+.1f} percentage point ML adjustment.")
    else:
        notes.append(f"ML probability at {ml_probability:.2%} does not trigger an extra allocation change.")

    if not pd.isna(astro_momentum):
        if astro_momentum >= dynamic["momentum_positive_threshold"]:
            adjusted_btc += dynamic["momentum_positive_delta"]
            notes.append(f"Astro momentum is supportive at {astro_momentum:.2f}, applying a {dynamic['momentum_positive_delta']:+.1f} point momentum adjustment.")
        elif astro_momentum <= dynamic["momentum_negative_threshold"]:
            adjusted_btc += dynamic["momentum_negative_delta"]
            notes.append(f"Astro momentum is weak at {astro_momentum:.2f}, applying a {dynamic['momentum_negative_delta']:+.1f} point momentum adjustment.")
        else:
            notes.append(f"Astro momentum is neutral-to-positive at {astro_momentum:.2f}, so no momentum adjustment was applied.")
    else:
        notes.append("Momentum unavailable, so no momentum adjustment was applied.")

    if {directional_group(outlook_30d), directional_group(outlook_90d), directional_group(outlook_365d)} == {"positive"}:
        adjusted_btc += dynamic["alignment_bonus"]
        notes.append(f"All major horizons align positively, so BTC gets a {dynamic['alignment_bonus']:.1f}-point alignment bonus.")
    else:
        notes.append("The 30D, 90D, and 365D outlooks do not fully align positively, so no alignment bonus was applied.")

    caps = []
    if risk_level == "High":
        caps.append(dynamic["cap_high_risk"])
        notes.append(f"Risk level is High, so BTC allocation is capped at {dynamic['cap_high_risk']:.1f}%.")
    elif risk_level == "Medium":
        caps.append(dynamic["cap_medium_risk"])
        notes.append(f"Risk level is Medium, so BTC allocation is capped at {dynamic['cap_medium_risk']:.1f}%.")
    if outlook_30d == "Volatility Caution":
        caps.append(dynamic["cap_30d_volatility_caution"])
        notes.append(f"The 30D outlook is Volatility Caution, so BTC allocation is capped at {dynamic['cap_30d_volatility_caution']:.1f}%.")
    if outlook_30d == "Transition / Low Conviction" and outlook_365d == "Constructive Drift":
        caps.append(dynamic["cap_transition_constructive"])
        notes.append(f"The 30D outlook is Transition / Low Conviction while 365D stays Constructive Drift, so BTC allocation is capped at {dynamic['cap_transition_constructive']:.1f}%.")

    adjusted_btc = max(0.0, min(100.0, adjusted_btc))
    if caps:
        adjusted_btc = min(adjusted_btc, min(caps))

    constructive_floor = dynamic.get("constructive_drift_floor_non_high_risk")
    if constructive_floor is not None and taxonomy_v4 == "Constructive Drift" and risk_level != "High":
        if adjusted_btc < constructive_floor:
            adjusted_btc = float(constructive_floor)
            notes.append(f"Trend-preserving rule keeps Constructive Drift at or above {constructive_floor:.1f}% outside High risk conditions.")

    return {
        "base_btc_allocation": base_btc,
        "adjusted_btc_allocation": adjusted_btc,
        "cash_allocation": 100.0 - adjusted_btc,
        "allocation_posture": allocation_posture_from_pct(adjusted_btc),
        "explanation_notes": notes,
    }


def apply_variant_allocation(
    variant_key: str,
    taxonomy_v4: str,
    ml_probability: float,
    confidence: float,
    risk_level: str,
    astro_momentum: float,
    outlook_30d: str,
    outlook_90d: str,
    outlook_365d: str,
) -> Dict[str, object]:
    config = VARIANT_CONFIGS[variant_key]
    if config["kind"] == "v1_baseline":
        return v1_baseline_allocation(
            taxonomy_v4,
            ml_probability,
            confidence,
            risk_level,
            astro_momentum,
            outlook_30d,
            outlook_90d,
            outlook_365d,
        )
    return rule_variant_allocation(
        config,
        taxonomy_v4,
        ml_probability,
        confidence,
        risk_level,
        astro_momentum,
        outlook_30d,
        outlook_90d,
        outlook_365d,
    )


def backtest_variant(historical: pd.DataFrame, variant_key: str) -> pd.DataFrame:
    frame = historical.copy()
    allocations: List[float] = []
    for _, row in frame.iterrows():
        allocation = apply_variant_allocation(
            variant_key,
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
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["btc_allocation_pct"] = allocations
    frame["cash_allocation_pct"] = 100.0 - frame["btc_allocation_pct"]
    frame["btc_exposure"] = frame["btc_allocation_pct"] / 100.0
    frame["btc_daily_return"] = frame["price"].pct_change().fillna(0.0)
    frame["strategy_return"] = frame["btc_exposure"].shift(1).fillna(frame["btc_exposure"].iloc[0]) * frame["btc_daily_return"]
    frame["buy_hold_return"] = frame["btc_daily_return"]
    return frame


def summarize_variant_results(historical: pd.DataFrame, variant_key: str, variant_label: str) -> Dict[str, object]:
    strategy_metrics = compute_metrics(historical["strategy_return"], historical["btc_exposure"])
    buy_hold_metrics = compute_metrics(historical["buy_hold_return"], pd.Series(np.ones(len(historical))))
    capture_ratio = strategy_metrics["total_return"] / buy_hold_metrics["total_return"] if buy_hold_metrics["total_return"] > 1e-12 else np.nan
    drawdown_improvement = strategy_metrics["max_drawdown"] - buy_hold_metrics["max_drawdown"]
    return {
        "variant_key": variant_key,
        "variant_label": variant_label,
        **strategy_metrics,
        "comparison_vs_buy_hold_total_return": strategy_metrics["total_return"] - buy_hold_metrics["total_return"],
        "comparison_vs_buy_hold_CAGR": strategy_metrics["CAGR"] - buy_hold_metrics["CAGR"],
        "comparison_vs_buy_hold_max_drawdown": drawdown_improvement,
        "comparison_vs_buy_hold_sharpe": strategy_metrics["Sharpe ratio"] - buy_hold_metrics["Sharpe ratio"],
        "comparison_vs_buy_hold_sortino": strategy_metrics["Sortino ratio"] - buy_hold_metrics["Sortino ratio"],
        "buy_hold_total_return": buy_hold_metrics["total_return"],
        "buy_hold_CAGR": buy_hold_metrics["CAGR"],
        "buy_hold_max_drawdown": buy_hold_metrics["max_drawdown"],
        "buy_hold_sharpe": buy_hold_metrics["Sharpe ratio"],
        "buy_hold_sortino": buy_hold_metrics["Sortino ratio"],
        "return_capture_ratio_vs_buy_hold": capture_ratio,
        "drawdown_improvement_points": drawdown_improvement,
    }


def annual_summary_for_variant(historical: pd.DataFrame, variant_key: str, variant_label: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    temp = historical.copy()
    temp["year"] = temp["date"].dt.year
    for year, group in temp.groupby("year"):
        strategy_metrics = compute_metrics(group["strategy_return"], group["btc_exposure"])
        buy_hold_metrics = compute_metrics(group["buy_hold_return"], pd.Series(np.ones(len(group))))
        rows.append({
            "variant_key": variant_key,
            "variant_label": variant_label,
            "year": int(year),
            "strategy": "Allocation Strategy",
            **strategy_metrics,
        })
        rows.append({
            "variant_key": variant_key,
            "variant_label": variant_label,
            "year": int(year),
            "strategy": "Buy & Hold",
            **buy_hold_metrics,
        })
    return pd.DataFrame(rows)


def stress_summary_for_variant(historical: pd.DataFrame, variant_key: str, variant_label: str) -> pd.DataFrame:
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
        rows.append({
            "variant_key": variant_key,
            "variant_label": variant_label,
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
        })
    return pd.DataFrame(rows)


def choose_recommended_variant(results_df: pd.DataFrame) -> pd.Series:
    candidates = results_df.copy()
    candidates["meets_primary"] = (
        (candidates["Sharpe ratio"] > candidates["buy_hold_sharpe"])
        & (candidates["Sortino ratio"] > candidates["buy_hold_sortino"])
    )
    candidates["meets_secondary"] = candidates["drawdown_improvement_points"] >= 0.20
    candidates["meets_third"] = candidates["return_capture_ratio_vs_buy_hold"] >= 0.60
    candidates["score"] = (
        candidates["meets_primary"].astype(int) * 1000
        + candidates["meets_secondary"].astype(int) * 100
        + candidates["meets_third"].astype(int) * 10
        + candidates["comparison_vs_buy_hold_sharpe"].fillna(-999)
        + candidates["comparison_vs_buy_hold_sortino"].fillna(-999)
        + candidates["drawdown_improvement_points"].fillna(-999)
        + candidates["return_capture_ratio_vs_buy_hold"].fillna(-999) / 10.0
    )
    candidates = candidates.sort_values(
        ["score", "comparison_vs_buy_hold_sharpe", "comparison_vs_buy_hold_sortino", "drawdown_improvement_points", "return_capture_ratio_vs_buy_hold"],
        ascending=[False, False, False, False, False],
    )
    return candidates.iloc[0]


def build_current_allocation_v2(
    current_date: pd.Timestamp,
    recommended_variant: pd.Series,
    mapping_df: pd.DataFrame,
    dashboard_current: Dict[str, object],
    dashboard_summary: Dict[str, object],
    future_timeline: pd.DataFrame,
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

    allocation = apply_variant_allocation(
        str(recommended_variant["variant_key"]),
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
    next_review_date = dashboard_current.get("next_turning_point", {}).get("turning_point_date") or dashboard_current.get("current_window", {}).get("end_date")
    key_risks = [
        str(mapping_row.get("caveat", "")),
        f"30D outlook: {outlook_30d}",
        f"90D outlook: {outlook_90d}",
        f"365D outlook: {outlook_365d}",
    ]
    return {
        "current_date": current_date.date().isoformat(),
        "recommended_variant": str(recommended_variant["variant_label"]),
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


def build_timeline_v2(intelligence_v4: pd.DataFrame, dashboard_summary: Dict[str, object], variant_key: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    outlook_30d = str(dashboard_summary.get("30D Outlook", {}).get("dominant_taxonomy", "Transition / Low Conviction"))
    outlook_90d = str(dashboard_summary.get("90D Outlook", {}).get("dominant_taxonomy", "Transition / Low Conviction"))
    outlook_365d = str(dashboard_summary.get("365D Outlook", {}).get("dominant_taxonomy", "Transition / Low Conviction"))
    for _, row in intelligence_v4.iterrows():
        risk_level = "High" if float(row.get("average_risk_score", 0.0)) >= 0.75 else "Medium" if float(row.get("average_risk_score", 0.0)) >= 0.35 else "Low"
        allocation = apply_variant_allocation(
            variant_key,
            str(row["taxonomy_v4"]),
            float(row["average_ml_probability"]),
            float(row["average_confidence"]),
            risk_level,
            float(row.get("average_astro_score", np.nan)),
            outlook_30d,
            outlook_90d,
            outlook_365d,
        )
        rows.append({
            "variant_key": variant_key,
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
        })
    return pd.DataFrame(rows)


def run_grid_search(historical: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    buy_hold_metrics = compute_metrics(historical["price"].pct_change().fillna(0.0), pd.Series(np.ones(len(historical))))
    config_id = 0
    for constructive in [70.0, 80.0, 90.0]:
        for expansion in [90.0, 100.0]:
            for recovery in [75.0, 85.0, 100.0]:
                for transition in [30.0, 50.0, 60.0]:
                    for caution in [0.0, 10.0, 20.0, 30.0]:
                        config_id += 1
                        variant_key = f"GRID_{config_id:03d}"
                        config = {
                            "kind": "rule_variant",
                            "label": variant_key,
                            "base_allocations": {
                                "High Conviction Expansion": expansion,
                                "Constructive Drift": constructive,
                                "Recovery / Reversal Setup": recovery,
                                "Transition / Low Conviction": transition,
                                "Volatility Caution": caution,
                            },
                            "dynamic": {**COMMON_DYNAMIC},
                        }
                        VARIANT_CONFIGS[variant_key] = config
                        hist = backtest_variant(historical, variant_key)
                        metrics = compute_metrics(hist["strategy_return"], hist["btc_exposure"])
                        annual = annual_summary_for_variant(hist, variant_key, variant_key)
                        positive_years = int(((annual["strategy"] == "Allocation Strategy") & (annual["total_return"] > 0)).sum())
                        rows.append({
                            "variant_key": variant_key,
                            "constructive_drift": constructive,
                            "high_conviction_expansion": expansion,
                            "recovery_reversal_setup": recovery,
                            "transition_low_conviction": transition,
                            "volatility_caution": caution,
                            **metrics,
                            "return_capture_ratio_vs_buy_hold": metrics["total_return"] / buy_hold_metrics["total_return"] if buy_hold_metrics["total_return"] > 1e-12 else np.nan,
                            "drawdown_improvement_points": metrics["max_drawdown"] - buy_hold_metrics["max_drawdown"],
                            "overfit_risk": (
                                "High"
                                if positive_years <= 2 or (metrics["total_return"] > 8 and metrics["number_of_allocation_changes"] > 550)
                                else "Medium"
                                if positive_years == 3
                                else "Low"
                            ),
                        })
    grid_df = pd.DataFrame(rows)
    if not grid_df.empty:
        grid_df["rank_sharpe"] = grid_df["Sharpe ratio"].rank(ascending=False, method="dense")
        grid_df["rank_sortino"] = grid_df["Sortino ratio"].rank(ascending=False, method="dense")
        grid_df["rank_return_drawdown"] = grid_df["return_max_drawdown_ratio"].rank(ascending=False, method="dense")
        grid_df["rank_total_return"] = grid_df["total_return"].rank(ascending=False, method="dense")
    return grid_df.sort_values(
        ["Sharpe ratio", "Sortino ratio", "return_max_drawdown_ratio", "total_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def write_report(
    recommended_variant: pd.Series,
    current_allocation: Dict[str, object],
    variant_results: pd.DataFrame,
    annual_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    grid_df: pd.DataFrame,
) -> None:
    buy_hold_row = variant_results.loc[variant_results["variant_key"] == "BUY_HOLD"].iloc[0]
    variant_only = variant_results[variant_results["variant_key"] != "BUY_HOLD"].copy()
    best_named = variant_only.sort_values(
        ["comparison_vs_buy_hold_sharpe", "comparison_vs_buy_hold_sortino", "drawdown_improvement_points"],
        ascending=[False, False, False],
    ).iloc[0]
    v1_row = variant_only.loc[variant_only["variant_key"] == "A_Conservative_v1_Baseline"].iloc[0]
    rec = recommended_variant
    capture_enough = bool(rec["return_capture_ratio_vs_buy_hold"] >= 0.60)
    weakest_period = stress_df[stress_df["variant_key"] == rec["variant_key"]].sort_values("strategy_total_return").iloc[0]
    next_step = (
        "tune rules further"
        if not capture_enough or rec["total_return"] <= buy_hold_row["total_return"]
        else "add transaction costs"
        if rec["Sharpe ratio"] > buy_hold_row["Sharpe ratio"] and rec["Sortino ratio"] > buy_hold_row["Sortino ratio"]
        else "create paper trading monitor"
    )
    grid_note = "No grid search results." if grid_df.empty else (
        f"Best grid candidate was `{grid_df.iloc[0]['variant_key']}` with Sharpe `{grid_df.iloc[0]['Sharpe ratio']:.4f}` and overfit risk `{grid_df.iloc[0]['overfit_risk']}`."
    )
    lines = [
        "# Portfolio Allocation Engine v2 Optimization",
        "",
        "## Objective",
        "Improve Portfolio Allocation Engine v1 by capturing more BTC upside while still reducing drawdown versus Buy & Hold.",
        "",
        "## Current Recommended Allocation",
        f"- Recommended variant: `{current_allocation['recommended_variant']}`",
        f"- Current BTC / Cash allocation: `{current_allocation['adjusted_btc_allocation']:.1f}% / {current_allocation['cash_allocation']:.1f}%`",
        f"- Current taxonomy: `{current_allocation['current_taxonomy']}`",
        f"- Current confidence: `{current_allocation['current_confidence']:.2%}`",
        f"- Current ML probability: `{current_allocation['current_ml_probability']:.2%}`",
        f"- Explanation: {current_allocation['explanation']}",
        "",
        "## Validation Answers",
        f"A. Which allocation variant performs best? `{rec['variant_label']}` based on the requested priority stack.",
        f"B. Does v2 beat v1? {'Yes' if rec['Sharpe ratio'] > v1_row['Sharpe ratio'] and rec['total_return'] > v1_row['total_return'] else 'Partially'}; v2 comparison vs v1 total return delta = `{rec['total_return'] - v1_row['total_return']:.4f}`, Sharpe delta = `{rec['Sharpe ratio'] - v1_row['Sharpe ratio']:.4f}`.",
        f"C. Does v2 beat Buy & Hold on total return? {'Yes' if rec['total_return'] > buy_hold_row['total_return'] else 'No'}.",
        f"D. Does v2 reduce drawdown versus Buy & Hold? {'Yes' if rec['max_drawdown'] > buy_hold_row['max_drawdown'] else 'No'}, drawdown improvement = `{rec['drawdown_improvement_points']:.4f}`.",
        f"E. Does v2 improve Sharpe / Sortino? Sharpe delta = `{rec['comparison_vs_buy_hold_sharpe']:.4f}`, Sortino delta = `{rec['comparison_vs_buy_hold_sortino']:.4f}`.",
        f"F. Does v2 capture enough upside? {'Yes' if capture_enough else 'No'}; return capture ratio vs Buy & Hold = `{rec['return_capture_ratio_vs_buy_hold']:.4f}`.",
        f"G. Which market regime causes underperformance? `{weakest_period['period']}` is the weakest stress-test period for the recommended rule.",
        f"H. Is recommended v2 ready for paper trading? {'Yes, for monitored paper testing only.' if capture_enough and rec['Sharpe ratio'] > buy_hold_row['Sharpe ratio'] else 'Not yet; further rule tuning is needed before paper trading.'}",
        f"I. Recommended next step: `{next_step}`",
        "",
        "## Variant Results",
        dataframe_to_markdown(variant_results),
        "",
        "## Best Named Variant",
        dataframe_to_markdown(pd.DataFrame([best_named])),
        "",
        "## Grid Search Note",
        f"- {grid_note}",
        "",
        "## Stress Test",
        dataframe_to_markdown(stress_df),
        "",
        "## Annual Comparison",
        dataframe_to_markdown(annual_df.head(24)),
        "",
        "## Key Risks",
        *[f"- {risk}" for risk in current_allocation["key_risks"]],
        "",
        "## Reliability Note",
        "This optimization is still a research layer. No transaction costs, taxes, or execution frictions are modeled yet, and the grid search was intentionally small to limit overfitting.",
    ]
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    mapping_df = pd.read_csv(MAPPING_PATH)
    intelligence_v4 = pd.read_csv(INTELLIGENCE_V4_PATH, parse_dates=["start_date", "end_date"])
    dashboard_current = load_json(DASHBOARD_CURRENT_PATH)
    dashboard_summary = load_json(DASHBOARD_SUMMARY_PATH)
    _ = load_json(DASHBOARD_TIMELINE_PATH)
    _ = load_json(DASHBOARD_RISK_CALENDAR_PATH)
    future_timeline = pd.read_csv(FUTURE_TIMELINE_PATH, parse_dates=["date"])

    historical = build_historical_allocation_frame()
    variant_frames: Dict[str, pd.DataFrame] = {}
    variant_result_rows: List[Dict[str, object]] = []
    annual_rows: List[pd.DataFrame] = []
    stress_rows: List[pd.DataFrame] = []

    buy_hold_metrics = compute_metrics(historical["price"].pct_change().fillna(0.0), pd.Series(np.ones(len(historical))))
    variant_result_rows.append({
        "variant_key": "BUY_HOLD",
        "variant_label": "Buy & Hold",
        **buy_hold_metrics,
        "comparison_vs_buy_hold_total_return": 0.0,
        "comparison_vs_buy_hold_CAGR": 0.0,
        "comparison_vs_buy_hold_max_drawdown": 0.0,
        "comparison_vs_buy_hold_sharpe": 0.0,
        "comparison_vs_buy_hold_sortino": 0.0,
        "buy_hold_total_return": buy_hold_metrics["total_return"],
        "buy_hold_CAGR": buy_hold_metrics["CAGR"],
        "buy_hold_max_drawdown": buy_hold_metrics["max_drawdown"],
        "buy_hold_sharpe": buy_hold_metrics["Sharpe ratio"],
        "buy_hold_sortino": buy_hold_metrics["Sortino ratio"],
        "return_capture_ratio_vs_buy_hold": 1.0,
        "drawdown_improvement_points": 0.0,
    })

    for variant_key, config in VARIANT_CONFIGS.items():
        hist = backtest_variant(historical, variant_key)
        variant_frames[variant_key] = hist
        variant_result_rows.append(summarize_variant_results(hist, variant_key, str(config["label"])))
        annual_rows.append(annual_summary_for_variant(hist, variant_key, str(config["label"])))
        stress_rows.append(stress_summary_for_variant(hist, variant_key, str(config["label"])))

    variant_results = pd.DataFrame(variant_result_rows).sort_values(
        ["comparison_vs_buy_hold_sharpe", "comparison_vs_buy_hold_sortino", "drawdown_improvement_points", "return_capture_ratio_vs_buy_hold"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    annual_df = pd.concat(annual_rows, ignore_index=True)
    stress_df = pd.concat(stress_rows, ignore_index=True)
    grid_df = run_grid_search(historical)

    recommended_variant = choose_recommended_variant(variant_results[variant_results["variant_key"] != "BUY_HOLD"].copy())
    current_date = determine_current_date(future_timeline)
    current_allocation = build_current_allocation_v2(
        current_date,
        recommended_variant,
        mapping_df,
        dashboard_current,
        dashboard_summary,
        future_timeline,
    )
    timeline_v2 = build_timeline_v2(intelligence_v4, dashboard_summary, str(recommended_variant["variant_key"]))

    CURRENT_OUTPUT.write_text(json.dumps(current_allocation, indent=2) + "\n", encoding="utf-8")
    TIMELINE_OUTPUT.write_text(timeline_v2.to_csv(index=False), encoding="utf-8")
    RESULTS_OUTPUT.write_text(variant_results.to_csv(index=False), encoding="utf-8")
    ANNUAL_OUTPUT.write_text(annual_df.to_csv(index=False), encoding="utf-8")
    STRESS_OUTPUT.write_text(stress_df.to_csv(index=False), encoding="utf-8")
    GRID_OUTPUT.write_text(grid_df.to_csv(index=False), encoding="utf-8")
    write_report(recommended_variant, current_allocation, variant_results, annual_df, stress_df, grid_df)

    print(f"Wrote {CURRENT_OUTPUT}")
    print(f"Wrote {TIMELINE_OUTPUT}")
    print(f"Wrote {RESULTS_OUTPUT}")
    print(f"Wrote {ANNUAL_OUTPUT}")
    print(f"Wrote {STRESS_OUTPUT}")
    print(f"Wrote {GRID_OUTPUT}")
    print(f"Wrote {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
