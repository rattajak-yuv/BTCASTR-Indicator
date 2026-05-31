from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

LIVE_FORECAST_PATH = Path("data/live_forecast.csv")
TIMELINE_PATH = Path("data/future_forecast_timeline.csv")
TURNING_POINTS_PATH = Path("data/turning_points.csv")
FORECAST_WINDOWS_PATH = Path("data/forecast_windows.csv")

OUTPUT_CSV_PATH = Path("data/forecast_intelligence.csv")
OUTPUT_REPORT_PATH = Path("data/forecast_intelligence_report.md")

CLASS_PRIORITY = {
    "Strong Bull": 7,
    "Bull Expansion": 6,
    "Accumulation": 5,
    "Neutral": 4,
    "Transition": 3,
    "High Risk": 2,
    "Bearish": 1,
}

RISK_LEVEL_MAP = {"Low": 0.0, "Moderate": 0.5, "High": 1.0}

CLASS_GUIDANCE = {
    "Strong Bull": (
        "Conviction is broad enough to support active long exposure, with pullbacks "
        "more likely to be consolidations than major trend failures."
    ),
    "Bull Expansion": (
        "Directional conditions are constructive and improving, favoring measured "
        "risk-on positioning while respecting shorter-term reversals."
    ),
    "Accumulation": (
        "The backdrop is quietly constructive, suggesting staged accumulation rather "
        "than aggressive momentum chasing."
    ),
    "Neutral": (
        "Directional edge is limited, so patience and selective exposure are more "
        "appropriate than large tactical bets."
    ),
    "Transition": (
        "The market is in a handoff phase with higher signal churn, so flexibility "
        "matters more than conviction."
    ),
    "High Risk": (
        "Risk conditions are elevated enough to prioritize capital preservation, "
        "tighter risk controls, and faster reaction speed."
    ),
    "Bearish": (
        "Downside pressure is dominant enough to justify a defensive posture until "
        "the signal base improves."
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
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_markdown_value(row[col]) for col in df.columns) + " |")
    return "\n".join([header_row, separator_row] + rows)


def horizon_summary_label(days_ahead: int) -> str:
    if days_ahead <= 30:
        return "30D"
    if days_ahead <= 90:
        return "90D"
    if days_ahead <= 180:
        return "180D"
    return "365D"


def parse_window_timeseries(
    window_row: pd.Series,
    timeline: pd.DataFrame,
    turning_points: pd.DataFrame,
) -> Dict[str, object]:
    mask = (timeline["date"] >= window_row["start_date"]) & (timeline["date"] <= window_row["end_date"])
    window_slice = timeline.loc[mask].copy()
    if window_slice.empty:
        raise ValueError(
            f"No timeline rows found for forecast window {window_row['start_date']} -> {window_row['end_date']}"
        )

    tp_mask = (
        (turning_points["turning_point_date"] >= window_row["start_date"])
        & (turning_points["turning_point_date"] <= window_row["end_date"])
    )
    tp_slice = turning_points.loc[tp_mask].copy()

    avg_probability = float(window_slice["ml_probability"].mean())
    avg_confidence = float(window_slice["confidence_score"].mean())
    avg_astro_score = float(window_slice["astro_score"].mean())
    max_probability = float(window_slice["ml_probability"].max())
    min_probability = float(window_slice["ml_probability"].min())
    probability_change = float(window_slice["ml_probability"].iloc[-1] - window_slice["ml_probability"].iloc[0])
    astro_score_change = float(window_slice["astro_score"].iloc[-1] - window_slice["astro_score"].iloc[0])
    avg_risk_score = float(window_slice["risk_level"].map(RISK_LEVEL_MAP).fillna(0.0).mean())
    high_risk_share = float((window_slice["risk_level"] == "High").mean())
    signal_mix = window_slice["signal"].value_counts(normalize=True).to_dict()

    return {
        "duration_days": int((window_row["end_date"] - window_row["start_date"]).days) + 1,
        "avg_ml_probability": avg_probability,
        "avg_confidence": avg_confidence,
        "avg_astro_score": avg_astro_score,
        "max_ml_probability": max_probability,
        "min_ml_probability": min_probability,
        "probability_change": probability_change,
        "astro_score_change": astro_score_change,
        "avg_risk_score": avg_risk_score,
        "high_risk_share": high_risk_share,
        "turning_point_count": int(len(tp_slice)),
        "turning_point_types": ", ".join(tp_slice["turning_point_type"].astype(str).unique().tolist()) or "",
        "bullish_share": float(signal_mix.get("Bullish", 0.0)),
        "neutral_share": float(signal_mix.get("Neutral", 0.0)),
        "bearish_share": float(signal_mix.get("Bearish", 0.0)),
        "within_30d": bool(window_slice["within_30d"].any()),
        "within_90d": bool(window_slice["within_90d"].any()),
        "within_180d": bool(window_slice["within_180d"].any()),
        "within_365d": bool(window_slice["within_365d"].any()),
    }


def classify_window(base_window_type: str, metrics: Dict[str, object]) -> str:
    avg_prob = float(metrics["avg_ml_probability"])
    avg_conf = float(metrics["avg_confidence"])
    avg_astro = float(metrics["avg_astro_score"])
    duration = int(metrics["duration_days"])
    turning_points = int(metrics["turning_point_count"])
    probability_range = float(metrics["max_ml_probability"]) - float(metrics["min_ml_probability"])
    high_risk_share = float(metrics["high_risk_share"])
    probability_change = abs(float(metrics["probability_change"]))

    if base_window_type == "Bullish Window":
        if avg_prob >= 0.60 and avg_conf >= 0.38 and avg_astro >= 1.0 and duration >= 7:
            return "Strong Bull"
        if avg_prob >= 0.575 and (avg_astro >= 0.25 or duration >= 5):
            return "Bull Expansion"
        return "Accumulation"

    if base_window_type == "Bearish / Risk Window":
        if avg_prob <= 0.41 and duration >= 5 and high_risk_share < 0.5:
            return "Bearish"
        return "High Risk"

    if avg_prob >= 0.545 and avg_astro >= 0.35:
        return "Accumulation"
    if turning_points >= 2 or duration <= 5 or probability_range >= 0.08 or probability_change >= 0.05:
        return "Transition"
    return "Neutral"


def investor_posture_for_class(window_class: str) -> str:
    posture_map = {
        "Strong Bull": "Long Bias",
        "Bull Expansion": "Constructive Long Bias",
        "Accumulation": "Gradual Accumulation",
        "Neutral": "Balanced / Wait",
        "Transition": "Flexible / Tactical",
        "High Risk": "Defensive",
        "Bearish": "Defensive / Short Bias",
    }
    return posture_map[window_class]


def build_narrative(
    window_class: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    metrics: Dict[str, object],
    turning_point_types: str,
) -> str:
    avg_prob = float(metrics["avg_ml_probability"])
    avg_conf = float(metrics["avg_confidence"])
    avg_astro = float(metrics["avg_astro_score"])
    duration = int(metrics["duration_days"])
    avg_risk_score = float(metrics["avg_risk_score"])

    risk_label = "low" if avg_risk_score < 0.2 else "moderate" if avg_risk_score < 0.6 else "high"
    turning_phrase = (
        f" The window contains {metrics['turning_point_count']} turning-point events, led by {turning_point_types}."
        if turning_point_types
        else ""
    )
    class_context = CLASS_GUIDANCE[window_class]

    return (
        f"From {start_date.date()} to {end_date.date()}, the forecast maps to {window_class.lower()} for "
        f"{duration} days. Average ML probability is {avg_prob:.2%}, average confidence is {avg_conf:.2%}, "
        f"and the average astro score is {avg_astro:.2f}, which together imply {risk_label}-risk conditions."
        f" {class_context}{turning_phrase}"
    )


def classify_windows(
    windows: pd.DataFrame,
    timeline: pd.DataFrame,
    turning_points: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for _, row in windows.iterrows():
        metrics = parse_window_timeseries(row, timeline, turning_points)
        window_class = classify_window(str(row["window_type"]), metrics)
        narrative = build_narrative(
            window_class,
            row["start_date"],
            row["end_date"],
            metrics,
            str(metrics["turning_point_types"]),
        )
        records.append(
            {
                "start_date": row["start_date"].date().isoformat(),
                "end_date": row["end_date"].date().isoformat(),
                "base_window_type": row["window_type"],
                "window_class": window_class,
                "duration_days": metrics["duration_days"],
                "average_confidence": metrics["avg_confidence"],
                "average_ml_probability": metrics["avg_ml_probability"],
                "average_astro_score": metrics["avg_astro_score"],
                "probability_change": metrics["probability_change"],
                "astro_score_change": metrics["astro_score_change"],
                "average_risk_score": metrics["avg_risk_score"],
                "high_risk_share": metrics["high_risk_share"],
                "turning_point_count": metrics["turning_point_count"],
                "turning_point_types": metrics["turning_point_types"],
                "bullish_share": metrics["bullish_share"],
                "neutral_share": metrics["neutral_share"],
                "bearish_share": metrics["bearish_share"],
                "within_30d": metrics["within_30d"],
                "within_90d": metrics["within_90d"],
                "within_180d": metrics["within_180d"],
                "within_365d": metrics["within_365d"],
                "investor_posture": investor_posture_for_class(window_class),
                "narrative": narrative,
            }
        )
    return pd.DataFrame(records)


def summarize_horizon_outlook(
    classified_windows: pd.DataFrame,
    horizon_end: pd.Timestamp,
    label: str,
) -> str:
    mask = pd.to_datetime(classified_windows["start_date"]) <= horizon_end
    slice_df = classified_windows.loc[mask].copy()
    if slice_df.empty:
        return f"{label}: No forecast windows are available in this horizon."

    dominant = (
        slice_df.groupby("window_class")["duration_days"].sum().sort_values(ascending=False).index[0]
    )
    first_window = slice_df.iloc[0]
    last_window = slice_df.iloc[-1]
    avg_prob = float(slice_df["average_ml_probability"].mean())
    avg_conf = float(slice_df["average_confidence"].mean())
    total_turns = int(slice_df["turning_point_count"].sum())

    transition_note = ""
    if first_window["window_class"] != last_window["window_class"]:
        transition_note = (
            f" It starts in {first_window['window_class'].lower()} and rotates toward "
            f"{last_window['window_class'].lower()} by the end of the horizon."
        )

    return (
        f"{label}: The dominant posture is {dominant.lower()}, with average ML probability at {avg_prob:.2%} "
        f"and average confidence at {avg_conf:.2%}.{transition_note} "
        f"Across the horizon, {total_turns} turning-point events are flagged."
    )


def find_next_window(
    classified_windows: pd.DataFrame,
    allowed_classes: Iterable[str],
    forecast_date: pd.Timestamp,
) -> pd.Series | None:
    allowed = set(allowed_classes)
    windows = classified_windows.copy()
    windows["start_date_dt"] = pd.to_datetime(windows["start_date"])
    future_windows = windows[
        (windows["start_date_dt"] >= forecast_date) & (windows["window_class"].isin(allowed))
    ].sort_values(["start_date_dt", "average_confidence"], ascending=[True, False])
    if future_windows.empty:
        return None
    return future_windows.iloc[0]


def build_event_summary(
    label: str,
    window_row: pd.Series | None,
) -> str:
    if window_row is None:
        return f"{label}: No qualifying event appears in the active forecast horizon."

    return (
        f"{label}: {window_row['window_class']} begins on {window_row['start_date']} and runs through "
        f"{window_row['end_date']}. Average confidence is {float(window_row['average_confidence']):.2%}, "
        f"with investor posture set to {window_row['investor_posture']}."
    )


def build_investor_summaries(
    live_forecast: pd.Series,
    weekly_outlook: str,
    monthly_outlook: str,
    next_risk_event: str,
    next_bullish_opportunity: str,
) -> Dict[str, str]:
    current_signal = str(live_forecast["current_signal"])
    bias = str(live_forecast["recommended_bias"])
    risk_level = str(live_forecast["risk_level"])

    return {
        "short_term_investor": (
            f"Short-term investors currently face a {current_signal.lower()} setup with a {bias.lower()} posture. "
            f"{weekly_outlook}"
        ),
        "swing_investor": (
            f"Swing investors should frame the next month around the coming window shifts. "
            f"{monthly_outlook}"
        ),
        "risk_manager": (
            f"Risk managers should treat the current environment as {risk_level.lower()} risk. "
            f"{next_risk_event} {next_bullish_opportunity}"
        ),
    }


def write_report(
    live_forecast: pd.Series,
    classified_windows: pd.DataFrame,
    turning_points: pd.DataFrame,
    weekly_outlook: str,
    monthly_outlook: str,
    next_risk_event: str,
    next_bullish_opportunity: str,
    investor_summaries: Dict[str, str],
) -> None:
    report_windows = classified_windows[
        [
            "start_date",
            "end_date",
            "window_class",
            "average_confidence",
            "average_ml_probability",
            "investor_posture",
        ]
    ]
    report_turns = turning_points.head(12).copy()
    if not report_turns.empty:
        report_turns["turning_point_date"] = report_turns["turning_point_date"].dt.date.astype(str)

    lines = [
        "# Forecast Intelligence Layer v1",
        "",
        "## Current Signal",
        (
            f"- Forecast date: {live_forecast['forecast_date']}\n"
            f"- Current signal: {live_forecast['current_signal']}\n"
            f"- Current probability: {float(live_forecast['current_probability']):.2%}\n"
            f"- Confidence score: {float(live_forecast['confidence_score']):.2%}\n"
            f"- Market view: {live_forecast['market_view']}\n"
            f"- Risk level: {live_forecast['risk_level']}\n"
            f"- Recommended bias: {live_forecast['recommended_bias']}"
        ),
        "",
        "## Outlook",
        f"- {weekly_outlook}",
        f"- {monthly_outlook}",
        f"- {next_risk_event}",
        f"- {next_bullish_opportunity}",
        "",
        "## Investor-Facing Summaries",
        f"- Short-term investor: {investor_summaries['short_term_investor']}",
        f"- Swing investor: {investor_summaries['swing_investor']}",
        f"- Risk manager: {investor_summaries['risk_manager']}",
        "",
        "## Classified Forecast Windows",
        dataframe_to_markdown(report_windows),
        "",
        "## Upcoming Turning Points",
        dataframe_to_markdown(report_turns) if not report_turns.empty else "No turning points were detected.",
    ]

    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    live_forecast = pd.read_csv(LIVE_FORECAST_PATH).iloc[0]
    timeline = pd.read_csv(TIMELINE_PATH, parse_dates=["date"])
    turning_points = pd.read_csv(TURNING_POINTS_PATH, parse_dates=["turning_point_date"])
    windows = pd.read_csv(FORECAST_WINDOWS_PATH, parse_dates=["start_date", "end_date"])

    classified_windows = classify_windows(windows, timeline, turning_points)

    forecast_date = pd.to_datetime(live_forecast["forecast_date"])
    weekly_outlook = summarize_horizon_outlook(classified_windows, forecast_date + pd.Timedelta(days=7), "Weekly outlook")
    monthly_outlook = summarize_horizon_outlook(classified_windows, forecast_date + pd.Timedelta(days=30), "Monthly outlook")

    next_risk_window = find_next_window(classified_windows, ["High Risk", "Bearish"], forecast_date)
    next_bull_window = find_next_window(
        classified_windows,
        ["Strong Bull", "Bull Expansion", "Accumulation"],
        forecast_date,
    )

    next_risk_event = build_event_summary("Next major risk event", next_risk_window)
    next_bullish_opportunity = build_event_summary("Next major bullish opportunity", next_bull_window)
    investor_summaries = build_investor_summaries(
        live_forecast,
        weekly_outlook,
        monthly_outlook,
        next_risk_event,
        next_bullish_opportunity,
    )

    classified_windows["window_priority"] = classified_windows["window_class"].map(CLASS_PRIORITY)
    classified_windows["forecast_horizon_bucket"] = classified_windows["start_date"].apply(
        lambda value: horizon_summary_label((pd.to_datetime(value) - forecast_date).days)
    )
    if next_risk_window is not None:
        classified_windows["is_next_major_risk_window"] = (
            classified_windows["start_date"] == next_risk_window["start_date"]
        )
    else:
        classified_windows["is_next_major_risk_window"] = False
    if next_bull_window is not None:
        classified_windows["is_next_major_bullish_opportunity"] = (
            classified_windows["start_date"] == next_bull_window["start_date"]
        )
    else:
        classified_windows["is_next_major_bullish_opportunity"] = False

    classified_windows.to_csv(OUTPUT_CSV_PATH, index=False)
    write_report(
        live_forecast,
        classified_windows,
        turning_points,
        weekly_outlook,
        monthly_outlook,
        next_risk_event,
        next_bullish_opportunity,
        investor_summaries,
    )


if __name__ == "__main__":
    main()
