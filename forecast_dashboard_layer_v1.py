from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LIVE_FORECAST_PATH = Path("data/live_forecast.csv")
FORECAST_WINDOWS_PATH = Path("data/forecast_windows.csv")
TURNING_POINTS_PATH = Path("data/turning_points.csv")
FORECAST_INTELLIGENCE_V2_PATH = Path("data/forecast_intelligence_v2.csv")

CURRENT_STATE_PATH = Path("data/dashboard_current_state.json")
TIMELINE_PATH = Path("data/dashboard_timeline.json")
RISK_CALENDAR_PATH = Path("data/dashboard_risk_calendar.json")
SUMMARY_PATH = Path("data/dashboard_summary.json")

TAXONOMY_PRIORITY = {
    "Constructive / Positive Drift": 5,
    "Neutral / Tactical": 4,
    "False Bull / Exhaustion Risk": 3,
    "High Risk": 2,
    "Bearish": 1,
}

TURNING_POINT_PRIORITY = {
    "signal_flip": 5,
    "confidence_shock": 4,
    "bullish_window_break": 4,
    "bearish_window_relief": 3,
    "momentum_breakdown_down": 4,
    "momentum_breakout_up": 3,
    "momentum_neutral_cross": 2,
}


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


def row_to_dict(row: pd.Series, keys: List[str]) -> Dict[str, object]:
    return {key: clean_value(row[key]) for key in keys}


def find_current_window(intelligence: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.Series:
    matches = intelligence[
        (intelligence["start_date"] <= forecast_date)
        & (intelligence["end_date"] >= forecast_date)
    ].copy()
    if matches.empty:
        raise ValueError("No current forecast window found for forecast date")
    matches["taxonomy_priority"] = matches["taxonomy_v2"].map(TAXONOMY_PRIORITY)
    return matches.sort_values(
        ["start_date", "taxonomy_priority"],
        ascending=[True, False],
    ).iloc[0]


def find_next_window(
    intelligence: pd.DataFrame,
    forecast_date: pd.Timestamp,
    labels: List[str],
) -> Optional[pd.Series]:
    windows = intelligence[
        (intelligence["end_date"] >= forecast_date) & (intelligence["taxonomy_v2"].isin(labels))
    ].copy()
    if windows.empty:
        return None
    windows["taxonomy_priority"] = windows["taxonomy_v2"].map(TAXONOMY_PRIORITY)
    return windows.sort_values(
        ["start_date", "taxonomy_priority", "average_confidence"],
        ascending=[True, False, False],
    ).iloc[0]


def summarize_outlook(intelligence: pd.DataFrame, forecast_date: pd.Timestamp, days: int) -> Dict[str, object]:
    horizon_end = forecast_date + pd.Timedelta(days=days - 1)
    horizon = intelligence[intelligence["start_date"] <= horizon_end].copy()
    if horizon.empty:
        return {
            "label": f"{days}D Outlook",
            "horizon_days": days,
            "dominant_taxonomy": None,
            "average_confidence": None,
            "window_count": 0,
            "summary": "No forecast windows are available in this horizon.",
        }

    duration_by_taxonomy = (
        horizon.groupby("taxonomy_v2")["duration_days"].sum().sort_values(ascending=False)
    )
    dominant_taxonomy = duration_by_taxonomy.index[0]
    avg_confidence = float(horizon["average_confidence"].mean())
    avg_probability = float(horizon["average_ml_probability"].mean())
    first_taxonomy = str(horizon.iloc[0]["taxonomy_v2"])
    last_taxonomy = str(horizon.iloc[-1]["taxonomy_v2"])

    transition_note = ""
    if first_taxonomy != last_taxonomy:
        transition_note = (
            f" The horizon starts in {first_taxonomy.lower()} and rotates toward "
            f"{last_taxonomy.lower()}."
        )

    return {
        "label": f"{days}D Outlook",
        "horizon_days": days,
        "dominant_taxonomy": dominant_taxonomy,
        "average_confidence": avg_confidence,
        "average_probability": avg_probability,
        "window_count": int(len(horizon)),
        "summary": (
            f"Dominant state is {dominant_taxonomy.lower()} with mean confidence "
            f"{avg_confidence:.2%} and mean probability {avg_probability:.2%}.{transition_note}"
        ),
    }


def classify_turning_point_severity(row: pd.Series, forecast_date: pd.Timestamp) -> str:
    if row["turning_point_date"] == forecast_date:
        return "immediate"
    if row["turning_point_type"] in {"signal_flip", "bullish_window_break", "momentum_breakdown_down"}:
        return "high"
    if row["turning_point_type"] in {"bearish_window_relief", "confidence_shock", "momentum_breakout_up"}:
        return "medium"
    return "low"


def build_current_state(
    live_forecast: pd.Series,
    current_window: pd.Series,
    next_turning_point: Optional[pd.Series],
    next_constructive_window: Optional[pd.Series],
    next_high_risk_window: Optional[pd.Series],
) -> Dict[str, object]:
    payload = {
        "forecast_date": clean_value(live_forecast["forecast_date"]),
        "current_signal": clean_value(live_forecast["current_signal"]),
        "current_probability": clean_value(live_forecast["current_probability"]),
        "current_confidence": clean_value(live_forecast["confidence_score"]),
        "current_taxonomy": clean_value(current_window["taxonomy_v2"]),
        "market_view": clean_value(live_forecast["market_view"]),
        "risk_level": clean_value(live_forecast["risk_level"]),
        "recommended_bias": clean_value(live_forecast["recommended_bias"]),
        "current_window": row_to_dict(
            current_window,
            [
                "start_date",
                "end_date",
                "window_class",
                "taxonomy_v2",
                "average_confidence",
                "average_ml_probability",
                "average_astro_score",
                "v2_posture",
                "narrative_v2",
            ],
        ),
        "next_turning_point": None,
        "next_constructive_window": None,
        "next_high_risk_window": None,
    }

    if next_turning_point is not None:
        payload["next_turning_point"] = row_to_dict(
            next_turning_point,
            [
                "turning_point_date",
                "turning_point_type",
                "old_signal",
                "new_signal",
                "confidence",
                "explanation",
                "severity",
            ],
        )
    if next_constructive_window is not None:
        payload["next_constructive_window"] = row_to_dict(
            next_constructive_window,
            [
                "start_date",
                "end_date",
                "taxonomy_v2",
                "average_confidence",
                "average_ml_probability",
                "v2_posture",
                "taxonomy_reason",
            ],
        )
    if next_high_risk_window is not None:
        payload["next_high_risk_window"] = row_to_dict(
            next_high_risk_window,
            [
                "start_date",
                "end_date",
                "taxonomy_v2",
                "average_confidence",
                "average_ml_probability",
                "v2_posture",
                "taxonomy_reason",
            ],
        )
    return payload


def build_timeline_payload(
    intelligence: pd.DataFrame,
    outlook_30d: Dict[str, object],
    outlook_90d: Dict[str, object],
    outlook_365d: Dict[str, object],
) -> Dict[str, object]:
    windows = []
    for _, row in intelligence.iterrows():
        windows.append(
            row_to_dict(
                row,
                [
                    "start_date",
                    "end_date",
                    "window_class",
                    "taxonomy_v2",
                    "duration_days",
                    "average_confidence",
                    "average_ml_probability",
                    "average_astro_score",
                    "average_risk_score",
                    "v2_posture",
                    "taxonomy_reason",
                    "narrative_v2",
                    "is_next_major_bullish_opportunity_v2",
                    "is_next_major_risk_window_v2",
                ],
            )
        )

    return {
        "outlooks": {
            "30d": outlook_30d,
            "90d": outlook_90d,
            "365d": outlook_365d,
        },
        "windows": windows,
    }


def build_risk_calendar_payload(
    intelligence: pd.DataFrame,
    turning_points: pd.DataFrame,
) -> Dict[str, object]:
    risk_windows = intelligence[
        intelligence["taxonomy_v2"].isin(
            ["High Risk", "Bearish", "False Bull / Exhaustion Risk"]
        )
    ].copy()
    constructive_windows = intelligence[
        intelligence["taxonomy_v2"] == "Constructive / Positive Drift"
    ].copy()

    turning_events = []
    for _, row in turning_points.iterrows():
        turning_events.append(
            row_to_dict(
                row,
                [
                    "turning_point_date",
                    "turning_point_type",
                    "old_signal",
                    "new_signal",
                    "confidence",
                    "severity",
                    "explanation",
                ],
            )
        )

    risk_entries = []
    for _, row in risk_windows.iterrows():
        risk_entries.append(
            row_to_dict(
                row,
                [
                    "start_date",
                    "end_date",
                    "taxonomy_v2",
                    "duration_days",
                    "average_confidence",
                    "average_ml_probability",
                    "taxonomy_reason",
                ],
            )
        )

    constructive_entries = []
    for _, row in constructive_windows.iterrows():
        constructive_entries.append(
            row_to_dict(
                row,
                [
                    "start_date",
                    "end_date",
                    "taxonomy_v2",
                    "duration_days",
                    "average_confidence",
                    "average_ml_probability",
                    "taxonomy_reason",
                ],
            )
        )

    return {
        "turning_points": turning_events,
        "risk_windows": risk_entries,
        "constructive_windows": constructive_entries,
    }


def build_summary_payload(
    live_forecast: pd.Series,
    current_window: pd.Series,
    next_turning_point: Optional[pd.Series],
    next_constructive_window: Optional[pd.Series],
    next_high_risk_window: Optional[pd.Series],
    outlook_30d: Dict[str, object],
    outlook_90d: Dict[str, object],
    outlook_365d: Dict[str, object],
) -> Dict[str, object]:
    return {
        "forecast_date": clean_value(live_forecast["forecast_date"]),
        "Current Signal": clean_value(live_forecast["current_signal"]),
        "Current Confidence": clean_value(live_forecast["confidence_score"]),
        "Current Taxonomy": clean_value(current_window["taxonomy_v2"]),
        "Next Turning Point": clean_value(next_turning_point["turning_point_date"]) if next_turning_point is not None else None,
        "Next Constructive Window": clean_value(next_constructive_window["start_date"]) if next_constructive_window is not None else None,
        "Next High Risk Window": clean_value(next_high_risk_window["start_date"]) if next_high_risk_window is not None else None,
        "30D Outlook": outlook_30d,
        "90D Outlook": outlook_90d,
        "365D Outlook": outlook_365d,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def main() -> None:
    live_forecast = pd.read_csv(LIVE_FORECAST_PATH).iloc[0]
    _ = pd.read_csv(FORECAST_WINDOWS_PATH)
    turning_points = pd.read_csv(TURNING_POINTS_PATH, parse_dates=["turning_point_date"])
    intelligence = pd.read_csv(
        FORECAST_INTELLIGENCE_V2_PATH,
        parse_dates=["start_date", "end_date"],
    )

    forecast_date = pd.to_datetime(live_forecast["forecast_date"])
    current_window = find_current_window(intelligence, forecast_date)

    if not turning_points.empty:
        turning_points["severity"] = turning_points.apply(
            classify_turning_point_severity,
            axis=1,
            forecast_date=forecast_date,
        )
        next_turning_point = turning_points[
            turning_points["turning_point_date"] >= forecast_date
        ].sort_values(["turning_point_date", "confidence"], ascending=[True, False]).head(1)
        next_turning_point_row = next_turning_point.iloc[0] if not next_turning_point.empty else None
    else:
        turning_points["severity"] = pd.Series(dtype=str)
        next_turning_point_row = None

    next_constructive_window = find_next_window(
        intelligence,
        forecast_date,
        ["Constructive / Positive Drift"],
    )
    next_high_risk_window = find_next_window(
        intelligence,
        forecast_date,
        ["High Risk"],
    )

    outlook_30d = summarize_outlook(intelligence, forecast_date, 30)
    outlook_90d = summarize_outlook(intelligence, forecast_date, 90)
    outlook_365d = summarize_outlook(intelligence, forecast_date, 365)

    current_state_payload = build_current_state(
        live_forecast,
        current_window,
        next_turning_point_row,
        next_constructive_window,
        next_high_risk_window,
    )
    timeline_payload = build_timeline_payload(
        intelligence,
        outlook_30d,
        outlook_90d,
        outlook_365d,
    )
    risk_calendar_payload = build_risk_calendar_payload(intelligence, turning_points)
    summary_payload = build_summary_payload(
        live_forecast,
        current_window,
        next_turning_point_row,
        next_constructive_window,
        next_high_risk_window,
        outlook_30d,
        outlook_90d,
        outlook_365d,
    )

    write_json(CURRENT_STATE_PATH, current_state_payload)
    write_json(TIMELINE_PATH, timeline_payload)
    write_json(RISK_CALENDAR_PATH, risk_calendar_payload)
    write_json(SUMMARY_PATH, summary_payload)


if __name__ == "__main__":
    main()
