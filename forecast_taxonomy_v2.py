from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

FORECAST_CALIBRATION_PATH = Path("data/forecast_calibration.csv")
FORECAST_CALIBRATION_REPORT_PATH = Path("data/forecast_calibration_report.md")
FORECAST_INTELLIGENCE_V1_PATH = Path("data/forecast_intelligence.csv")
FORECAST_WINDOWS_PATH = Path("data/forecast_windows.csv")
FUTURE_TIMELINE_PATH = Path("data/future_forecast_timeline.csv")

OUTPUT_CSV_PATH = Path("data/forecast_intelligence_v2.csv")
OUTPUT_REPORT_PATH = Path("data/forecast_intelligence_v2_report.md")

TAXONOMY_PRIORITY = {
    "Constructive / Positive Drift": 5,
    "Neutral / Tactical": 4,
    "False Bull / Exhaustion Risk": 3,
    "High Risk": 2,
    "Bearish": 1,
}

POSTURE_MAP = {
    "Constructive / Positive Drift": "Constructive Long Bias",
    "Neutral / Tactical": "Tactical / Wait",
    "High Risk": "Defensive",
    "Bearish": "Short Bias / Defensive",
    "False Bull / Exhaustion Risk": "Fade Strength / Defensive",
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


def build_class_evidence(calibration_df: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        calibration_df.groupby("window_class", as_index=False)
        .agg(
            avg_forward_return=("average_forward_return", "mean"),
            median_forward_return=("median_forward_return", "mean"),
            avg_volatility=("volatility", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_max_gain=("max_gain", "mean"),
            avg_max_loss=("max_loss", "mean"),
            horizon_count=("horizon_days", "nunique"),
            window_count=("window_count", "max"),
            sample_count=("sample_count", "sum"),
            positive_horizon_share=("average_forward_return", lambda series: float((series > 0).mean())),
            negative_horizon_share=("average_forward_return", lambda series: float((series < 0).mean())),
            weak_distinction=("cluster_size", lambda series: bool(series.max() > 1)),
            merged_with=("merged_with", lambda series: next((value for value in series if isinstance(value, str)), "")),
        )
    )

    per_horizon = calibration_df.pivot_table(
        index="window_class",
        columns="horizon_days",
        values=["average_forward_return", "win_rate", "volatility"],
        aggfunc="mean",
    )
    per_horizon.columns = [f"{metric}_{int(horizon)}d" for metric, horizon in per_horizon.columns]
    per_horizon = per_horizon.reset_index()

    evidence = aggregated.merge(per_horizon, on="window_class", how="left")
    volatility_median = float(evidence["avg_volatility"].median())
    evidence["elevated_volatility"] = evidence["avg_volatility"] >= volatility_median
    evidence["bullish_looking"] = evidence["window_class"].isin(
        ["Strong Bull", "Bull Expansion", "Accumulation"]
    )
    return evidence


def classify_taxonomy_v2(row: pd.Series) -> str:
    avg_return = float(row["avg_forward_return"])
    avg_win = float(row["avg_win_rate"])
    avg_vol = float(row["avg_volatility"])
    weak_distinction = bool(row["weak_distinction"])
    bullish_looking = bool(row["bullish_looking"])
    negative_share = float(row["negative_horizon_share"])
    elevated_volatility = bool(row["elevated_volatility"])

    if bullish_looking and avg_return < 0 and avg_win < 0.45:
        return "False Bull / Exhaustion Risk"

    if avg_return <= -0.02 and avg_win < 0.45 and negative_share >= (2 / 3):
        return "Bearish"

    if avg_return <= 0.01 and elevated_volatility:
        return "High Risk"

    if avg_return > 0 and avg_win > 0.55:
        if weak_distinction and avg_return < 0.04:
            return "Neutral / Tactical"
        return "Constructive / Positive Drift"

    if weak_distinction:
        return "Neutral / Tactical"

    if avg_return > 0:
        return "Constructive / Positive Drift"

    if avg_return < 0:
        return "Bearish"

    return "Neutral / Tactical"


def taxonomy_reason(row: pd.Series) -> str:
    label = row["taxonomy_v2"]
    avg_return = float(row["avg_forward_return"])
    avg_win = float(row["avg_win_rate"])
    avg_vol = float(row["avg_volatility"])
    return_7d = float(row.get("average_forward_return_7d", np.nan))
    return_14d = float(row.get("average_forward_return_14d", np.nan))
    return_30d = float(row.get("average_forward_return_30d", np.nan))

    if label == "Constructive / Positive Drift":
        return (
            f"Historical outcomes remain positive across key horizons "
            f"(7D={return_7d:.2%}, 14D={return_14d:.2%}, 30D={return_30d:.2%}) "
            f"with average win rate {avg_win:.2%}."
        )
    if label == "Neutral / Tactical":
        return (
            f"Historical returns are positive but not sharply distinct, with average return {avg_return:.2%}, "
            f"win rate {avg_win:.2%}, and volatility {avg_vol:.2%}."
        )
    if label == "High Risk":
        return (
            f"Historical returns are weak at {avg_return:.2%} while average volatility stays elevated at "
            f"{avg_vol:.2%}, which supports a defensive risk framing."
        )
    if label == "Bearish":
        return (
            f"Historical forward returns are consistently negative "
            f"(7D={return_7d:.2%}, 14D={return_14d:.2%}, 30D={return_30d:.2%}) "
            f"with average win rate only {avg_win:.2%}."
        )
    return (
        f"The label appears bullish on the surface, but realized forward returns stayed negative "
        f"(avg {avg_return:.2%}) and win rate remained weak at {avg_win:.2%}."
    )


def build_window_narrative(row: pd.Series) -> str:
    label = row["taxonomy_v2"]
    date_span = f"{row['start_date']} to {row['end_date']}"
    evidence = row["taxonomy_reason"]
    if label == "Constructive / Positive Drift":
        return (
            f"From {date_span}, the calibrated outlook is constructive rather than merely optimistic. "
            f"{evidence} The stance favors measured long exposure instead of aggressive chasing."
        )
    if label == "Neutral / Tactical":
        return (
            f"From {date_span}, the calibrated outlook is neutral and tactical. "
            f"{evidence} Positioning should stay flexible until cleaner edge appears."
        )
    if label == "High Risk":
        return (
            f"From {date_span}, the calibrated outlook shifts to high risk. "
            f"{evidence} Capital preservation matters more than directional conviction."
        )
    if label == "Bearish":
        return (
            f"From {date_span}, the calibrated outlook is bearish. "
            f"{evidence} Defensive or short-biased positioning is more justified than buying dips."
        )
    return (
        f"From {date_span}, the calibrated outlook is false bull / exhaustion risk. "
        f"{evidence} Any apparent upside strength should be treated as fragile until proven otherwise."
    )


def choose_next_window(df: pd.DataFrame, labels: List[str]) -> pd.Series | None:
    slice_df = df[df["taxonomy_v2"].isin(labels)].copy()
    if slice_df.empty:
        return None
    slice_df["priority"] = slice_df["taxonomy_v2"].map(TAXONOMY_PRIORITY)
    return slice_df.sort_values(["start_date", "priority"], ascending=[True, False]).iloc[0]


def build_event_summary(prefix: str, row: pd.Series | None) -> str:
    if row is None:
        return f"{prefix}: No qualifying window is currently visible in the forecast horizon."
    return (
        f"{prefix}: {row['taxonomy_v2']} from {row['start_date']} to {row['end_date']} "
        f"with average confidence {float(row['average_confidence']):.2%}."
    )


def write_report(
    v2_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    next_bullish: str,
    next_risk: str,
) -> None:
    comparison_rows = (
        v2_df.groupby(["window_class", "taxonomy_v2"], as_index=False)
        .agg(
            windows=("start_date", "size"),
            total_days=("duration_days", "sum"),
            avg_confidence=("average_confidence", "mean"),
        )
        .sort_values(["taxonomy_v2", "window_class"])
    )

    changed_rows = v2_df[v2_df["taxonomy_changed"]].copy()
    changed_rows = changed_rows[
        [
            "start_date",
            "end_date",
            "window_class",
            "taxonomy_v2",
            "taxonomy_reason",
        ]
    ]

    lines = [
        "# Forecast Taxonomy v2",
        "",
        "## Mapping",
        dataframe_to_markdown(mapping_df),
        "",
        "## Outlook",
        f"- {next_bullish}",
        f"- {next_risk}",
        "",
        "## V1 vs V2 Comparison",
        dataframe_to_markdown(comparison_rows),
        "",
        "## Changed Windows",
        dataframe_to_markdown(changed_rows.head(20)) if not changed_rows.empty else "No windows changed label under the calibrated taxonomy.",
    ]

    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    calibration_df = pd.read_csv(FORECAST_CALIBRATION_PATH)
    _ = FORECAST_CALIBRATION_REPORT_PATH.read_text(encoding="utf-8")
    intelligence_v1 = pd.read_csv(FORECAST_INTELLIGENCE_V1_PATH)
    forecast_windows = pd.read_csv(FORECAST_WINDOWS_PATH)
    _ = pd.read_csv(FUTURE_TIMELINE_PATH, parse_dates=["date"])

    evidence_df = build_class_evidence(calibration_df)
    evidence_df["taxonomy_v2"] = evidence_df.apply(classify_taxonomy_v2, axis=1)

    rename_map = {
        "average_forward_return_7d": "average_forward_return_7d",
        "average_forward_return_14d": "average_forward_return_14d",
        "average_forward_return_30d": "average_forward_return_30d",
        "win_rate_7d": "win_rate_7d",
        "win_rate_14d": "win_rate_14d",
        "win_rate_30d": "win_rate_30d",
        "volatility_7d": "volatility_7d",
        "volatility_14d": "volatility_14d",
        "volatility_30d": "volatility_30d",
    }
    evidence_df = evidence_df.rename(columns=rename_map)
    evidence_df["taxonomy_reason"] = evidence_df.apply(taxonomy_reason, axis=1)
    evidence_df["v2_posture"] = evidence_df["taxonomy_v2"].map(POSTURE_MAP)

    mapping_df = evidence_df[
        [
            "window_class",
            "taxonomy_v2",
            "avg_forward_return",
            "avg_win_rate",
            "avg_volatility",
            "window_count",
            "weak_distinction",
            "taxonomy_reason",
        ]
    ].copy()

    v2_df = intelligence_v1.merge(
        evidence_df[
            [
                "window_class",
                "taxonomy_v2",
                "v2_posture",
                "taxonomy_reason",
                "avg_forward_return",
                "avg_win_rate",
                "avg_volatility",
                "average_forward_return_7d",
                "average_forward_return_14d",
                "average_forward_return_30d",
                "win_rate_7d",
                "win_rate_14d",
                "win_rate_30d",
            ]
        ],
        on="window_class",
        how="left",
    ).merge(
        forecast_windows[["start_date", "end_date", "key_driver_summary"]],
        on=["start_date", "end_date"],
        how="left",
    )

    v2_df["taxonomy_changed"] = v2_df["window_class"] != v2_df["taxonomy_v2"]
    v2_df["v2_window_priority"] = v2_df["taxonomy_v2"].map(TAXONOMY_PRIORITY)
    v2_df["narrative_v2"] = v2_df.apply(build_window_narrative, axis=1)

    next_bullish_window = choose_next_window(v2_df, ["Constructive / Positive Drift"])
    next_risk_window = choose_next_window(v2_df, ["High Risk", "Bearish", "False Bull / Exhaustion Risk"])

    v2_df["is_next_major_bullish_opportunity_v2"] = False
    v2_df["is_next_major_risk_window_v2"] = False
    if next_bullish_window is not None:
        v2_df.loc[
            (v2_df["start_date"] == next_bullish_window["start_date"])
            & (v2_df["end_date"] == next_bullish_window["end_date"]),
            "is_next_major_bullish_opportunity_v2",
        ] = True
    if next_risk_window is not None:
        v2_df.loc[
            (v2_df["start_date"] == next_risk_window["start_date"])
            & (v2_df["end_date"] == next_risk_window["end_date"]),
            "is_next_major_risk_window_v2",
        ] = True

    ordered_columns = [
        "start_date",
        "end_date",
        "base_window_type",
        "window_class",
        "taxonomy_v2",
        "taxonomy_changed",
        "duration_days",
        "average_confidence",
        "average_ml_probability",
        "average_astro_score",
        "average_risk_score",
        "v2_posture",
        "avg_forward_return",
        "avg_win_rate",
        "avg_volatility",
        "average_forward_return_7d",
        "average_forward_return_14d",
        "average_forward_return_30d",
        "win_rate_7d",
        "win_rate_14d",
        "win_rate_30d",
        "key_driver_summary",
        "taxonomy_reason",
        "narrative_v2",
        "is_next_major_bullish_opportunity_v2",
        "is_next_major_risk_window_v2",
    ]
    v2_df = v2_df[ordered_columns]
    v2_df.to_csv(OUTPUT_CSV_PATH, index=False)

    next_bullish_text = build_event_summary("Next constructive opportunity", next_bullish_window)
    next_risk_text = build_event_summary("Next calibrated risk event", next_risk_window)
    write_report(v2_df, mapping_df, next_bullish_text, next_risk_text)


if __name__ == "__main__":
    main()
