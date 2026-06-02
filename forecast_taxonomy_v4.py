from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from forecast_taxonomy_v3 import dataframe_to_markdown

REGIME_AUDIT_PATH = Path("data/taxonomy_regime_audit.csv")
TRANSITION_MATRIX_PATH = Path("data/taxonomy_transition_matrix.csv")
RENAME_RECOMMENDATION_PATH = Path("data/taxonomy_rename_recommendation.csv")
DEFENSIVE_DEEP_DIVE_PATH = Path("data/defensive_weak_trend_deep_dive.csv")
REGIME_AUDIT_REPORT_PATH = Path("data/taxonomy_regime_audit_report.md")
FORECAST_INTELLIGENCE_V3_PATH = Path("data/forecast_intelligence_v3.csv")
CURRENT_STATE_PATH = Path("data/dashboard_current_state.json")
SUMMARY_PATH = Path("data/dashboard_summary.json")
TIMELINE_PATH = Path("data/dashboard_timeline.json")
RISK_CALENDAR_PATH = Path("data/dashboard_risk_calendar.json")

OUTPUT_MAPPING_PATH = Path("data/forecast_taxonomy_v4_mapping.csv")
OUTPUT_CSV_PATH = Path("data/forecast_intelligence_v4.csv")
OUTPUT_REPORT_PATH = Path("data/forecast_intelligence_v4_report.md")

TAXONOMY_V4_MAP = {
    "Constructive Drift": "Constructive Drift",
    "High Momentum Expansion": "High Conviction Expansion",
    "Tactical Neutral": "Transition / Low Conviction",
    "Defensive / Weak Trend": "Recovery / Reversal Setup",
    "High Volatility Risk": "Volatility Caution",
}

TAXONOMY_V4_DETAILS = {
    "Constructive Drift": {
        "meaning": "Stable positive drift",
        "investor_posture": "Measured long bias",
        "exposure_language": "Moderate risk-on",
        "caveat": "Broadly positive and more stable than the faster expansion state.",
        "color_hex": "#2E7D32",
        "priority": 5,
    },
    "High Conviction Expansion": {
        "meaning": "High-probability expansion window",
        "investor_posture": "Risk-on with confirmation",
        "exposure_language": "Moderate to aggressive risk-on",
        "caveat": "Historically positive, but fragile because the sample is concentrated in fewer active years.",
        "color_hex": "#D97706",
        "priority": 6,
    },
    "Transition / Low Conviction": {
        "meaning": "Low-conviction transition state",
        "investor_posture": "Tactical / wait",
        "exposure_language": "Selective exposure only",
        "caveat": "This bucket absorbs many observations and likely includes multiple sub-regimes.",
        "color_hex": "#C9A227",
        "priority": 3,
    },
    "Recovery / Reversal Setup": {
        "meaning": "Historically strong post-stress recovery / reversal setup",
        "investor_posture": "Opportunistic accumulation",
        "exposure_language": "High opportunity but confirm with price action",
        "caveat": "Do not treat this as a defensive label; the opportunity profile is strong but can still be noisy intrawindow.",
        "color_hex": "#7C3AED",
        "priority": 4,
    },
    "Volatility Caution": {
        "meaning": "Volatility dominates directional edge",
        "investor_posture": "Capital preservation",
        "exposure_language": "Low exposure / defensive",
        "caveat": "This remains the clearest caution state in the current taxonomy family.",
        "color_hex": "#7F1D1D",
        "priority": 2,
    },
}


def safe_float(value) -> float:
    return np.nan if pd.isna(value) else float(value)


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {}


def build_mapping_table(regime_audit: pd.DataFrame, rename_df: pd.DataFrame) -> pd.DataFrame:
    regime_lookup = regime_audit.set_index("taxonomy_v3")
    rename_lookup = rename_df.set_index("taxonomy_v3")
    rows: List[Dict[str, object]] = []
    for v3_label, v4_label in TAXONOMY_V4_MAP.items():
        details = TAXONOMY_V4_DETAILS[v4_label]
        regime_row = regime_lookup.loc[v3_label] if v3_label in regime_lookup.index else pd.Series(dtype=object)
        rename_row = rename_lookup.loc[v3_label] if v3_label in rename_lookup.index else pd.Series(dtype=object)
        rows.append(
            {
                "taxonomy_v3": v3_label,
                "taxonomy_v4": v4_label,
                "meaning": details["meaning"],
                "investor_posture": details["investor_posture"],
                "exposure_language": details["exposure_language"],
                "caveat": details["caveat"],
                "color_hex": details["color_hex"],
                "priority": details["priority"],
                "sample_count": regime_row.get("sample_count", np.nan),
                "sample_share": regime_row.get("sample_share", np.nan),
                "average_ml_probability": regime_row.get("average_ml_probability", np.nan),
                "average_astro_momentum": regime_row.get("average_astro_momentum", np.nan),
                "forward_return_30d": regime_row.get("forward_return_30d", np.nan),
                "forward_return_60d": regime_row.get("forward_return_60d", np.nan),
                "forward_return_90d": regime_row.get("forward_return_90d", np.nan),
                "win_rate_30d": regime_row.get("win_rate_30d", np.nan),
                "stability_assessment": regime_row.get("stability_assessment", ""),
                "recommended_action": rename_row.get("recommended_action", ""),
                "legacy_suggested_label": rename_row.get("suggested_label", ""),
                "primary_issue": rename_row.get("primary_issue", ""),
                "evidence_summary": rename_row.get("evidence_summary", ""),
            }
        )
    return pd.DataFrame(rows)


def build_v4_reason(v4_label: str, mapping_row: pd.Series) -> str:
    ret30 = safe_float(mapping_row.get("forward_return_30d", np.nan))
    ret60 = safe_float(mapping_row.get("forward_return_60d", np.nan))
    ret90 = safe_float(mapping_row.get("forward_return_90d", np.nan))
    win30 = safe_float(mapping_row.get("win_rate_30d", np.nan))
    stability = mapping_row.get("stability_assessment", "")
    details = TAXONOMY_V4_DETAILS[v4_label]

    if v4_label == "Constructive Drift":
        return (
            f"Historical outcomes are steadily positive (30D={ret30:.2%}, 60D={ret60:.2%}, 90D={ret90:.2%}) "
            f"with 30D win rate {win30:.2%}. This supports a measured long bias rather than aggressive chasing."
        )
    if v4_label == "High Conviction Expansion":
        return (
            f"Historical outcomes are strong (30D={ret30:.2%}, 60D={ret60:.2%}) with 30D win rate {win30:.2%}, "
            f"but stability is {stability.lower()} because the active sample is concentrated."
        )
    if v4_label == "Transition / Low Conviction":
        return (
            f"Historical outcomes are only moderately positive (30D={ret30:.2%}, 60D={ret60:.2%}) "
            f"and this state absorbs a large share of windows, so patience and selectivity matter."
        )
    if v4_label == "Recovery / Reversal Setup":
        return (
            f"Historical forward returns are unexpectedly strong after stress (30D={ret30:.2%}, 60D={ret60:.2%}, "
            f"90D={ret90:.2%}) with 30D win rate {win30:.2%}. This behaves like a recovery / reversal setup, not a defensive trend."
        )
    return (
        f"Historical returns are weakest here (30D={ret30:.2%}, 60D={ret60:.2%}, 90D={ret90:.2%}) "
        f"and stability is {stability.lower()}. Volatility dominates directional edge."
    )


def build_v4_narrative(row: pd.Series, mapping_row: pd.Series) -> str:
    label = row["taxonomy_v4"]
    details = TAXONOMY_V4_DETAILS[label]
    reason = row["taxonomy_v4_reason"]
    date_span = f"{row['start_date']} to {row['end_date']}"
    return (
        f"From {date_span}, the outlook is {label.lower()}. {details['meaning']}. "
        f"{reason} Investor posture: {details['investor_posture'].lower()}. "
        f"Exposure language: {details['exposure_language'].lower()}. "
        f"Caveat: {details['caveat']}"
    )


def transform_intelligence_v4(intelligence_v3: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    mapping_lookup = mapping_df.set_index("taxonomy_v3")
    df = intelligence_v3.copy()
    source_col = "taxonomy_v3" if "taxonomy_v3" in df.columns else "taxonomy_v2"
    df["taxonomy_v4_source"] = df[source_col]
    df["taxonomy_v4"] = df["taxonomy_v4_source"].map(TAXONOMY_V4_MAP).fillna(df["taxonomy_v4_source"])
    df["v4_posture"] = df["taxonomy_v4"].map(lambda x: TAXONOMY_V4_DETAILS[x]["investor_posture"])
    df["taxonomy_v4_meaning"] = df["taxonomy_v4"].map(lambda x: TAXONOMY_V4_DETAILS[x]["meaning"])
    df["taxonomy_v4_exposure_language"] = df["taxonomy_v4"].map(lambda x: TAXONOMY_V4_DETAILS[x]["exposure_language"])
    df["taxonomy_v4_caveat"] = df["taxonomy_v4"].map(lambda x: TAXONOMY_V4_DETAILS[x]["caveat"])
    df["taxonomy_v4_color"] = df["taxonomy_v4"].map(lambda x: TAXONOMY_V4_DETAILS[x]["color_hex"])
    df["taxonomy_v4_priority"] = df["taxonomy_v4"].map(lambda x: TAXONOMY_V4_DETAILS[x]["priority"])
    df["taxonomy_v4_reason"] = df.apply(
        lambda row: build_v4_reason(
            row["taxonomy_v4"],
            mapping_lookup.loc[row["taxonomy_v4_source"]] if row["taxonomy_v4_source"] in mapping_lookup.index else pd.Series(dtype=object),
        ),
        axis=1,
    )
    df["narrative_v4"] = df.apply(
        lambda row: build_v4_narrative(
            row,
            mapping_lookup.loc[row["taxonomy_v4_source"]] if row["taxonomy_v4_source"] in mapping_lookup.index else pd.Series(dtype=object),
        ),
        axis=1,
    )
    df["is_next_major_bullish_opportunity_v4"] = df["taxonomy_v4"].isin(
        ["Constructive Drift", "High Conviction Expansion", "Recovery / Reversal Setup"]
    )
    df["is_next_major_risk_window_v4"] = df["taxonomy_v4"].isin(["Volatility Caution"])
    return df


def write_report(
    mapping_df: pd.DataFrame,
    intelligence_v4: pd.DataFrame,
    dashboard_current: Dict[str, object],
    dashboard_summary: Dict[str, object],
    dashboard_timeline: Dict[str, object],
    dashboard_risk_calendar: Dict[str, object],
    defensive_deep_dive: pd.DataFrame,
    transition_matrix: pd.DataFrame,
) -> None:
    current_label = dashboard_current.get("current_taxonomy", "")
    next_30 = dashboard_summary.get("30D Outlook", {})
    next_90 = dashboard_summary.get("90D Outlook", {})
    next_365 = dashboard_summary.get("365D Outlook", {})
    focus_transitions = transition_matrix[
        transition_matrix["from_taxonomy"].isin(TAXONOMY_V4_MAP.keys())
    ].head(12)
    lines = [
        "# Forecast Taxonomy v4 Semantic Revision",
        "",
        "## Objective",
        "Update the taxonomy interpretation layer so the dashboard language matches validated historical behavior, without changing model predictions or forecast calculations.",
        "",
        "## Core Semantic Changes",
        "- `High Momentum Expansion` -> `High Conviction Expansion`",
        "- `Tactical Neutral` -> `Transition / Low Conviction`",
        "- `Defensive / Weak Trend` -> `Recovery / Reversal Setup`",
        "- `High Volatility Risk` -> `Volatility Caution`",
        "- `Constructive Drift` remains unchanged",
        "",
        "## Mapping Table",
        dataframe_to_markdown(mapping_df),
        "",
        "## Current Dashboard Context",
        f"- Current taxonomy before JSON refresh: `{current_label}`",
        f"- 30D dominant outlook after semantic revision: `{next_30.get('dominant_taxonomy', '')}`",
        f"- 90D dominant outlook after semantic revision: `{next_90.get('dominant_taxonomy', '')}`",
        f"- 365D dominant outlook after semantic revision: `{next_365.get('dominant_taxonomy', '')}`",
        "",
        "## Why The Major Rename Matters",
        "The regime audit showed that `Defensive / Weak Trend` had strong forward returns and an 85.71% 30D win rate. That is inconsistent with a defensive label, so the new name emphasizes recovery and reversal opportunity instead of weakness.",
        "",
        "## Defensive / Weak Trend Deep Dive Snapshot",
        dataframe_to_markdown(defensive_deep_dive.head(12)),
        "",
        "## Transition Snapshot",
        dataframe_to_markdown(
            focus_transitions[
                [
                    "from_taxonomy",
                    "to_taxonomy",
                    "transition_count",
                    "average_return_7d",
                    "average_return_14d",
                    "average_return_30d",
                    "win_rate",
                ]
            ]
        ),
        "",
        "## Current Forecast Windows Under v4",
        dataframe_to_markdown(
            intelligence_v4[
                [
                    "start_date",
                    "end_date",
                    "taxonomy_v4",
                    "v4_posture",
                    "average_confidence",
                    "average_ml_probability",
                    "taxonomy_v4_exposure_language",
                ]
            ].head(20)
        ),
        "",
        "## Supporting Files Refreshed",
        f"- `{CURRENT_STATE_PATH}`",
        f"- `{SUMMARY_PATH}`",
        f"- `{TIMELINE_PATH}`",
        f"- `{RISK_CALENDAR_PATH}`",
        "",
        "## Source Note",
        REGIME_AUDIT_REPORT_PATH.read_text(encoding="utf-8").splitlines()[0],
    ]
    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    regime_audit = pd.read_csv(REGIME_AUDIT_PATH)
    transition_matrix = pd.read_csv(TRANSITION_MATRIX_PATH)
    rename_df = pd.read_csv(RENAME_RECOMMENDATION_PATH)
    defensive_deep_dive = pd.read_csv(DEFENSIVE_DEEP_DIVE_PATH)
    intelligence_v3 = pd.read_csv(FORECAST_INTELLIGENCE_V3_PATH)
    dashboard_current = load_json(CURRENT_STATE_PATH)
    dashboard_summary = load_json(SUMMARY_PATH)
    dashboard_timeline = load_json(TIMELINE_PATH)
    dashboard_risk_calendar = load_json(RISK_CALENDAR_PATH)

    mapping_df = build_mapping_table(regime_audit, rename_df)
    intelligence_v4 = transform_intelligence_v4(intelligence_v3, mapping_df)

    mapping_df.to_csv(OUTPUT_MAPPING_PATH, index=False)
    intelligence_v4.to_csv(OUTPUT_CSV_PATH, index=False)
    write_report(
        mapping_df,
        intelligence_v4,
        dashboard_current,
        dashboard_summary,
        dashboard_timeline,
        dashboard_risk_calendar,
        defensive_deep_dive,
        transition_matrix,
    )

    print(f"Wrote {OUTPUT_MAPPING_PATH}")
    print(f"Wrote {OUTPUT_CSV_PATH}")
    print(f"Wrote {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
