from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ALPHA_AUDIT_REPORT_PATH = Path("data/astro_alpha_audit_report.md")
TAXONOMY_AUDIT_PATH = Path("data/taxonomy_alpha_audit.csv")
MOMENTUM_AUDIT_PATH = Path("data/astro_momentum_alpha_audit.csv")
TURNING_POINT_AUDIT_PATH = Path("data/turning_point_alpha_audit.csv")
FORECAST_INTELLIGENCE_V2_PATH = Path("data/forecast_intelligence_v2.csv")

MAPPING_OUTPUT_PATH = Path("data/forecast_taxonomy_v3_mapping.csv")
OUTPUT_CSV_PATH = Path("data/forecast_intelligence_v3.csv")
OUTPUT_REPORT_PATH = Path("data/forecast_intelligence_v3_report.md")

TAXONOMY_V3_MAP = {
    "False Bull / Exhaustion Risk": "High Momentum Expansion",
    "Constructive / Positive Drift": "Constructive Drift",
    "Neutral / Tactical": "Tactical Neutral",
    "Bearish": "Defensive / Weak Trend",
    "High Risk": "High Volatility Risk",
}

TAXONOMY_V3_POSTURE = {
    "High Momentum Expansion": "Momentum Long Bias",
    "Constructive Drift": "Constructive Long Bias",
    "Tactical Neutral": "Tactical / Wait",
    "Defensive / Weak Trend": "Defensive / Short Bias",
    "High Volatility Risk": "Defensive / Volatility Control",
}

TAXONOMY_V3_COLOR = {
    "High Momentum Expansion": "#D97706",
    "Constructive Drift": "#2E7D32",
    "Tactical Neutral": "#C9A227",
    "Defensive / Weak Trend": "#C62828",
    "High Volatility Risk": "#7F1D1D",
}

TAXONOMY_V3_PRIORITY = {
    "High Momentum Expansion": 6,
    "Constructive Drift": 5,
    "Tactical Neutral": 4,
    "High Volatility Risk": 2,
    "Defensive / Weak Trend": 1,
}


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


def taxonomy_v3_reason(row: pd.Series) -> str:
    label = row["taxonomy_v3"]
    ret7 = float(row.get("average_forward_return_7d", np.nan))
    ret14 = float(row.get("average_forward_return_14d", np.nan))
    ret30 = float(row.get("average_forward_return_30d", np.nan))
    win30 = float(row.get("win_rate_30d", np.nan))

    if label == "High Momentum Expansion":
        return (
            f"Alpha audit evidence is positive rather than fragile: 7D={ret7:.2%}, 14D={ret14:.2%}, "
            f"30D={ret30:.2%}, with 30D win rate {win30:.2%}. Strength should be respected, not faded."
        )
    if label == "Constructive Drift":
        return (
            f"Historical outcomes stay constructive across horizons (7D={ret7:.2%}, 14D={ret14:.2%}, "
            f"30D={ret30:.2%}) with steady follow-through rather than explosive upside."
        )
    if label == "Tactical Neutral":
        return (
            f"Historical returns are mixed to mildly positive (7D={ret7:.2%}, 14D={ret14:.2%}, 30D={ret30:.2%}), "
            f"so the state is best treated tactically rather than as a high-conviction trend."
        )
    if label == "Defensive / Weak Trend":
        return (
            f"Historical downside and weak follow-through remain the base case here "
            f"(7D={ret7:.2%}, 14D={ret14:.2%}, 30D={ret30:.2%})."
        )
    return (
        f"Volatility dominates directional edge in this state. Historical outcomes are weaker and less reliable "
        f"(7D={ret7:.2%}, 14D={ret14:.2%}, 30D={ret30:.2%})."
    )


def build_narrative_v3(row: pd.Series) -> str:
    label = row["taxonomy_v3"]
    date_span = f"{row['start_date']} to {row['end_date']}"
    reason = row["taxonomy_v3_reason"]

    if label == "High Momentum Expansion":
        return (
            f"From {date_span}, the outlook shifts into high momentum expansion. {reason} "
            "This is a trend-following state where upside strength historically persisted."
        )
    if label == "Constructive Drift":
        return (
            f"From {date_span}, the outlook is constructive drift. {reason} "
            "The evidence supports measured long exposure with patience rather than chase behavior."
        )
    if label == "Tactical Neutral":
        return (
            f"From {date_span}, the outlook is tactical neutral. {reason} "
            "Flexibility and selective positioning matter more than full directional commitment."
        )
    if label == "Defensive / Weak Trend":
        return (
            f"From {date_span}, the outlook is defensive / weak trend. {reason} "
            "The historical profile argues for smaller risk budgets and caution on long exposure."
        )
    return (
        f"From {date_span}, the outlook is high volatility risk. {reason} "
        "Risk control matters more than conviction until the signal base stabilizes."
    )


def build_mapping_table(taxonomy_audit: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for legacy_label, v3_label in TAXONOMY_V3_MAP.items():
        audit_row = taxonomy_audit.loc[taxonomy_audit["label"] == legacy_label]
        if audit_row.empty:
            continue
        audit_row = audit_row.iloc[0]
        rows.append(
            {
                "legacy_taxonomy": legacy_label,
                "taxonomy_v3": v3_label,
                "v3_posture": TAXONOMY_V3_POSTURE[v3_label],
                "color_hex": TAXONOMY_V3_COLOR[v3_label],
                "average_return_7d": float(audit_row["average_return_7d"]),
                "average_return_14d": float(audit_row["average_return_14d"]),
                "average_return_30d": float(audit_row["average_return_30d"]),
                "win_rate_30d": float(audit_row["win_rate_30d"]),
                "sample_count_30d": int(float(audit_row["sample_count_30d"])),
                "mapping_rationale": taxonomy_v3_reason(
                    pd.Series(
                        {
                            "taxonomy_v3": v3_label,
                            "average_forward_return_7d": audit_row["average_return_7d"],
                            "average_forward_return_14d": audit_row["average_return_14d"],
                            "average_forward_return_30d": audit_row["average_return_30d"],
                            "win_rate_30d": audit_row["win_rate_30d"],
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    mapping_df: pd.DataFrame,
    intelligence_v3: pd.DataFrame,
    taxonomy_audit: pd.DataFrame,
    momentum_audit: pd.DataFrame,
    turning_audit: pd.DataFrame,
) -> None:
    current_window = intelligence_v3.iloc[0] if not intelligence_v3.empty else pd.Series(dtype=object)
    next_positive = intelligence_v3[
        intelligence_v3["taxonomy_v3"].isin(["Constructive Drift", "High Momentum Expansion"])
    ].head(1)
    next_defensive = intelligence_v3[
        intelligence_v3["taxonomy_v3"].isin(["Defensive / Weak Trend", "High Volatility Risk"])
    ].head(1)

    lines = [
        "# Forecast Taxonomy v3",
        "",
        "## Objective",
        "This version updates taxonomy names and investor interpretation to match the Astro Alpha Audit evidence, without changing any forecast calculations.",
        "",
        "## Core Change",
        "- `False Bull / Exhaustion Risk` was renamed because the alpha audit showed strong positive historical performance rather than exhaustion.",
        "- The interpretation layer now favors evidence-based investor language instead of cautionary naming that conflicts with realized outcomes.",
        "",
        "## Mapping",
        dataframe_to_markdown(mapping_df),
        "",
        "## Current Read",
        f"- Current taxonomy v3: `{current_window.get('taxonomy_v3', '')}`",
        f"- Current posture: `{current_window.get('v3_posture', '')}`",
        f"- Current narrative: {current_window.get('narrative_v3', '')}",
        "",
        "## Next Windows",
        (
            f"- Next positive window: `{next_positive.iloc[0]['taxonomy_v3']}` from "
            f"`{next_positive.iloc[0]['start_date']}` to `{next_positive.iloc[0]['end_date']}`"
            if not next_positive.empty
            else "- Next positive window: none"
        ),
        (
            f"- Next defensive window: `{next_defensive.iloc[0]['taxonomy_v3']}` from "
            f"`{next_defensive.iloc[0]['start_date']}` to `{next_defensive.iloc[0]['end_date']}`"
            if not next_defensive.empty
            else "- Next defensive window: none"
        ),
        "",
        "## Supporting Audit Snapshots",
        "### Taxonomy Alpha Audit",
        dataframe_to_markdown(
            taxonomy_audit[
                [
                    "label",
                    "average_return_7d",
                    "average_return_14d",
                    "average_return_30d",
                    "win_rate_30d",
                    "sample_count_30d",
                ]
            ]
        ),
        "",
        "### Momentum Alpha Audit Highlights",
        dataframe_to_markdown(
            momentum_audit[
                [
                    "label",
                    "average_return_30d",
                    "win_rate_30d",
                    "sample_count_30d",
                ]
            ].sort_values("average_return_30d", ascending=False).head(5)
        ),
        "",
        "### Turning Point Highlights",
        dataframe_to_markdown(
            turning_audit[
                [
                    "audit_group",
                    "label",
                    "average_return_30d",
                    "win_rate_30d",
                    "sample_count_30d",
                ]
            ].sort_values("average_return_30d", ascending=False).head(6)
        ),
        "",
        "## Source Note",
        ALPHA_AUDIT_REPORT_PATH.read_text(encoding="utf-8").splitlines()[0],
    ]
    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    taxonomy_audit = pd.read_csv(TAXONOMY_AUDIT_PATH)
    momentum_audit = pd.read_csv(MOMENTUM_AUDIT_PATH)
    turning_audit = pd.read_csv(TURNING_POINT_AUDIT_PATH)
    intelligence_v2 = pd.read_csv(FORECAST_INTELLIGENCE_V2_PATH)

    mapping_df = build_mapping_table(taxonomy_audit)
    mapping_lookup = mapping_df.set_index("legacy_taxonomy")

    intelligence_v3 = intelligence_v2.copy()
    intelligence_v3["taxonomy_v2_source"] = intelligence_v3["taxonomy_v2"]
    intelligence_v3["taxonomy_v3"] = intelligence_v3["taxonomy_v2"].map(TAXONOMY_V3_MAP).fillna(intelligence_v3["taxonomy_v2"])
    intelligence_v3["v3_posture"] = intelligence_v3["taxonomy_v3"].map(TAXONOMY_V3_POSTURE).fillna(intelligence_v3["v2_posture"])

    reasons = []
    for _, row in intelligence_v3.iterrows():
        legacy = row["taxonomy_v2_source"]
        if legacy in mapping_lookup.index:
            audit_row = mapping_lookup.loc[legacy]
            reasons.append(
                taxonomy_v3_reason(
                    pd.Series(
                        {
                            "taxonomy_v3": row["taxonomy_v3"],
                            "average_forward_return_7d": audit_row["average_return_7d"],
                            "average_forward_return_14d": audit_row["average_return_14d"],
                            "average_forward_return_30d": audit_row["average_return_30d"],
                            "win_rate_30d": audit_row["win_rate_30d"],
                        }
                    )
                )
            )
        else:
            reasons.append(str(row.get("taxonomy_reason", "")))
    intelligence_v3["taxonomy_v3_reason"] = reasons
    intelligence_v3["narrative_v3"] = intelligence_v3.apply(build_narrative_v3, axis=1)
    intelligence_v3["taxonomy_v3_priority"] = intelligence_v3["taxonomy_v3"].map(TAXONOMY_V3_PRIORITY)

    mapping_df.to_csv(MAPPING_OUTPUT_PATH, index=False)
    intelligence_v3.to_csv(OUTPUT_CSV_PATH, index=False)
    write_report(mapping_df, intelligence_v3, taxonomy_audit, momentum_audit, turning_audit)

    print(f"Saved {MAPPING_OUTPUT_PATH}")
    print(f"Saved {OUTPUT_CSV_PATH}")
    print(f"Saved {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
