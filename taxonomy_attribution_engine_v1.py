from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from astro_alpha_audit_v1 import build_full_feature_frame_safe, dataframe_to_markdown
from build_ml_dataset import (
    ASPECT_STRENGTH_COLUMNS,
    NATAL_TARGET_STRENGTH_COLUMNS,
    PLANET_SIGNAL_COLUMNS,
    RAW_SCORE_COLUMNS,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

FORECAST_INTELLIGENCE_V3_PATH = DATA_DIR / "forecast_intelligence_v3.csv"
FORECAST_TIMELINE_PATH = DATA_DIR / "future_forecast_timeline.csv"
OUTPUT_ATTRIBUTION_PATH = DATA_DIR / "taxonomy_attribution.csv"
OUTPUT_FEATURE_IMPORTANCE_PATH = DATA_DIR / "taxonomy_feature_importance.csv"
OUTPUT_REPORT_PATH = DATA_DIR / "taxonomy_attribution_report.md"

TAXONOMY_ORDER = [
    "Constructive Drift",
    "High Momentum Expansion",
    "Tactical Neutral",
    "Defensive / Weak Trend",
    "High Volatility Risk",
]

CORE_ASTRO_FEATURES = [
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
    "raw_astro_total_strength",
    "raw_astro_directional_signal",
    "raw_astro_event_count",
    "house_activation_strength",
] + RAW_SCORE_COLUMNS


def format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def safe_range_text(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return ""
    return f"{clean.quantile(0.25):.2f} to {clean.quantile(0.75):.2f}"


def build_daily_taxonomy_frame(
    intelligence_v3: pd.DataFrame,
    future_timeline: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, window in intelligence_v3.iterrows():
        start = pd.to_datetime(window["start_date"])
        end = pd.to_datetime(window["end_date"])
        window_dates = future_timeline[
            (future_timeline["date"] >= start) & (future_timeline["date"] <= end)
        ]["date"]
        for dt in window_dates:
            rows.append(
                {
                    "date": dt,
                    "taxonomy_v3": window["taxonomy_v3"],
                    "taxonomy_v3_priority": window.get("taxonomy_v3_priority", np.nan),
                    "v3_posture": window.get("v3_posture", ""),
                    "taxonomy_v3_reason": window.get("taxonomy_v3_reason", ""),
                    "narrative_v3": window.get("narrative_v3", ""),
                }
            )
    return pd.DataFrame(rows)


def build_future_feature_frame() -> pd.DataFrame:
    intelligence_v3 = pd.read_csv(
        FORECAST_INTELLIGENCE_V3_PATH,
        parse_dates=["start_date", "end_date"],
    )
    future_timeline = pd.read_csv(
        FORECAST_TIMELINE_PATH,
        parse_dates=["date"],
    )
    full_df = build_full_feature_frame_safe()

    future_dates = future_timeline["date"].dropna().sort_values().unique()
    future_frame = full_df[full_df["date"].isin(future_dates)].copy()
    future_frame = future_frame.sort_values("date").reset_index(drop=True)
    future_frame = future_frame.merge(
        future_timeline[
            [
                "date",
                "ml_probability",
                "signal",
                "confidence_score",
                "astro_score",
                "forecast_strength",
                "risk_level",
            ]
        ],
        on="date",
        how="left",
    )

    taxonomy_daily = build_daily_taxonomy_frame(intelligence_v3, future_timeline)
    future_frame = future_frame.merge(taxonomy_daily, on="date", how="left")
    return future_frame


def get_interpretable_feature_columns(df: pd.DataFrame) -> List[str]:
    candidates = CORE_ASTRO_FEATURES + PLANET_SIGNAL_COLUMNS + ASPECT_STRENGTH_COLUMNS + NATAL_TARGET_STRENGTH_COLUMNS
    columns: List[str] = []
    for col in candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            columns.append(col)
    return list(dict.fromkeys(columns))


def make_feature_family(feature: str) -> str:
    if feature in PLANET_SIGNAL_COLUMNS:
        return "planet"
    if feature in ASPECT_STRENGTH_COLUMNS:
        return "aspect"
    if feature in NATAL_TARGET_STRENGTH_COLUMNS:
        return "natal_target"
    if feature in RAW_SCORE_COLUMNS:
        return "raw_score"
    return "core_astro"


def build_feature_importance(
    frame: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    overall_means = frame[features].mean(numeric_only=True)
    overall_std = frame[features].std(numeric_only=True, ddof=0).replace(0, np.nan)

    rows: List[Dict[str, object]] = []
    for state in TAXONOMY_ORDER:
        state_slice = frame[frame["taxonomy_v3"] == state].copy()
        for feature in features:
            feature_series = pd.to_numeric(state_slice[feature], errors="coerce")
            if feature_series.dropna().empty:
                continue
            state_mean = float(feature_series.mean())
            base_mean = float(overall_means.get(feature, np.nan))
            base_std = float(overall_std.get(feature, np.nan))
            differential = state_mean - base_mean
            zscore = differential / base_std if pd.notna(base_std) and base_std > 0 else np.nan
            rows.append(
                {
                    "taxonomy_v3": state,
                    "feature": feature,
                    "feature_family": make_feature_family(feature),
                    "state_mean": state_mean,
                    "overall_mean": base_mean,
                    "differential": differential,
                    "zscore_diff": zscore,
                    "abs_zscore_diff": abs(zscore) if pd.notna(zscore) else np.nan,
                    "direction": "positive" if differential >= 0 else "negative",
                }
            )

    importance = pd.DataFrame(rows)
    if importance.empty:
        return importance
    return importance.sort_values(
        ["taxonomy_v3", "abs_zscore_diff", "feature"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def top_feature_list(
    importance: pd.DataFrame,
    state: str,
    direction: str,
    family_filter: List[str] | None = None,
    top_n: int = 3,
) -> str:
    subset = importance[importance["taxonomy_v3"] == state].copy()
    subset = subset[subset["direction"] == direction]
    subset = subset[subset["zscore_diff"].notna()]
    if family_filter is not None:
        subset = subset[subset["feature_family"].isin(family_filter)]
    subset = subset.sort_values("abs_zscore_diff", ascending=False).head(top_n)
    if subset.empty:
        return ""
    return ", ".join(f"{row.feature} ({row.zscore_diff:+.2f})" for row in subset.itertuples())


def summarize_state(
    frame: pd.DataFrame,
    importance: pd.DataFrame,
    state: str,
) -> Dict[str, object]:
    state_slice = frame[frame["taxonomy_v3"] == state].copy()
    if state_slice.empty:
        return {
            "taxonomy_v3": state,
            "sample_count": 0,
            "average_astro_momentum": np.nan,
            "average_ml_probability": np.nan,
            "typical_momentum_range": "",
            "typical_probability_range": "",
            "top_positive_astro_features": "",
            "top_negative_astro_features": "",
            "most_influential_planets": "",
            "most_influential_aspects": "",
        }

    return {
        "taxonomy_v3": state,
        "sample_count": int(len(state_slice)),
        "average_astro_momentum": float(state_slice["astro_momentum_v2_smooth"].mean()),
        "average_ml_probability": float(state_slice["ml_probability"].mean()),
        "average_confidence": float(state_slice["confidence_score"].mean()),
        "average_astro_score": float(state_slice["astro_score"].mean()),
        "typical_momentum_range": safe_range_text(state_slice["astro_momentum_v2_smooth"]),
        "typical_probability_range": safe_range_text(state_slice["ml_probability"]),
        "top_positive_astro_features": top_feature_list(importance, state, "positive", family_filter=["core_astro", "raw_score"]),
        "top_negative_astro_features": top_feature_list(importance, state, "negative", family_filter=["core_astro", "raw_score"]),
        "most_influential_planets": top_feature_list(importance, state, "positive", family_filter=["planet"]),
        "most_influential_aspects": top_feature_list(importance, state, "positive", family_filter=["aspect"]),
    }


def build_report(
    summary_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> str:
    lines = [
        "# Taxonomy Attribution Engine v1",
        "",
        "This report explains the current Forecast Taxonomy v3 states using the live future feature frame, ML probabilities, and astro feature inputs.",
        "",
        "## State Summary",
        dataframe_to_markdown(
            summary_df[
                [
                    "taxonomy_v3",
                    "sample_count",
                    "average_astro_momentum",
                    "average_ml_probability",
                    "typical_momentum_range",
                    "typical_probability_range",
                    "most_influential_planets",
                    "most_influential_aspects",
                ]
            ]
        ),
        "",
        "## Per-State Narrative",
    ]

    for row in summary_df.itertuples():
        lines.extend(
            [
                f"### {row.taxonomy_v3}",
                (
                    f"- Average astro momentum: `{row.average_astro_momentum:.2f}`"
                    if pd.notna(row.average_astro_momentum)
                    else "- Average astro momentum: unavailable"
                ),
                (
                    f"- Average ML probability: `{row.average_ml_probability:.2%}`"
                    if pd.notna(row.average_ml_probability)
                    else "- Average ML probability: unavailable"
                ),
                f"- Typical momentum range: `{row.typical_momentum_range or 'N/A'}`",
                f"- Typical probability range: `{row.typical_probability_range or 'N/A'}`",
                f"- Top positive astro features: {row.top_positive_astro_features or 'N/A'}",
                f"- Top negative astro features: {row.top_negative_astro_features or 'N/A'}",
                f"- Most influential planets: {row.most_influential_planets or 'N/A'}",
                f"- Most influential aspects: {row.most_influential_aspects or 'N/A'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Feature Importance Snapshot",
            dataframe_to_markdown(
                importance_df[
                    [
                        "taxonomy_v3",
                        "feature",
                        "feature_family",
                        "state_mean",
                        "overall_mean",
                        "differential",
                        "zscore_diff",
                        "direction",
                    ]
                ].head(40)
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    future_frame = build_future_feature_frame()
    features = get_interpretable_feature_columns(future_frame)
    importance_df = build_feature_importance(future_frame, features)
    summary_rows = [summarize_state(future_frame, importance_df, state) for state in TAXONOMY_ORDER]
    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(OUTPUT_ATTRIBUTION_PATH, index=False)
    importance_df.to_csv(OUTPUT_FEATURE_IMPORTANCE_PATH, index=False)
    OUTPUT_REPORT_PATH.write_text(build_report(summary_df, importance_df), encoding="utf-8")

    print(f"Saved {OUTPUT_ATTRIBUTION_PATH}")
    print(f"Saved {OUTPUT_FEATURE_IMPORTANCE_PATH}")
    print(f"Saved {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
