from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from forecast_intelligence_v1 import dataframe_to_markdown
from taxonomy_performance_validation_v1 import (
    TARGET_TAXONOMIES,
    build_historical_taxonomy_daily,
    classify_stability,
    load_dashboard_summary,
    read_resilient_csv,
)


ROOT = Path(".")
ML_DATASET_PATH = ROOT / "data" / "ml_dataset.csv"
FORECAST_INTELLIGENCE_V3_PATH = ROOT / "data" / "forecast_intelligence_v3.csv"
FUTURE_TIMELINE_PATH = ROOT / "data" / "future_forecast_timeline.csv"
TAXONOMY_ATTRIBUTION_PATH = ROOT / "data" / "taxonomy_attribution.csv"
TAXONOMY_FEATURE_IMPORTANCE_PATH = ROOT / "data" / "taxonomy_feature_importance.csv"
VALIDATION_PATH = ROOT / "data" / "taxonomy_performance_validation.csv"
YEARLY_PATH = ROOT / "data" / "taxonomy_performance_by_year.csv"
EXPOSURE_PATH = ROOT / "data" / "taxonomy_exposure_recommendation.csv"
VALIDATION_REPORT_PATH = ROOT / "data" / "taxonomy_performance_validation_report.md"

OUTPUT_AUDIT_PATH = ROOT / "data" / "taxonomy_regime_audit.csv"
OUTPUT_TRANSITION_PATH = ROOT / "data" / "taxonomy_transition_matrix.csv"
OUTPUT_RENAME_PATH = ROOT / "data" / "taxonomy_rename_recommendation.csv"
OUTPUT_DEFENSIVE_PATH = ROOT / "data" / "defensive_weak_trend_deep_dive.csv"
OUTPUT_REPORT_PATH = ROOT / "data" / "taxonomy_regime_audit_report.md"

PLANET_SIGNAL_COLUMNS = [
    "sun_signal",
    "moon_signal",
    "mercury_signal",
    "venus_signal",
    "mars_signal",
    "jupiter_signal",
    "saturn_signal",
    "uranus_signal",
    "neptune_signal",
    "pluto_signal",
]
ASPECT_STRENGTH_COLUMNS = [
    "conjunction_strength",
    "trine_strength",
    "sextile_strength",
    "square_strength",
    "opposition_strength",
]
NATAL_TARGET_COLUMNS = [
    "sun_target_strength",
    "moon_target_strength",
    "asc_target_strength",
    "mc_target_strength",
]
ASPECT_COUNT_COLUMNS = [
    "aspect_count_conjunction",
    "aspect_count_house_position",
    "aspect_count_opposition",
    "aspect_count_sextile",
    "aspect_count_square",
    "aspect_count_trine",
]
FORWARD_HORIZONS = [7, 14, 30, 60, 90]
FOCUS_TRANSITIONS = {
    ("Defensive / Weak Trend", "Constructive Drift"),
    ("Defensive / Weak Trend", "Tactical Neutral"),
    ("Tactical Neutral", "High Momentum Expansion"),
    ("Constructive Drift", "High Momentum Expansion"),
    ("High Momentum Expansion", "Tactical Neutral"),
    ("High Volatility Risk", "Constructive Drift"),
}


def load_support_frames():
    dataset = read_resilient_csv(ML_DATASET_PATH)
    intelligence_v3 = pd.read_csv(FORECAST_INTELLIGENCE_V3_PATH)
    future_timeline = pd.read_csv(FUTURE_TIMELINE_PATH)
    attribution = pd.read_csv(TAXONOMY_ATTRIBUTION_PATH)
    feature_importance = pd.read_csv(TAXONOMY_FEATURE_IMPORTANCE_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    yearly = pd.read_csv(YEARLY_PATH)
    exposure = pd.read_csv(EXPOSURE_PATH)
    report_text = VALIDATION_REPORT_PATH.read_text(encoding="utf-8")
    return dataset, intelligence_v3, future_timeline, attribution, feature_importance, validation, yearly, exposure, report_text


def enrich_historical_daily() -> pd.DataFrame:
    historical = build_historical_taxonomy_daily()
    dataset = read_resilient_csv(ML_DATASET_PATH)

    extra_cols = [
        "price",
        "btc_return_7d",
        "btc_return_14d",
        "btc_return_30d",
        "astro_bullish_score",
        "astro_bearish_score",
        "astro_reversal_score",
        "astro_volatility_score",
        "astro_compression_score",
        "astro_trend_start_score",
        "astro_trend_end_score",
        "astro_momentum_v2",
        "astro_momentum_v2_smooth",
        "raw_astro_total_strength",
        "raw_astro_directional_signal",
        "bullish",
        "bearish",
        "reversal",
        "volatility",
        "compression",
        "trend_start",
        "trend_end",
    ] + PLANET_SIGNAL_COLUMNS + ASPECT_STRENGTH_COLUMNS + NATAL_TARGET_COLUMNS
    available = [col for col in extra_cols if col in dataset.columns]
    historical = historical.merge(
        dataset[["date"] + available],
        on="date",
        how="left",
        suffixes=("", "_dataset"),
    )

    historical = historical.sort_values("date").reset_index(drop=True)
    for planet_signal in PLANET_SIGNAL_COLUMNS:
        if planet_signal not in historical.columns:
            planet_name = planet_signal.replace("_signal", "").capitalize()
            source_cols = [col for col in historical.columns if col.startswith("planet_") and col.endswith(f"_{planet_name}")]
            historical[planet_signal] = historical[source_cols].mean(axis=1) if source_cols else np.nan
    for aspect_signal in ASPECT_STRENGTH_COLUMNS:
        if aspect_signal not in historical.columns:
            aspect_name = aspect_signal.replace("_strength", "")
            source_col = f"aspect_count_{aspect_name}"
            historical[aspect_signal] = historical[source_col] if source_col in historical.columns else np.nan
    if "raw_astro_event_count" not in historical.columns:
        event_cols = [col for col in ASPECT_COUNT_COLUMNS if col in historical.columns]
        if event_cols:
            historical["raw_astro_event_count"] = historical[event_cols].sum(axis=1)
        else:
            historical["raw_astro_event_count"] = np.nan
    if "house_activation_strength" not in historical.columns:
        if "aspect_count_house_position" in historical.columns:
            historical["house_activation_strength"] = historical["aspect_count_house_position"]
        else:
            historical["house_activation_strength"] = np.nan
    historical["astro_momentum_slope_7d"] = historical["astro_momentum_v2_smooth"].diff(7)
    historical["prior_drawdown_30d"] = (
        historical["price"] / historical["price"].rolling(30, min_periods=5).max()
    ) - 1.0
    return historical


def dominant_from_columns(frame: pd.DataFrame, columns: List[str], top_n: int = 3) -> str:
    available = [col for col in columns if col in frame.columns]
    if not available or frame.empty:
        return ""
    means = frame[available].mean().sort_values(ascending=False).head(top_n)
    parts = [f"{idx} ({value:+.2f})" for idx, value in means.items() if pd.notna(value)]
    return ", ".join(parts)


def dominant_from_feature_importance(feature_df: pd.DataFrame, taxonomy: str, family: str, top_n: int = 3) -> str:
    subset = feature_df[
        (feature_df["taxonomy_v3"] == taxonomy) & (feature_df["feature_family"] == family)
    ].copy()
    if subset.empty:
        return ""
    subset = subset.sort_values("abs_zscore_diff", ascending=False).head(top_n)
    return ", ".join(
        f"{row['feature']} ({row['zscore_diff']:+.2f})"
        for _, row in subset.iterrows()
    )


def build_regime_audit(
    historical: pd.DataFrame,
    validation: pd.DataFrame,
    yearly: pd.DataFrame,
    attribution: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    validation_30 = validation[validation["horizon_days"] == 30].copy()
    validation_lookup = validation_30.set_index("taxonomy_v3")
    yearly_summary = yearly.groupby("taxonomy_v3").agg(
        positive_year_share=("average_forward_return_30d", lambda s: float((s > 0).mean())),
        active_years=("year", "nunique"),
        best_year_return_30d=("average_forward_return_30d", "max"),
        worst_year_return_30d=("average_forward_return_30d", "min"),
    )
    attribution_lookup = attribution.set_index("taxonomy_v3")

    rows: List[Dict[str, object]] = []
    for taxonomy in TARGET_TAXONOMIES:
        subset = historical[historical["taxonomy_v3"] == taxonomy].copy()
        attr = attribution_lookup.loc[taxonomy] if taxonomy in attribution_lookup.index else pd.Series(dtype=object)
        val = validation_lookup.loc[taxonomy] if taxonomy in validation_lookup.index else pd.Series(dtype=object)
        yearly_row = yearly_summary.loc[taxonomy] if taxonomy in yearly_summary.index else pd.Series(dtype=object)
        rows.append(
            {
                "taxonomy_v3": taxonomy,
                "sample_count": int(len(subset)),
                "sample_share": float(len(subset) / max(len(historical), 1)),
                "average_astro_momentum": float(subset["astro_momentum_v2_smooth"].mean()),
                "average_momentum_slope_7d": float(subset["astro_momentum_slope_7d"].mean()),
                "average_ml_probability": float(subset["ml_probability"].mean()),
                "average_confidence": float(subset["confidence_score"].mean()),
                "average_astro_score": float(subset["astro_score"].mean()),
                "average_compression_score": float(subset["astro_compression_score"].mean()),
                "average_bullish_score": float(subset["astro_bullish_score"].mean()),
                "average_bearish_score": float(subset["astro_bearish_score"].mean()),
                "average_reversal_score": float(subset["astro_reversal_score"].mean()),
                "average_volatility_score": float(subset["astro_volatility_score"].mean()),
                "average_event_count": float(subset["raw_astro_event_count"].mean()),
                "average_house_activation_strength": float(subset["house_activation_strength"].mean()),
                "average_prior_return_7d": float(subset["btc_return_7d"].mean()),
                "average_prior_return_14d": float(subset["btc_return_14d"].mean()),
                "average_prior_return_30d": float(subset["btc_return_30d"].mean()),
                "average_prior_drawdown_30d": float(subset["prior_drawdown_30d"].mean()),
                "forward_return_7d": float(subset["future_return_7d"].mean()),
                "forward_return_14d": float(subset["future_return_14d"].mean()),
                "forward_return_30d": float(subset["future_return_30d"].mean()),
                "forward_return_60d": float(subset["future_return_60d"].mean()),
                "forward_return_90d": float(subset["future_return_90d"].mean()),
                "win_rate_30d": float((subset["future_return_30d"] > 0).mean()),
                "dominant_planets": attr.get("most_influential_planets", "") or dominant_from_feature_importance(feature_df, taxonomy, "planet"),
                "dominant_aspects": attr.get("most_influential_aspects", "") or dominant_from_feature_importance(feature_df, taxonomy, "aspect"),
                "dominant_natal_targets": dominant_from_feature_importance(feature_df, taxonomy, "natal_target"),
                "typical_momentum_range": attr.get("typical_momentum_range", ""),
                "typical_probability_range": attr.get("typical_probability_range", ""),
                "stability_assessment": classify_stability(yearly, taxonomy),
                "positive_year_share": yearly_row.get("positive_year_share", np.nan),
                "active_years": yearly_row.get("active_years", np.nan),
                "best_year_return_30d": yearly_row.get("best_year_return_30d", np.nan),
                "worst_year_return_30d": yearly_row.get("worst_year_return_30d", np.nan),
                "return_volatility_ratio_30d": val.get("return_volatility_ratio", np.nan),
            }
        )
    return pd.DataFrame(rows)


def build_defensive_deep_dive(historical: pd.DataFrame) -> pd.DataFrame:
    focus = historical[historical["taxonomy_v3"] == "Defensive / Weak Trend"].copy()
    other = historical[historical["taxonomy_v3"] != "Defensive / Weak Trend"].copy()
    metrics = [
        ("average_prior_return_7d", "btc_return_7d"),
        ("average_prior_return_14d", "btc_return_14d"),
        ("average_prior_return_30d", "btc_return_30d"),
        ("average_prior_drawdown_30d", "prior_drawdown_30d"),
        ("average_astro_momentum", "astro_momentum_v2_smooth"),
        ("average_momentum_slope_7d", "astro_momentum_slope_7d"),
        ("average_ml_probability", "ml_probability"),
        ("average_bullish_score", "astro_bullish_score"),
        ("average_bearish_score", "astro_bearish_score"),
        ("average_reversal_score", "astro_reversal_score"),
        ("average_compression_score", "astro_compression_score"),
        ("average_volatility_score", "astro_volatility_score"),
        ("saturn_signal", "saturn_signal"),
        ("opposition_strength", "opposition_strength"),
        ("average_forward_return_30d", "future_return_30d"),
        ("average_forward_return_60d", "future_return_60d"),
        ("average_forward_return_90d", "future_return_90d"),
    ]
    rows = []
    for label, col in metrics:
        focus_mean = float(focus[col].mean())
        other_mean = float(other[col].mean())
        rows.append(
            {
                "metric": label,
                "defensive_weak_trend_mean": focus_mean,
                "other_taxonomies_mean": other_mean,
                "delta": focus_mean - other_mean,
            }
        )
    return pd.DataFrame(rows)


def classify_defensive_archetype(regime_df: pd.DataFrame, deep_dive: pd.DataFrame) -> str:
    row = regime_df.loc[regime_df["taxonomy_v3"] == "Defensive / Weak Trend"].iloc[0]
    lookup = deep_dive.set_index("metric")
    prior_30 = float(lookup.loc["average_prior_return_30d", "defensive_weak_trend_mean"])
    drawdown = float(lookup.loc["average_prior_drawdown_30d", "defensive_weak_trend_mean"])
    momentum = float(lookup.loc["average_astro_momentum", "defensive_weak_trend_mean"])
    momentum_slope = float(lookup.loc["average_momentum_slope_7d", "defensive_weak_trend_mean"])
    bullish = float(lookup.loc["average_bullish_score", "defensive_weak_trend_mean"])
    bearish = float(lookup.loc["average_bearish_score", "defensive_weak_trend_mean"])
    reversal = float(lookup.loc["average_reversal_score", "defensive_weak_trend_mean"])
    compression = float(lookup.loc["average_compression_score", "defensive_weak_trend_mean"])
    volatility = float(lookup.loc["average_volatility_score", "defensive_weak_trend_mean"])

    if prior_30 < -0.05 and drawdown < -0.10 and momentum <= 0.5 and bullish < bearish:
        if compression > 1.0 or reversal > 1.0 or volatility > 0.8:
            return "Capitulation Recovery"
        return "Deep Value Recovery"
    if prior_30 < 0 and momentum_slope >= 0 and bearish > bullish:
        return "Bearish Exhaustion Reversal"
    if compression > 1.0 and volatility < 0.8:
        return "Post-Panic Accumulation"
    return "Not truly Defensive"


def build_window_sequence(historical: pd.DataFrame) -> pd.DataFrame:
    ordered = historical.sort_values("date").reset_index(drop=True).copy()
    ordered["taxonomy_shift"] = ordered["taxonomy_v3"] != ordered["taxonomy_v3"].shift(1)
    ordered["window_id"] = ordered["taxonomy_shift"].cumsum()
    grouped = (
        ordered.groupby("window_id", as_index=False)
        .agg(
            start_date=("date", "first"),
            end_date=("date", "last"),
            taxonomy_v3=("taxonomy_v3", "first"),
            future_return_7d=("future_return_7d", "first"),
            future_return_14d=("future_return_14d", "first"),
            future_return_30d=("future_return_30d", "first"),
        )
        .sort_values("start_date")
        .reset_index(drop=True)
    )
    return grouped


def build_transition_matrix(historical: pd.DataFrame) -> pd.DataFrame:
    windows = build_window_sequence(historical)
    rows: List[Dict[str, object]] = []
    for idx in range(1, len(windows)):
        prev_row = windows.iloc[idx - 1]
        row = windows.iloc[idx]
        rows.append(
            {
                "from_taxonomy": prev_row["taxonomy_v3"],
                "to_taxonomy": row["taxonomy_v3"],
                "transition_date": row["start_date"],
                "future_return_7d": row["future_return_7d"],
                "future_return_14d": row["future_return_14d"],
                "future_return_30d": row["future_return_30d"],
                "win_30d": float(row["future_return_30d"] > 0),
                "focus_transition": (prev_row["taxonomy_v3"], row["taxonomy_v3"]) in FOCUS_TRANSITIONS,
            }
        )
    transitions = pd.DataFrame(rows)
    if transitions.empty:
        return transitions
    summary = (
        transitions.groupby(["from_taxonomy", "to_taxonomy", "focus_transition"], as_index=False)
        .agg(
            transition_count=("transition_date", "size"),
            average_return_7d=("future_return_7d", "mean"),
            average_return_14d=("future_return_14d", "mean"),
            average_return_30d=("future_return_30d", "mean"),
            win_rate=("win_30d", "mean"),
            sample_count=("transition_date", "size"),
        )
        .sort_values(["focus_transition", "transition_count", "average_return_30d"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return summary


def build_rename_recommendations(
    regime_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    defensive_archetype: str,
) -> pd.DataFrame:
    exposure_lookup = exposure_df.set_index("taxonomy_v3")
    rows = []
    for _, row in regime_df.iterrows():
        taxonomy = row["taxonomy_v3"]
        recommended_label = taxonomy
        issue = ""
        action = "keep"
        evidence = ""

        if taxonomy == "Defensive / Weak Trend":
            issue = "Label semantics conflict with strong positive forward returns."
            recommended_label = defensive_archetype
            action = "rename"
            evidence = (
                f"30D/60D/90D returns are {row['forward_return_30d']:.2%}/{row['forward_return_60d']:.2%}/{row['forward_return_90d']:.2%} "
                f"despite low momentum {row['average_astro_momentum']:.2f} and low ML probability {row['average_ml_probability']:.2%}."
            )
        elif taxonomy == "Tactical Neutral":
            issue = "Large share plus mild positive drift suggest catch-all behavior."
            recommended_label = "Transition / Low Conviction"
            action = "rename or split"
            evidence = (
                f"Observation share is {row['sample_share']:.2%} with 30D return {row['forward_return_30d']:.2%} "
                f"and only middling 30D win rate {row['win_rate_30d']:.2%}."
            )
        elif taxonomy == "High Momentum Expansion":
            issue = "State looks valid, but stability is fragile."
            recommended_label = "High Conviction Expansion"
            action = "soft rename"
            evidence = (
                f"30D edge is {row['forward_return_30d']:.2%} with win rate {row['win_rate_30d']:.2%}, "
                f"but active years are only {int(row['active_years'])}."
            )
        elif taxonomy == "High Volatility Risk":
            issue = "Semantics are directionally correct."
            recommended_label = "Volatility Caution"
            action = "optional soft rename"
            evidence = (
                f"30D return is only {row['forward_return_30d']:.2%} and 30D return/volatility is {row['return_volatility_ratio_30d']:.4f}."
            )
        else:
            issue = "State is broadly consistent and relatively stable."
            recommended_label = "Constructive Drift"
            action = "keep"
            evidence = (
                f"Large sample share {row['sample_share']:.2%} and positive multi-horizon returns support the current label."
            )

        rows.append(
            {
                "taxonomy_v3": taxonomy,
                "current_label": taxonomy,
                "recommended_action": action,
                "suggested_label": recommended_label,
                "primary_issue": issue,
                "allocation_ready": bool(exposure_lookup.loc[taxonomy, "stability_assessment"] == "Relatively stable" and taxonomy not in {"Defensive / Weak Trend", "Tactical Neutral"}),
                "evidence_summary": evidence,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    regime_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    rename_df: pd.DataFrame,
    defensive_deep_dive: pd.DataFrame,
    dashboard_summary: Dict[str, object],
    intelligence_v3: pd.DataFrame,
    future_timeline: pd.DataFrame,
) -> None:
    current_30d = dashboard_summary.get("30D Outlook", {})
    current_90d = dashboard_summary.get("90D Outlook", {})
    current_365d = dashboard_summary.get("365D Outlook", {})
    defensive_label = rename_df.loc[rename_df["taxonomy_v3"] == "Defensive / Weak Trend", "suggested_label"].iloc[0]

    tactical_share = regime_df.loc[regime_df["taxonomy_v3"] == "Tactical Neutral", "sample_share"].iloc[0]
    hme_row = regime_df.loc[regime_df["taxonomy_v3"] == "High Momentum Expansion"].iloc[0]
    constructive_row = regime_df.loc[regime_df["taxonomy_v3"] == "Constructive Drift"].iloc[0]
    hvr_row = regime_df.loc[regime_df["taxonomy_v3"] == "High Volatility Risk"].iloc[0]

    stable_for_allocation = regime_df[
        regime_df["stability_assessment"] == "Relatively stable"
    ]["taxonomy_v3"].tolist()

    lines = [
        "# Taxonomy Regime Audit Engine v1",
        "",
        "## Objective",
        "Diagnose why Forecast Taxonomy v3 still contains semantic conflicts before turning taxonomy states into real portfolio allocation decisions.",
        "",
        "## Current Dashboard Context",
        f"- 30D dominant taxonomy: `{current_30d.get('dominant_taxonomy', 'N/A')}`",
        f"- 90D dominant taxonomy: `{current_90d.get('dominant_taxonomy', 'N/A')}`",
        f"- 365D dominant taxonomy: `{current_365d.get('dominant_taxonomy', 'N/A')}`",
        "",
        "## Validation Answers",
        f"A. Why does Defensive / Weak Trend produce strong returns? It behaves like `{defensive_label}`: prior returns and drawdown are weak, current ML probability is low, and the state appears after stress rather than during durable downside continuation.",
        "B. Is Defensive / Weak Trend mislabeled? Yes. The forward-return profile contradicts the label.",
        f"C. Is Tactical Neutral a real regime or a catch-all/default state? It looks partly catch-all. It absorbs `{tactical_share:.2%}` of observations and sits between constructive and expansion states.",
        f"D. Does High Momentum Expansion remain valid? Yes, but it is more fragile than Constructive Drift and should not be treated as the sole foundation for allocation.",
        f"E. Which taxonomy labels should be renamed? Defensive / Weak Trend -> `{defensive_label}`; Tactical Neutral -> `Transition / Low Conviction`; optional softer renames for High Momentum Expansion and High Volatility Risk.",
        f"F. Which taxonomy states are stable enough for allocation? `{', '.join(stable_for_allocation)}` are statistically more stable, but semantics are still unresolved for Defensive / Weak Trend and Tactical Neutral.",
        "G. Should we proceed to Portfolio Allocation Engine after this? No. Revise taxonomy semantics first.",
        "",
        "## Key Findings",
        f"- Constructive Drift: positive and comparatively stable with 30D return `{constructive_row['forward_return_30d']:.2%}` across a large sample.",
        f"- High Momentum Expansion: still investable on raw edge (`{hme_row['forward_return_30d']:.2%}` at 30D), but concentrated in fewer years.",
        f"- Tactical Neutral: positive drift is real, so the label is too passive for a supposed neutral/default bucket.",
        f"- High Volatility Risk: weakest directional edge and the clearest defensive/caution state.",
        "",
        "## Regime Audit",
        dataframe_to_markdown(regime_df),
        "",
        "## Defensive / Weak Trend Deep Dive",
        dataframe_to_markdown(defensive_deep_dive),
        "",
        "## Transition Matrix",
        dataframe_to_markdown(transition_df.head(20)),
        "",
        "## Rename Recommendations",
        dataframe_to_markdown(rename_df),
        "",
        "## Current Future Window Context",
        dataframe_to_markdown(
            intelligence_v3[
                [
                    "start_date",
                    "end_date",
                    "taxonomy_v3",
                    "v3_posture",
                    "average_confidence",
                    "average_ml_probability",
                ]
            ].head(12)
        ),
        "",
        "## Current Future Timeline Snapshot",
        dataframe_to_markdown(
            future_timeline[
                ["date", "astro_score", "ml_probability", "signal", "confidence_score", "risk_level"]
            ].head(12)
        ),
    ]
    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    (
        _dataset,
        intelligence_v3,
        future_timeline,
        attribution,
        feature_importance,
        validation,
        yearly,
        exposure,
        _report_text,
    ) = load_support_frames()

    historical = enrich_historical_daily()
    regime_df = build_regime_audit(historical, validation, yearly, attribution, feature_importance)
    defensive_deep_dive = build_defensive_deep_dive(historical)
    defensive_archetype = classify_defensive_archetype(regime_df, defensive_deep_dive)
    transition_df = build_transition_matrix(historical)
    rename_df = build_rename_recommendations(regime_df, exposure, defensive_archetype)
    dashboard_summary = load_dashboard_summary()

    regime_df.to_csv(OUTPUT_AUDIT_PATH, index=False)
    transition_df.to_csv(OUTPUT_TRANSITION_PATH, index=False)
    rename_df.to_csv(OUTPUT_RENAME_PATH, index=False)
    defensive_deep_dive.to_csv(OUTPUT_DEFENSIVE_PATH, index=False)
    write_report(regime_df, transition_df, rename_df, defensive_deep_dive, dashboard_summary, intelligence_v3, future_timeline)

    print(f"Wrote {OUTPUT_AUDIT_PATH}")
    print(f"Wrote {OUTPUT_TRANSITION_PATH}")
    print(f"Wrote {OUTPUT_RENAME_PATH}")
    print(f"Wrote {OUTPUT_DEFENSIVE_PATH}")
    print(f"Wrote {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
