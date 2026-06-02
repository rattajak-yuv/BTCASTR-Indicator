from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from forecast_intelligence_v1 import classify_windows, dataframe_to_markdown
from forecast_system_v1 import (
    build_forecast_windows,
    detect_turning_points,
    load_selected_features,
    risk_level_from_features,
    run_historical_walk_forward_predictions,
)
from forecast_taxonomy_v2 import build_class_evidence, classify_taxonomy_v2
from forecast_taxonomy_v3 import TAXONOMY_V3_MAP


ROOT = Path(".")
ML_DATASET_PATH = ROOT / "data" / "ml_dataset.csv"
SELECTED_FEATURES_PATH = ROOT / "data" / "selected_features.csv"
FORECAST_CALIBRATION_PATH = ROOT / "data" / "forecast_calibration.csv"
FORECAST_INTELLIGENCE_V3_PATH = ROOT / "data" / "forecast_intelligence_v3.csv"
FUTURE_TIMELINE_PATH = ROOT / "data" / "future_forecast_timeline.csv"
DASHBOARD_SUMMARY_PATH = ROOT / "data" / "dashboard_summary.json"
TAXONOMY_ATTRIBUTION_PATH = ROOT / "data" / "taxonomy_attribution.csv"
TAXONOMY_FEATURE_IMPORTANCE_PATH = ROOT / "data" / "taxonomy_feature_importance.csv"

OUTPUT_VALIDATION_PATH = ROOT / "data" / "taxonomy_performance_validation.csv"
OUTPUT_YEARLY_PATH = ROOT / "data" / "taxonomy_performance_by_year.csv"
OUTPUT_EXPOSURE_PATH = ROOT / "data" / "taxonomy_exposure_recommendation.csv"
OUTPUT_REPORT_PATH = ROOT / "data" / "taxonomy_performance_validation_report.md"

HORIZONS = [7, 14, 30, 60, 90]
TARGET_TAXONOMIES = [
    "Constructive Drift",
    "High Momentum Expansion",
    "Tactical Neutral",
    "Defensive / Weak Trend",
    "High Volatility Risk",
]
CATEGORICAL_COLUMNS = {
    "date",
    "astro_regime_v2",
    "signal",
    "regime",
    "market_regime",
    "volatility_state",
    "applied_weight_profile",
}


def read_resilient_csv(path: Path) -> pd.DataFrame:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    cleaned_lines = [
        line for line in raw_lines
        if not line.startswith("<<<<<<<")
        and not line.startswith("=======")
        and not line.startswith(">>>>>>>")
    ]

    header_index = next(
        (idx for idx, line in enumerate(cleaned_lines) if line.startswith("date,")),
        None,
    )
    if header_index is None:
        raise ValueError(f"Could not locate a valid CSV header in {path}")

    header_line = cleaned_lines[header_index]
    body_lines = [line for line in cleaned_lines[header_index + 1 :] if line != header_line]
    csv_text = "\n".join([header_line] + body_lines) + "\n"

    df = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip", engine="python")
    if "date" in df.columns:
        date_text = df["date"].astype(str).str.strip()
        bad_mask = (
            date_text.eq("date")
            | date_text.str.startswith("<<<<<<<")
            | date_text.str.startswith("=======")
            | date_text.str.startswith(">>>>>>>")
        )
        df = df.loc[~bad_mask].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    for col in df.columns:
        if col in CATEGORICAL_COLUMNS:
            continue
        if col == "date":
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any() or pd.api.types.is_numeric_dtype(df[col]):
            df[col] = converted
    return df


def load_dashboard_summary() -> Dict[str, object]:
    with open(DASHBOARD_SUMMARY_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_record(value) -> Dict[str, object]:
    return value if isinstance(value, dict) else {}


def max_drawdown_from_returns(returns: pd.Series) -> float:
    clean = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    equity = (1.0 + clean).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    return float(drawdown.min())


def compounded_return(returns: pd.Series) -> float:
    clean = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() - 1.0)


def build_historical_taxonomy_daily() -> pd.DataFrame:
    dataset = read_resilient_csv(ML_DATASET_PATH)
    feature_cols = load_selected_features(dataset)

    historical_df = dataset[dataset["price"].notna()].copy()
    historical_df = historical_df.replace([np.inf, -np.inf], np.nan)
    historical_df = historical_df.sort_values("date").reset_index(drop=True)

    historical_predictions = run_historical_walk_forward_predictions(historical_df, feature_cols)
    historical_predictions["risk_level"] = risk_level_from_features(
        historical_predictions,
        historical_volatility_reference=historical_df["astro_volatility_score"].dropna(),
        confidence_score=historical_predictions["confidence_score"],
    )
    historical_predictions["within_30d"] = False
    historical_predictions["within_90d"] = False
    historical_predictions["within_180d"] = False
    historical_predictions["within_365d"] = False

    turning_points = detect_turning_points(
        historical_predictions,
        historical_momentum=historical_df["astro_momentum_v2_smooth"].dropna(),
    )
    if not turning_points.empty:
        turning_points["turning_point_date"] = pd.to_datetime(turning_points["turning_point_date"])

    windows = build_forecast_windows(historical_predictions)
    windows["start_date"] = pd.to_datetime(windows["start_date"])
    windows["end_date"] = pd.to_datetime(windows["end_date"])
    classified_windows = classify_windows(windows, historical_predictions, turning_points)
    classified_windows["start_date"] = pd.to_datetime(classified_windows["start_date"])
    classified_windows["end_date"] = pd.to_datetime(classified_windows["end_date"])
    classified_windows["window_id"] = np.arange(len(classified_windows))

    calibration_df = pd.read_csv(FORECAST_CALIBRATION_PATH)
    class_evidence = build_class_evidence(calibration_df)
    class_evidence["taxonomy_v2"] = class_evidence.apply(classify_taxonomy_v2, axis=1)
    class_map = dict(zip(class_evidence["window_class"], class_evidence["taxonomy_v2"]))

    classified_windows["taxonomy_v2"] = classified_windows["window_class"].map(class_map)
    classified_windows["taxonomy_v3"] = classified_windows["taxonomy_v2"].map(TAXONOMY_V3_MAP)

    merge_cols = [
        "date",
        "price",
        "astro_momentum_v2_smooth",
        "astro_volatility_score",
        "astro_compression_score",
        "future_return_7d",
        "future_return_14d",
        "future_return_30d",
        "future_return_60d",
        "future_return_90d",
    ]
    classified_daily = historical_predictions.merge(
        historical_df[merge_cols],
        on="date",
        how="left",
        suffixes=("", "_hist"),
    )
    classified_daily["window_id"] = -1
    classified_daily["window_class"] = ""
    classified_daily["taxonomy_v2"] = ""
    classified_daily["taxonomy_v3"] = ""

    for _, window in classified_windows.iterrows():
        mask = (
            (classified_daily["date"] >= window["start_date"])
            & (classified_daily["date"] <= window["end_date"])
        )
        classified_daily.loc[mask, "window_id"] = int(window["window_id"])
        classified_daily.loc[mask, "window_class"] = str(window["window_class"])
        classified_daily.loc[mask, "taxonomy_v2"] = str(window["taxonomy_v2"])
        classified_daily.loc[mask, "taxonomy_v3"] = str(window["taxonomy_v3"])

    classified_daily = classified_daily[classified_daily["taxonomy_v3"].isin(TARGET_TAXONOMIES)].copy()
    classified_daily["year"] = classified_daily["date"].dt.year
    return classified_daily


def summarize_taxonomy_performance(classified_daily: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total_rows = max(len(classified_daily), 1)

    for taxonomy in TARGET_TAXONOMIES:
        taxonomy_slice = classified_daily[classified_daily["taxonomy_v3"] == taxonomy].copy()
        sample_share = len(taxonomy_slice) / total_rows

        for horizon in HORIZONS:
            return_col = f"future_return_{horizon}d"
            returns = taxonomy_slice[return_col].replace([np.inf, -np.inf], np.nan).dropna()
            if returns.empty:
                rows.append(
                    {
                        "taxonomy_v3": taxonomy,
                        "horizon_days": horizon,
                        "sample_count": 0,
                        "sample_share": sample_share,
                        "average_return": np.nan,
                        "median_return": np.nan,
                        "win_rate": np.nan,
                        "volatility": np.nan,
                        "max_gain": np.nan,
                        "max_loss": np.nan,
                        "compounded_return": np.nan,
                        "max_drawdown": np.nan,
                        "return_volatility_ratio": np.nan,
                        "return_drawdown_ratio": np.nan,
                    }
                )
                continue

            mean_return = float(returns.mean())
            volatility = float(returns.std(ddof=0))
            compounded = compounded_return(returns)
            max_dd = max_drawdown_from_returns(returns)
            max_loss = float(returns.min())
            rows.append(
                {
                    "taxonomy_v3": taxonomy,
                    "horizon_days": horizon,
                    "sample_count": int(len(returns)),
                    "sample_share": sample_share,
                    "average_return": mean_return,
                    "median_return": float(returns.median()),
                    "win_rate": float((returns > 0).mean()),
                    "volatility": volatility,
                    "max_gain": float(returns.max()),
                    "max_loss": max_loss,
                    "compounded_return": compounded,
                    "max_drawdown": max_dd,
                    "return_volatility_ratio": mean_return / volatility if volatility > 1e-12 else np.nan,
                    "return_drawdown_ratio": (
                        mean_return / abs(max_loss)
                        if max_loss < -1e-12
                        else np.nan
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    for horizon in [30, 60, 90]:
        mask = summary["horizon_days"] == horizon
        if mask.any():
            summary.loc[mask, f"rank_avg_return_{horizon}d"] = summary.loc[mask, "average_return"].rank(
                ascending=False,
                method="dense",
                na_option="bottom",
            )
    summary["rank_win_rate"] = summary.groupby("horizon_days")["win_rate"].rank(
        ascending=False, method="dense", na_option="bottom"
    )
    summary["rank_return_volatility_ratio"] = summary.groupby("horizon_days")[
        "return_volatility_ratio"
    ].rank(ascending=False, method="dense", na_option="bottom")
    return summary


def summarize_yearly_stability(classified_daily: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for taxonomy in TARGET_TAXONOMIES:
        taxonomy_slice = classified_daily[classified_daily["taxonomy_v3"] == taxonomy].copy()
        for year, group in taxonomy_slice.groupby("year"):
            returns = group["future_return_30d"].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "taxonomy_v3": taxonomy,
                    "year": int(year),
                    "sample_count": int(len(returns)),
                    "average_forward_return_30d": float(returns.mean()) if not returns.empty else np.nan,
                    "win_rate_30d": float((returns > 0).mean()) if not returns.empty else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["taxonomy_v3", "year"]).reset_index(drop=True)


def taxonomy_horizon_lookup(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(index="taxonomy_v3", columns="horizon_days")
    pivot.columns = [f"{metric}_{horizon}d" for metric, horizon in pivot.columns]
    return pivot.reset_index()


def classify_stability(yearly_df: pd.DataFrame, taxonomy: str) -> str:
    subset = yearly_df[(yearly_df["taxonomy_v3"] == taxonomy) & (yearly_df["sample_count"] >= 5)].copy()
    if subset.empty:
        return "Insufficient history"
    positive_share = float((subset["average_forward_return_30d"] > 0).mean())
    active_years = int(len(subset))
    if active_years >= 4 and positive_share >= 0.60:
        return "Relatively stable"
    if active_years >= 3 and positive_share >= 0.45:
        return "Cycle-dependent"
    return "Fragile"


def build_exposure_recommendations(summary: pd.DataFrame, yearly_df: pd.DataFrame) -> pd.DataFrame:
    horizon_view = taxonomy_horizon_lookup(summary)
    rows: List[Dict[str, object]] = []

    for _, row in horizon_view.iterrows():
        taxonomy = row["taxonomy_v3"]
        avg30 = float(row.get("average_return_30d", np.nan))
        avg60 = float(row.get("average_return_60d", np.nan))
        avg90 = float(row.get("average_return_90d", np.nan))
        win30 = float(row.get("win_rate_30d", np.nan))
        rv30 = float(row.get("return_volatility_ratio_30d", np.nan))
        vol30 = float(row.get("volatility_30d", np.nan))
        sample_share = float(row.get("sample_share_30d", np.nan))
        stability_label = classify_stability(yearly_df, taxonomy)

        if taxonomy == "High Volatility Risk" and (avg30 <= 0.03 or rv30 < 0.10):
            exposure = "0-25% BTC"
            rationale = "Volatility dominates edge; this state should stay capital-preservation oriented."
        elif taxonomy == "Defensive / Weak Trend" and avg30 > 0.05 and avg60 > 0.10:
            exposure = "100% BTC"
            rationale = (
                "Empirical returns are strong despite the defensive label. Treat this as a taxonomy-semantics conflict, "
                "not as production-ready risk-on guidance."
            )
        elif avg60 < 0 and avg90 < 0:
            exposure = "0% BTC"
            rationale = "Medium-term forward returns remain negative, which argues for a defensive posture."
        elif avg30 > 0.08 and avg60 > 0.12 and avg90 > 0.18 and win30 >= 0.60:
            exposure = "100% BTC"
            rationale = "The state shows strong multi-horizon upside persistence and qualifies as full risk-on."
        elif avg30 > 0.03 and avg60 > 0.05 and win30 >= 0.55:
            exposure = "50-75% BTC"
            rationale = "The state is positive but not explosive; measured risk-on exposure is more appropriate."
        elif avg30 > 0 and avg60 > 0 and avg90 > 0:
            exposure = "0-50% BTC"
            rationale = "The edge is mild or tactical, so sizing should stay selective rather than fully committed."
        else:
            exposure = "0% BTC"
            rationale = "The state does not justify long exposure without additional confirmation."

        rows.append(
            {
                "taxonomy_v3": taxonomy,
                "suggested_exposure": exposure,
                "research_only": True,
                "average_return_30d": avg30,
                "average_return_60d": avg60,
                "average_return_90d": avg90,
                "win_rate_30d": win30,
                "return_volatility_ratio_30d": rv30,
                "volatility_30d": vol30,
                "sample_share_30d": sample_share,
                "stability_assessment": stability_label,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def top_taxonomy(summary: pd.DataFrame, horizon: int, metric: str) -> pd.Series:
    subset = summary[summary["horizon_days"] == horizon].copy()
    subset = subset.dropna(subset=[metric])
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.sort_values(metric, ascending=False).iloc[0]


def bottom_taxonomy(summary: pd.DataFrame, horizon: int, metric: str) -> pd.Series:
    subset = summary[summary["horizon_days"] == horizon].copy()
    subset = subset.dropna(subset=[metric])
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.sort_values(metric, ascending=True).iloc[0]


def determine_next_step(exposure_df: pd.DataFrame, yearly_df: pd.DataFrame) -> str:
    best = exposure_df.sort_values("average_return_90d", ascending=False).iloc[0]
    stable_positive = exposure_df[
        (exposure_df["stability_assessment"] == "Relatively stable")
        & (exposure_df["average_return_60d"] > 0)
        & (exposure_df["average_return_90d"] > 0)
    ]
    if best["taxonomy_v3"] in {"Defensive / Weak Trend", "High Volatility Risk"}:
        return "revise taxonomy"
    if not stable_positive.empty:
        return "proceed to Portfolio Allocation Engine"
    if best["taxonomy_v3"] == "High Momentum Expansion" and best["stability_assessment"] != "Relatively stable":
        return "add Astro Momentum confirmation"
    if exposure_df["average_return_30d"].max() <= 0:
        return "stop and rethink taxonomy if no edge is found"
    return "revise taxonomy"


def load_attribution_lookup() -> pd.DataFrame:
    return pd.read_csv(TAXONOMY_ATTRIBUTION_PATH)


def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(TAXONOMY_FEATURE_IMPORTANCE_PATH)


def write_report(
    summary: pd.DataFrame,
    yearly_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    attribution_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    dashboard_summary = load_dashboard_summary()
    current_30d = safe_record(dashboard_summary.get("30D Outlook"))
    current_90d = safe_record(dashboard_summary.get("90D Outlook"))
    current_365d = safe_record(dashboard_summary.get("365D Outlook"))

    best_30 = top_taxonomy(summary, 30, "average_return")
    best_60 = top_taxonomy(summary, 60, "average_return")
    best_90 = top_taxonomy(summary, 90, "average_return")
    best_rv = top_taxonomy(summary, 30, "return_volatility_ratio")
    worst_30 = bottom_taxonomy(summary, 30, "average_return")

    tactical_30 = summary[
        (summary["taxonomy_v3"] == "Tactical Neutral") & (summary["horizon_days"] == 30)
    ].iloc[0]
    tactical_share = float(tactical_30["sample_share"])
    tactical_neutral_assessment = (
        "Tactical Neutral behaves as a true neutral/tactical state."
        if abs(float(tactical_30["average_return"])) < 0.02 and 0.45 <= float(tactical_30["win_rate"]) <= 0.58
        else "Tactical Neutral is not fully neutral and may be absorbing mild positive drift."
    )

    hme_30 = summary[
        (summary["taxonomy_v3"] == "High Momentum Expansion") & (summary["horizon_days"] == 30)
    ].iloc[0]
    if float(hme_30["average_return"]) > 0.10 and float(hme_30["win_rate"]) >= 0.60:
        hme_treatment = "full risk-on"
    elif float(hme_30["average_return"]) > 0.03 and float(hme_30["win_rate"]) >= 0.55:
        hme_treatment = "partial risk-on"
    else:
        hme_treatment = "wait for confirmation"

    defensive_trigger = exposure_df[
        exposure_df["suggested_exposure"].isin(["0% BTC", "0-25% BTC"])
        & ~exposure_df["taxonomy_v3"].isin(["Defensive / Weak Trend"])
    ]["taxonomy_v3"].tolist()

    strongest_taxonomy = best_90["taxonomy_v3"] if not best_90.empty else ""
    strongest_attr = attribution_df.loc[attribution_df["taxonomy_v3"] == strongest_taxonomy]
    strongest_attr_row = strongest_attr.iloc[0] if not strongest_attr.empty else pd.Series(dtype=object)
    strongest_features = feature_df[
        feature_df["taxonomy_v3"] == strongest_taxonomy
    ].sort_values("abs_zscore_diff", ascending=False).head(5)

    semantic_conflict = strongest_taxonomy in {"Defensive / Weak Trend", "High Volatility Risk"}
    predictive_power = (
        "Yes, the taxonomy separates return outcomes, but the directional semantics are not trustworthy yet."
        if (not best_90.empty and not worst_30.empty and float(best_90["average_return"]) > float(worst_30["average_return"]))
        else "No clear predictive power was found."
    )
    stable_enough = any(exposure_df["stability_assessment"] == "Relatively stable") and not semantic_conflict
    next_step = determine_next_step(exposure_df, yearly_df)

    lines = [
        "# Taxonomy Performance Validation Engine v1",
        "",
        "## Objective",
        "Validate whether Forecast Taxonomy v3 has enough real forward-return edge to support a later portfolio allocation layer, without changing model logic or taxonomy definitions.",
        "",
        "## Current Dashboard Outlook Context",
        f"- 30D dominant taxonomy: `{current_30d.get('dominant_taxonomy', 'N/A')}`",
        f"- 90D dominant taxonomy: `{current_90d.get('dominant_taxonomy', 'N/A')}`",
        f"- 365D dominant taxonomy: `{current_365d.get('dominant_taxonomy', 'N/A')}`",
        "",
        "## Validation Answers",
        f"A. Predictive power: {predictive_power}",
        f"B. Strongest investable edge: `{strongest_taxonomy}`",
        f"C. Tactical Neutral assessment: {tactical_neutral_assessment} Sample share is `{tactical_share:.2%}` of historical taxonomy observations.",
        f"D. High Momentum Expansion treatment: `{hme_treatment}`",
        f"E. Defensive behavior should be triggered by: `{', '.join(defensive_trigger) if defensive_trigger else 'No taxonomy cleanly maps to defensive behavior without semantic revision.'}`",
        (
            "F. Taxonomy stability: Stable enough to begin allocation research."
            if stable_enough
            else "F. Taxonomy stability: Not stable enough for allocation until the taxonomy semantics are revised."
        ),
        f"G. Recommended next step: `{next_step}`",
        "",
        "## Top Taxonomy Rankings",
        f"- Best 30D average return: `{best_30.get('taxonomy_v3', 'N/A')}` at `{float(best_30.get('average_return', np.nan)):.2%}`",
        f"- Best 60D average return: `{best_60.get('taxonomy_v3', 'N/A')}` at `{float(best_60.get('average_return', np.nan)):.2%}`",
        f"- Best 90D average return: `{best_90.get('taxonomy_v3', 'N/A')}` at `{float(best_90.get('average_return', np.nan)):.2%}`",
        f"- Best 30D return/volatility: `{best_rv.get('taxonomy_v3', 'N/A')}` at `{float(best_rv.get('return_volatility_ratio', np.nan)):.4f}`",
        f"- Weakest 30D edge: `{worst_30.get('taxonomy_v3', 'N/A')}` at `{float(worst_30.get('average_return', np.nan)):.2%}`",
        "",
        "## Strongest Edge Attribution",
        f"- Strongest taxonomy: `{strongest_taxonomy}`",
        f"- Average astro momentum: `{float(strongest_attr_row.get('average_astro_momentum', np.nan)):.2f}`",
        f"- Average ML probability: `{float(strongest_attr_row.get('average_ml_probability', np.nan)):.2%}`",
        f"- Dominant planets: `{strongest_attr_row.get('most_influential_planets', 'N/A')}`",
        f"- Dominant aspects: `{strongest_attr_row.get('most_influential_aspects', 'N/A')}`",
        "",
        "### Strongest Edge Top Features",
        dataframe_to_markdown(
            strongest_features[
                ["feature", "feature_family", "differential", "zscore_diff", "direction"]
            ] if not strongest_features.empty else pd.DataFrame()
        ),
        "",
        "## Exposure Recommendation (Research Only)",
        dataframe_to_markdown(exposure_df),
        "",
        "## Performance Summary",
        dataframe_to_markdown(
            summary[
                summary["horizon_days"].isin([30, 60, 90])
            ][
                [
                    "taxonomy_v3",
                    "horizon_days",
                    "sample_count",
                    "average_return",
                    "win_rate",
                    "volatility",
                    "return_volatility_ratio",
                    "return_drawdown_ratio",
                ]
            ]
        ),
        "",
        "## Yearly Stability (30D Forward Returns)",
        dataframe_to_markdown(yearly_df),
    ]
    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    classified_daily = build_historical_taxonomy_daily()
    summary = summarize_taxonomy_performance(classified_daily)
    yearly_df = summarize_yearly_stability(classified_daily)
    exposure_df = build_exposure_recommendations(summary, yearly_df)
    attribution_df = load_attribution_lookup()
    feature_df = load_feature_importance()

    summary.to_csv(OUTPUT_VALIDATION_PATH, index=False)
    yearly_df.to_csv(OUTPUT_YEARLY_PATH, index=False)
    exposure_df.to_csv(OUTPUT_EXPOSURE_PATH, index=False)
    write_report(summary, yearly_df, exposure_df, attribution_df, feature_df)

    print(f"Wrote {OUTPUT_VALIDATION_PATH}")
    print(f"Wrote {OUTPUT_YEARLY_PATH}")
    print(f"Wrote {OUTPUT_EXPOSURE_PATH}")
    print(f"Wrote {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
