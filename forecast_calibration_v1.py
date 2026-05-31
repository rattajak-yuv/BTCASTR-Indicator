from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from forecast_intelligence_v1 import CLASS_PRIORITY, classify_windows, dataframe_to_markdown
from forecast_system_v1 import (
    build_forecast_windows,
    build_full_feature_frame,
    detect_turning_points,
    load_selected_features,
    risk_level_from_features,
    run_historical_walk_forward_predictions,
)

FORECAST_INTELLIGENCE_PATH = Path("data/forecast_intelligence.csv")
FUTURE_TIMELINE_PATH = Path("data/future_forecast_timeline.csv")
OUTPUT_CSV_PATH = Path("data/forecast_calibration.csv")
OUTPUT_REPORT_PATH = Path("data/forecast_calibration_report.md")

CALIBRATION_HORIZONS = [7, 14, 30]
DEFAULT_CLASS_ORDER = [
    "Strong Bull",
    "Bull Expansion",
    "Accumulation",
    "Neutral",
    "Transition",
    "High Risk",
    "Bearish",
]
PERMUTATION_COUNT = 2000
MIN_PAIRWISE_OBS = 25
RNG_SEED = 42
MIN_WINDOWS_FOR_MERGE = 5


@dataclass
class PairwiseResult:
    class_a: str
    class_b: str
    horizon_days: int
    n_a: int
    n_b: int
    mean_diff: float
    effect_size: float
    p_value: float
    is_distinct: bool


def format_markdown_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def union_find_groups(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> Dict[str, List[str]]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(left, right)

    groups: Dict[str, List[str]] = {}
    for node in nodes:
        groups.setdefault(find(node), []).append(node)
    return groups


def sorted_class_labels(current_labels: Iterable[str]) -> List[str]:
    current = list(dict.fromkeys(str(label) for label in current_labels if pd.notna(label)))
    ordered = [label for label in DEFAULT_CLASS_ORDER if label in current]
    ordered.extend(label for label in current if label not in ordered)
    return ordered


def build_historical_window_classification() -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_df = build_full_feature_frame()
    feature_cols = load_selected_features(full_df)

    historical_df = full_df[full_df["price"].notna()].copy()
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

    historical_windows = build_forecast_windows(historical_predictions)
    historical_windows["start_date"] = pd.to_datetime(historical_windows["start_date"])
    historical_windows["end_date"] = pd.to_datetime(historical_windows["end_date"])

    classified_windows = classify_windows(historical_windows, historical_predictions, turning_points)
    classified_windows["start_date"] = pd.to_datetime(classified_windows["start_date"])
    classified_windows["end_date"] = pd.to_datetime(classified_windows["end_date"])
    classified_windows["window_id"] = np.arange(len(classified_windows))

    future_return_cols = [f"future_return_{horizon}d" for horizon in CALIBRATION_HORIZONS]
    historical_returns = full_df[["date"] + future_return_cols].copy()
    classified_daily = historical_predictions.merge(historical_returns, on="date", how="left", suffixes=("", "_full"))
    classified_daily["window_id"] = -1
    classified_daily["window_class"] = ""

    for _, window in classified_windows.iterrows():
        mask = (
            (classified_daily["date"] >= window["start_date"])
            & (classified_daily["date"] <= window["end_date"])
        )
        classified_daily.loc[mask, "window_id"] = int(window["window_id"])
        classified_daily.loc[mask, "window_class"] = str(window["window_class"])

    classified_daily = classified_daily[classified_daily["window_class"] != ""].copy()
    classified_daily["window_class"] = classified_daily["window_class"].astype(str)
    return classified_daily, classified_windows


def summarize_class_behavior(classified_daily: pd.DataFrame, class_labels: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label in class_labels:
        class_slice = classified_daily[classified_daily["window_class"] == label].copy()
        window_count = int(class_slice["window_id"].nunique()) if not class_slice.empty else 0
        for horizon in CALIBRATION_HORIZONS:
            return_col = f"future_return_{horizon}d"
            returns = class_slice[return_col].replace([np.inf, -np.inf], np.nan).dropna()
            if returns.empty:
                rows.append(
                    {
                        "window_class": label,
                        "horizon_days": horizon,
                        "sample_count": 0,
                        "window_count": window_count,
                        "average_forward_return": np.nan,
                        "median_forward_return": np.nan,
                        "volatility": np.nan,
                        "win_rate": np.nan,
                        "max_gain": np.nan,
                        "max_loss": np.nan,
                    }
                )
                continue

            rows.append(
                {
                    "window_class": label,
                    "horizon_days": horizon,
                    "sample_count": int(len(returns)),
                    "window_count": window_count,
                    "average_forward_return": float(returns.mean()),
                    "median_forward_return": float(returns.median()),
                    "volatility": float(returns.std(ddof=0)),
                    "win_rate": float((returns > 0).mean()),
                    "max_gain": float(returns.max()),
                    "max_loss": float(returns.min()),
                }
            )
    summary = pd.DataFrame(rows)
    summary["average_return_rank"] = (
        summary.groupby("horizon_days")["average_forward_return"]
        .rank(ascending=False, method="dense", na_option="bottom")
    )
    return summary


def permutation_mean_test(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    seed: int,
    permutations: int = PERMUTATION_COUNT,
) -> Tuple[float, float, float]:
    clean_a = pd.Series(series_a).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    clean_b = pd.Series(series_b).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

    if len(clean_a) < MIN_PAIRWISE_OBS or len(clean_b) < MIN_PAIRWISE_OBS:
        return np.nan, np.nan, np.nan

    observed = float(clean_a.mean() - clean_b.mean())
    pooled_var = (
        ((len(clean_a) - 1) * clean_a.var(ddof=1)) + ((len(clean_b) - 1) * clean_b.var(ddof=1))
    ) / max(len(clean_a) + len(clean_b) - 2, 1)
    pooled_std = float(np.sqrt(max(pooled_var, 0.0)))
    effect_size = observed / pooled_std if pooled_std > 1e-12 else np.nan

    combined = np.concatenate([clean_a, clean_b])
    split = len(clean_a)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(combined)
        diff = permuted[:split].mean() - permuted[split:].mean()
        if abs(diff) >= abs(observed):
            exceedances += 1
    p_value = (exceedances + 1) / (permutations + 1)
    return observed, effect_size, float(p_value)


def evaluate_pairwise_distinctness(
    classified_daily: pd.DataFrame,
    class_labels: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for horizon in CALIBRATION_HORIZONS:
        return_col = f"future_return_{horizon}d"
        for left_index, class_a in enumerate(class_labels):
            for class_b in class_labels[left_index + 1 :]:
                returns_a = classified_daily.loc[classified_daily["window_class"] == class_a, return_col]
                returns_b = classified_daily.loc[classified_daily["window_class"] == class_b, return_col]
                mean_diff, effect_size, p_value = permutation_mean_test(
                    returns_a,
                    returns_b,
                    seed=RNG_SEED + horizon + left_index,
                )
                is_distinct = bool(
                    pd.notna(p_value)
                    and p_value < 0.05
                    and pd.notna(effect_size)
                    and abs(effect_size) >= 0.20
                )
                rows.append(
                    {
                        "class_a": class_a,
                        "class_b": class_b,
                        "horizon_days": horizon,
                        "n_a": int(returns_a.notna().sum()),
                        "n_b": int(returns_b.notna().sum()),
                        "mean_diff": mean_diff,
                        "effect_size": effect_size,
                        "p_value": p_value,
                        "is_distinct": is_distinct,
                    }
                )
    return pd.DataFrame(rows)


def find_similarity_edges(summary_df: pd.DataFrame, pairwise_df: pd.DataFrame, class_labels: List[str]) -> List[Tuple[str, str]]:
    metrics_for_distance = [
        "average_forward_return",
        "median_forward_return",
        "volatility",
        "win_rate",
    ]
    metric_frame = summary_df[["window_class", "horizon_days"] + metrics_for_distance].copy()
    for metric in metrics_for_distance:
        metric_series = metric_frame[metric].replace([np.inf, -np.inf], np.nan)
        std = metric_series.std(ddof=0)
        if pd.isna(std) or std <= 1e-12:
            metric_frame[f"{metric}_z"] = 0.0
        else:
            metric_frame[f"{metric}_z"] = (metric_series - metric_series.mean()) / std

    similarity_edges: List[Tuple[str, str]] = []
    window_counts = (
        summary_df.groupby("window_class")["window_count"].max().to_dict()
    )
    for left_index, class_a in enumerate(class_labels):
        for class_b in class_labels[left_index + 1 :]:
            if window_counts.get(class_a, 0) < MIN_WINDOWS_FOR_MERGE:
                continue
            if window_counts.get(class_b, 0) < MIN_WINDOWS_FOR_MERGE:
                continue
            pair_slice = pairwise_df[
                (pairwise_df["class_a"] == class_a) & (pairwise_df["class_b"] == class_b)
            ]
            comparable = pair_slice["p_value"].notna().sum()
            if comparable == 0:
                continue
            if bool(pair_slice["is_distinct"].fillna(False).any()):
                continue

            merged_metrics = metric_frame[metric_frame["window_class"].isin([class_a, class_b])]
            pivot = merged_metrics.pivot(index="horizon_days", columns="window_class")
            distances = []
            for horizon in CALIBRATION_HORIZONS:
                if horizon not in pivot.index:
                    continue
                for metric in metrics_for_distance:
                    column_name = f"{metric}_z"
                    try:
                        value_a = pivot.loc[horizon, (column_name, class_a)]
                        value_b = pivot.loc[horizon, (column_name, class_b)]
                    except KeyError:
                        continue
                    if pd.notna(value_a) and pd.notna(value_b):
                        distances.append(abs(float(value_a) - float(value_b)))
            avg_distance = float(np.mean(distances)) if distances else np.nan
            if pd.notna(avg_distance) and avg_distance <= 0.75:
                similarity_edges.append((class_a, class_b))
    return similarity_edges


def cluster_name(classes: List[str]) -> str:
    ordered = sorted(classes, key=lambda label: CLASS_PRIORITY.get(label, 0), reverse=True)
    class_set = set(ordered)
    if class_set.issubset({"Strong Bull", "Bull Expansion", "Accumulation"}):
        if len(class_set) == 1:
            return ordered[0]
        return "Constructive Bull"
    if class_set.issubset({"Neutral", "Transition"}):
        if len(class_set) == 1:
            return ordered[0]
        return "Neutral / Transition"
    if class_set.issubset({"High Risk", "Bearish"}):
        if len(class_set) == 1:
            return ordered[0]
        return "Risk-Off"
    if len(class_set) == 1:
        return ordered[0]
    return " / ".join(ordered)


def build_taxonomy_mapping(class_labels: List[str], similarity_edges: List[Tuple[str, str]]) -> pd.DataFrame:
    groups = union_find_groups(class_labels, similarity_edges)
    rows: List[Dict[str, object]] = []
    for _, members in groups.items():
        ordered_members = sorted(members, key=lambda label: CLASS_PRIORITY.get(label, 0), reverse=True)
        recommended = cluster_name(ordered_members)
        for member in ordered_members:
            rows.append(
                {
                    "window_class": member,
                    "recommended_taxonomy": recommended,
                    "merged_with": ", ".join(label for label in ordered_members if label != member),
                    "cluster_size": len(ordered_members),
                }
            )
    return pd.DataFrame(rows)


def attach_pairwise_context(summary_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        label = row["window_class"]
        horizon = row["horizon_days"]
        horizon_pairs = pairwise_df[
            (pairwise_df["horizon_days"] == horizon)
            & ((pairwise_df["class_a"] == label) | (pairwise_df["class_b"] == label))
        ].copy()
        distinct_peers = []
        similar_peers = []
        for _, pair in horizon_pairs.iterrows():
            other = pair["class_b"] if pair["class_a"] == label else pair["class_a"]
            if bool(pair["is_distinct"]):
                distinct_peers.append(other)
            elif pd.notna(pair["p_value"]):
                similar_peers.append(other)
        enriched = row.to_dict()
        enriched["statistically_distinct_from"] = ", ".join(sorted(set(distinct_peers)))
        enriched["behaviorally_similar_to"] = ", ".join(sorted(set(similar_peers)))
        rows.append(enriched)
    return pd.DataFrame(rows)


def write_report(
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    observed_windows: pd.DataFrame,
    reference_classes: List[str],
) -> None:
    best_rows = (
        summary_df[summary_df["sample_count"] > 0]
        .sort_values(["horizon_days", "average_forward_return"], ascending=[True, False])
        .groupby("horizon_days", as_index=False)
        .head(1)[
            [
                "window_class",
                "horizon_days",
                "sample_count",
                "average_forward_return",
                "median_forward_return",
                "volatility",
                "win_rate",
            ]
        ]
    )

    pairwise_summary = (
        pairwise_df.groupby(["class_a", "class_b"], as_index=False)
        .agg(
            distinct_horizons=("is_distinct", "sum"),
            tested_horizons=("p_value", lambda series: int(series.notna().sum())),
            average_p_value=("p_value", "mean"),
            average_effect_size=("effect_size", "mean"),
        )
        .sort_values(["distinct_horizons", "average_effect_size"], ascending=[False, False])
    )

    taxonomy_lines = []
    for _, row in taxonomy_df.sort_values(
        ["recommended_taxonomy", "window_class"],
        ascending=[True, False],
    ).iterrows():
        merge_text = f" merged with {row['merged_with']}" if row["merged_with"] else " kept as a standalone class"
        taxonomy_lines.append(f"- `{row['window_class']}` -> `{row['recommended_taxonomy']}`:{merge_text}.")

    distinct_count = int(pairwise_df["is_distinct"].fillna(False).sum())
    comparable_count = int(pairwise_df["p_value"].notna().sum())
    observed_class_count = int(observed_windows["window_class"].nunique())
    underpowered = (
        summary_df.groupby("window_class")["window_count"].max().reset_index()
    )
    underpowered = underpowered[underpowered["window_count"] < MIN_WINDOWS_FOR_MERGE]
    underpowered_text = (
        ", ".join(
            f"{row['window_class']} ({int(row['window_count'])} windows)"
            for _, row in underpowered.iterrows()
        )
        if not underpowered.empty
        else "None"
    )

    lines = [
        "# Forecast Calibration Engine v1",
        "",
        "## Summary",
        f"- Historical window classes observed: `{observed_class_count}`",
        f"- Reference taxonomy classes evaluated: `{', '.join(reference_classes)}`",
        f"- Pairwise class comparisons with enough data: `{comparable_count}`",
        f"- Pairwise comparisons marked statistically distinct: `{distinct_count}`",
        f"- Underpowered classes excluded from merge recommendations: `{underpowered_text}`",
        "",
        "## Best Historical Class By Horizon",
        dataframe_to_markdown(best_rows) if not best_rows.empty else "No valid historical calibration rows were available.",
        "",
        "## Calibration By Class",
        dataframe_to_markdown(
            summary_df[
                [
                    "window_class",
                    "horizon_days",
                    "sample_count",
                    "average_forward_return",
                    "median_forward_return",
                    "volatility",
                    "win_rate",
                    "max_gain",
                    "max_loss",
                    "recommended_taxonomy",
                ]
            ]
        ),
        "",
        "## Pairwise Distinctness",
        dataframe_to_markdown(pairwise_summary.head(20)) if not pairwise_summary.empty else "No pairwise comparisons met the minimum sample threshold.",
        "",
        "## Recommended Calibrated Taxonomy",
        *taxonomy_lines,
    ]

    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    current_intelligence = pd.read_csv(FORECAST_INTELLIGENCE_PATH)
    _ = pd.read_csv(FUTURE_TIMELINE_PATH)

    classified_daily, historical_windows = build_historical_window_classification()
    observed_labels = sorted_class_labels(
        list(current_intelligence["window_class"].dropna().astype(str).unique())
        + list(classified_daily["window_class"].dropna().astype(str).unique())
    )

    summary_df = summarize_class_behavior(classified_daily, observed_labels)
    pairwise_df = evaluate_pairwise_distinctness(classified_daily, observed_labels)
    similarity_edges = find_similarity_edges(summary_df, pairwise_df, observed_labels)
    taxonomy_df = build_taxonomy_mapping(observed_labels, similarity_edges)

    final_df = summary_df.merge(taxonomy_df, on="window_class", how="left")
    final_df = attach_pairwise_context(final_df, pairwise_df)
    final_df = final_df.sort_values(
        ["horizon_days", "average_forward_return", "window_class"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    write_report(final_df, pairwise_df, taxonomy_df, historical_windows, observed_labels)


if __name__ == "__main__":
    main()
