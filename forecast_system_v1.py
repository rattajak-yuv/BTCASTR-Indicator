import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from build_ml_dataset import (
    DAILY_PATH,
    RAW_PATH,
    RAW_SCORE_COLUMNS,
    PLANET_SIGNAL_COLUMNS,
    ASPECT_STRENGTH_COLUMNS,
    NATAL_TARGET_STRENGTH_COLUMNS,
    add_rolling_features,
    build_raw_aspect_features,
)

HISTORICAL_DATA_PATH = "data/ml_dataset.csv"
SELECTED_FEATURES_PATH = "data/selected_features.csv"

LIVE_FORECAST_PATH = "data/live_forecast.csv"
TIMELINE_PATH = "data/future_forecast_timeline.csv"
TURNING_POINTS_PATH = "data/turning_points.csv"
FORECAST_WINDOWS_PATH = "data/forecast_windows.csv"
REPORT_PATH = "data/forecast_system_report.md"

TARGET_HORIZON_DAYS = 7
TRAIN_YEARS = 5
LONG_THRESHOLD = 0.57
SHORT_THRESHOLD = 0.43
FORECAST_DAYS = 365

NON_FEATURE_COLUMNS = {
    "date",
    "astro_regime_v2",
    "signal",
    "regime",
    "market_regime",
    "volatility_state",
    "applied_weight_profile",
    "price",
    "strategy_total_return",
    "buy_hold_total_return",
    "strategy_max_drawdown",
    "buy_hold_max_drawdown",
}

warnings.filterwarnings(
    "ignore",
    message=(
        "`sklearn.utils.parallel.delayed` should be used with "
        "`sklearn.utils.parallel.Parallel`"
    ),
)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def sharpe_like(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std() == 0:
        return np.nan
    return float((clean.mean() / clean.std()) * np.sqrt(365))


def robust_scale(series: pd.Series, fallback: float = 1.0, quantile: float = 0.75) -> float:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().abs()
    if clean.empty:
        return fallback
    scale = clean.quantile(quantile)
    if pd.isna(scale) or scale <= 1e-6:
        return fallback
    return float(scale)


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


def load_selected_features(df: pd.DataFrame) -> List[str]:
    sf = pd.read_csv(SELECTED_FEATURES_PATH)
    if "feature" not in sf.columns:
        raise ValueError("selected_features.csv must contain a 'feature' column")

    selected = sf["feature"].dropna().astype(str).unique().tolist()
    selected = [
        feature for feature in selected
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])
    ]
    if not selected:
        raise ValueError("No selected features found in the forecast feature frame")
    return selected


def build_full_feature_frame() -> pd.DataFrame:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    raw = pd.read_csv(RAW_PATH)
    raw_features = build_raw_aspect_features(raw)

    df = daily.merge(raw_features, on="date", how="left", suffixes=("", "_raw"))

    raw_feature_cols = [
        c for c in df.columns
        if (
            c.startswith("planet_")
            or c.startswith("aspect_count_")
            or c in PLANET_SIGNAL_COLUMNS
            or c in ASPECT_STRENGTH_COLUMNS
            or c in NATAL_TARGET_STRENGTH_COLUMNS
            or c in {
                "house_activation_strength",
                "raw_astro_total_strength",
                "raw_astro_directional_signal",
                "raw_astro_event_count",
            }
        )
    ]
    df[raw_feature_cols] = df[raw_feature_cols].fillna(0)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    base_feature_cols = [
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
    ]
    base_feature_cols = [col for col in base_feature_cols if col in df.columns]

    for col in base_feature_cols:
        df = add_rolling_features(df, col)

    df["btc_return_1d"] = df["price"].pct_change()
    df["btc_return_3d"] = df["price"].pct_change(3)
    df["btc_return_7d"] = df["price"].pct_change(7)
    df["btc_return_14d"] = df["price"].pct_change(14)
    df["btc_return_30d"] = df["price"].pct_change(30)
    df["btc_vol_7d"] = df["btc_return_1d"].rolling(7).std()
    df["btc_vol_14d"] = df["btc_return_1d"].rolling(14).std()
    df["btc_vol_30d"] = df["btc_return_1d"].rolling(30).std()

    for horizon in [3, 7, 14, 30, 60, 90]:
        df[f"future_return_{horizon}d"] = df["price"].shift(-horizon) / df["price"] - 1
        df[f"future_direction_{horizon}d"] = (df[f"future_return_{horizon}d"] > 0).astype(int)

    latest_price_date = df.loc[df["price"].notna(), "date"].max()
    future_mask = df["date"] > latest_price_date

    # Future price-derived features should not leak unknown price paths.
    # Keep them deterministic and neutral: zero return assumptions, latest realized volatility carried forward.
    for col in ["btc_return_1d", "btc_return_3d", "btc_return_7d", "btc_return_14d", "btc_return_30d"]:
        df.loc[future_mask, col] = 0.0

    for col in ["btc_vol_7d", "btc_vol_14d", "btc_vol_30d"]:
        df[col] = df[col].ffill()

    df["has_price"] = df["price"].notna().astype(int)
    return df


def is_valid_feature_column(df: pd.DataFrame, col: str) -> bool:
    if col in NON_FEATURE_COLUMNS:
        return False
    if col.startswith("future_"):
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    if df[col].isna().all():
        return False
    return True


def build_fold_schedule(df: pd.DataFrame, train_years: int) -> List[Tuple[int, int, int]]:
    years = sorted(df["date"].dt.year.dropna().unique().tolist())
    return [
        (years[idx - train_years], years[idx - 1], years[idx])
        for idx in range(train_years, len(years))
    ]


def compute_tree_probability_std(model: RandomForestClassifier, x: pd.DataFrame) -> np.ndarray:
    x_values = x.to_numpy()
    tree_probabilities = np.vstack(
        [tree.predict_proba(x_values)[:, 1] for tree in model.estimators_]
    )
    return tree_probabilities.std(axis=0)


def compute_confidence(prob_up: np.ndarray, tree_std: np.ndarray) -> np.ndarray:
    margin_strength = np.clip(np.abs(prob_up - 0.5) / 0.5, 0.0, 1.0)
    dispersion_penalty = 1.0 - np.clip(tree_std / 0.5, 0.0, 1.0)
    confidence = 0.6 * margin_strength + 0.4 * dispersion_penalty
    return np.clip(confidence, 0.0, 1.0)


def compute_astro_score(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["astro_bullish_score"] - frame["astro_bearish_score"])
        + 0.50 * (frame["astro_trend_start_score"] - frame["astro_trend_end_score"])
        + 0.35 * frame["astro_momentum_v2_smooth"]
        - 0.20 * frame["astro_reversal_score"]
        - 0.10 * frame["astro_compression_score"]
    )


def probability_to_signal(prob_up: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            prob_up >= LONG_THRESHOLD,
            "Bullish",
            np.where(prob_up <= SHORT_THRESHOLD, "Bearish", "Neutral"),
        ),
        index=prob_up.index,
    )


def forecast_strength_label(prob_up: pd.Series, confidence_score: pd.Series) -> pd.Series:
    raw_strength = confidence_score * (np.abs(prob_up - 0.5) / 0.5)
    return pd.Series(
        np.where(
            raw_strength >= 0.75,
            "Very Strong",
            np.where(raw_strength >= 0.55, "Strong", np.where(raw_strength >= 0.35, "Moderate", "Weak")),
        ),
        index=prob_up.index,
    )


def risk_level_from_features(
    frame: pd.DataFrame,
    historical_volatility_reference: pd.Series,
    confidence_score: pd.Series,
) -> pd.Series:
    q33 = historical_volatility_reference.quantile(0.33)
    q67 = historical_volatility_reference.quantile(0.67)

    vol_score = np.where(
        frame["astro_volatility_score"] >= q67,
        0.85,
        np.where(frame["astro_volatility_score"] <= q33, 0.25, 0.55),
    )
    uncertainty_penalty = 1.0 - confidence_score
    bearish_penalty = np.where(frame["signal"] == "Bearish", 0.15, 0.0)
    total_risk = np.clip(vol_score * 0.70 + uncertainty_penalty * 0.30 + bearish_penalty, 0.0, 1.0)

    return pd.Series(
        np.where(total_risk >= 0.70, "High", np.where(total_risk >= 0.40, "Moderate", "Low")),
        index=frame.index,
    )


def recommended_bias_from_row(row: pd.Series) -> str:
    if row["signal"] == "Bullish":
        if row["risk_level"] == "High":
            return "Defensive"
        return "Long Bias"
    if row["signal"] == "Bearish":
        if row["confidence_score"] >= 0.60:
            return "Short Bias"
        return "Defensive"
    if row["risk_level"] == "High":
        return "Defensive"
    return "Neutral"


def market_view_from_row(row: pd.Series) -> str:
    if row["signal"] == "Bullish":
        if row["confidence_score"] >= 0.65:
            return "Constructive upside setup"
        return "Mild upside bias"
    if row["signal"] == "Bearish":
        if row["confidence_score"] >= 0.65:
            return "Elevated downside pressure"
        return "Cautious downside drift"
    if row["risk_level"] == "High":
        return "Choppy transition / elevated risk"
    return "Balanced consolidation"


def run_historical_walk_forward_predictions(
    historical_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    target_col = f"future_direction_{TARGET_HORIZON_DAYS}d"
    return_col = f"future_return_{TARGET_HORIZON_DAYS}d"
    data = historical_df.dropna(subset=feature_cols + [target_col, return_col]).copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.sort_values("date").reset_index(drop=True)

    folds = build_fold_schedule(data, train_years=TRAIN_YEARS)
    rows = []

    for fold_index, (train_start_year, train_end_year, test_year) in enumerate(folds, start=1):
        train_mask = (
            (data["date"].dt.year >= train_start_year)
            & (data["date"].dt.year <= train_end_year)
        )
        test_mask = data["date"].dt.year == test_year

        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()
        if len(train) < 300 or len(test) == 0:
            continue

        x_train = train[feature_cols]
        y_train = train[target_col].astype(int)
        x_test = test[feature_cols]

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42 + TARGET_HORIZON_DAYS,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)

        prob_up = model.predict_proba(x_test)[:, 1]
        tree_std = compute_tree_probability_std(model, x_test)
        confidence_score = compute_confidence(prob_up, tree_std)

        out = test[
            [
                "date",
                "price",
                "astro_bullish_score",
                "astro_bearish_score",
                "astro_reversal_score",
                "astro_volatility_score",
                "astro_compression_score",
                "astro_trend_start_score",
                "astro_trend_end_score",
                "astro_momentum_v2_smooth",
                return_col,
                target_col,
            ]
        ].copy()
        out["ml_probability"] = prob_up
        out["confidence_score"] = confidence_score
        out["signal"] = probability_to_signal(out["ml_probability"])
        out["actual_direction"] = out[target_col].astype(int)
        out["actual_future_return"] = out[return_col].astype(float)
        out["fold_id"] = fold_index
        out["astro_score"] = compute_astro_score(out)
        rows.append(out)

    historical_predictions = pd.concat(rows, ignore_index=True)
    historical_predictions = historical_predictions.sort_values("date").reset_index(drop=True)
    return historical_predictions


def fit_final_model_and_forecast(
    historical_df: pd.DataFrame,
    future_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    target_col = f"future_direction_{TARGET_HORIZON_DAYS}d"
    train = historical_df.dropna(subset=feature_cols + [target_col]).copy()
    x_train = train[feature_cols]
    y_train = train[target_col].astype(int)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42 + TARGET_HORIZON_DAYS,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    x_future = future_df[feature_cols].copy()
    prob_up = model.predict_proba(x_future)[:, 1]
    tree_std = compute_tree_probability_std(model, x_future)
    confidence_score = compute_confidence(prob_up, tree_std)

    out = future_df[
        [
            "date",
            "astro_bullish_score",
            "astro_bearish_score",
            "astro_reversal_score",
            "astro_volatility_score",
            "astro_compression_score",
            "astro_trend_start_score",
            "astro_trend_end_score",
            "astro_momentum_v2_smooth",
            "astro_regime_v2",
        ]
    ].copy()
    out["astro_score"] = compute_astro_score(out)
    out["ml_probability"] = prob_up
    out["confidence_score"] = confidence_score
    out["signal"] = probability_to_signal(out["ml_probability"])
    out["forecast_strength"] = forecast_strength_label(out["ml_probability"], out["confidence_score"])
    return out


def detect_turning_points(timeline_df: pd.DataFrame, historical_momentum: pd.Series) -> pd.DataFrame:
    rows = []
    timeline = timeline_df.sort_values("date").reset_index(drop=True).copy()
    q25 = float(historical_momentum.quantile(0.25))
    q75 = float(historical_momentum.quantile(0.75))

    for idx in range(1, len(timeline)):
        prev_row = timeline.iloc[idx - 1]
        row = timeline.iloc[idx]
        confidence_jump = float(row["confidence_score"] - prev_row["confidence_score"])

        if row["signal"] != prev_row["signal"]:
            if prev_row["signal"] == "Bullish" and row["signal"] != "Bullish":
                turning_type = "bullish_window_break"
            elif prev_row["signal"] == "Bearish" and row["signal"] != "Bearish":
                turning_type = "bearish_window_relief"
            else:
                turning_type = "signal_flip"

            rows.append(
                {
                    "turning_point_date": row["date"].date().isoformat(),
                    "old_signal": prev_row["signal"],
                    "new_signal": row["signal"],
                    "turning_point_type": turning_type,
                    "confidence": row["confidence_score"],
                    "explanation": (
                        f"Signal shifted from {prev_row['signal']} to {row['signal']} "
                        f"as ML probability moved to {row['ml_probability']:.4f}."
                    ),
                }
            )

        if abs(confidence_jump) >= 0.20:
            rows.append(
                {
                    "turning_point_date": row["date"].date().isoformat(),
                    "old_signal": prev_row["signal"],
                    "new_signal": row["signal"],
                    "turning_point_type": "confidence_shock",
                    "confidence": row["confidence_score"],
                    "explanation": (
                        f"Confidence changed sharply by {confidence_jump:+.4f}, "
                        f"moving from {prev_row['confidence_score']:.4f} to {row['confidence_score']:.4f}."
                    ),
                }
            )

        crossed_up = prev_row["astro_momentum_v2_smooth"] < q75 <= row["astro_momentum_v2_smooth"]
        crossed_down = prev_row["astro_momentum_v2_smooth"] > q25 >= row["astro_momentum_v2_smooth"]
        crossed_zero = (prev_row["astro_momentum_v2_smooth"] <= 0 < row["astro_momentum_v2_smooth"]) or (
            prev_row["astro_momentum_v2_smooth"] >= 0 > row["astro_momentum_v2_smooth"]
        )

        if crossed_up:
            rows.append(
                {
                    "turning_point_date": row["date"].date().isoformat(),
                    "old_signal": prev_row["signal"],
                    "new_signal": row["signal"],
                    "turning_point_type": "momentum_breakout_up",
                    "confidence": row["confidence_score"],
                    "explanation": (
                        f"Astro momentum crossed above the upper historical threshold ({q75:.4f}), "
                        f"reaching {row['astro_momentum_v2_smooth']:.4f}."
                    ),
                }
            )

        if crossed_down:
            rows.append(
                {
                    "turning_point_date": row["date"].date().isoformat(),
                    "old_signal": prev_row["signal"],
                    "new_signal": row["signal"],
                    "turning_point_type": "momentum_breakdown_down",
                    "confidence": row["confidence_score"],
                    "explanation": (
                        f"Astro momentum dropped through the lower historical threshold ({q25:.4f}), "
                        f"falling to {row['astro_momentum_v2_smooth']:.4f}."
                    ),
                }
            )

        if crossed_zero:
            rows.append(
                {
                    "turning_point_date": row["date"].date().isoformat(),
                    "old_signal": prev_row["signal"],
                    "new_signal": row["signal"],
                    "turning_point_type": "momentum_neutral_cross",
                    "confidence": row["confidence_score"],
                    "explanation": (
                        f"Astro momentum crossed the neutral line and is now {row['astro_momentum_v2_smooth']:.4f}."
                    ),
                }
            )

    turning_points = pd.DataFrame(rows)
    if turning_points.empty:
        return pd.DataFrame(
            columns=[
                "turning_point_date",
                "old_signal",
                "new_signal",
                "turning_point_type",
                "confidence",
                "explanation",
            ]
        )

    turning_points = turning_points.sort_values(
        ["turning_point_date", "turning_point_type"]
    ).reset_index(drop=True)
    return turning_points


def build_forecast_windows(timeline_df: pd.DataFrame) -> pd.DataFrame:
    timeline = timeline_df.sort_values("date").reset_index(drop=True).copy()
    timeline["window_type"] = np.where(
        timeline["signal"] == "Bullish",
        "Bullish Window",
        np.where(timeline["signal"] == "Bearish", "Bearish / Risk Window", "Neutral / Transition Window"),
    )

    windows = []
    start_idx = 0

    for idx in range(1, len(timeline) + 1):
        boundary = idx == len(timeline) or timeline.loc[idx, "window_type"] != timeline.loc[start_idx, "window_type"]
        if not boundary:
            continue

        window_slice = timeline.iloc[start_idx:idx]
        window_type = window_slice["window_type"].iloc[0]
        avg_confidence = float(window_slice["confidence_score"].mean())
        avg_prob = float(window_slice["ml_probability"].mean())
        avg_astro = float(window_slice["astro_score"].mean())
        avg_momentum = float(window_slice["astro_momentum_v2_smooth"].mean())
        avg_volatility = float(window_slice["astro_volatility_score"].mean())

        if window_type == "Bullish Window":
            key_driver_summary = (
                f"Positive astro score ({avg_astro:.2f}), supportive momentum ({avg_momentum:.2f}), "
                f"and mean ML probability ({avg_prob:.2f}) above the bullish threshold."
            )
        elif window_type == "Bearish / Risk Window":
            key_driver_summary = (
                f"Negative or fragile directional balance with mean astro score ({avg_astro:.2f}), "
                f"ML probability ({avg_prob:.2f}) near/below bearish territory, and volatility score ({avg_volatility:.2f})."
            )
        else:
            key_driver_summary = (
                f"Mixed directional signals with astro score ({avg_astro:.2f}), "
                f"momentum ({avg_momentum:.2f}), and ML probability ({avg_prob:.2f}) clustering near neutral."
            )

        windows.append(
            {
                "start_date": window_slice["date"].iloc[0].date().isoformat(),
                "end_date": window_slice["date"].iloc[-1].date().isoformat(),
                "window_type": window_type,
                "average_confidence": avg_confidence,
                "key_driver_summary": key_driver_summary,
            }
        )
        start_idx = idx

    return pd.DataFrame(windows)


def compute_calibration_statistics(historical_predictions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    active = historical_predictions[historical_predictions["signal"] != "Neutral"].copy()
    if active.empty:
        empty_stats = pd.DataFrame(columns=["confidence_bucket", "signals", "accuracy"])
        summary = {
            "average_return_after_bullish": np.nan,
            "average_return_after_bearish": np.nan,
            "false_signal_rate": np.nan,
        }
        return empty_stats, summary

    active["predicted_direction"] = np.where(active["signal"] == "Bullish", 1, 0)
    active["correct_signal"] = active["predicted_direction"] == active["actual_direction"]

    bucket_edges = [0.00, 0.40, 0.55, 0.70, 0.85, 1.01]
    bucket_labels = [
        "0.00-0.40",
        "0.40-0.55",
        "0.55-0.70",
        "0.70-0.85",
        "0.85-1.00",
    ]
    active["confidence_bucket"] = pd.cut(
        active["confidence_score"],
        bins=bucket_edges,
        labels=bucket_labels,
        include_lowest=True,
        right=False,
    )

    bucket_stats = (
        active.groupby("confidence_bucket", observed=False)
        .agg(signals=("correct_signal", "size"), accuracy=("correct_signal", "mean"))
        .reset_index()
    )

    average_return_after_bullish = float(
        active.loc[active["signal"] == "Bullish", "actual_future_return"].mean()
    )
    average_return_after_bearish = float(
        active.loc[active["signal"] == "Bearish", "actual_future_return"].mean()
    )
    false_signal_rate = float((~active["correct_signal"]).mean())

    summary = {
        "average_return_after_bullish": average_return_after_bullish,
        "average_return_after_bearish": average_return_after_bearish,
        "false_signal_rate": false_signal_rate,
    }
    return bucket_stats, summary


def save_outputs(
    live_forecast: pd.DataFrame,
    timeline_df: pd.DataFrame,
    turning_points: pd.DataFrame,
    forecast_windows: pd.DataFrame,
    calibration_buckets: pd.DataFrame,
    calibration_summary: Dict[str, float],
):
    live_forecast.to_csv(LIVE_FORECAST_PATH, index=False)
    timeline_df.to_csv(TIMELINE_PATH, index=False)
    turning_points.to_csv(TURNING_POINTS_PATH, index=False)
    forecast_windows.to_csv(FORECAST_WINDOWS_PATH, index=False)

    next_turning_point = turning_points.iloc[0] if not turning_points.empty else None
    bullish_windows = forecast_windows[forecast_windows["window_type"] == "Bullish Window"].copy()
    bearish_windows = forecast_windows[forecast_windows["window_type"] == "Bearish / Risk Window"].copy()

    next_bullish = bullish_windows.iloc[0] if not bullish_windows.empty else None
    next_bearish = bearish_windows.iloc[0] if not bearish_windows.empty else None

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Forecast System v1\n\n")
        handle.write("## Current Signal\n\n")
        handle.write(f"- Forecast date: `{live_forecast.loc[0, 'forecast_date']}`\n")
        handle.write(f"- Current signal: `{live_forecast.loc[0, 'current_signal']}`\n")
        handle.write(f"- Current probability: `{live_forecast.loc[0, 'current_probability']:.4f}`\n")
        handle.write(f"- Confidence score: `{live_forecast.loc[0, 'confidence_score']:.4f}`\n")
        handle.write(f"- Market view: `{live_forecast.loc[0, 'market_view']}`\n")
        handle.write(f"- Risk level: `{live_forecast.loc[0, 'risk_level']}`\n")
        handle.write(f"- Recommended bias: `{live_forecast.loc[0, 'recommended_bias']}`\n\n")

        handle.write("## Timeline Outlook\n\n")
        if next_turning_point is not None:
            handle.write(
                f"- Next major turning point: `{next_turning_point['turning_point_date']}` "
                f"({next_turning_point['turning_point_type']})\n"
            )
        if next_bullish is not None:
            handle.write(
                f"- Next bullish window: `{next_bullish['start_date']}` to `{next_bullish['end_date']}` "
                f"with average confidence `{next_bullish['average_confidence']:.4f}`\n"
            )
        if next_bearish is not None:
            handle.write(
                f"- Next bearish / risk window: `{next_bearish['start_date']}` to `{next_bearish['end_date']}` "
                f"with average confidence `{next_bearish['average_confidence']:.4f}`\n"
            )
        handle.write("\n")

        handle.write("## Forecast Windows\n\n")
        handle.write(dataframe_to_markdown(forecast_windows))
        handle.write("\n\n## Turning Points\n\n")
        handle.write(dataframe_to_markdown(turning_points.head(20)))
        handle.write("\n\n## Historical Calibration\n\n")
        handle.write(dataframe_to_markdown(calibration_buckets))
        handle.write("\n\n")
        handle.write(
            f"- Average return after bullish signals ({TARGET_HORIZON_DAYS}D): "
            f"`{calibration_summary['average_return_after_bullish']:.4f}`\n"
        )
        handle.write(
            f"- Average return after bearish signals ({TARGET_HORIZON_DAYS}D): "
            f"`{calibration_summary['average_return_after_bearish']:.4f}`\n"
        )
        handle.write(
            f"- False signal rate: `{calibration_summary['false_signal_rate']:.4f}`\n"
        )


def main():
    full_df = build_full_feature_frame()
    feature_cols = load_selected_features(full_df)

    historical_df = full_df[full_df["price"].notna()].copy()
    historical_df = historical_df.replace([np.inf, -np.inf], np.nan)
    historical_df = historical_df.dropna(subset=feature_cols + [f"future_direction_{TARGET_HORIZON_DAYS}d"])
    historical_df = historical_df.sort_values("date").reset_index(drop=True)

    latest_price_date = full_df.loc[full_df["price"].notna(), "date"].max()
    forecast_start = latest_price_date + pd.Timedelta(days=1)
    forecast_end = forecast_start + pd.Timedelta(days=FORECAST_DAYS - 1)

    future_df = full_df[
        (full_df["date"] >= forecast_start)
        & (full_df["date"] <= forecast_end)
    ].copy()
    future_df = future_df.sort_values("date").reset_index(drop=True)

    if len(future_df) == 0:
        raise ValueError("No future astro rows available for forecasting")

    for feature in feature_cols:
        if feature not in future_df.columns:
            future_df[feature] = 0.0

    future_df[feature_cols] = future_df[feature_cols].ffill().fillna(0.0)

    historical_predictions = run_historical_walk_forward_predictions(historical_df, feature_cols)
    timeline_df = fit_final_model_and_forecast(historical_df, future_df, feature_cols)

    historical_vol_reference = historical_df["astro_volatility_score"].dropna()
    timeline_df["risk_level"] = risk_level_from_features(
        timeline_df,
        historical_volatility_reference=historical_vol_reference,
        confidence_score=timeline_df["confidence_score"],
    )
    timeline_df["recommended_bias"] = timeline_df.apply(recommended_bias_from_row, axis=1)
    timeline_df["market_view"] = timeline_df.apply(market_view_from_row, axis=1)
    timeline_df["days_ahead"] = (timeline_df["date"] - forecast_start).dt.days.astype(int)
    timeline_df["within_30d"] = timeline_df["days_ahead"] < 30
    timeline_df["within_90d"] = timeline_df["days_ahead"] < 90
    timeline_df["within_180d"] = timeline_df["days_ahead"] < 180
    timeline_df["within_365d"] = timeline_df["days_ahead"] < 365

    turning_points = detect_turning_points(
        timeline_df,
        historical_momentum=historical_df["astro_momentum_v2_smooth"].dropna(),
    )
    forecast_windows = build_forecast_windows(timeline_df)
    calibration_buckets, calibration_summary = compute_calibration_statistics(historical_predictions)

    live_row = timeline_df.iloc[0]
    next_turning_point = turning_points.iloc[0]["turning_point_date"] if not turning_points.empty else ""
    next_bullish = (
        forecast_windows.loc[forecast_windows["window_type"] == "Bullish Window", "start_date"].iloc[0]
        if not forecast_windows[forecast_windows["window_type"] == "Bullish Window"].empty
        else ""
    )
    next_bearish = (
        forecast_windows.loc[forecast_windows["window_type"] == "Bearish / Risk Window", "start_date"].iloc[0]
        if not forecast_windows[forecast_windows["window_type"] == "Bearish / Risk Window"].empty
        else ""
    )

    live_forecast = pd.DataFrame(
        [
            {
                "forecast_date": live_row["date"].date().isoformat(),
                "current_signal": live_row["signal"],
                "current_probability": live_row["ml_probability"],
                "confidence_score": live_row["confidence_score"],
                "market_view": live_row["market_view"],
                "risk_level": live_row["risk_level"],
                "recommended_bias": live_row["recommended_bias"],
                "next_major_turning_point": next_turning_point,
                "next_bullish_window": next_bullish,
                "next_bearish_risk_window": next_bearish,
            }
        ]
    )

    timeline_output = timeline_df[
        [
            "date",
            "days_ahead",
            "within_30d",
            "within_90d",
            "within_180d",
            "within_365d",
            "astro_score",
            "ml_probability",
            "signal",
            "confidence_score",
            "forecast_strength",
            "risk_level",
        ]
    ].copy()
    timeline_output["date"] = timeline_output["date"].dt.date.astype(str)

    save_outputs(
        live_forecast=live_forecast,
        timeline_df=timeline_output,
        turning_points=turning_points,
        forecast_windows=forecast_windows,
        calibration_buckets=calibration_buckets,
        calibration_summary=calibration_summary,
    )

    print(f"Saved {LIVE_FORECAST_PATH}")
    print(f"Saved {TIMELINE_PATH}")
    print(f"Saved {TURNING_POINTS_PATH}")
    print(f"Saved {FORECAST_WINDOWS_PATH}")
    print(f"Saved {REPORT_PATH}")
    print(
        f"Current signal {live_forecast.loc[0, 'current_signal']} | "
        f"next turning point {live_forecast.loc[0, 'next_major_turning_point']}"
    )


if __name__ == "__main__":
    main()
