import argparse
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

DATA_PATH = "data/ml_dataset.csv"
SELECTED_FEATURES_PATH = "data/selected_features.csv"
THRESHOLD_RESULTS_PATH = "data/ml_threshold_tuning_results.csv"

PREDICTION_PATH = "data/ml_predictions.csv"
SUMMARY_PATH = "data/ml_model_summary.csv"
IMPORTANCE_PATH = "data/ml_feature_importance.csv"
SIGNAL_FILTER_RESULTS_PATH = "data/signal_filter_results.csv"
SIGNAL_FILTER_SUMMARY_PATH = "data/signal_filter_summary.csv"
SIGNAL_FILTER_V2_RESULTS_PATH = "data/signal_filter_v2_results.csv"
SIGNAL_FILTER_V2_SUMMARY_PATH = "data/signal_filter_v2_summary.csv"

HORIZONS = [3, 7, 14, 30, 60, 90]

TRAIN_WINDOW = 730
TEST_WINDOW = 90
STEP_SIZE = 90
MIN_HOLD_DAYS_OPTIONS = [3, 7, 14]
REGIME_FILTER_MODES = ["none", "soft", "strict"]
TREND_CONFIRMATION_MODES = ["none", "soft", "strict"]
VOLATILITY_FILTER_MODES = ["none", "soft"]
REVERSAL_FILTER_MODES = ["none", "soft"]

PROBA_THRESHOLDS = {
    3: {"long": 0.56, "short": 0.44},
    7: {"long": 0.57, "short": 0.43},
    14: {"long": 0.58, "short": 0.42},
    30: {"long": 0.60, "short": 0.40},
    60: {"long": 0.62, "short": 0.38},
    90: {"long": 0.63, "short": 0.37},
}

LONG_ALLOWED_REGIMES = {"uptrend", "strong_uptrend"}
SHORT_ALLOWED_REGIMES = {"downtrend", "crash_risk"}
SIGNAL_VOTE_LONG_MIN = 2
SIGNAL_VOTE_SHORT_MAX = -2
SIGNAL_QUALITY_ENTRY_THRESHOLD = 1.15

REGIME_COMPONENT_MAP = {
    "strong_uptrend": 1.25,
    "uptrend": 0.75,
    "sideways": 0.0,
    "compression_zone": -0.10,
    "reversal_zone": -0.20,
    "downtrend": -0.75,
    "crash_risk": -1.25,
}

warnings.filterwarnings(
    "ignore",
    message=(
        "`sklearn.utils.parallel.delayed` should be used with "
        "`sklearn.utils.parallel.Parallel`"
    ),
)

NON_FEATURE_COLUMNS = {
    "date",
    "astro_regime_v2",
    "signal",
    "regime",
    "price",
    "strategy_total_return",
    "buy_hold_total_return",
    "strategy_max_drawdown",
    "buy_hold_max_drawdown",
}


def max_drawdown(equity):
    peak = equity.cummax()
    return ((equity / peak) - 1).min()


def sharpe_like(returns):
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(365)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BTC Astro ML models using either all valid features or a selected feature subset."
    )
    parser.add_argument(
        "--feature-set",
        choices=["all", "selected"],
        default="selected",
        help="Feature source to use for training.",
    )
    return parser.parse_args()


def validate_thresholds(horizon, long_th, short_th):
    if pd.isna(long_th) or pd.isna(short_th):
        raise ValueError(f"Thresholds for horizon {horizon}D must not be NaN")

    if short_th >= long_th:
        raise ValueError(
            f"Invalid thresholds for horizon {horizon}D: short threshold must be below long threshold"
        )


def load_threshold_config():
    thresholds = {
        horizon: {
            "long": values["long"],
            "short": values["short"],
            "source": "default",
        }
        for horizon, values in PROBA_THRESHOLDS.items()
    }

    if not os.path.exists(THRESHOLD_RESULTS_PATH):
        print(
            f"Threshold file not found at {THRESHOLD_RESULTS_PATH}. "
            "Falling back to default thresholds."
        )
        return thresholds

    tuning = pd.read_csv(THRESHOLD_RESULTS_PATH)

    required_columns = [
        "horizon",
        "long_threshold",
        "short_threshold",
        "balanced_score",
    ]
    missing_columns = [c for c in required_columns if c not in tuning.columns]
    if missing_columns:
        raise ValueError(
            f"{THRESHOLD_RESULTS_PATH} is missing required columns: {missing_columns}"
        )

    if tuning.empty:
        print(
            f"Threshold file at {THRESHOLD_RESULTS_PATH} is empty. "
            "Falling back to default thresholds."
        )
        return thresholds

    ranking_columns = ["horizon", "balanced_score"]
    ranking_order = [True, False]

    if "return_drawdown_ratio" in tuning.columns:
        ranking_columns.append("return_drawdown_ratio")
        ranking_order.append(False)

    if "total_return" in tuning.columns:
        ranking_columns.append("total_return")
        ranking_order.append(False)

    if "number_of_trades" in tuning.columns:
        ranking_columns.append("number_of_trades")
        ranking_order.append(True)

    ranking_columns.extend(["long_threshold", "short_threshold"])
    ranking_order.extend([True, True])

    ranked = tuning.sort_values(ranking_columns, ascending=ranking_order)
    best_by_horizon = ranked.groupby("horizon", as_index=False).head(1)

    for _, row in best_by_horizon.iterrows():
        horizon = int(row["horizon"])

        if horizon not in thresholds:
            continue

        long_th = float(row["long_threshold"])
        short_th = float(row["short_threshold"])
        validate_thresholds(horizon, long_th, short_th)

        thresholds[horizon] = {
            "long": long_th,
            "short": short_th,
            "source": "tuned",
        }

    return thresholds


def normalize_regime_label(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def is_valid_feature_column(df, col):
    if col in NON_FEATURE_COLUMNS:
        return False

    if col.startswith("future_"):
        return False

    if not pd.api.types.is_numeric_dtype(df[col]):
        return False

    series = df[col]
    if series.isna().all():
        return False

    return True


def load_all_features(df):
    feature_cols = sorted(
        col for col in df.columns
        if is_valid_feature_column(df, col)
    )

    if len(feature_cols) == 0:
        raise ValueError("No valid numeric features found in ml_dataset.csv")

    return feature_cols


def load_selected_features(df):
    if not os.path.exists(SELECTED_FEATURES_PATH):
        raise FileNotFoundError(f"Missing {SELECTED_FEATURES_PATH}")

    sf = pd.read_csv(SELECTED_FEATURES_PATH)

    if "feature" not in sf.columns:
        raise ValueError("selected_features.csv must contain a 'feature' column")

    selected = sf["feature"].dropna().astype(str).unique().tolist()

    selected = [
        f for f in selected
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    if len(selected) == 0:
        raise ValueError("No selected features found in ml_dataset.csv")

    return selected


def resolve_feature_columns(df, feature_set):
    if feature_set == "all":
        return load_all_features(df), "all_features"

    return load_selected_features(df), "selected_features"


def create_signal(prob_up, horizon, threshold_config):
    long_th = threshold_config[horizon]["long"]
    short_th = threshold_config[horizon]["short"]

    if prob_up >= long_th:
        return 1
    elif prob_up <= short_th:
        return -1
    return 0


def create_ml_vote(prob_up, horizon, threshold_config):
    long_th = threshold_config[horizon]["long"]
    short_th = threshold_config[horizon]["short"]

    if prob_up >= long_th:
        return 2
    if prob_up <= short_th:
        return -2

    if prob_up > 0.55:
        return 1
    if prob_up < 0.45:
        return -1

    return 0


def create_momentum_vote(momentum_value):
    if pd.isna(momentum_value):
        return 0
    if momentum_value > 0:
        return 1
    if momentum_value < 0:
        return -1
    return 0


def create_trend_vote(trend_start_score, trend_end_score):
    if pd.isna(trend_start_score) or pd.isna(trend_end_score):
        return 0
    if trend_start_score > trend_end_score:
        return 1
    if trend_end_score > trend_start_score:
        return -1
    return 0


def create_regime_vote(regime_value):
    regime_label = normalize_regime_label(regime_value)
    if regime_label in LONG_ALLOWED_REGIMES:
        return 1
    if regime_label in SHORT_ALLOWED_REGIMES:
        return -1
    return 0


def build_signal_vote_frame(out, horizon, threshold_config):
    vote_df = out.copy()
    vote_df["ml_vote_component"] = vote_df["ml_prob_up"].apply(
        lambda value: create_ml_vote(value, horizon, threshold_config)
    )
    vote_df["momentum_vote_component"] = vote_df["astro_momentum_v2_smooth"].apply(
        create_momentum_vote
    )
    vote_df["trend_vote_component"] = vote_df.apply(
        lambda row: create_trend_vote(
            row["astro_trend_start_score"],
            row["astro_trend_end_score"],
        ),
        axis=1,
    )
    vote_df["regime_vote_component"] = vote_df["astro_regime_v2"].apply(
        create_regime_vote
    )
    vote_df["signal_vote_score"] = (
        vote_df["ml_vote_component"]
        + vote_df["momentum_vote_component"]
        + vote_df["trend_vote_component"]
        + vote_df["regime_vote_component"]
    )

    vote_df["long_regime_allowed"] = vote_df["astro_regime_v2"].apply(
        lambda value: normalize_regime_label(value) in LONG_ALLOWED_REGIMES
    )
    vote_df["short_regime_allowed"] = vote_df["astro_regime_v2"].apply(
        lambda value: normalize_regime_label(value) in SHORT_ALLOWED_REGIMES
    )
    vote_df["long_trend_confirmed"] = (
        vote_df["astro_trend_start_score"] > vote_df["astro_trend_end_score"]
    )
    vote_df["short_trend_confirmed"] = (
        vote_df["astro_trend_end_score"] > vote_df["astro_trend_start_score"]
    )

    vote_df["signal_position_raw"] = 0
    long_mask = (
        (vote_df["signal_vote_score"] >= SIGNAL_VOTE_LONG_MIN)
        & vote_df["long_regime_allowed"]
        & vote_df["long_trend_confirmed"]
    )
    short_mask = (
        (vote_df["signal_vote_score"] <= SIGNAL_VOTE_SHORT_MAX)
        & vote_df["short_regime_allowed"]
        & vote_df["short_trend_confirmed"]
    )

    vote_df.loc[long_mask, "signal_position_raw"] = 1
    vote_df.loc[short_mask, "signal_position_raw"] = -1

    return vote_df


def apply_min_hold_period(position_series, min_hold_days):
    desired_positions = position_series.fillna(0).astype(int).tolist()
    held_positions = []
    current_position = 0
    hold_days = 0

    for desired_position in desired_positions:
        if current_position == 0:
            if desired_position != 0:
                current_position = desired_position
                hold_days = 1
            else:
                hold_days = 0
        else:
            if desired_position == current_position:
                hold_days += 1
            elif hold_days >= min_hold_days:
                current_position = desired_position
                hold_days = 1 if current_position != 0 else 0
            else:
                hold_days += 1

        held_positions.append(current_position)

    return pd.Series(held_positions, index=position_series.index, dtype=int)


def robust_scale(series, fallback=1.0, quantile=0.75):
    clean = (
        pd.Series(series)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .abs()
    )

    if clean.empty:
        return fallback

    scale = clean.quantile(quantile)
    if pd.isna(scale) or scale <= 1e-6:
        return fallback

    return float(scale)


def clip_series(series, limit):
    return series.clip(lower=-limit, upper=limit)


def build_signal_quality_components(group_df, horizon, threshold_config):
    g = group_df.copy()
    long_th = threshold_config[horizon]["long"]
    short_th = threshold_config[horizon]["short"]
    long_scale = max(long_th - 0.5, 0.01)
    short_scale = max(0.5 - short_th, 0.01)

    g["ml_quality_component"] = np.where(
        g["ml_prob_up"] >= 0.5,
        (g["ml_prob_up"] - 0.5) / long_scale,
        (g["ml_prob_up"] - 0.5) / short_scale,
    )
    g["ml_quality_component"] = clip_series(g["ml_quality_component"], 2.5)

    trend_delta = g["astro_trend_start_score"] - g["astro_trend_end_score"]
    g["trend_delta"] = trend_delta

    momentum_scale = robust_scale(g["astro_momentum_v2_smooth"])
    trend_scale = robust_scale(trend_delta)
    reversal_scale = robust_scale(g["astro_reversal_score"])
    volatility_scale = robust_scale(g["astro_volatility_score"])
    compression_scale = robust_scale(g["astro_compression_score"])

    g["momentum_quality_component"] = clip_series(
        g["astro_momentum_v2_smooth"] / momentum_scale,
        1.5,
    )
    g["trend_quality_component"] = clip_series(
        trend_delta / trend_scale,
        1.5,
    )
    g["regime_quality_component"] = g["astro_regime_v2"].apply(
        lambda value: REGIME_COMPONENT_MAP.get(normalize_regime_label(value), 0.0)
    )
    g["reversal_penalty_component"] = (
        g["astro_reversal_score"].abs() / reversal_scale
    ).clip(0, 1.75)
    g["volatility_penalty_component"] = (
        g["astro_volatility_score"].abs() / volatility_scale
    ).clip(0, 1.75)
    g["compression_penalty_component"] = (
        g["astro_compression_score"].abs() / compression_scale
    ).clip(0, 1.5)

    g["signal_quality_score_base"] = (
        g["ml_quality_component"] * 1.35
        + g["momentum_quality_component"] * 0.45
        + g["trend_quality_component"] * 0.20
        + g["regime_quality_component"] * 0.10
    )

    return g


def evaluate_signal_filters_v2(pred_df, feature_set_name, threshold_config, v1_summary_df):
    regime_mode_weights = {"none": 0.0, "soft": 0.35, "strict": 0.80}
    trend_mode_weights = {"none": 0.0, "soft": 0.45, "strict": 0.95}
    volatility_mode_weights = {"none": 0.0, "soft": 0.30}
    reversal_mode_weights = {"none": 0.0, "soft": 0.40}

    results = []

    for horizon, group in pred_df.groupby("horizon"):
        base_group = group.sort_values("date").reset_index(drop=True).copy()
        base_group = build_signal_quality_components(
            base_group,
            horizon=horizon,
            threshold_config=threshold_config,
        )

        print(
            f"Evaluating signal filters v2 for {horizon}D | "
            f"thresholds long={threshold_config[horizon]['long']:.2f} "
            f"short={threshold_config[horizon]['short']:.2f}"
        )

        for min_hold_days in MIN_HOLD_DAYS_OPTIONS:
            for regime_filter_mode in REGIME_FILTER_MODES:
                for trend_confirmation_mode in TREND_CONFIRMATION_MODES:
                    for volatility_filter_mode in VOLATILITY_FILTER_MODES:
                        for reversal_filter_mode in REVERSAL_FILTER_MODES:
                            g = base_group.copy()

                            score = g["signal_quality_score_base"].copy()
                            score += (
                                regime_mode_weights[regime_filter_mode]
                                * g["regime_quality_component"]
                            )
                            score += (
                                trend_mode_weights[trend_confirmation_mode]
                                * g["trend_quality_component"]
                            )

                            score_sign = np.sign(score)
                            score_sign = score_sign.where(score_sign != 0, np.sign(g["ml_quality_component"]))
                            score_sign = score_sign.where(score_sign != 0, 1.0)

                            penalty = (
                                volatility_mode_weights[volatility_filter_mode]
                                * g["volatility_penalty_component"]
                                + reversal_mode_weights[reversal_filter_mode]
                                * g["reversal_penalty_component"]
                                + 0.15 * g["compression_penalty_component"]
                            )
                            score = score - penalty * score_sign
                            g["signal_quality_score"] = score

                            g["signal_position_raw_v2"] = 0
                            g.loc[
                                g["signal_quality_score"] >= SIGNAL_QUALITY_ENTRY_THRESHOLD,
                                "signal_position_raw_v2",
                            ] = 1
                            g.loc[
                                g["signal_quality_score"] <= -SIGNAL_QUALITY_ENTRY_THRESHOLD,
                                "signal_position_raw_v2",
                            ] = -1

                            g["btc_return_1d"] = g["price"].pct_change().fillna(0)
                            desired_position = g["signal_position_raw_v2"].shift(1).fillna(0)
                            g["signal_position_v2"] = apply_min_hold_period(
                                desired_position,
                                min_hold_days=min_hold_days,
                            )
                            g["signal_strategy_return_v2"] = (
                                g["btc_return_1d"] * g["signal_position_v2"]
                            )
                            g["signal_strategy_equity_v2"] = (
                                1 + g["signal_strategy_return_v2"]
                            ).cumprod()
                            g["signal_strategy_drawdown_v2"] = (
                                g["signal_strategy_equity_v2"]
                                / g["signal_strategy_equity_v2"].cummax()
                            ) - 1

                            total_return = g["signal_strategy_equity_v2"].iloc[-1] - 1
                            max_dd = g["signal_strategy_drawdown_v2"].min()
                            sharpe = sharpe_like(g["signal_strategy_return_v2"])
                            trades = int(
                                (g["signal_position_v2"].diff().fillna(0) != 0).sum()
                            )

                            dd_abs = abs(max_dd) if pd.notna(max_dd) else np.nan
                            return_dd_ratio = (
                                total_return / dd_abs
                                if pd.notna(dd_abs) and dd_abs != 0
                                else np.nan
                            )
                            balanced_score = (
                                total_return * 0.30
                                + (sharpe if pd.notna(sharpe) else 0) * 0.35
                                + (return_dd_ratio if pd.notna(return_dd_ratio) else 0) * 0.20
                                - (dd_abs if pd.notna(dd_abs) else 0) * 1.25
                                - trades * 0.002
                            )

                            results.append(
                                {
                                    "feature_set": feature_set_name,
                                    "horizon_days": horizon,
                                    "min_hold_days": min_hold_days,
                                    "regime_filter_mode": regime_filter_mode,
                                    "trend_confirmation_mode": trend_confirmation_mode,
                                    "volatility_filter_mode": volatility_filter_mode,
                                    "reversal_filter_mode": reversal_filter_mode,
                                    "long_probability_threshold": threshold_config[horizon]["long"],
                                    "short_probability_threshold": threshold_config[horizon]["short"],
                                    "threshold_source": threshold_config[horizon]["source"],
                                    "signal_quality_entry_threshold": SIGNAL_QUALITY_ENTRY_THRESHOLD,
                                    "total_return": total_return,
                                    "max_drawdown": max_dd,
                                    "sharpe_like": sharpe,
                                    "return_drawdown_ratio": return_dd_ratio,
                                    "balanced_score": balanced_score,
                                    "number_of_trades": trades,
                                    "long_signal_days": int((g["signal_position_v2"] == 1).sum()),
                                    "short_signal_days": int((g["signal_position_v2"] == -1).sum()),
                                    "flat_signal_days": int((g["signal_position_v2"] == 0).sum()),
                                    "avg_signal_quality_score": g["signal_quality_score"].mean(),
                                    "abs_avg_signal_quality_score": g["signal_quality_score"].abs().mean(),
                                    "avg_ml_quality_component": g["ml_quality_component"].mean(),
                                    "avg_momentum_quality_component": g["momentum_quality_component"].mean(),
                                    "avg_trend_quality_component": g["trend_quality_component"].mean(),
                                    "avg_regime_quality_component": g["regime_quality_component"].mean(),
                                    "avg_reversal_penalty_component": g["reversal_penalty_component"].mean(),
                                    "avg_volatility_penalty_component": g["volatility_penalty_component"].mean(),
                                    "avg_compression_penalty_component": g["compression_penalty_component"].mean(),
                                    "prediction_start": g["date"].min(),
                                    "prediction_end": g["date"].max(),
                                }
                            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["balanced_score", "return_drawdown_ratio", "total_return", "number_of_trades"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    results_df["overall_rank"] = np.arange(1, len(results_df) + 1)
    results_df["rank_within_horizon"] = results_df.groupby("horizon_days").cumcount() + 1

    summary_df = (
        results_df.sort_values(
            ["horizon_days", "balanced_score", "return_drawdown_ratio", "total_return", "number_of_trades"],
            ascending=[True, False, False, False, True],
        )
        .groupby("horizon_days", as_index=False)
        .head(1)
        .sort_values(
            ["balanced_score", "return_drawdown_ratio", "total_return", "number_of_trades"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    summary_df["summary_rank"] = np.arange(1, len(summary_df) + 1)

    v1_compare = v1_summary_df[
        [
            "horizon_days",
            "min_hold_days",
            "balanced_score",
            "return_drawdown_ratio",
            "total_return",
            "number_of_trades",
        ]
    ].rename(
        columns={
            "min_hold_days": "v1_min_hold_days",
            "balanced_score": "v1_balanced_score",
            "return_drawdown_ratio": "v1_return_drawdown_ratio",
            "total_return": "v1_total_return",
            "number_of_trades": "v1_number_of_trades",
        }
    )

    summary_df = summary_df.merge(v1_compare, on="horizon_days", how="left")
    summary_df["balanced_score_delta_vs_v1"] = (
        summary_df["balanced_score"] - summary_df["v1_balanced_score"]
    )
    summary_df["return_drawdown_ratio_delta_vs_v1"] = (
        summary_df["return_drawdown_ratio"] - summary_df["v1_return_drawdown_ratio"]
    )
    summary_df["total_return_delta_vs_v1"] = (
        summary_df["total_return"] - summary_df["v1_total_return"]
    )
    summary_df["trade_delta_vs_v1"] = (
        summary_df["number_of_trades"] - summary_df["v1_number_of_trades"]
    )
    summary_df["improved_vs_v1"] = summary_df["balanced_score_delta_vs_v1"] > 0

    return results_df, summary_df


def evaluate_signal_filters(pred_df, feature_set_name, threshold_config):
    results = []

    for horizon, group in pred_df.groupby("horizon"):
        base_group = group.sort_values("date").reset_index(drop=True).copy()

        print(
            f"Evaluating signal filters for {horizon}D | "
            f"thresholds long={threshold_config[horizon]['long']:.2f} "
            f"short={threshold_config[horizon]['short']:.2f}"
        )

        for min_hold_days in MIN_HOLD_DAYS_OPTIONS:
            g = base_group.copy()
            g["btc_return_1d"] = g["price"].pct_change().fillna(0)
            desired_position = g["signal_position_raw"].shift(1).fillna(0)
            g["signal_position"] = apply_min_hold_period(
                desired_position,
                min_hold_days=min_hold_days,
            )
            g["signal_strategy_return"] = g["btc_return_1d"] * g["signal_position"]
            g["signal_strategy_equity"] = (1 + g["signal_strategy_return"]).cumprod()
            g["signal_strategy_drawdown"] = (
                g["signal_strategy_equity"] / g["signal_strategy_equity"].cummax()
            ) - 1

            total_return = g["signal_strategy_equity"].iloc[-1] - 1
            max_dd = g["signal_strategy_drawdown"].min()
            sharpe = sharpe_like(g["signal_strategy_return"])
            trades = int((g["signal_position"].diff().fillna(0) != 0).sum())

            dd_abs = abs(max_dd) if pd.notna(max_dd) else np.nan
            return_dd_ratio = (
                total_return / dd_abs
                if pd.notna(dd_abs) and dd_abs != 0
                else np.nan
            )

            balanced_score = (
                total_return * 0.30
                + (sharpe if pd.notna(sharpe) else 0) * 0.35
                + (return_dd_ratio if pd.notna(return_dd_ratio) else 0) * 0.20
                - (dd_abs if pd.notna(dd_abs) else 0) * 1.25
                - trades * 0.002
            )
            ratio_text = (
                f"{return_dd_ratio:.3f}"
                if pd.notna(return_dd_ratio)
                else "nan"
            )

            results.append(
                {
                    "feature_set": feature_set_name,
                    "horizon_days": horizon,
                    "min_hold_days": min_hold_days,
                    "long_probability_threshold": threshold_config[horizon]["long"],
                    "short_probability_threshold": threshold_config[horizon]["short"],
                    "threshold_source": threshold_config[horizon]["source"],
                    "vote_long_min": SIGNAL_VOTE_LONG_MIN,
                    "vote_short_max": SIGNAL_VOTE_SHORT_MAX,
                    "total_return": total_return,
                    "max_drawdown": max_dd,
                    "sharpe_like": sharpe,
                    "return_drawdown_ratio": return_dd_ratio,
                    "balanced_score": balanced_score,
                    "number_of_trades": trades,
                    "long_signal_days": int((g["signal_position"] == 1).sum()),
                    "short_signal_days": int((g["signal_position"] == -1).sum()),
                    "flat_signal_days": int((g["signal_position"] == 0).sum()),
                    "avg_signal_vote_score": g["signal_vote_score"].mean(),
                    "abs_avg_signal_vote_score": g["signal_vote_score"].abs().mean(),
                    "prediction_start": g["date"].min(),
                    "prediction_end": g["date"].max(),
                }
            )

            print(
                f"Signal filter {horizon}D | min_hold={min_hold_days} | "
                f"balanced_score={balanced_score:.3f} "
                f"return_dd_ratio={ratio_text} "
                f"trades={trades}"
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["balanced_score", "return_drawdown_ratio", "number_of_trades"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    results_df["overall_rank"] = np.arange(1, len(results_df) + 1)
    results_df["rank_within_horizon"] = (
        results_df.groupby("horizon_days").cumcount() + 1
    )

    summary_df = (
        results_df.sort_values(
            ["horizon_days", "balanced_score", "return_drawdown_ratio", "number_of_trades"],
            ascending=[True, False, False, True],
        )
        .groupby("horizon_days", as_index=False)
        .head(1)
        .sort_values(
            ["balanced_score", "return_drawdown_ratio", "number_of_trades"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    summary_df["summary_rank"] = np.arange(1, len(summary_df) + 1)

    return results_df, summary_df


def walk_forward_train(df, horizon, feature_cols, feature_set_name, threshold_config):
    target_col = f"future_direction_{horizon}d"

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    rows = []
    all_importances = []

    data = df.dropna(subset=["price", target_col]).copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.sort_values("date").reset_index(drop=True)

    start = TRAIN_WINDOW

    while start + TEST_WINDOW <= len(data):
        train_start = start - TRAIN_WINDOW
        train_end = start
        test_start = start
        test_end = start + TEST_WINDOW

        train = data.iloc[train_start:train_end].copy()
        test = data.iloc[test_start:test_end].copy()

        train = train.dropna(subset=feature_cols + [target_col])
        test = test.dropna(subset=feature_cols + [target_col])

        if len(train) < 300 or len(test) == 0:
            start += STEP_SIZE
            continue

        X_train = train[feature_cols]
        y_train = train[target_col].astype(int)

        X_test = test[feature_cols]
        y_test = test[target_col].astype(int)

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42 + horizon,
            n_jobs=-1,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        prob_up = model.predict_proba(X_test)[:, 1]
        pred = (prob_up >= 0.5).astype(int)

        out = test[
            [
                "date",
                "price",
                "astro_regime_v2",
                "astro_momentum_v2_smooth",
                "astro_trend_start_score",
                "astro_trend_end_score",
                "astro_reversal_score",
                "astro_volatility_score",
                "astro_compression_score",
            ]
        ].copy()
        out["horizon"] = horizon
        out["ml_prob_up"] = prob_up
        out["ml_pred_direction"] = pred
        out["ml_position_raw"] = [
            create_signal(p, horizon, threshold_config) for p in prob_up
        ]
        out["actual_direction"] = y_test.values
        out["walk_train_start"] = train["date"].min()
        out["walk_train_end"] = train["date"].max()
        out["long_probability_threshold"] = threshold_config[horizon]["long"]
        out["short_probability_threshold"] = threshold_config[horizon]["short"]
        out["threshold_source"] = threshold_config[horizon]["source"]
        out = build_signal_vote_frame(out, horizon, threshold_config)

        rows.append(out)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)

        imp = pd.DataFrame({
            "horizon": horizon,
            "feature": feature_cols,
            "importance": model.feature_importances_,
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "test_start": test["date"].min(),
            "test_end": test["date"].max(),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "feature_set": feature_set_name,
        })

        all_importances.append(imp)

        print(
            f"Horizon {horizon}D | "
            f"{test['date'].min().date()} to {test['date'].max().date()} | "
            f"Acc={acc:.3f} Prec={prec:.3f} Recall={rec:.3f}"
        )

        start += STEP_SIZE

    if not rows:
        raise ValueError(f"No predictions generated for horizon {horizon}")

    pred_df = pd.concat(rows, ignore_index=True)
    imp_df = pd.concat(all_importances, ignore_index=True)

    return pred_df, imp_df


def backtest_ml(pred_df):
    all_rows = []

    for horizon, group in pred_df.groupby("horizon"):
        g = group.sort_values("date").reset_index(drop=True).copy()

        g["btc_return_1d"] = g["price"].pct_change().fillna(0)

        # shift one day to avoid look-ahead
        g["ml_position"] = g["ml_position_raw"].shift(1).fillna(0)

        g["ml_strategy_return"] = g["btc_return_1d"] * g["ml_position"]
        g["buy_hold_return"] = g["btc_return_1d"]

        g["ml_strategy_equity"] = (1 + g["ml_strategy_return"]).cumprod()
        g["buy_hold_equity_ml_period"] = (1 + g["buy_hold_return"]).cumprod()

        g["ml_strategy_drawdown"] = (
            g["ml_strategy_equity"] / g["ml_strategy_equity"].cummax()
        ) - 1

        g["buy_hold_drawdown_ml_period"] = (
            g["buy_hold_equity_ml_period"] / g["buy_hold_equity_ml_period"].cummax()
        ) - 1

        all_rows.append(g)

    return pd.concat(all_rows, ignore_index=True)


def summarize(pred_df, feature_set_name, threshold_config):
    summaries = []

    for horizon, g in pred_df.groupby("horizon"):
        g = g.sort_values("date").reset_index(drop=True)

        total_return = g["ml_strategy_equity"].iloc[-1] - 1
        buy_hold_return = g["buy_hold_equity_ml_period"].iloc[-1] - 1

        max_dd = g["ml_strategy_drawdown"].min()
        bh_dd = g["buy_hold_drawdown_ml_period"].min()

        sharpe = sharpe_like(g["ml_strategy_return"])
        bh_sharpe = sharpe_like(g["buy_hold_return"])

        trades = int((g["ml_position_raw"].diff().fillna(0) != 0).sum())

        acc = accuracy_score(
            g["actual_direction"].astype(int),
            g["ml_pred_direction"].astype(int),
        )

        prec = precision_score(
            g["actual_direction"].astype(int),
            g["ml_pred_direction"].astype(int),
            zero_division=0,
        )

        rec = recall_score(
            g["actual_direction"].astype(int),
            g["ml_pred_direction"].astype(int),
            zero_division=0,
        )

        dd_abs = abs(max_dd) if pd.notna(max_dd) else np.nan
        return_dd_ratio = total_return / dd_abs if dd_abs and dd_abs != 0 else np.nan

        balanced_score = (
            total_return * 0.30
            + (sharpe if pd.notna(sharpe) else 0) * 0.35
            + (return_dd_ratio if pd.notna(return_dd_ratio) else 0) * 0.20
            - (dd_abs if pd.notna(dd_abs) else 0) * 1.25
            - trades * 0.002
        )

        summaries.append({
            "model": "RandomForestClassifier",
            "feature_set": feature_set_name,
            "horizon_days": horizon,
            "train_window_days": TRAIN_WINDOW,
            "test_window_days": TEST_WINDOW,
            "long_probability_threshold": threshold_config[horizon]["long"],
            "short_probability_threshold": threshold_config[horizon]["short"],
            "threshold_source": threshold_config[horizon]["source"],
            "ml_total_return": total_return,
            "buy_hold_return_same_period": buy_hold_return,
            "ml_max_drawdown": max_dd,
            "buy_hold_max_drawdown_same_period": bh_dd,
            "ml_sharpe_like": sharpe,
            "buy_hold_sharpe_like": bh_sharpe,
            "return_drawdown_ratio": return_dd_ratio,
            "balanced_score": balanced_score,
            "number_of_trades": trades,
            "direction_accuracy": acc,
            "direction_precision": prec,
            "direction_recall": rec,
            "prediction_start": g["date"].min(),
            "prediction_end": g["date"].max(),
        })

    return pd.DataFrame(summaries).sort_values("balanced_score", ascending=False)


def main():
    args = parse_args()

    print("Loading ML dataset...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    feature_cols, feature_set_name = resolve_feature_columns(df, args.feature_set)
    threshold_config = load_threshold_config()

    print(f"Using feature set: {feature_set_name}")
    print(f"Feature count: {len(feature_cols):,}")

    all_preds = []
    all_imps = []

    for horizon in HORIZONS:
        threshold_details = threshold_config[horizon]
        print(f"\nTraining horizon: {horizon}D")
        print(
            f"Using thresholds for {horizon}D | "
            f"long={threshold_details['long']:.2f} "
            f"short={threshold_details['short']:.2f} "
            f"source={threshold_details['source']}"
        )
        pred, imp = walk_forward_train(
            df,
            horizon,
            feature_cols,
            feature_set_name,
            threshold_config,
        )
        all_preds.append(pred)
        all_imps.append(imp)

    pred_df = pd.concat(all_preds, ignore_index=True)
    imp_df = pd.concat(all_imps, ignore_index=True)

    pred_df = backtest_ml(pred_df)
    summary = summarize(pred_df, feature_set_name, threshold_config)
    signal_filter_results, signal_filter_summary = evaluate_signal_filters(
        pred_df,
        feature_set_name,
        threshold_config,
    )
    signal_filter_v2_results, signal_filter_v2_summary = evaluate_signal_filters_v2(
        pred_df,
        feature_set_name,
        threshold_config,
        signal_filter_summary,
    )

    importance = (
        imp_df.groupby(["horizon", "feature", "feature_set"])["importance"]
        .mean()
        .reset_index()
        .sort_values(["horizon", "importance"], ascending=[True, False])
    )

    os.makedirs("data", exist_ok=True)

    pred_df.to_csv(PREDICTION_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    signal_filter_results.to_csv(SIGNAL_FILTER_RESULTS_PATH, index=False)
    signal_filter_summary.to_csv(SIGNAL_FILTER_SUMMARY_PATH, index=False)
    signal_filter_v2_results.to_csv(SIGNAL_FILTER_V2_RESULTS_PATH, index=False)
    signal_filter_v2_summary.to_csv(SIGNAL_FILTER_V2_SUMMARY_PATH, index=False)

    print(f"Saved: {PREDICTION_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {IMPORTANCE_PATH}")
    print(f"Saved: {SIGNAL_FILTER_RESULTS_PATH}")
    print(f"Saved: {SIGNAL_FILTER_SUMMARY_PATH}")
    print(f"Saved: {SIGNAL_FILTER_V2_RESULTS_PATH}")
    print(f"Saved: {SIGNAL_FILTER_V2_SUMMARY_PATH}")

    print(summary.to_string(index=False))
    print(importance.head(40).to_string(index=False))
    print(signal_filter_summary.to_string(index=False))
    print(signal_filter_v2_summary.to_string(index=False))


if __name__ == "__main__":
    main()
