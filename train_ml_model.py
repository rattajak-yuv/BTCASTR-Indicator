import argparse
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

DATA_PATH = "data/ml_dataset.csv"
SELECTED_FEATURES_PATH = "data/selected_features.csv"

PREDICTION_PATH = "data/ml_predictions.csv"
SUMMARY_PATH = "data/ml_model_summary.csv"
IMPORTANCE_PATH = "data/ml_feature_importance.csv"

HORIZONS = [3, 7, 14, 30, 60, 90]

TRAIN_WINDOW = 730
TEST_WINDOW = 90
STEP_SIZE = 90

PROBA_THRESHOLDS = {
    3: {"long": 0.56, "short": 0.44},
    7: {"long": 0.57, "short": 0.43},
    14: {"long": 0.58, "short": 0.42},
    30: {"long": 0.60, "short": 0.40},
    60: {"long": 0.62, "short": 0.38},
    90: {"long": 0.63, "short": 0.37},
}

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


def create_signal(prob_up, horizon):
    long_th = PROBA_THRESHOLDS[horizon]["long"]
    short_th = PROBA_THRESHOLDS[horizon]["short"]

    if prob_up >= long_th:
        return 1
    elif prob_up <= short_th:
        return -1
    return 0


def walk_forward_train(df, horizon, feature_cols, feature_set_name):
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

        out = test[["date", "price"]].copy()
        out["horizon"] = horizon
        out["ml_prob_up"] = prob_up
        out["ml_pred_direction"] = pred
        out["ml_position_raw"] = [create_signal(p, horizon) for p in prob_up]
        out["actual_direction"] = y_test.values
        out["walk_train_start"] = train["date"].min()
        out["walk_train_end"] = train["date"].max()

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


def summarize(pred_df, feature_set_name):
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
            "long_probability_threshold": PROBA_THRESHOLDS[horizon]["long"],
            "short_probability_threshold": PROBA_THRESHOLDS[horizon]["short"],
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

    print(f"Using feature set: {feature_set_name}")
    print(f"Feature count: {len(feature_cols):,}")

    all_preds = []
    all_imps = []

    for horizon in HORIZONS:
        print(f"\nTraining horizon: {horizon}D")
        pred, imp = walk_forward_train(df, horizon, feature_cols, feature_set_name)
        all_preds.append(pred)
        all_imps.append(imp)

    pred_df = pd.concat(all_preds, ignore_index=True)
    imp_df = pd.concat(all_imps, ignore_index=True)

    pred_df = backtest_ml(pred_df)
    summary = summarize(pred_df, feature_set_name)

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

    print(f"Saved: {PREDICTION_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {IMPORTANCE_PATH}")

    print(summary.to_string(index=False))
    print(importance.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
