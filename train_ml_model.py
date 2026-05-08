import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.inspection import permutation_importance

DATA_PATH = "data/ml_dataset.csv"
PREDICTION_PATH = "data/ml_predictions.csv"
SUMMARY_PATH = "data/ml_model_summary.csv"
IMPORTANCE_PATH = "data/ml_feature_importance.csv"

HORIZON = 14
TARGET_COL = f"future_direction_{HORIZON}d"

TRAIN_WINDOW = 730
TEST_WINDOW = 90
STEP_SIZE = 90

MIN_PROBA_LONG = 0.58
MAX_PROBA_SHORT = 0.42


def max_drawdown(equity):
    peak = equity.cummax()
    return ((equity / peak) - 1).min()


def sharpe_like(returns):
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(365)


def get_feature_columns(df):
    exclude_prefixes = [
        "future_return_",
        "future_direction_",
        "future_drawdown_",
    ]

    exclude_cols = {
        "date",
        "price",
        "signal",
        "position",
        "returns",
        "strategy_returns",
        "strategy_equity",
        "buy_hold_equity",
        "strategy_drawdown",
        "buy_hold_drawdown",
        "strategy_total_return",
        "buy_hold_total_return",
        "strategy_max_drawdown",
        "buy_hold_max_drawdown",
        "astro_regime_v2",
        "regime",
        TARGET_COL,
    }

    feature_cols = []

    for c in df.columns:
        if c in exclude_cols:
            continue

        if any(c.startswith(prefix) for prefix in exclude_prefixes):
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    return feature_cols


def create_signal(prob_up):
    if prob_up >= MIN_PROBA_LONG:
        return 1
    elif prob_up <= MAX_PROBA_SHORT:
        return -1
    return 0


def walk_forward_train(df, feature_cols):
    rows = []
    all_importances = []

    df = df.sort_values("date").reset_index(drop=True)

    start = TRAIN_WINDOW

    while start + TEST_WINDOW <= len(df):
        train_start = start - TRAIN_WINDOW
        train_end = start
        test_start = start
        test_end = start + TEST_WINDOW

        train = df.iloc[train_start:train_end].copy()
        test = df.iloc[test_start:test_end].copy()

        train = train.dropna(subset=feature_cols + [TARGET_COL])
        test = test.dropna(subset=feature_cols + [TARGET_COL])

        if len(train) < 300 or len(test) == 0:
            start += STEP_SIZE
            continue

        X_train = train[feature_cols]
        y_train = train[TARGET_COL].astype(int)

        X_test = test[feature_cols]
        y_test = test[TARGET_COL].astype(int)

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        prob_up = model.predict_proba(X_test)[:, 1]
        pred = (prob_up >= 0.5).astype(int)

        test_out = test[["date", "price"]].copy()
        test_out["ml_prob_up"] = prob_up
        test_out["ml_pred_direction"] = pred
        test_out["ml_position_raw"] = [create_signal(p) for p in prob_up]
        test_out["actual_direction"] = y_test.values
        test_out["walk_train_start"] = train["date"].min()
        test_out["walk_train_end"] = train["date"].max()

        rows.append(test_out)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)

        fold_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "test_start": test["date"].min(),
            "test_end": test["date"].max(),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
        })

        all_importances.append(fold_importance)

        print(
            f"Fold {test['date'].min().date()} to {test['date'].max().date()} | "
            f"Acc={acc:.3f} Prec={prec:.3f} Recall={rec:.3f}"
        )

        start += STEP_SIZE

    if not rows:
        raise ValueError("No walk-forward predictions generated")

    pred_df = pd.concat(rows, ignore_index=True)
    imp_df = pd.concat(all_importances, ignore_index=True)

    return pred_df, imp_df


def backtest_ml(pred_df):
    pred_df = pred_df.sort_values("date").reset_index(drop=True)

    pred_df["btc_return_1d"] = pred_df["price"].pct_change().fillna(0)

    # shift position 1 day to avoid look-ahead
    pred_df["ml_position"] = pred_df["ml_position_raw"].shift(1).fillna(0)

    pred_df["ml_strategy_return"] = pred_df["btc_return_1d"] * pred_df["ml_position"]
    pred_df["buy_hold_return"] = pred_df["btc_return_1d"]

    pred_df["ml_strategy_equity"] = (1 + pred_df["ml_strategy_return"]).cumprod()
    pred_df["buy_hold_equity_ml_period"] = (1 + pred_df["buy_hold_return"]).cumprod()

    pred_df["ml_strategy_drawdown"] = (
        pred_df["ml_strategy_equity"] / pred_df["ml_strategy_equity"].cummax()
    ) - 1

    pred_df["buy_hold_drawdown_ml_period"] = (
        pred_df["buy_hold_equity_ml_period"] / pred_df["buy_hold_equity_ml_period"].cummax()
    ) - 1

    return pred_df


def summarize(pred_df, imp_df):
    total_return = pred_df["ml_strategy_equity"].iloc[-1] - 1
    buy_hold_return = pred_df["buy_hold_equity_ml_period"].iloc[-1] - 1

    max_dd = pred_df["ml_strategy_drawdown"].min()
    bh_dd = pred_df["buy_hold_drawdown_ml_period"].min()

    sharpe = sharpe_like(pred_df["ml_strategy_return"])
    bh_sharpe = sharpe_like(pred_df["buy_hold_return"])

    n_trades = int((pred_df["ml_position_raw"].diff().fillna(0) != 0).sum())

    accuracy = accuracy_score(
        pred_df["actual_direction"].astype(int),
        pred_df["ml_pred_direction"].astype(int),
    )

    precision = precision_score(
        pred_df["actual_direction"].astype(int),
        pred_df["ml_pred_direction"].astype(int),
        zero_division=0,
    )

    recall = recall_score(
        pred_df["actual_direction"].astype(int),
        pred_df["ml_pred_direction"].astype(int),
        zero_division=0,
    )

    summary = pd.DataFrame([{
        "model": "RandomForestClassifier",
        "horizon_days": HORIZON,
        "train_window_days": TRAIN_WINDOW,
        "test_window_days": TEST_WINDOW,
        "long_probability_threshold": MIN_PROBA_LONG,
        "short_probability_threshold": MAX_PROBA_SHORT,
        "ml_total_return": total_return,
        "buy_hold_return_same_period": buy_hold_return,
        "ml_max_drawdown": max_dd,
        "buy_hold_max_drawdown_same_period": bh_dd,
        "ml_sharpe_like": sharpe,
        "buy_hold_sharpe_like": bh_sharpe,
        "number_of_trades": n_trades,
        "direction_accuracy": accuracy,
        "direction_precision": precision,
        "direction_recall": recall,
        "prediction_start": pred_df["date"].min(),
        "prediction_end": pred_df["date"].max(),
    }])

    importance_summary = (
        imp_df.groupby("feature")["importance"]
        .mean()
        .reset_index()
        .sort_values("importance", ascending=False)
    )

    return summary, importance_summary


def main():
    print("Loading ML dataset...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df = df.dropna(subset=["price", TARGET_COL]).copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = get_feature_columns(df)

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_cols):,}")
    print(f"Target: {TARGET_COL}")

    pred_df, imp_df = walk_forward_train(df, feature_cols)
    pred_df = backtest_ml(pred_df)

    summary, importance_summary = summarize(pred_df, imp_df)

    os.makedirs("data", exist_ok=True)

    pred_df.to_csv(PREDICTION_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    importance_summary.to_csv(IMPORTANCE_PATH, index=False)

    print(f"Saved: {PREDICTION_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {IMPORTANCE_PATH}")

    print(summary.to_string(index=False))
    print(importance_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
