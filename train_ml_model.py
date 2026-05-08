import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

DATA_PATH = "data/ml_dataset.csv"
PREDICTION_PATH = "data/ml_predictions.csv"
SUMMARY_PATH = "data/ml_model_summary.csv"
IMPORTANCE_PATH = "data/ml_feature_importance.csv"

HORIZONS = [3, 7, 14, 30, 60, 90]

TRAIN_WINDOW = 730
TEST_WINDOW = 90
STEP_SIZE = 90

PROBA_THRESHOLDS = {
    3:  {"long": 0.56, "short": 0.44},
    7:  {"long": 0.57, "short": 0.43},
    14: {"long": 0.58, "short": 0.42},
    30: {"long": 0.60, "short": 0.40},
    60: {"long": 0.62, "short": 0.38},
    90: {"long": 0.63, "short": 0.37},
}


def max_drawdown(equity):
    peak = equity.cummax()
    return ((equity / peak) - 1).min()


def sharpe_like(returns):
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(365)


def get_feature_columns(df, target_col):
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
        target_col,
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


def create_signal(prob_up, horizon):
    long_th = PROBA_THRESHOLDS[horizon]["long"]
    short_th = PROBA_THRESHOLDS[horizon]["short"]

    if prob_up >= long_th:
        return 1
    elif prob_up <= short_th:
        return -1
    return 0


def walk_forward_train(df, horizon):
    target_col = f"future_direction_{horizon}d"

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    feature_cols = get_feature_columns(df, target_col)

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
            n_estimators=400,
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

        # Shift position to avoid look-ahead
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


def summarize(pred_df, imp_df):
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

    summary = pd.DataFrame(summaries).sort_values("balanced_score", ascending=False)

    importance_summary = (
        imp_df.groupby(["horizon", "feature"])["importance"]
        .mean()
        .reset_index()
        .sort_values(["horizon", "importance"], ascending=[True, False])
    )

    return summary, importance_summary


def main():
    print("Loading ML dataset...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    all_preds = []
    all_imps = []

    for horizon in HORIZONS:
        print(f"\nTraining horizon: {horizon}D")
        pred, imp = walk_forward_train(df, horizon)
        all_preds.append(pred)
        all_imps.append(imp)

    pred_df = pd.concat(all_preds, ignore_index=True)
    imp_df = pd.concat(all_imps, ignore_index=True)

    pred_df = backtest_ml(pred_df)
    summary, importance = summarize(pred_df, imp_df)

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
