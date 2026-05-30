import os
import numpy as np
import pandas as pd

PREDICTION_PATH = "data/ml_predictions.csv"
OUTPUT_PATH = "data/ml_threshold_tuning_results.csv"

LONG_THRESHOLDS = [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.68]
SHORT_THRESHOLDS = [0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.35, 0.32]


def max_drawdown(equity):
    peak = equity.cummax()
    return ((equity / peak) - 1).min()


def sharpe_like(returns):
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(365)


def create_position(prob_up, long_th, short_th):
    if prob_up >= long_th:
        return 1
    if prob_up <= short_th:
        return -1
    return 0


def run_threshold_backtest(df, horizon, long_th, short_th):
    g = df[df["horizon"] == horizon].copy()
    g = g.sort_values("date").reset_index(drop=True)

    if g.empty:
        return None

    g["position_raw"] = g["ml_prob_up"].apply(
        lambda x: create_position(x, long_th, short_th)
    )

    g["position"] = g["position_raw"].shift(1).fillna(0)
    g["btc_return_1d"] = g["price"].pct_change().fillna(0)

    g["strategy_return"] = g["btc_return_1d"] * g["position"]
    g["buy_hold_return"] = g["btc_return_1d"]

    g["strategy_equity"] = (1 + g["strategy_return"]).cumprod()
    g["buy_hold_equity"] = (1 + g["buy_hold_return"]).cumprod()

    total_return = g["strategy_equity"].iloc[-1] - 1
    buy_hold_return = g["buy_hold_equity"].iloc[-1] - 1

    mdd = max_drawdown(g["strategy_equity"])
    bh_mdd = max_drawdown(g["buy_hold_equity"])

    sharpe = sharpe_like(g["strategy_return"])
    bh_sharpe = sharpe_like(g["buy_hold_return"])

    trades = int((g["position_raw"].diff().fillna(0) != 0).sum())

    dd_abs = abs(mdd) if pd.notna(mdd) else np.nan
    return_dd_ratio = total_return / dd_abs if pd.notna(dd_abs) and dd_abs != 0 else np.nan

    # Balanced objective:
    # reward return, sharpe, return/DD
    # penalize drawdown and overtrading
    balanced_score = (
        total_return * 0.30
        + (sharpe if pd.notna(sharpe) else 0) * 0.35
        + (return_dd_ratio if pd.notna(return_dd_ratio) else 0) * 0.20
        - (dd_abs if pd.notna(dd_abs) else 0) * 1.25
        - trades * 0.002
    )

    return {
        "horizon": horizon,
        "long_threshold": long_th,
        "short_threshold": short_th,
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "max_drawdown": mdd,
        "buy_hold_max_drawdown": bh_mdd,
        "sharpe_like": sharpe,
        "buy_hold_sharpe_like": bh_sharpe,
        "return_drawdown_ratio": return_dd_ratio,
        "number_of_trades": trades,
        "balanced_score": balanced_score,
        "prediction_start": g["date"].min(),
        "prediction_end": g["date"].max(),
    }


def main():
    print("Loading ML predictions...")
    df = pd.read_csv(PREDICTION_PATH)
    df["date"] = pd.to_datetime(df["date"])

    required = ["horizon", "date", "price", "ml_prob_up"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column in ml_predictions.csv: {c}")

    results = []

    for horizon in sorted(df["horizon"].dropna().unique()):
        print(f"Tuning horizon: {horizon}D")

        for long_th in LONG_THRESHOLDS:
            for short_th in SHORT_THRESHOLDS:
                if short_th >= long_th:
                    continue

                result = run_threshold_backtest(
                    df=df,
                    horizon=horizon,
                    long_th=long_th,
                    short_th=short_th,
                )

                if result:
                    results.append(result)

    out = pd.DataFrame(results)

    if out.empty:
        raise ValueError("No threshold tuning results generated")

    out = out.sort_values(
        ["horizon", "balanced_score"],
        ascending=[True, False]
    ).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")

    for horizon in sorted(out["horizon"].unique()):
        print(f"\nBest threshold for {horizon}D:")
        print(
            out[out["horizon"] == horizon]
            .head(5)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
