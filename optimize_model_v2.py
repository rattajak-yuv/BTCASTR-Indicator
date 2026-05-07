import os
import numpy as np
import pandas as pd

DATA_PATH = "data/bitcoin_astro_daily_score.csv"
OUTPUT_PATH = "data/model_optimization_v2_results.csv"

INDICATORS = [
    "astro_momentum_v2",
    "astro_momentum_v2_smooth",
    "astro_bullish_score",
    "astro_bearish_score",
    "astro_reversal_score",
    "astro_volatility_score",
    "astro_compression_score",
    "astro_trend_start_score",
    "astro_trend_end_score",
]

SMOOTH_SPANS = [3, 5, 8, 10, 14, 21, 30]
LONG_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SHORT_THRESHOLDS = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]


def max_drawdown(equity):
    peak = equity.cummax()
    return ((equity / peak) - 1).min()


def sharpe_like(returns):
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(365)


def run_backtest(df, indicator, span, long_th, short_th):
    test = df[["date", "price", indicator]].dropna().copy()
    if test.empty:
        return None

    test["signal_line"] = test[indicator].ewm(span=span, adjust=False).mean()

    test["position_raw"] = np.select(
        [
            test["signal_line"] >= long_th,
            test["signal_line"] <= short_th,
        ],
        [1, -1],
        default=0,
    )

    test["position"] = test["position_raw"].shift(1).fillna(0)
    test["returns"] = test["price"].pct_change().fillna(0)
    test["strategy_returns"] = test["returns"] * test["position"]

    test["strategy_equity"] = (1 + test["strategy_returns"]).cumprod()
    test["buy_hold_equity"] = (1 + test["returns"]).cumprod()

    total_return = test["strategy_equity"].iloc[-1] - 1
    buy_hold_return = test["buy_hold_equity"].iloc[-1] - 1

    mdd = max_drawdown(test["strategy_equity"])
    bh_mdd = max_drawdown(test["buy_hold_equity"])

    sharpe = sharpe_like(test["strategy_returns"])
    trades = int((test["position_raw"].diff().fillna(0) != 0).sum())

    dd_abs = abs(mdd) if pd.notna(mdd) else 999
    return_dd = total_return / dd_abs if dd_abs != 0 else np.nan

    balanced_score = (
        total_return * 0.30
        + (sharpe if pd.notna(sharpe) else 0) * 0.40
        + (return_dd if pd.notna(return_dd) else 0) * 0.20
        - dd_abs * 1.50
        - trades * 0.002
    )

    return {
        "indicator": indicator,
        "span": span,
        "long_threshold": long_th,
        "short_threshold": short_th,
        "mode": "long_short_balanced_v2",
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "max_drawdown": mdd,
        "buy_hold_max_drawdown": bh_mdd,
        "sharpe_like": sharpe,
        "return_drawdown_ratio": return_dd,
        "number_of_trades": trades,
        "balanced_score": balanced_score,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    results = []

    for indicator in INDICATORS:
        if indicator not in df.columns:
            print(f"Skip missing indicator: {indicator}")
            continue

        for span in SMOOTH_SPANS:
            for long_th in LONG_THRESHOLDS:
                for short_th in SHORT_THRESHOLDS:
                    if abs(abs(long_th) - abs(short_th)) > 1.0:
                        continue

                    result = run_backtest(df, indicator, span, long_th, short_th)
                    if result:
                        results.append(result)

    out = pd.DataFrame(results)

    if out.empty:
        raise ValueError("No optimization results generated")

    out = out.sort_values(
        by=["balanced_score", "return_drawdown_ratio", "sharpe_like"],
        ascending=False
    ).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
