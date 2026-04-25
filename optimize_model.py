import os
import numpy as np
import pandas as pd

DATA_PATH = "data/bitcoin_astro_daily_score.csv"
OUTPUT_PATH = "data/model_optimization_results.csv"

INDICATORS = [
    "astro_momentum",
    "expansion_score",
    "contraction_score",
    "narrative_score",
    "trigger_score",
]

SMOOTH_SPANS = [3, 5, 8, 10, 14, 21, 30]
LONG_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SHORT_THRESHOLDS = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1
    return dd.min()


def annualized_return(equity: pd.Series, n_days: int) -> float:
    if n_days <= 0 or equity.empty:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0]
    return total_return ** (365 / n_days) - 1


def sharpe_like(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.std() == 0 or returns.empty:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(365)


def count_trades(position: pd.Series) -> int:
    return int((position.diff().fillna(0) != 0).sum())


def run_backtest(df: pd.DataFrame, indicator: str, span: int, long_th: float, short_th: float):
    test = df[["date", "price", indicator]].dropna().copy()
    if test.empty:
        return None

    test["signal_line"] = test[indicator].ewm(span=span, adjust=False).mean()

    # Long + Short + Flat logic
    # +1 = long BTC
    # -1 = short BTC
    # 0 = flat
    test["position_raw"] = np.select(
        [
            test["signal_line"] >= long_th,
            test["signal_line"] <= short_th,
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    # shift to avoid look-ahead bias
    test["position"] = test["position_raw"].shift(1).fillna(0)

    test["returns"] = test["price"].pct_change().fillna(0)
    test["strategy_returns"] = test["returns"] * test["position"]

    test["buy_hold_equity"] = (1 + test["returns"]).cumprod()
    test["strategy_equity"] = (1 + test["strategy_returns"]).cumprod()

    total_return = test["strategy_equity"].iloc[-1] - 1
    buy_hold_return = test["buy_hold_equity"].iloc[-1] - 1

    mdd = max_drawdown(test["strategy_equity"])
    bh_mdd = max_drawdown(test["buy_hold_equity"])

    n_days = (test["date"].max() - test["date"].min()).days
    cagr = annualized_return(test["strategy_equity"], n_days)
    bh_cagr = annualized_return(test["buy_hold_equity"], n_days)

    sharpe = sharpe_like(test["strategy_returns"])
    trades = count_trades(test["position_raw"])

    # Balanced score:
    # reward return and sharpe, penalize drawdown and overtrading
    dd_penalty = abs(mdd) if pd.notna(mdd) else 999
    trade_penalty = trades / 1000

    if dd_penalty == 0:
        return_dd_ratio = np.nan
    else:
        return_dd_ratio = total_return / dd_penalty

    balanced_score = (
        (cagr if pd.notna(cagr) else 0) * 2.0
        + (sharpe if pd.notna(sharpe) else 0) * 0.5
        + (return_dd_ratio if pd.notna(return_dd_ratio) else 0) * 0.2
        - dd_penalty * 1.5
        - trade_penalty
    )

    return {
        "indicator": indicator,
        "span": span,
        "long_threshold": long_th,
        "short_threshold": short_th,
        "mode": "long_short_balanced",
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "cagr": cagr,
        "buy_hold_cagr": bh_cagr,
        "max_drawdown": mdd,
        "buy_hold_max_drawdown": bh_mdd,
        "sharpe_like": sharpe,
        "return_drawdown_ratio": return_dd_ratio,
        "number_of_trades": trades,
        "balanced_score": balanced_score,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    results = []

    for indicator in INDICATORS:
        if indicator not in df.columns:
            continue

        for span in SMOOTH_SPANS:
            for long_th in LONG_THRESHOLDS:
                for short_th in SHORT_THRESHOLDS:
                    # keep symmetric-ish configs for balanced model
                    if abs(abs(long_th) - abs(short_th)) > 1.0:
                        continue

                    result = run_backtest(
                        df=df,
                        indicator=indicator,
                        span=span,
                        long_th=long_th,
                        short_th=short_th,
                    )

                    if result is not None:
                        results.append(result)

    out = pd.DataFrame(results)

    out = out.sort_values(
        by=["balanced_score", "return_drawdown_ratio", "cagr"],
        ascending=False,
    ).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
