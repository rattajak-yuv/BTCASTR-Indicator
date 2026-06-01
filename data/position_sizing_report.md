# Position Sizing Engine v2

## Setup

- Base horizon: `7D`
- Ensemble confidence source: `balanced_score_weighted_voting`
- Current Long/Short baseline balanced score: `271.4743`

## Method Comparison

| sizing_method | balanced_score | return_drawdown_ratio | total_return | max_drawdown | volatility | accuracy | trades |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed 100% | 271.4743 | 677.7616 | 455.7202 | -0.6724 | 0.5110 | 0.5801 | 239 |
| Hybrid Sizing | 23.0568 | 68.9928 | 34.2408 | -0.4963 | 0.2835 | 0.5806 | 450 |
| Confidence-Based Sizing | 15.2921 | 45.0864 | 24.2185 | -0.5372 | 0.2905 | 0.5801 | 387 |
| Volatility-Adjusted Sizing | 8.8262 | 29.1194 | 12.6980 | -0.4361 | 0.2338 | 0.5888 | 355 |

## Winners

- Best Aggressive Strategy: `Fixed 100%`
  Total return: `455.7202`
  Balanced score: `271.4743`
  Volatility: `0.5110`
- Best Balanced Strategy: `Fixed 100%`
  Total return: `455.7202`
  Balanced score: `271.4743`
  Return/drawdown ratio: `677.7616`
- Best Conservative Strategy: `Volatility-Adjusted Sizing`
  Total return: `12.6980`
  Return/drawdown ratio: `29.1194`
  Volatility: `0.2338`

## Baseline Comparison

- Best balanced sizing method vs current long/short baseline: `0.0000` balanced-score delta, `0.0000` return/drawdown delta, `0.0000` total-return delta.
- Position sizing did not beat the current Long/Short baseline on balanced score, even if it improved one or more secondary risk metrics.
