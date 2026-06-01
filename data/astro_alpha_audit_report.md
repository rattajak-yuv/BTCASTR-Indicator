# Astro Alpha Audit & Institutional Backtest Framework v1

## Context
- Historical analysis was rebuilt from the current real repo's engine code, not from the legacy blueprint.
- The saved `bitcoin_astro_daily_score.csv`, `astro_aspects_raw.csv`, and `ml_dataset.csv` snapshots contain git conflict markers in the current workspace, so this audit resolved those conflicts in-memory without editing the source files.
- Results are based on the out-of-sample historical prediction path produced by `Robust Astro Engine v1` logic.

## Core Answers
- A. Astro Momentum v2 Smooth standalone alpha: Astro Momentum v2 Smooth showed its best 30D average return in `momentum_gt_2_5` at `13.34%` across `207` samples.
- B. Taxonomy v2 interpretation value: Taxonomy does separate regimes enough to support portfolio mapping, but the label semantics are not fully trustworthy yet because at least one defensive label still delivered materially positive forward returns.
- C. Turning Point timing value: The best timing event was `Bearish -> Neutral` with `18.33%` average 30D forward return.
- D. Strategy beating Buy & Hold: The strongest strategy was `Taxonomy Spot` with return/drawdown `47.85` and total return `2828.50%`.
- E. Best risk-adjusted strategy: `Taxonomy Spot` with return/drawdown `47.85`.
- F. System is currently most useful for: `Spot investor`.
- G. Recommended next step: `revise taxonomy`.

## Strategy Scorecard
| strategy | total_return | cagr | max_drawdown | sharpe_ratio | sortino_ratio | volatility | win_rate | number_of_trades | average_trade_return | exposure_ratio | return_drawdown_ratio | beats_buy_hold_total_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Taxonomy Spot | 28.2850 | 0.5786 | -0.5911 | 1.3117 | 1.6213 | 0.4128 | 0.5239 | 87 | 0.0495 | 0.5624 | 47.8493 | True |
| Taxonomy Long/Short | 20.2845 | 0.5119 | -0.5246 | 1.1945 | 1.4677 | 0.4192 | 0.5237 | 86 | 0.0470 | 0.5819 | 38.6657 | True |
| Buy & Hold | 19.0262 | 0.4995 | -0.7663 | 0.9706 | 1.3222 | 0.6147 | 0.5107 | 1 | 19.0262 | 1.0000 | 24.8271 | False |
| Hybrid Taxonomy + Momentum | 7.3276 | 0.3318 | -0.3450 | 1.0513 | 1.1083 | 0.3215 | 0.5291 | 155 | 0.0161 | 0.3932 | 21.2420 | False |
| Astro Momentum Spot | 5.9628 | 0.3000 | -0.5628 | 0.7890 | 0.9411 | 0.4767 | 0.5190 | 130 | 0.0205 | 0.7128 | 10.5956 | False |

## Momentum Audit Highlights
| label | event_count | average_return_7d | average_return_14d | average_return_30d | win_rate_30d | sample_count_30d |
| --- | --- | --- | --- | --- | --- | --- |
| momentum_gt_2_5 | 220 | 0.0316 | 0.0613 | 0.1334 | 0.6812 | 207 |
| momentum_gt_2_0 | 445 | 0.0208 | 0.0378 | 0.1008 | 0.5794 | 428 |
| momentum_recovering_from_low_zone | 57 | 0.0168 | 0.0316 | 0.0830 | 0.6316 | 57 |
| momentum_rolling_over_from_high_zone | 54 | 0.0262 | 0.0374 | 0.0782 | 0.5686 | 51 |
| momentum_slope_turns_positive | 141 | 0.0134 | 0.0273 | 0.0621 | 0.5683 | 139 |
| momentum_crosses_below_zero | 22 | 0.0126 | 0.0126 | 0.0609 | 0.5455 | 22 |
| momentum_slope_turns_negative | 142 | 0.0175 | 0.0254 | 0.0515 | 0.5324 | 139 |
| momentum_crosses_above_zero | 22 | 0.0390 | 0.0601 | 0.0460 | 0.5000 | 22 |

## Taxonomy Audit Highlights
| label | event_count | average_return_7d | average_return_14d | average_return_30d | win_rate_30d | sample_count_30d |
| --- | --- | --- | --- | --- | --- | --- |
| False Bull / Exhaustion Risk | 192 | 0.0355 | 0.0720 | 0.1682 | 0.7344 | 192 |
| Constructive / Positive Drift | 1121 | 0.0181 | 0.0377 | 0.0807 | 0.6266 | 1098 |
| Bearish | 116 | -0.0135 | -0.0154 | 0.0331 | 0.5259 | 116 |
| High Risk | 572 | -0.0011 | 0.0012 | 0.0214 | 0.5000 | 572 |
| Neutral / Tactical | 699 | 0.0081 | 0.0112 | 0.0053 | 0.4664 | 699 |

## Turning Point Audit Highlights
| audit_group | label | event_count | average_return_7d | average_return_14d | average_return_30d | win_rate_30d | sample_count_30d |
| --- | --- | --- | --- | --- | --- | --- | --- |
| signal_transition | Bearish -> Neutral | 29 | 0.0408 | 0.1007 | 0.1833 | 0.7586 | 29 |
| turning_point_type | bearish_window_relief | 29 | 0.0543 | 0.1075 | 0.1774 | 0.7586 | 29 |
| turning_point_type | momentum_breakdown_down | 24 | 0.0296 | 0.0288 | 0.1038 | 0.6250 | 24 |
| signal_transition | Neutral -> Bearish | 32 | 0.0398 | 0.0469 | 0.0893 | 0.5625 | 32 |
| turning_point_type | momentum_neutral_cross | 44 | 0.0258 | 0.0363 | 0.0535 | 0.5227 | 44 |
| turning_point_type | signal_flip | 67 | 0.0247 | 0.0261 | 0.0433 | 0.4925 | 67 |
| turning_point_type | momentum_breakout_up | 28 | -0.0058 | -0.0006 | 0.0106 | 0.4643 | 28 |
| signal_transition | Neutral -> Bullish | 38 | 0.0131 | 0.0063 | 0.0003 | 0.4211 | 38 |
| turning_point_type | bullish_window_break | 40 | 0.0094 | 0.0022 | -0.0031 | 0.4500 | 40 |
| signal_transition | Bullish -> Neutral | 42 | 0.0050 | -0.0016 | -0.0091 | 0.4286 | 42 |

## Annual / Out-of-Sample Breakdown
- Best strategy by total return: `Taxonomy Spot` at `2828.50%`.
- Best annual row: `Buy & Hold` in `2020` with `307.96%` return.
- Worst annual row: `Buy & Hold` in `2022` with `-65.14%` return.
- Positive strategy-years: `28`.
- Strategy-years beating Buy & Hold: `16`.

## Caveats
- Backtests are frictionless and do not include fees, slippage, funding, or borrow costs.
- Daily strategy returns apply each day's forecast exposure to the next day's BTC return, which is a conservative and explicit timing assumption.
- Taxonomy labels were reconstructed from historical forecast windows using the current calibrated taxonomy mapping, so this audit measures the current dashboard interpretation layer rather than inventing a new one.
