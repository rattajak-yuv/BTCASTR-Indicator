# Portfolio Allocation Engine v1

## Objective
Translate Forecast Taxonomy v4 into research-based BTC / Cash allocation guidance and test whether it improves risk-adjusted outcomes versus Buy & Hold.

## Current Allocation
- Current date: `2026-06-02`
- Current BTC / Cash allocation: `55.0% / 45.0%`
- Current taxonomy: `Constructive Drift`
- Current signal: `Neutral`
- Current confidence: `29.98%`
- Current ML probability: `54.18%`
- Current risk level: `Low`

## Validation Answers
A. What is the current BTC / Cash allocation? `55.0% BTC / 45.0% cash`.
B. Why is this allocation recommended? Base BTC allocation starts at 70.0% from the `Constructive Drift` rule. Confidence is low at 29.98%, so BTC allocation is reduced by 15 percentage points. ML probability at 54.18% does not trigger an additional change. Astro momentum is neutral-to-positive at 1.36, so no momentum adjustment was applied. The 30D, 90D, and 365D outlooks do not all align positively, so no alignment bonus is applied.
C. Which taxonomy states drive allocation most? `Recovery / Reversal Setup`, `High Conviction Expansion`, and `Constructive Drift` carry the strongest 30D edge in the current mapping table.
D. Does allocation strategy beat Buy & Hold? No on total return.
E. Does allocation reduce drawdown? Yes based on max drawdown comparison.
F. Does allocation improve Sharpe / Sortino? Sharpe improvement = `0.0754`, Sortino improvement = `0.0652`.
G. Where does allocation fail? Raw total return does not exceed Buy & Hold.; Return / max drawdown ratio does not improve versus Buy & Hold.
H. Is the system ready for paper trading? Not yet; the rule set still needs refinement before paper trading.
I. Recommended next step: `revise rules`

## Backtest Results
| strategy | total_return | CAGR | max_drawdown | Sharpe ratio | Sortino ratio | annual_volatility | win_rate | exposure_ratio | turnover | number_of_allocation_changes | return_max_drawdown_ratio | comparison_vs_buy_hold_total_return | comparison_vs_buy_hold_CAGR | comparison_vs_buy_hold_max_drawdown | comparison_vs_buy_hold_sharpe | comparison_vs_buy_hold_sortino |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Allocation Strategy | 4.3338 | 0.2540 | -0.3931 | 1.0460 | 1.3875 | 0.2450 | 0.4996 | 0.3600 | 84.4000 | 459 | 11.0238 | -14.6950 | -0.2456 | 0.3732 | 0.0754 | 0.0652 |
| Buy & Hold | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 0.6147 | 0.5104 | 1.0000 | 0.0000 | 0 | 24.8306 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Allocation Driver States
| taxonomy_v4 | forward_return_30d | forward_return_60d | stability_assessment |
| --- | --- | --- | --- |
| Recovery / Reversal Setup | 0.1362 | 0.3082 | Relatively stable |
| High Conviction Expansion | 0.1070 | 0.1616 | Fragile |
| Constructive Drift | 0.0789 | 0.1193 | Relatively stable |
| Transition / Low Conviction | 0.0325 | 0.1231 | Relatively stable |
| Volatility Caution | 0.0123 | 0.0681 | Fragile |

## Stress Test
| period | start_date | end_date | strategy_total_return | buy_hold_total_return | strategy_max_drawdown | buy_hold_max_drawdown | strategy_sharpe | buy_hold_sharpe | strategy_sortino | buy_hold_sortino | strategy_exposure_ratio | strategy_turnover | allocation_changes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 bull market | 2020-01-01 | 2020-12-31 | 1.0337 | 3.0316 | -0.0886 | -0.5186 | 2.8345 | 2.3246 | 4.0855 | 2.5625 | 0.3615 | 13.4500 | 69 |
| 2021 peak / drawdown | 2021-01-01 | 2021-12-31 | 0.4285 | 0.5967 | -0.2863 | -0.5306 | 1.1696 | 0.9839 | 1.6756 | 1.5211 | 0.4027 | 11.2000 | 66 |
| 2022 bear market | 2022-01-01 | 2022-12-31 | -0.3207 | -0.6427 | -0.3531 | -0.6689 | -1.4016 | -1.2982 | -1.6677 | -1.6748 | 0.3592 | 12.2500 | 74 |
| 2023 recovery | 2023-01-01 | 2023-12-31 | 0.4983 | 1.5542 | -0.0937 | -0.2006 | 2.4997 | 2.3637 | 4.4340 | 3.9848 | 0.2568 | 7.0000 | 57 |
| 2024 bull / ETF cycle | 2024-01-01 | 2024-12-31 | 0.2190 | 1.2105 | -0.1677 | -0.2618 | 0.9305 | 1.7456 | 1.2152 | 2.9056 | 0.4014 | 12.3000 | 45 |
| 2025-2026 available period | 2025-01-01 | 2026-05-24 | 0.1237 | -0.1760 | -0.2588 | -0.4974 | 0.4795 | -0.0872 | 0.6991 | -0.1236 | 0.4989 | 18.6500 | 85 |

## Annual Snapshot
| year | strategy | total_return | CAGR | max_drawdown | Sharpe ratio | Sortino ratio | annual_volatility | win_rate | exposure_ratio | turnover | number_of_allocation_changes | return_max_drawdown_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | Allocation Strategy | 0.3170 | 0.3170 | -0.0909 | 2.1043 | 2.9048 | 0.1352 | 0.4712 | 0.1849 | 9.0000 | 59 | 3.4882 |
| 2019 | Buy & Hold | 0.8716 | 0.8716 | -0.4898 | 1.2594 | 1.8411 | 0.6792 | 0.5205 | 1.0000 | 0.0000 | 0 | 1.7794 |
| 2020 | Allocation Strategy | 1.0337 | 1.0297 | -0.0886 | 2.8345 | 4.0855 | 0.2619 | 0.5492 | 0.3615 | 13.4500 | 69 | 11.6618 |
| 2020 | Buy & Hold | 3.0316 | 3.0163 | -0.5186 | 2.3246 | 2.5625 | 0.7194 | 0.5628 | 1.0000 | 0.0000 | 0 | 5.8455 |
| 2021 | Allocation Strategy | 0.4285 | 0.4285 | -0.2863 | 1.1696 | 1.6756 | 0.3605 | 0.5014 | 0.4027 | 11.2000 | 66 | 1.4965 |
| 2021 | Buy & Hold | 0.5967 | 0.5967 | -0.5306 | 0.9839 | 1.5211 | 0.8030 | 0.5123 | 1.0000 | 0.0000 | 0 | 1.1245 |
| 2022 | Allocation Strategy | -0.3207 | -0.3207 | -0.3531 | -1.4016 | -1.6677 | 0.2528 | 0.4603 | 0.3592 | 12.2500 | 74 | -0.9084 |
| 2022 | Buy & Hold | -0.6427 | -0.6427 | -0.6689 | -1.2982 | -1.6748 | 0.6345 | 0.4658 | 1.0000 | 0.0000 | 0 | -0.9607 |
| 2023 | Allocation Strategy | 0.4983 | 0.4983 | -0.0937 | 2.4997 | 4.4340 | 0.1674 | 0.4986 | 0.2568 | 7.0000 | 57 | 5.3185 |
| 2023 | Buy & Hold | 1.5542 | 1.5542 | -0.2006 | 2.3637 | 3.9848 | 0.4371 | 0.4986 | 1.0000 | 0.0000 | 0 | 7.7485 |
| 2024 | Allocation Strategy | 0.2190 | 0.2183 | -0.1677 | 0.9305 | 1.2152 | 0.2443 | 0.5219 | 0.4014 | 12.3000 | 45 | 1.3057 |
| 2024 | Buy & Hold | 1.2105 | 1.2058 | -0.2618 | 1.7456 | 2.9056 | 0.5348 | 0.5219 | 1.0000 | 0.0000 | 0 | 4.6236 |
| 2025 | Allocation Strategy | 0.2078 | 0.2078 | -0.1312 | 1.0232 | 1.7408 | 0.2049 | 0.4986 | 0.4641 | 11.3000 | 45 | 1.5838 |
| 2025 | Buy & Hold | -0.0634 | -0.0634 | -0.3215 | 0.0528 | 0.0773 | 0.4186 | 0.4986 | 1.0000 | 0.0000 | 0 | -0.1971 |
| 2026 | Allocation Strategy | -0.0697 | -0.1683 | -0.2156 | -0.5105 | -0.6729 | 0.2824 | 0.4895 | 0.5878 | 7.2500 | 39 | -0.3232 |
| 2026 | Buy & Hold | -0.1203 | -0.2790 | -0.3531 | -0.3804 | -0.5165 | 0.5122 | 0.4895 | 1.0000 | 0.0000 | 0 | -0.3407 |

## Key Risks
- Broadly positive and more stable than the faster expansion state.
- 30D outlook: High Conviction Expansion
- 90D outlook: Transition / Low Conviction
- 365D outlook: Constructive Drift

## Research Limitation
This v1 backtest assumes zero transaction costs and next-day application of daily allocation decisions. It should be treated as a research allocation layer, not execution-ready portfolio logic.
