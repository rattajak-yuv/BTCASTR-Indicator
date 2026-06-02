# Portfolio Allocation Engine v2 Optimization

## Objective
Improve Portfolio Allocation Engine v1 by capturing more BTC upside while still reducing drawdown versus Buy & Hold.

## Current Recommended Allocation
- Recommended variant: `Trend-Preserving v2`
- Current BTC / Cash allocation: `85.0% / 15.0%`
- Current taxonomy: `Constructive Drift`
- Current confidence: `29.98%`
- Current ML probability: `54.18%`
- Explanation: Base BTC allocation starts at 90.0% from the `Constructive Drift` rule. Confidence is low at 29.98%, applying a -5.0 percentage point confidence adjustment. ML probability at 54.18% does not trigger an extra allocation change. Astro momentum is neutral-to-positive at 1.36, so no momentum adjustment was applied. The 30D, 90D, and 365D outlooks do not fully align positively, so no alignment bonus was applied.

## Validation Answers
A. Which allocation variant performs best? `Trend-Preserving v2` based on the requested priority stack.
B. Does v2 beat v1? Yes; v2 comparison vs v1 total return delta = `7.6954`, Sharpe delta = `0.1210`.
C. Does v2 beat Buy & Hold on total return? No.
D. Does v2 reduce drawdown versus Buy & Hold? Yes, drawdown improvement = `0.2420`.
E. Does v2 improve Sharpe / Sortino? Sharpe delta = `0.1964`, Sortino delta = `0.2275`.
F. Does v2 capture enough upside? Yes; return capture ratio vs Buy & Hold = `0.6322`.
G. Which market regime causes underperformance? `2022 bear market` is the weakest stress-test period for the recommended rule.
H. Is recommended v2 ready for paper trading? Yes, for monitored paper testing only.
I. Recommended next step: `tune rules further`

## Variant Results
| variant_key | variant_label | total_return | CAGR | max_drawdown | Sharpe ratio | Sortino ratio | annual_volatility | win_rate | exposure_ratio | turnover | number_of_allocation_changes | return_max_drawdown_ratio | comparison_vs_buy_hold_total_return | comparison_vs_buy_hold_CAGR | comparison_vs_buy_hold_max_drawdown | comparison_vs_buy_hold_sharpe | comparison_vs_buy_hold_sortino | buy_hold_total_return | buy_hold_CAGR | buy_hold_max_drawdown | buy_hold_sharpe | buy_hold_sortino | return_capture_ratio_vs_buy_hold | drawdown_improvement_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_Trend_Preserving_v2 | Trend-Preserving v2 | 12.0292 | 0.4149 | -0.5244 | 1.1670 | 1.5498 | 0.3496 | 0.5015 | 0.4948 | 93.5500 | 448 | 22.9406 | -6.9996 | -0.0847 | 0.2420 | 0.1964 | 0.2275 | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 0.6322 | 0.2420 |
| E_Recovery_Aggressive_v2 | Recovery Aggressive v2 | 7.8525 | 0.3429 | -0.4516 | 1.1397 | 1.5360 | 0.2974 | 0.5015 | 0.4399 | 80.7000 | 443 | 17.3881 | -11.1764 | -0.1567 | 0.3147 | 0.1691 | 0.2138 | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 0.4127 | 0.3147 |
| B_Upside_Capture_v2 | Upside Capture v2 | 8.3472 | 0.3528 | -0.4968 | 1.1081 | 1.5225 | 0.3183 | 0.5104 | 0.4827 | 79.7000 | 446 | 16.8005 | -10.6816 | -0.1468 | 0.2695 | 0.1375 | 0.2003 | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 0.4387 | 0.2695 |
| F_Balanced_v2 | Balanced v2 | 7.6861 | 0.3394 | -0.4747 | 1.1078 | 1.5086 | 0.3059 | 0.5104 | 0.4584 | 77.0500 | 449 | 16.1919 | -11.3427 | -0.1601 | 0.2917 | 0.1372 | 0.1863 | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 0.4039 | 0.2917 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 4.3338 | 0.2540 | -0.3931 | 1.0460 | 1.3875 | 0.2450 | 0.4996 | 0.3600 | 84.4000 | 459 | 11.0238 | -14.6950 | -0.2456 | 0.3732 | 0.0754 | 0.0652 | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 0.2277 | 0.3732 |
| D_Drawdown_Guard_v2 | Drawdown Guard v2 | 6.2445 | 0.3070 | -0.4941 | 1.0092 | 1.2080 | 0.3139 | 0.3722 | 0.4146 | 89.4500 | 362 | 12.6383 | -12.7843 | -0.1926 | 0.2723 | 0.0385 | -0.1143 | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 0.3282 | 0.2723 |
| BUY_HOLD | Buy & Hold | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 0.6147 | 0.5104 | 1.0000 | 0.0000 | 0 | 24.8306 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 1.0000 | 0.0000 |

## Best Named Variant
| variant_key | variant_label | total_return | CAGR | max_drawdown | Sharpe ratio | Sortino ratio | annual_volatility | win_rate | exposure_ratio | turnover | number_of_allocation_changes | return_max_drawdown_ratio | comparison_vs_buy_hold_total_return | comparison_vs_buy_hold_CAGR | comparison_vs_buy_hold_max_drawdown | comparison_vs_buy_hold_sharpe | comparison_vs_buy_hold_sortino | buy_hold_total_return | buy_hold_CAGR | buy_hold_max_drawdown | buy_hold_sharpe | buy_hold_sortino | return_capture_ratio_vs_buy_hold | drawdown_improvement_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_Trend_Preserving_v2 | Trend-Preserving v2 | 12.0292 | 0.4149 | -0.5244 | 1.1670 | 1.5498 | 0.3496 | 0.5015 | 0.4948 | 93.5500 | 448 | 22.9406 | -6.9996 | -0.0847 | 0.2420 | 0.1964 | 0.2275 | 19.0288 | 0.4996 | -0.7663 | 0.9706 | 1.3223 | 0.6322 | 0.2420 |

## Grid Search Note
- Best grid candidate was `GRID_145` with Sharpe `1.2674` and overfit risk `Low`.

## Stress Test
| variant_key | variant_label | period | start_date | end_date | strategy_total_return | buy_hold_total_return | strategy_max_drawdown | buy_hold_max_drawdown | strategy_sharpe | buy_hold_sharpe | strategy_sortino | buy_hold_sortino | strategy_exposure_ratio | strategy_turnover | allocation_changes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2020 bull market | 2020-01-01 | 2020-12-31 | 1.0337 | 3.0316 | -0.0886 | -0.5186 | 2.8345 | 2.3246 | 4.0855 | 2.5625 | 0.3615 | 13.4500 | 69 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2021 peak / drawdown | 2021-01-01 | 2021-12-31 | 0.4285 | 0.5967 | -0.2863 | -0.5306 | 1.1696 | 0.9839 | 1.6756 | 1.5211 | 0.4027 | 11.2000 | 66 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2022 bear market | 2022-01-01 | 2022-12-31 | -0.3207 | -0.6427 | -0.3531 | -0.6689 | -1.4016 | -1.2982 | -1.6677 | -1.6748 | 0.3592 | 12.2500 | 74 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2023 recovery | 2023-01-01 | 2023-12-31 | 0.4983 | 1.5542 | -0.0937 | -0.2006 | 2.4997 | 2.3637 | 4.4340 | 3.9848 | 0.2568 | 7.0000 | 57 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2024 bull / ETF cycle | 2024-01-01 | 2024-12-31 | 0.2190 | 1.2105 | -0.1677 | -0.2618 | 0.9305 | 1.7456 | 1.2152 | 2.9056 | 0.4014 | 12.3000 | 45 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2025-2026 available period | 2025-01-01 | 2026-05-24 | 0.1237 | -0.1760 | -0.2588 | -0.4974 | 0.4795 | -0.0872 | 0.6991 | -0.1236 | 0.4989 | 18.6500 | 85 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2020 bull market | 2020-01-01 | 2020-12-31 | 1.5390 | 3.0316 | -0.1288 | -0.5186 | 2.7111 | 2.3246 | 3.8136 | 2.5625 | 0.5104 | 14.6000 | 71 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2021 peak / drawdown | 2021-01-01 | 2021-12-31 | 0.7289 | 0.5967 | -0.3163 | -0.5306 | 1.4352 | 0.9839 | 2.1766 | 1.5211 | 0.5233 | 10.2500 | 67 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2022 bear market | 2022-01-01 | 2022-12-31 | -0.4225 | -0.6427 | -0.4588 | -0.6689 | -1.4654 | -1.2982 | -1.7279 | -1.6748 | 0.4725 | 12.1000 | 73 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2023 recovery | 2023-01-01 | 2023-12-31 | 0.7695 | 1.5542 | -0.1239 | -0.2006 | 2.5657 | 2.3637 | 4.6577 | 3.9848 | 0.3697 | 5.8500 | 48 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2024 bull / ETF cycle | 2024-01-01 | 2024-12-31 | 0.2567 | 1.2105 | -0.2271 | -0.2618 | 0.9205 | 1.7456 | 1.2507 | 2.9056 | 0.5156 | 11.1000 | 45 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2025-2026 available period | 2025-01-01 | 2026-05-24 | 0.1624 | -0.1760 | -0.2947 | -0.4974 | 0.5185 | -0.0872 | 0.7790 | -0.1236 | 0.6455 | 17.3000 | 79 |
| C_Trend_Preserving_v2 | Trend-Preserving v2 | 2020 bull market | 2020-01-01 | 2020-12-31 | 1.9087 | 3.0316 | -0.1385 | -0.5186 | 2.9020 | 2.3246 | 4.1962 | 2.5625 | 0.5385 | 16.7500 | 72 |
| C_Trend_Preserving_v2 | Trend-Preserving v2 | 2021 peak / drawdown | 2021-01-01 | 2021-12-31 | 0.9780 | 0.5967 | -0.3667 | -0.5306 | 1.5927 | 0.9839 | 2.3601 | 1.5211 | 0.5842 | 12.1000 | 70 |
| C_Trend_Preserving_v2 | Trend-Preserving v2 | 2022 bear market | 2022-01-01 | 2022-12-31 | -0.4610 | -0.6427 | -0.4987 | -0.6689 | -1.4559 | -1.2982 | -1.6972 | -1.6748 | 0.5151 | 13.4000 | 74 |
| C_Trend_Preserving_v2 | Trend-Preserving v2 | 2023 recovery | 2023-01-01 | 2023-12-31 | 0.7734 | 1.5542 | -0.1138 | -0.2006 | 2.6281 | 2.3637 | 4.9546 | 3.9848 | 0.3019 | 6.5000 | 48 |
| C_Trend_Preserving_v2 | Trend-Preserving v2 | 2024 bull / ETF cycle | 2024-01-01 | 2024-12-31 | 0.2877 | 1.2105 | -0.2403 | -0.2618 | 0.9626 | 1.7456 | 1.3125 | 2.9056 | 0.5253 | 13.3000 | 48 |
| C_Trend_Preserving_v2 | Trend-Preserving v2 | 2025-2026 available period | 2025-01-01 | 2026-05-24 | 0.1814 | -0.1760 | -0.3175 | -0.4974 | 0.5433 | -0.0872 | 0.8009 | -0.1236 | 0.6737 | 18.4500 | 71 |
| D_Drawdown_Guard_v2 | Drawdown Guard v2 | 2020 bull market | 2020-01-01 | 2020-12-31 | 1.5772 | 3.0316 | -0.1329 | -0.5186 | 2.7884 | 2.3246 | 3.7277 | 2.5625 | 0.4701 | 17.2000 | 66 |
| D_Drawdown_Guard_v2 | Drawdown Guard v2 | 2021 peak / drawdown | 2021-01-01 | 2021-12-31 | 0.7924 | 0.5967 | -0.3408 | -0.5306 | 1.4800 | 0.9839 | 2.1339 | 1.5211 | 0.5101 | 11.3000 | 60 |
| D_Drawdown_Guard_v2 | Drawdown Guard v2 | 2022 bear market | 2022-01-01 | 2022-12-31 | -0.4292 | -0.6427 | -0.4683 | -0.6689 | -1.4579 | -1.2982 | -1.6130 | -1.6748 | 0.4489 | 13.7000 | 66 |
| D_Drawdown_Guard_v2 | Drawdown Guard v2 | 2023 recovery | 2023-01-01 | 2023-12-31 | 0.6371 | 1.5542 | -0.1116 | -0.2006 | 2.3758 | 2.3637 | 2.9353 | 3.9848 | 0.2127 | 5.8500 | 24 |
| D_Drawdown_Guard_v2 | Drawdown Guard v2 | 2024 bull / ETF cycle | 2024-01-01 | 2024-12-31 | 0.0867 | 1.2105 | -0.2324 | -0.2618 | 0.4362 | 1.7456 | 0.5368 | 2.9056 | 0.4429 | 13.9000 | 46 |
| D_Drawdown_Guard_v2 | Drawdown Guard v2 | 2025-2026 available period | 2025-01-01 | 2026-05-24 | 0.2190 | -0.1760 | -0.2884 | -0.4974 | 0.6443 | -0.0872 | 0.9307 | -0.1236 | 0.6031 | 20.3000 | 79 |
| E_Recovery_Aggressive_v2 | Recovery Aggressive v2 | 2020 bull market | 2020-01-01 | 2020-12-31 | 1.5147 | 3.0316 | -0.1200 | -0.5186 | 2.9024 | 2.3246 | 4.1836 | 2.5625 | 0.4678 | 14.7500 | 71 |
| E_Recovery_Aggressive_v2 | Recovery Aggressive v2 | 2021 peak / drawdown | 2021-01-01 | 2021-12-31 | 0.7203 | 0.5967 | -0.2902 | -0.5306 | 1.5098 | 0.9839 | 2.2278 | 1.5211 | 0.4689 | 9.9500 | 64 |
| E_Recovery_Aggressive_v2 | Recovery Aggressive v2 | 2022 bear market | 2022-01-01 | 2022-12-31 | -0.3899 | -0.6427 | -0.4246 | -0.6689 | -1.4268 | -1.2982 | -1.6542 | -1.6748 | 0.4345 | 11.9000 | 73 |
| E_Recovery_Aggressive_v2 | Recovery Aggressive v2 | 2023 recovery | 2023-01-01 | 2023-12-31 | 0.6342 | 1.5542 | -0.1392 | -0.2006 | 2.2314 | 2.3637 | 3.6204 | 3.9848 | 0.3149 | 6.8000 | 48 |
| E_Recovery_Aggressive_v2 | Recovery Aggressive v2 | 2024 bull / ETF cycle | 2024-01-01 | 2024-12-31 | 0.2653 | 1.2105 | -0.2100 | -0.2618 | 0.9650 | 1.7456 | 1.3048 | 2.9056 | 0.4937 | 11.4000 | 45 |
| E_Recovery_Aggressive_v2 | Recovery Aggressive v2 | 2025-2026 available period | 2025-01-01 | 2026-05-24 | 0.1523 | -0.1760 | -0.2887 | -0.4974 | 0.5085 | -0.0872 | 0.7665 | -0.1236 | 0.6173 | 16.4500 | 79 |
| F_Balanced_v2 | Balanced v2 | 2020 bull market | 2020-01-01 | 2020-12-31 | 1.5377 | 3.0316 | -0.1209 | -0.5186 | 2.8411 | 2.3246 | 4.0917 | 2.5625 | 0.4872 | 14.1500 | 72 |
| F_Balanced_v2 | Balanced v2 | 2021 peak / drawdown | 2021-01-01 | 2021-12-31 | 0.7006 | 0.5967 | -0.3162 | -0.5306 | 1.4304 | 0.9839 | 2.1360 | 1.5211 | 0.5011 | 9.6000 | 67 |
| F_Balanced_v2 | Balanced v2 | 2022 bear market | 2022-01-01 | 2022-12-31 | -0.4075 | -0.6427 | -0.4425 | -0.6689 | -1.4402 | -1.2982 | -1.6994 | -1.6748 | 0.4595 | 11.3000 | 74 |
| F_Balanced_v2 | Balanced v2 | 2023 recovery | 2023-01-01 | 2023-12-31 | 0.6789 | 1.5542 | -0.1236 | -0.2006 | 2.4602 | 2.3637 | 4.3060 | 3.9848 | 0.3301 | 5.9000 | 47 |
| F_Balanced_v2 | Balanced v2 | 2024 bull / ETF cycle | 2024-01-01 | 2024-12-31 | 0.2587 | 1.2105 | -0.2158 | -0.2618 | 0.9403 | 1.7456 | 1.2784 | 2.9056 | 0.5000 | 11.1000 | 47 |
| F_Balanced_v2 | Balanced v2 | 2025-2026 available period | 2025-01-01 | 2026-05-24 | 0.1456 | -0.1760 | -0.2926 | -0.4974 | 0.4902 | -0.0872 | 0.7334 | -0.1236 | 0.6214 | 16.2500 | 79 |

## Annual Comparison
| variant_key | variant_label | year | strategy | total_return | CAGR | max_drawdown | Sharpe ratio | Sortino ratio | annual_volatility | win_rate | exposure_ratio | turnover | number_of_allocation_changes | return_max_drawdown_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2019 | Allocation Strategy | 0.3170 | 0.3170 | -0.0909 | 2.1043 | 2.9048 | 0.1352 | 0.4712 | 0.1849 | 9.0000 | 59 | 3.4882 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2019 | Buy & Hold | 0.8716 | 0.8716 | -0.4898 | 1.2594 | 1.8411 | 0.6792 | 0.5205 | 1.0000 | 0.0000 | 0 | 1.7794 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2020 | Allocation Strategy | 1.0337 | 1.0297 | -0.0886 | 2.8345 | 4.0855 | 0.2619 | 0.5492 | 0.3615 | 13.4500 | 69 | 11.6618 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2020 | Buy & Hold | 3.0316 | 3.0163 | -0.5186 | 2.3246 | 2.5625 | 0.7194 | 0.5628 | 1.0000 | 0.0000 | 0 | 5.8455 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2021 | Allocation Strategy | 0.4285 | 0.4285 | -0.2863 | 1.1696 | 1.6756 | 0.3605 | 0.5014 | 0.4027 | 11.2000 | 66 | 1.4965 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2021 | Buy & Hold | 0.5967 | 0.5967 | -0.5306 | 0.9839 | 1.5211 | 0.8030 | 0.5123 | 1.0000 | 0.0000 | 0 | 1.1245 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2022 | Allocation Strategy | -0.3207 | -0.3207 | -0.3531 | -1.4016 | -1.6677 | 0.2528 | 0.4603 | 0.3592 | 12.2500 | 74 | -0.9084 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2022 | Buy & Hold | -0.6427 | -0.6427 | -0.6689 | -1.2982 | -1.6748 | 0.6345 | 0.4658 | 1.0000 | 0.0000 | 0 | -0.9607 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2023 | Allocation Strategy | 0.4983 | 0.4983 | -0.0937 | 2.4997 | 4.4340 | 0.1674 | 0.4986 | 0.2568 | 7.0000 | 57 | 5.3185 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2023 | Buy & Hold | 1.5542 | 1.5542 | -0.2006 | 2.3637 | 3.9848 | 0.4371 | 0.4986 | 1.0000 | 0.0000 | 0 | 7.7485 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2024 | Allocation Strategy | 0.2190 | 0.2183 | -0.1677 | 0.9305 | 1.2152 | 0.2443 | 0.5219 | 0.4014 | 12.3000 | 45 | 1.3057 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2024 | Buy & Hold | 1.2105 | 1.2058 | -0.2618 | 1.7456 | 2.9056 | 0.5348 | 0.5219 | 1.0000 | 0.0000 | 0 | 4.6236 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2025 | Allocation Strategy | 0.2078 | 0.2078 | -0.1312 | 1.0232 | 1.7408 | 0.2049 | 0.4986 | 0.4641 | 11.3000 | 45 | 1.5838 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2025 | Buy & Hold | -0.0634 | -0.0634 | -0.3215 | 0.0528 | 0.0773 | 0.4186 | 0.4986 | 1.0000 | 0.0000 | 0 | -0.1971 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2026 | Allocation Strategy | -0.0697 | -0.1683 | -0.2156 | -0.5105 | -0.6729 | 0.2824 | 0.4895 | 0.5878 | 7.2500 | 39 | -0.3232 |
| A_Conservative_v1_Baseline | Conservative v1 Baseline | 2026 | Buy & Hold | -0.1203 | -0.2790 | -0.3531 | -0.3804 | -0.5165 | 0.5122 | 0.4895 | 1.0000 | 0.0000 | 0 | -0.3407 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2019 | Allocation Strategy | 0.4265 | 0.4265 | -0.1133 | 1.9932 | 3.1784 | 0.1870 | 0.5205 | 0.2781 | 7.9500 | 59 | 3.7648 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2019 | Buy & Hold | 0.8716 | 0.8716 | -0.4898 | 1.2594 | 1.8411 | 0.6792 | 0.5205 | 1.0000 | 0.0000 | 0 | 1.7794 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2020 | Allocation Strategy | 1.5390 | 1.5325 | -0.1288 | 2.7111 | 3.8136 | 0.3678 | 0.5628 | 0.5104 | 14.6000 | 71 | 11.9477 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2020 | Buy & Hold | 3.0316 | 3.0163 | -0.5186 | 2.3246 | 2.5625 | 0.7194 | 0.5628 | 1.0000 | 0.0000 | 0 | 5.8455 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2021 | Allocation Strategy | 0.7289 | 0.7289 | -0.3163 | 1.4352 | 2.1766 | 0.4530 | 0.5123 | 0.5233 | 10.2500 | 67 | 2.3042 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2021 | Buy & Hold | 0.5967 | 0.5967 | -0.5306 | 0.9839 | 1.5211 | 0.8030 | 0.5123 | 1.0000 | 0.0000 | 0 | 1.1245 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2022 | Allocation Strategy | -0.4225 | -0.4225 | -0.4588 | -1.4654 | -1.7279 | 0.3356 | 0.4658 | 0.4725 | 12.1000 | 73 | -0.9210 |
| B_Upside_Capture_v2 | Upside Capture v2 | 2022 | Buy & Hold | -0.6427 | -0.6427 | -0.6689 | -1.2982 | -1.6748 | 0.6345 | 0.4658 | 1.0000 | 0.0000 | 0 | -0.9607 |

## Key Risks
- Broadly positive and more stable than the faster expansion state.
- 30D outlook: High Conviction Expansion
- 90D outlook: Transition / Low Conviction
- 365D outlook: Constructive Drift

## Reliability Note
This optimization is still a research layer. No transaction costs, taxes, or execution frictions are modeled yet, and the grid search was intentionally small to limit overfitting.
