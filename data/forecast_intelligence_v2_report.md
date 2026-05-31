# Forecast Taxonomy v2

## Mapping
| window_class | taxonomy_v2 | avg_forward_return | avg_win_rate | avg_volatility | window_count | weak_distinction | taxonomy_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Accumulation | Constructive / Positive Drift | 0.0521 | 0.6392 | 0.1175 | 34 | True | Historical outcomes remain positive across key horizons (7D=4.08%, 14D=5.11%, 30D=6.44%) with average win rate 63.92%. |
| Bearish | Bearish | -0.0370 | 0.3485 | 0.1163 | 4 | False | Historical forward returns are consistently negative (7D=-3.24%, 14D=-4.37%, 30D=-3.49%) with average win rate only 34.85%. |
| Bull Expansion | Neutral / Tactical | 0.0356 | 0.5486 | 0.1490 | 39 | True | Historical returns are positive but not sharply distinct, with average return 3.56%, win rate 54.86%, and volatility 14.90%. |
| High Risk | High Risk | 0.0003 | 0.4744 | 0.1214 | 34 | False | Historical returns are weak at 0.03% while average volatility stays elevated at 12.14%, which supports a defensive risk framing. |
| Neutral | Neutral / Tactical | 0.0383 | 0.6590 | 0.1129 | 8 | True | Historical returns are positive but not sharply distinct, with average return 3.83%, win rate 65.90%, and volatility 11.29%. |
| Strong Bull | False Bull / Exhaustion Risk | -0.0261 | 0.3333 | 0.0662 | 2 | False | The label appears bullish on the surface, but realized forward returns stayed negative (avg -2.61%) and win rate remained weak at 33.33%. |
| Transition | Constructive / Positive Drift | 0.0468 | 0.6034 | 0.1373 | 69 | True | Historical outcomes remain positive across key horizons (7D=1.59%, 14D=3.46%, 30D=8.98%) with average win rate 60.34%. |

## Outlook
- Next constructive opportunity: Constructive / Positive Drift from 2026-05-31 to 2026-06-04 with average confidence 29.46%.
- Next calibrated risk event: False Bull / Exhaustion Risk from 2026-06-05 to 2026-07-04 with average confidence 40.11%.

## V1 vs V2 Comparison
| window_class | taxonomy_v2 | windows | total_days | avg_confidence |
| --- | --- | --- | --- | --- |
| Bearish | Bearish | 1 | 17 | 0.3806 |
| Accumulation | Constructive / Positive Drift | 7 | 20 | 0.3818 |
| Transition | Constructive / Positive Drift | 10 | 207 | 0.3081 |
| Strong Bull | False Bull / Exhaustion Risk | 1 | 30 | 0.4011 |
| High Risk | High Risk | 3 | 10 | 0.3655 |
| Bull Expansion | Neutral / Tactical | 5 | 81 | 0.3780 |

## Changed Windows
| start_date | end_date | window_class | taxonomy_v2 | taxonomy_reason |
| --- | --- | --- | --- | --- |
| 2026-05-31 | 2026-06-04 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.59%, 14D=3.46%, 30D=8.98%) with average win rate 60.34%. |
| 2026-06-05 | 2026-07-04 | Strong Bull | False Bull / Exhaustion Risk | The label appears bullish on the surface, but realized forward returns stayed negative (avg -2.61%) and win rate remained weak at 33.33%. |
| 2026-07-05 | 2026-07-14 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.59%, 14D=3.46%, 30D=8.98%) with average win rate 60.34%. |
| 2026-08-01 | 2026-08-19 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.59%, 14D=3.46%, 30D=8.98%) with average win rate 60.34%. |
| 2026-08-20 | 2026-09-27 | Bull Expansion | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 3.56%, win rate 54.86%, and volatility 14.90%. |
| 2026-09-28 | 2026-10-02 | Accumulation | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=4.08%, 14D=5.11%, 30D=6.44%) with average win rate 63.92%. |
| 2026-10-03 | 2026-10-12 | Bull Expansion | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 3.56%, win rate 54.86%, and volatility 14.90%. |
| 2026-10-13 | 2026-11-14 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.59%, 14D=3.46%, 30D=8.98%) with average win rate 60.34%. |
| 2026-11-23 | 2026-12-12 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.59%, 14D=3.46%, 30D=8.98%) with average win rate 60.34%. |
| 2026-12-13 | 2026-12-13 | Accumulation | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=4.08%, 14D=5.11%, 30D=6.44%) with average win rate 63.92%. |
| 2026-12-14 | 2026-12-14 | Accumulation | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=4.08%, 14D=5.11%, 30D=6.44%) with average win rate 63.92%. |
| 2026-12-15 | 2026-12-15 | Accumulation | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=4.08%, 14D=5.11%, 30D=6.44%) with average win rate 63.92%. |
| 2026-12-16 | 2026-12-21 | Accumulation | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=4.08%, 14D=5.11%, 30D=6.44%) with average win rate 63.92%. |
| 2026-12-22 | 2026-12-22 | Accumulation | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=4.08%, 14D=5.11%, 30D=6.44%) with average win rate 63.92%. |
| 2026-12-23 | 2026-12-27 | Accumulation | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=4.08%, 14D=5.11%, 30D=6.44%) with average win rate 63.92%. |
| 2026-12-28 | 2027-01-03 | Bull Expansion | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 3.56%, win rate 54.86%, and volatility 14.90%. |
| 2027-01-04 | 2027-01-18 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.59%, 14D=3.46%, 30D=8.98%) with average win rate 60.34%. |
| 2027-01-19 | 2027-02-06 | Bull Expansion | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 3.56%, win rate 54.86%, and volatility 14.90%. |
| 2027-02-07 | 2027-02-12 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.59%, 14D=3.46%, 30D=8.98%) with average win rate 60.34%. |
| 2027-02-14 | 2027-02-14 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.59%, 14D=3.46%, 30D=8.98%) with average win rate 60.34%. |
