# Forecast Taxonomy v2

## Mapping
| window_class | taxonomy_v2 | avg_forward_return | avg_win_rate | avg_volatility | window_count | weak_distinction | taxonomy_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Accumulation | Neutral / Tactical | 0.0036 | 0.4286 | 0.1226 | 27 | True | Historical returns are positive but not sharply distinct, with average return 0.36%, win rate 42.86%, and volatility 12.26%. |
| Bearish | Neutral / Tactical | 0.0178 | 0.4957 | 0.1216 | 5 | True | Historical returns are positive but not sharply distinct, with average return 1.78%, win rate 49.57%, and volatility 12.16%. |
| Bull Expansion | Neutral / Tactical | 0.0155 | 0.5065 | 0.1416 | 27 | True | Historical returns are positive but not sharply distinct, with average return 1.55%, win rate 50.65%, and volatility 14.16%. |
| High Risk | Neutral / Tactical | 0.0020 | 0.4848 | 0.1260 | 31 | True | Historical returns are positive but not sharply distinct, with average return 0.20%, win rate 48.48%, and volatility 12.60%. |
| Neutral | Constructive / Positive Drift | 0.1332 | 0.7816 | 0.1435 | 5 | False | Historical outcomes remain positive across key horizons (7D=4.94%, 14D=10.46%, 30D=24.56%) with average win rate 78.16%. |
| Strong Bull | Constructive / Positive Drift | 0.0574 | 0.5839 | 0.1289 | 2 | False | Historical outcomes remain positive across key horizons (7D=2.36%, 14D=4.33%, 30D=10.52%) with average win rate 58.39%. |
| Transition | Constructive / Positive Drift | 0.0404 | 0.5938 | 0.1353 | 65 | False | Historical outcomes remain positive across key horizons (7D=1.85%, 14D=3.62%, 30D=6.64%) with average win rate 59.38%. |

## Outlook
- Next constructive opportunity: Constructive / Positive Drift from 2026-06-02 to 2026-06-05 with average confidence 29.16%.
- Next calibrated risk event: No qualifying window is currently visible in the forecast horizon.

## V1 vs V2 Comparison
| window_class | taxonomy_v2 | windows | total_days | avg_confidence |
| --- | --- | --- | --- | --- |
| Neutral | Constructive / Positive Drift | 1 | 10 | 0.3112 |
| Transition | Constructive / Positive Drift | 9 | 216 | 0.3094 |
| Accumulation | Neutral / Tactical | 3 | 7 | 0.3279 |
| Bull Expansion | Neutral / Tactical | 5 | 105 | 0.3649 |
| High Risk | Neutral / Tactical | 3 | 27 | 0.3745 |

## Changed Windows
| start_date | end_date | window_class | taxonomy_v2 | taxonomy_reason |
| --- | --- | --- | --- | --- |
| 2026-06-02 | 2026-06-05 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.85%, 14D=3.62%, 30D=6.64%) with average win rate 59.38%. |
| 2026-06-06 | 2026-07-05 | Bull Expansion | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 1.55%, win rate 50.65%, and volatility 14.16%. |
| 2026-07-06 | 2026-07-08 | Accumulation | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 0.36%, win rate 42.86%, and volatility 12.26%. |
| 2026-07-09 | 2026-07-09 | Accumulation | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 0.36%, win rate 42.86%, and volatility 12.26%. |
| 2026-07-10 | 2026-08-15 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.85%, 14D=3.62%, 30D=6.64%) with average win rate 59.38%. |
| 2026-08-16 | 2026-09-26 | Bull Expansion | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 1.55%, win rate 50.65%, and volatility 14.16%. |
| 2026-09-27 | 2026-10-06 | Neutral | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=4.94%, 14D=10.46%, 30D=24.56%) with average win rate 78.16%. |
| 2026-10-07 | 2026-10-11 | Bull Expansion | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 1.55%, win rate 50.65%, and volatility 14.16%. |
| 2026-10-12 | 2026-10-25 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.85%, 14D=3.62%, 30D=6.64%) with average win rate 59.38%. |
| 2026-10-26 | 2026-11-01 | Bull Expansion | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 1.55%, win rate 50.65%, and volatility 14.16%. |
| 2026-11-02 | 2026-11-14 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.85%, 14D=3.62%, 30D=6.64%) with average win rate 59.38%. |
| 2026-11-15 | 2026-11-23 | High Risk | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 0.20%, win rate 48.48%, and volatility 12.60%. |
| 2026-11-24 | 2027-01-16 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.85%, 14D=3.62%, 30D=6.64%) with average win rate 59.38%. |
| 2027-01-17 | 2027-02-06 | Bull Expansion | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 1.55%, win rate 50.65%, and volatility 14.16%. |
| 2027-02-07 | 2027-02-12 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.85%, 14D=3.62%, 30D=6.64%) with average win rate 59.38%. |
| 2027-02-13 | 2027-03-01 | High Risk | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 0.20%, win rate 48.48%, and volatility 12.60%. |
| 2027-03-02 | 2027-03-30 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.85%, 14D=3.62%, 30D=6.64%) with average win rate 59.38%. |
| 2027-03-31 | 2027-04-02 | Accumulation | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 0.36%, win rate 42.86%, and volatility 12.26%. |
| 2027-04-03 | 2027-04-27 | Transition | Constructive / Positive Drift | Historical outcomes remain positive across key horizons (7D=1.85%, 14D=3.62%, 30D=6.64%) with average win rate 59.38%. |
| 2027-04-28 | 2027-04-28 | High Risk | Neutral / Tactical | Historical returns are positive but not sharply distinct, with average return 0.20%, win rate 48.48%, and volatility 12.60%. |
