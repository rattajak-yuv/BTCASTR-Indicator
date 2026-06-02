# Taxonomy Regime Audit Engine v1

## Objective
Diagnose why Forecast Taxonomy v3 still contains semantic conflicts before turning taxonomy states into real portfolio allocation decisions.

## Current Dashboard Context
- 30D dominant taxonomy: `High Momentum Expansion`
- 90D dominant taxonomy: `Tactical Neutral`
- 365D dominant taxonomy: `Constructive Drift`

## Validation Answers
A. Why does Defensive / Weak Trend produce strong returns? It behaves like `Not truly Defensive`: prior returns and drawdown are weak, current ML probability is low, and the state appears after stress rather than during durable downside continuation.
B. Is Defensive / Weak Trend mislabeled? Yes. The forward-return profile contradicts the label.
C. Is Tactical Neutral a real regime or a catch-all/default state? It looks partly catch-all. It absorbs `28.30%` of observations and sits between constructive and expansion states.
D. Does High Momentum Expansion remain valid? Yes, but it is more fragile than Constructive Drift and should not be treated as the sole foundation for allocation.
E. Which taxonomy labels should be renamed? Defensive / Weak Trend -> `Not truly Defensive`; Tactical Neutral -> `Transition / Low Conviction`; optional softer renames for High Momentum Expansion and High Volatility Risk.
F. Which taxonomy states are stable enough for allocation? `Constructive Drift, Tactical Neutral, Defensive / Weak Trend` are statistically more stable, but semantics are still unresolved for Defensive / Weak Trend and Tactical Neutral.
G. Should we proceed to Portfolio Allocation Engine after this? No. Revise taxonomy semantics first.

## Key Findings
- Constructive Drift: positive and comparatively stable with 30D return `7.89%` across a large sample.
- High Momentum Expansion: still investable on raw edge (`10.70%` at 30D), but concentrated in fewer years.
- Tactical Neutral: positive drift is real, so the label is too passive for a supposed neutral/default bucket.
- High Volatility Risk: weakest directional edge and the clearest defensive/caution state.

## Regime Audit
| taxonomy_v3 | sample_count | sample_share | average_astro_momentum | average_momentum_slope_7d | average_ml_probability | average_confidence | average_astro_score | average_compression_score | average_bullish_score | average_bearish_score | average_reversal_score | average_volatility_score | average_event_count | average_house_activation_strength | average_prior_return_7d | average_prior_return_14d | average_prior_return_30d | average_prior_drawdown_30d | forward_return_7d | forward_return_14d | forward_return_30d | forward_return_60d | forward_return_90d | win_rate_30d | dominant_planets | dominant_aspects | dominant_natal_targets | typical_momentum_range | typical_probability_range | stability_assessment | positive_year_share | active_years | best_year_return_30d | worst_year_return_30d | return_volatility_ratio_30d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constructive Drift | 1083 | 0.4011 | 0.7244 | -0.0113 | 0.5036 | 0.2642 | 0.7867 | 0.6272 | 1.8320 | 1.1104 | 1.9290 | 1.6304 |  |  | 0.0229 | 0.0425 | 0.0708 | -0.0661 | 0.0203 | 0.0369 | 0.0789 | 0.1193 | 0.2166 | 0.6223 | mars_signal (+0.08), uranus_signal (+0.07), neptune_signal (+0.03) | square_strength (+0.22), opposition_strength (+0.03), trine_strength (+0.03) | mc_target_strength (+0.20), sun_target_strength (+0.18), asc_target_strength (+0.07) | 0.66 to 1.79 | 0.46 to 0.54 | Relatively stable | 0.7500 | 8.0000 | 0.2042 | -0.0646 | 0.3877 |
| High Momentum Expansion | 138 | 0.0511 | 2.3164 | -0.0257 | 0.7315 | 0.4935 | 3.1929 | 0.3575 | 3.4248 | 1.1295 | 3.3886 | 3.5216 |  |  | 0.0210 | 0.0393 | 0.0646 | -0.0613 | 0.0234 | 0.0448 | 0.1070 | 0.1616 | 0.1266 | 0.6377 | saturn_signal (+0.55), mars_signal (+0.42), uranus_signal (+0.17) | trine_strength (+0.79), opposition_strength (+0.44) | mc_target_strength (-0.57), asc_target_strength (-0.43), moon_target_strength (-0.17) | 0.59 to 1.14 | 0.60 to 0.63 | Fragile | 1.0000 | 1.0000 | 0.1070 | 0.1070 | 0.5365 |
| Tactical Neutral | 764 | 0.2830 | 0.5724 | 0.0154 | 0.6026 | 0.3460 | 0.7226 | 0.6391 | 1.6358 | 1.0614 | 1.4452 | 1.4211 |  |  | -0.0047 | -0.0072 | 0.0045 | -0.1197 | 0.0096 | 0.0230 | 0.0325 | 0.1231 | 0.1993 | 0.4935 | saturn_signal (+0.45), jupiter_signal (+0.24), neptune_signal (+0.10) | conjunction_strength (+0.45), sextile_strength (+0.39) | sun_target_strength (-0.35), mc_target_strength (-0.12), moon_target_strength (-0.12) | 0.23 to 1.54 | 0.58 to 0.61 | Relatively stable | 0.6250 | 8.0000 | 0.1787 | -0.0562 | 0.1607 |
| Defensive / Weak Trend | 70 | 0.0259 | -0.1540 | 0.2665 | 0.2863 | 0.4655 | -0.2505 | 0.5291 | 0.7646 | 0.8436 | 0.9042 | 1.0472 |  |  | 0.0092 | 0.0255 | 0.1088 | -0.0506 | 0.0043 | 0.0227 | 0.1362 | 0.3082 | 0.3199 | 0.8571 | saturn_signal (+0.55) | opposition_strength (+0.47) | mc_target_strength (-0.78), asc_target_strength (-0.60), sun_target_strength (-0.35) | 0.10 to 0.48 | 0.27 to 0.40 | Relatively stable | 1.0000 | 4.0000 | 0.3495 | 0.0917 | 1.0785 |
| High Volatility Risk | 645 | 0.2389 | 1.1743 | -0.0090 | 0.3694 | 0.4011 | 1.4585 | 0.4395 | 2.0258 | 0.8487 | 2.1691 | 2.0436 |  |  | 0.0097 | 0.0235 | 0.0692 | -0.0840 | -0.0036 | -0.0038 | 0.0123 | 0.0681 | 0.1186 | 0.4698 | neptune_signal (+0.60), jupiter_signal (+0.19) | sextile_strength (+1.11), conjunction_strength (+0.83), trine_strength (+0.00) | moon_target_strength (+1.73), mc_target_strength (-0.65), sun_target_strength (-0.35) | 0.53 to 1.12 | 0.42 to 0.43 | Fragile | 0.4286 | 7.0000 | 0.1355 | -0.0569 | 0.0695 |

## Defensive / Weak Trend Deep Dive
| metric | defensive_weak_trend_mean | other_taxonomies_mean | delta |
| --- | --- | --- | --- |
| average_prior_return_7d | 0.0092 | 0.0115 | -0.0023 |
| average_prior_return_14d | 0.0255 | 0.0232 | 0.0023 |
| average_prior_return_30d | 0.1088 | 0.0508 | 0.0580 |
| average_prior_drawdown_30d | -0.0506 | -0.0858 | 0.0351 |
| average_astro_momentum | -0.1540 | 0.8741 | -1.0281 |
| average_momentum_slope_7d | 0.2665 | -0.0037 | 0.2702 |
| average_ml_probability | 0.2863 | 0.5114 | -0.2251 |
| average_bullish_score | 0.7646 | 1.9061 | -1.1415 |
| average_bearish_score | 0.8436 | 1.0330 | -0.1894 |
| average_reversal_score | 0.9042 | 1.9240 | -1.0197 |
| average_compression_score | 0.5291 | 0.5705 | -0.0414 |
| average_volatility_score | 1.0472 | 1.7702 | -0.7230 |
| saturn_signal |  |  |  |
| opposition_strength |  |  |  |
| average_forward_return_30d | 0.1362 | 0.0503 | 0.0858 |
| average_forward_return_60d | 0.3082 | 0.1099 | 0.1983 |
| average_forward_return_90d | 0.3199 | 0.1818 | 0.1381 |

## Transition Matrix
| from_taxonomy | to_taxonomy | focus_transition | transition_count | average_return_7d | average_return_14d | average_return_30d | win_rate | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| High Volatility Risk | Constructive Drift | True | 18 | 0.0540 | 0.1200 | 0.2112 | 0.9444 | 18 |
| Defensive / Weak Trend | Constructive Drift | True | 4 | 0.0513 | 0.1231 | 0.2763 | 1.0000 | 4 |
| High Momentum Expansion | Tactical Neutral | True | 1 | 0.0162 | -0.0662 | 0.0156 | 1.0000 | 1 |
| Constructive Drift | High Momentum Expansion | True | 1 | -0.0261 | -0.0528 | -0.0302 | 0.0000 | 1 |
| Tactical Neutral | High Momentum Expansion | True | 1 | 0.0277 | -0.0900 | -0.1642 | 0.0000 | 1 |
| Tactical Neutral | Constructive Drift | False | 25 | 0.0191 | 0.0298 | 0.0414 | 0.5200 | 25 |
| Constructive Drift | Tactical Neutral | False | 24 | 0.0135 | 0.0327 | 0.0445 | 0.5000 | 24 |
| Constructive Drift | High Volatility Risk | False | 18 | 0.0265 | 0.0162 | 0.1277 | 0.6111 | 18 |
| Tactical Neutral | High Volatility Risk | False | 6 | 0.0303 | 0.0587 | 0.1241 | 0.5000 | 6 |
| High Volatility Risk | Tactical Neutral | False | 6 | -0.0499 | -0.0246 | 0.0514 | 0.3333 | 6 |
| Constructive Drift | Defensive / Weak Trend | False | 4 | 0.0108 | 0.0166 | 0.0832 | 1.0000 | 4 |
| High Momentum Expansion | Constructive Drift | False | 1 | -0.0519 | -0.0594 | -0.0231 | 0.0000 | 1 |

## Rename Recommendations
| taxonomy_v3 | current_label | recommended_action | suggested_label | primary_issue | allocation_ready | evidence_summary |
| --- | --- | --- | --- | --- | --- | --- |
| Constructive Drift | Constructive Drift | keep | Constructive Drift | State is broadly consistent and relatively stable. | True | Large sample share 40.11% and positive multi-horizon returns support the current label. |
| High Momentum Expansion | High Momentum Expansion | soft rename | High Conviction Expansion | State looks valid, but stability is fragile. | False | 30D edge is 10.70% with win rate 63.77%, but active years are only 1. |
| Tactical Neutral | Tactical Neutral | rename or split | Transition / Low Conviction | Large share plus mild positive drift suggest catch-all behavior. | False | Observation share is 28.30% with 30D return 3.25% and only middling 30D win rate 49.35%. |
| Defensive / Weak Trend | Defensive / Weak Trend | rename | Not truly Defensive | Label semantics conflict with strong positive forward returns. | False | 30D/60D/90D returns are 13.62%/30.82%/31.99% despite low momentum -0.15 and low ML probability 28.63%. |
| High Volatility Risk | High Volatility Risk | optional soft rename | Volatility Caution | Semantics are directionally correct. | False | 30D return is only 1.23% and 30D return/volatility is 0.0695. |

## Current Future Window Context
| start_date | end_date | taxonomy_v3 | v3_posture | average_confidence | average_ml_probability |
| --- | --- | --- | --- | --- | --- |
| 2026-05-31 | 2026-06-04 | Constructive Drift | Constructive Long Bias | 0.2946 | 0.5288 |
| 2026-06-05 | 2026-07-04 | High Momentum Expansion | Momentum Long Bias | 0.4011 | 0.6125 |
| 2026-07-05 | 2026-07-14 | Constructive Drift | Constructive Long Bias | 0.2700 | 0.4584 |
| 2026-07-15 | 2026-07-31 | Defensive / Weak Trend | Defensive / Short Bias | 0.3806 | 0.3449 |
| 2026-08-01 | 2026-08-19 | Constructive Drift | Constructive Long Bias | 0.2772 | 0.5350 |
| 2026-08-20 | 2026-09-27 | Tactical Neutral | Tactical / Wait | 0.4012 | 0.5983 |
| 2026-09-28 | 2026-10-02 | Constructive Drift | Constructive Long Bias | 0.3760 | 0.5678 |
| 2026-10-03 | 2026-10-12 | Tactical Neutral | Tactical / Wait | 0.3942 | 0.5812 |
| 2026-10-13 | 2026-11-14 | Constructive Drift | Constructive Long Bias | 0.3130 | 0.4996 |
| 2026-11-15 | 2026-11-22 | High Volatility Risk | Defensive / Volatility Control | 0.3996 | 0.4238 |
| 2026-11-23 | 2026-12-12 | Constructive Drift | Constructive Long Bias | 0.3375 | 0.5005 |
| 2026-12-13 | 2026-12-13 | Constructive Drift | Constructive Long Bias | 0.3848 | 0.5716 |

## Current Future Timeline Snapshot
| date | astro_score | ml_probability | signal | confidence_score | risk_level |
| --- | --- | --- | --- | --- | --- |
| 2026-05-31 | 1.7098 | 0.4750 | Neutral | 0.2708 | Low |
| 2026-06-01 | 1.6915 | 0.5051 | Neutral | 0.2513 | Low |
| 2026-06-02 | 1.6567 | 0.5418 | Neutral | 0.2998 | Low |
| 2026-06-03 | 1.6172 | 0.5539 | Neutral | 0.3155 | Low |
| 2026-06-04 | 1.5746 | 0.5684 | Neutral | 0.3355 | Low |
| 2026-06-05 | 1.5301 | 0.5786 | Bullish | 0.3508 | Low |
| 2026-06-06 | 1.4842 | 0.5943 | Bullish | 0.3733 | Low |
| 2026-06-07 | 1.4376 | 0.6170 | Bullish | 0.4029 | Low |
| 2026-06-08 | 1.3910 | 0.6205 | Bullish | 0.4069 | Low |
| 2026-06-09 | 1.3473 | 0.6262 | Bullish | 0.4138 | Low |
| 2026-06-10 | 1.3071 | 0.6326 | Bullish | 0.4209 | Low |
| 2026-06-11 | 1.3438 | 0.6257 | Bullish | 0.4112 | Low |
