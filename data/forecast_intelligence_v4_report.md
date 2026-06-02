# Forecast Taxonomy v4 Semantic Revision

## Objective
Update the taxonomy interpretation layer so the dashboard language matches validated historical behavior, without changing model predictions or forecast calculations.

## Core Semantic Changes
- `High Momentum Expansion` -> `High Conviction Expansion`
- `Tactical Neutral` -> `Transition / Low Conviction`
- `Defensive / Weak Trend` -> `Recovery / Reversal Setup`
- `High Volatility Risk` -> `Volatility Caution`
- `Constructive Drift` remains unchanged

## Mapping Table
| taxonomy_v3 | taxonomy_v4 | meaning | investor_posture | exposure_language | caveat | color_hex | priority | sample_count | sample_share | average_ml_probability | average_astro_momentum | forward_return_30d | forward_return_60d | forward_return_90d | win_rate_30d | stability_assessment | recommended_action | legacy_suggested_label | primary_issue | evidence_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constructive Drift | Constructive Drift | Stable positive drift | Measured long bias | Moderate risk-on | Broadly positive and more stable than the faster expansion state. | #2E7D32 | 5 | 1083 | 0.4011 | 0.5036 | 0.7244 | 0.0789 | 0.1193 | 0.2166 | 0.6223 | Relatively stable | keep | Constructive Drift | State is broadly consistent and relatively stable. | Large sample share 40.11% and positive multi-horizon returns support the current label. |
| High Momentum Expansion | High Conviction Expansion | High-probability expansion window | Risk-on with confirmation | Moderate to aggressive risk-on | Historically positive, but fragile because the sample is concentrated in fewer active years. | #D97706 | 6 | 138 | 0.0511 | 0.7315 | 2.3164 | 0.1070 | 0.1616 | 0.1266 | 0.6377 | Fragile | soft rename | High Conviction Expansion | State looks valid, but stability is fragile. | 30D edge is 10.70% with win rate 63.77%, but active years are only 1. |
| Tactical Neutral | Transition / Low Conviction | Low-conviction transition state | Tactical / wait | Selective exposure only | This bucket absorbs many observations and likely includes multiple sub-regimes. | #C9A227 | 3 | 764 | 0.2830 | 0.6026 | 0.5724 | 0.0325 | 0.1231 | 0.1993 | 0.4935 | Relatively stable | rename or split | Transition / Low Conviction | Large share plus mild positive drift suggest catch-all behavior. | Observation share is 28.30% with 30D return 3.25% and only middling 30D win rate 49.35%. |
| Defensive / Weak Trend | Recovery / Reversal Setup | Historically strong post-stress recovery / reversal setup | Opportunistic accumulation | High opportunity but confirm with price action | Do not treat this as a defensive label; the opportunity profile is strong but can still be noisy intrawindow. | #7C3AED | 4 | 70 | 0.0259 | 0.2863 | -0.1540 | 0.1362 | 0.3082 | 0.3199 | 0.8571 | Relatively stable | rename | Not truly Defensive | Label semantics conflict with strong positive forward returns. | 30D/60D/90D returns are 13.62%/30.82%/31.99% despite low momentum -0.15 and low ML probability 28.63%. |
| High Volatility Risk | Volatility Caution | Volatility dominates directional edge | Capital preservation | Low exposure / defensive | This remains the clearest caution state in the current taxonomy family. | #7F1D1D | 2 | 645 | 0.2389 | 0.3694 | 1.1743 | 0.0123 | 0.0681 | 0.1186 | 0.4698 | Fragile | optional soft rename | Volatility Caution | Semantics are directionally correct. | 30D return is only 1.23% and 30D return/volatility is 0.0695. |

## Current Dashboard Context
- Current taxonomy before JSON refresh: `Constructive Drift`
- 30D dominant outlook after semantic revision: `High Conviction Expansion`
- 90D dominant outlook after semantic revision: `Transition / Low Conviction`
- 365D dominant outlook after semantic revision: `Constructive Drift`

## Why The Major Rename Matters
The regime audit showed that `Defensive / Weak Trend` had strong forward returns and an 85.71% 30D win rate. That is inconsistent with a defensive label, so the new name emphasizes recovery and reversal opportunity instead of weakness.

## Defensive / Weak Trend Deep Dive Snapshot
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

## Transition Snapshot
| from_taxonomy | to_taxonomy | transition_count | average_return_7d | average_return_14d | average_return_30d | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| High Volatility Risk | Constructive Drift | 18 | 0.0540 | 0.1200 | 0.2112 | 0.9444 |
| Defensive / Weak Trend | Constructive Drift | 4 | 0.0513 | 0.1231 | 0.2763 | 1.0000 |
| High Momentum Expansion | Tactical Neutral | 1 | 0.0162 | -0.0662 | 0.0156 | 1.0000 |
| Constructive Drift | High Momentum Expansion | 1 | -0.0261 | -0.0528 | -0.0302 | 0.0000 |
| Tactical Neutral | High Momentum Expansion | 1 | 0.0277 | -0.0900 | -0.1642 | 0.0000 |
| Tactical Neutral | Constructive Drift | 25 | 0.0191 | 0.0298 | 0.0414 | 0.5200 |
| Constructive Drift | Tactical Neutral | 24 | 0.0135 | 0.0327 | 0.0445 | 0.5000 |
| Constructive Drift | High Volatility Risk | 18 | 0.0265 | 0.0162 | 0.1277 | 0.6111 |
| Tactical Neutral | High Volatility Risk | 6 | 0.0303 | 0.0587 | 0.1241 | 0.5000 |
| High Volatility Risk | Tactical Neutral | 6 | -0.0499 | -0.0246 | 0.0514 | 0.3333 |
| Constructive Drift | Defensive / Weak Trend | 4 | 0.0108 | 0.0166 | 0.0832 | 1.0000 |
| High Momentum Expansion | Constructive Drift | 1 | -0.0519 | -0.0594 | -0.0231 | 0.0000 |

## Current Forecast Windows Under v4
| start_date | end_date | taxonomy_v4 | v4_posture | average_confidence | average_ml_probability | taxonomy_v4_exposure_language |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 | 2026-06-04 | Transition / Low Conviction | Tactical / wait | 0.2744 | 0.5308 | Selective exposure only |
| 2026-06-05 | 2026-07-03 | Transition / Low Conviction | Tactical / wait | 0.3697 | 0.5995 | Selective exposure only |
| 2026-07-04 | 2026-07-29 | Transition / Low Conviction | Tactical / wait | 0.2868 | 0.5148 | Selective exposure only |
| 2026-07-30 | 2026-07-30 | Transition / Low Conviction | Tactical / wait | 0.3029 | 0.4185 | Selective exposure only |
| 2026-07-31 | 2026-08-15 | Transition / Low Conviction | Tactical / wait | 0.2645 | 0.5199 | Selective exposure only |
| 2026-08-16 | 2026-09-26 | Transition / Low Conviction | Tactical / wait | 0.3987 | 0.6034 | Selective exposure only |
| 2026-09-27 | 2026-10-05 | Constructive Drift | Measured long bias | 0.3185 | 0.5277 | Moderate risk-on |
| 2026-10-06 | 2026-10-11 | Transition / Low Conviction | Tactical / wait | 0.3973 | 0.5870 | Selective exposure only |
| 2026-10-12 | 2026-10-26 | Transition / Low Conviction | Tactical / wait | 0.3390 | 0.4731 | Selective exposure only |
| 2026-10-27 | 2026-10-27 | Constructive Drift | Measured long bias | 0.3349 | 0.5742 | Moderate risk-on |
| 2026-10-28 | 2026-10-29 | Constructive Drift | Measured long bias | 0.3311 | 0.5674 | Moderate risk-on |
| 2026-10-30 | 2026-11-01 | Transition / Low Conviction | Tactical / wait | 0.3454 | 0.5936 | Selective exposure only |
| 2026-11-02 | 2026-11-14 | Transition / Low Conviction | Tactical / wait | 0.3067 | 0.4734 | Selective exposure only |
| 2026-11-15 | 2026-11-23 | Transition / Low Conviction | Tactical / wait | 0.4211 | 0.4026 | Selective exposure only |
| 2026-11-24 | 2026-11-24 | Transition / Low Conviction | Tactical / wait | 0.3733 | 0.4315 | Selective exposure only |
| 2026-11-25 | 2026-11-25 | Transition / Low Conviction | Tactical / wait | 0.3774 | 0.4297 | Selective exposure only |
| 2026-11-26 | 2027-01-21 | Transition / Low Conviction | Tactical / wait | 0.3348 | 0.5179 | Selective exposure only |
| 2027-01-22 | 2027-02-06 | Transition / Low Conviction | Tactical / wait | 0.2995 | 0.5910 | Selective exposure only |
| 2027-02-07 | 2027-02-21 | Constructive Drift | Measured long bias | 0.3070 | 0.4451 | Moderate risk-on |
| 2027-02-22 | 2027-02-27 | Transition / Low Conviction | Tactical / wait | 0.3305 | 0.4269 | Selective exposure only |

## Supporting Files Refreshed
- `data\dashboard_current_state.json`
- `data\dashboard_summary.json`
- `data\dashboard_timeline.json`
- `data\dashboard_risk_calendar.json`

## Source Note
# Taxonomy Regime Audit Engine v1
