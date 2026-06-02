# Taxonomy Performance Validation Engine v1

## Objective
Validate whether Forecast Taxonomy v3 has enough real forward-return edge to support a later portfolio allocation layer, without changing model logic or taxonomy definitions.

## Current Dashboard Outlook Context
- 30D dominant taxonomy: `High Momentum Expansion`
- 90D dominant taxonomy: `Tactical Neutral`
- 365D dominant taxonomy: `Constructive Drift`

## Validation Answers
A. Predictive power: Yes, the taxonomy separates return outcomes, but the directional semantics are not trustworthy yet.
B. Strongest investable edge: `Defensive / Weak Trend`
C. Tactical Neutral assessment: Tactical Neutral is not fully neutral and may be absorbing mild positive drift. Sample share is `28.30%` of historical taxonomy observations.
D. High Momentum Expansion treatment: `full risk-on`
E. Defensive behavior should be triggered by: `High Volatility Risk`
F. Taxonomy stability: Not stable enough for allocation until the taxonomy semantics are revised.
G. Recommended next step: `revise taxonomy`

## Top Taxonomy Rankings
- Best 30D average return: `Defensive / Weak Trend` at `13.62%`
- Best 60D average return: `Defensive / Weak Trend` at `30.82%`
- Best 90D average return: `Defensive / Weak Trend` at `31.99%`
- Best 30D return/volatility: `Defensive / Weak Trend` at `1.0785`
- Weakest 30D edge: `High Volatility Risk` at `1.23%`

## Strongest Edge Attribution
- Strongest taxonomy: `Defensive / Weak Trend`
- Average astro momentum: `0.24`
- Average ML probability: `34.49%`
- Dominant planets: `saturn_signal (+0.55)`
- Dominant aspects: `opposition_strength (+0.47)`

### Strongest Edge Top Features
| feature | feature_family | differential | zscore_diff | direction |
| --- | --- | --- | --- | --- |
| raw_astro_total_strength | core_astro | -4.7931 | -1.4419 | negative |
| bullish | raw_score | -1.3255 | -1.3915 | negative |
| astro_bullish_score | core_astro | -1.3255 | -1.3915 | negative |
| astro_bullish_score_smooth | core_astro | -1.2637 | -1.3550 | negative |
| trend_start | raw_score | -0.9880 | -1.2419 | negative |

## Exposure Recommendation (Research Only)
| taxonomy_v3 | suggested_exposure | research_only | average_return_30d | average_return_60d | average_return_90d | win_rate_30d | return_volatility_ratio_30d | volatility_30d | sample_share_30d | stability_assessment | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constructive Drift | 50-75% BTC | True | 0.0789 | 0.1193 | 0.2166 | 0.6358 | 0.3877 | 0.2036 | 0.4011 | Relatively stable | The state is positive but not explosive; measured risk-on exposure is more appropriate. |
| Defensive / Weak Trend | 100% BTC | True | 0.1362 | 0.3082 | 0.3199 | 0.8571 | 1.0785 | 0.1262 | 0.0259 | Relatively stable | Empirical returns are strong despite the defensive label. Treat this as a taxonomy-semantics conflict, not as production-ready risk-on guidance. |
| High Momentum Expansion | 50-75% BTC | True | 0.1070 | 0.1616 | 0.1266 | 0.6377 | 0.5365 | 0.1994 | 0.0511 | Fragile | The state is positive but not explosive; measured risk-on exposure is more appropriate. |
| High Volatility Risk | 0-25% BTC | True | 0.0123 | 0.0681 | 0.1186 | 0.4698 | 0.0695 | 0.1763 | 0.2389 | Fragile | Volatility dominates edge; this state should stay capital-preservation oriented. |
| Tactical Neutral | 0-50% BTC | True | 0.0325 | 0.1231 | 0.1993 | 0.4935 | 0.1607 | 0.2024 | 0.2830 | Relatively stable | The edge is mild or tactical, so sizing should stay selective rather than fully committed. |

## Performance Summary
| taxonomy_v3 | horizon_days | sample_count | average_return | win_rate | volatility | return_volatility_ratio | return_drawdown_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constructive Drift | 30 | 1060 | 0.0789 | 0.6358 | 0.2036 | 0.3877 | 0.2385 |
| Constructive Drift | 60 | 1030 | 0.1193 | 0.5495 | 0.3449 | 0.3460 | 0.2322 |
| Constructive Drift | 90 | 1009 | 0.2166 | 0.5877 | 0.4987 | 0.4343 | 0.4216 |
| High Momentum Expansion | 30 | 138 | 0.1070 | 0.6377 | 0.1994 | 0.5365 | 0.5273 |
| High Momentum Expansion | 60 | 138 | 0.1616 | 0.5435 | 0.2801 | 0.5769 | 0.6820 |
| High Momentum Expansion | 90 | 138 | 0.1266 | 0.5870 | 0.2649 | 0.4778 | 0.5685 |
| Tactical Neutral | 30 | 764 | 0.0325 | 0.4935 | 0.2024 | 0.1607 | 0.0758 |
| Tactical Neutral | 60 | 764 | 0.1231 | 0.5942 | 0.3594 | 0.3426 | 0.2273 |
| Tactical Neutral | 90 | 755 | 0.1993 | 0.6199 | 0.5408 | 0.3685 | 0.3400 |
| Defensive / Weak Trend | 30 | 70 | 0.1362 | 0.8571 | 0.1262 | 1.0785 | 1.0918 |
| Defensive / Weak Trend | 60 | 70 | 0.3082 | 1.0000 | 0.1186 | 2.5982 |  |
| Defensive / Weak Trend | 90 | 70 | 0.3199 | 1.0000 | 0.2129 | 1.5029 |  |
| High Volatility Risk | 30 | 645 | 0.0123 | 0.4698 | 0.1763 | 0.0695 | 0.0239 |
| High Volatility Risk | 60 | 645 | 0.0681 | 0.5287 | 0.2873 | 0.2370 | 0.1740 |
| High Volatility Risk | 90 | 645 | 0.1186 | 0.5070 | 0.4082 | 0.2905 | 0.2789 |

## Yearly Stability (30D Forward Returns)
| taxonomy_v3 | year | sample_count | average_forward_return_30d | win_rate_30d |
| --- | --- | --- | --- | --- |
| Constructive Drift | 2019 | 106 | 0.2042 | 0.8585 |
| Constructive Drift | 2020 | 156 | 0.2040 | 0.7692 |
| Constructive Drift | 2021 | 122 | 0.1357 | 0.6967 |
| Constructive Drift | 2022 | 148 | -0.0646 | 0.2838 |
| Constructive Drift | 2023 | 57 | 0.1473 | 0.9123 |
| Constructive Drift | 2024 | 131 | 0.1366 | 0.8779 |
| Constructive Drift | 2025 | 256 | 0.0034 | 0.5078 |
| Constructive Drift | 2026 | 84 | -0.0472 | 0.4643 |
| Defensive / Weak Trend | 2020 | 7 | 0.1693 | 1.0000 |
| Defensive / Weak Trend | 2022 | 9 | 0.3495 | 1.0000 |
| Defensive / Weak Trend | 2023 | 43 | 0.0975 | 0.7674 |
| Defensive / Weak Trend | 2024 | 11 | 0.0917 | 1.0000 |
| High Momentum Expansion | 2024 | 138 | 0.1070 | 0.6377 |
| High Volatility Risk | 2019 | 212 | 0.0172 | 0.4858 |
| High Volatility Risk | 2020 | 55 | -0.0490 | 0.3636 |
| High Volatility Risk | 2021 | 59 | -0.0569 | 0.1695 |
| High Volatility Risk | 2022 | 37 | -0.0241 | 0.3514 |
| High Volatility Risk | 2023 | 234 | 0.0436 | 0.5726 |
| High Volatility Risk | 2024 | 38 | -0.0089 | 0.3421 |
| High Volatility Risk | 2025 | 10 | 0.1355 | 1.0000 |
| Tactical Neutral | 2019 | 47 | 0.1480 | 0.7447 |
| Tactical Neutral | 2020 | 148 | 0.1787 | 0.7365 |
| Tactical Neutral | 2021 | 184 | 0.0081 | 0.4022 |
| Tactical Neutral | 2022 | 171 | -0.0514 | 0.4854 |
| Tactical Neutral | 2023 | 31 | 0.1222 | 0.4839 |
| Tactical Neutral | 2024 | 48 | -0.0562 | 0.1667 |
| Tactical Neutral | 2025 | 99 | -0.0318 | 0.2929 |
| Tactical Neutral | 2026 | 36 | 0.0222 | 0.6667 |
