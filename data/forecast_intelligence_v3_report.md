# Forecast Taxonomy v3

## Objective
This version updates taxonomy names and investor interpretation to match the Astro Alpha Audit evidence, without changing any forecast calculations.

## Core Change
- `False Bull / Exhaustion Risk` was renamed because the alpha audit showed strong positive historical performance rather than exhaustion.
- The interpretation layer now favors evidence-based investor language instead of cautionary naming that conflicts with realized outcomes.

## Mapping
| legacy_taxonomy | taxonomy_v3 | v3_posture | color_hex | average_return_7d | average_return_14d | average_return_30d | win_rate_30d | sample_count_30d | mapping_rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False Bull / Exhaustion Risk | High Momentum Expansion | Momentum Long Bias | #D97706 | 0.0355 | 0.0720 | 0.1682 | 0.7344 | 192 | Alpha audit evidence is positive rather than fragile: 7D=3.55%, 14D=7.20%, 30D=16.82%, with 30D win rate 73.44%. Strength should be respected, not faded. |
| Constructive / Positive Drift | Constructive Drift | Constructive Long Bias | #2E7D32 | 0.0181 | 0.0377 | 0.0807 | 0.6266 | 1098 | Historical outcomes stay constructive across horizons (7D=1.81%, 14D=3.77%, 30D=8.07%) with steady follow-through rather than explosive upside. |
| Neutral / Tactical | Tactical Neutral | Tactical / Wait | #C9A227 | 0.0081 | 0.0112 | 0.0053 | 0.4664 | 699 | Historical returns are mixed to mildly positive (7D=0.81%, 14D=1.12%, 30D=0.53%), so the state is best treated tactically rather than as a high-conviction trend. |
| Bearish | Defensive / Weak Trend | Defensive / Short Bias | #C62828 | -0.0135 | -0.0154 | 0.0331 | 0.5259 | 116 | Historical downside and weak follow-through remain the base case here (7D=-1.35%, 14D=-1.54%, 30D=3.31%). |
| High Risk | High Volatility Risk | Defensive / Volatility Control | #7F1D1D | -0.0011 | 0.0012 | 0.0214 | 0.5000 | 572 | Volatility dominates directional edge in this state. Historical outcomes are weaker and less reliable (7D=-0.11%, 14D=0.12%, 30D=2.14%). |

## Current Read
- Current taxonomy v3: `Constructive Drift`
- Current posture: `Constructive Long Bias`
- Current narrative: From 2026-06-02 to 2026-06-05, the outlook is constructive drift. Historical outcomes stay constructive across horizons (7D=1.81%, 14D=3.77%, 30D=8.07%) with steady follow-through rather than explosive upside. The evidence supports measured long exposure with patience rather than chase behavior.

## Next Windows
- Next positive window: `Constructive Drift` from `2026-06-02` to `2026-06-05`
- Next defensive window: none

## Supporting Audit Snapshots
### Taxonomy Alpha Audit
| label | average_return_7d | average_return_14d | average_return_30d | win_rate_30d | sample_count_30d |
| --- | --- | --- | --- | --- | --- |
| Bearish | -0.0135 | -0.0154 | 0.0331 | 0.5259 | 116 |
| Constructive / Positive Drift | 0.0181 | 0.0377 | 0.0807 | 0.6266 | 1098 |
| False Bull / Exhaustion Risk | 0.0355 | 0.0720 | 0.1682 | 0.7344 | 192 |
| High Risk | -0.0011 | 0.0012 | 0.0214 | 0.5000 | 572 |
| Neutral / Tactical | 0.0081 | 0.0112 | 0.0053 | 0.4664 | 699 |

### Momentum Alpha Audit Highlights
| label | average_return_30d | win_rate_30d | sample_count_30d |
| --- | --- | --- | --- |
| momentum_gt_2_5 | 0.1334 | 0.6812 | 207 |
| momentum_gt_2_0 | 0.1008 | 0.5794 | 428 |
| momentum_recovering_from_low_zone | 0.0830 | 0.6316 | 57 |
| momentum_rolling_over_from_high_zone | 0.0782 | 0.5686 | 51 |
| momentum_slope_turns_positive | 0.0621 | 0.5683 | 139 |

### Turning Point Highlights
| audit_group | label | average_return_30d | win_rate_30d | sample_count_30d |
| --- | --- | --- | --- | --- |
| signal_transition | Bearish -> Neutral | 0.1833 | 0.7586 | 29 |
| turning_point_type | bearish_window_relief | 0.1774 | 0.7586 | 29 |
| turning_point_type | momentum_breakdown_down | 0.1038 | 0.6250 | 24 |
| signal_transition | Neutral -> Bearish | 0.0893 | 0.5625 | 32 |
| turning_point_type | momentum_neutral_cross | 0.0535 | 0.5227 | 44 |
| turning_point_type | signal_flip | 0.0433 | 0.4925 | 67 |

## Source Note
# Astro Alpha Audit & Institutional Backtest Framework v1
