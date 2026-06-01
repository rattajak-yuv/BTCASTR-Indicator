# Astro Auto-Optimization v2

Analysis only. No engine weights were changed.

## Method
- Discovered weights combine selected-feature ML importance, cross-horizon stability, and direct predictive contribution from `ml_dataset.csv` future-return correlations.
- Aspect current weights come from active `astro_model_config.json` aspect weights.
- Natal-target current weights come from active `astro_model_config.json` target weights.
- Planet current weights use an active rule-mass proxy derived from the live config's rule scores, target coverage, and aspect coverage because the engine does not currently expose a single explicit planet multiplier.
- Post-recovery selected raw aspect features: 5, selected planet features: 15, selected natal-target features: 4.

## Discovered Planet Weights
| component_name | current_weight | discovered_weight | confidence_score | recommendation    |
| -------------- | -------------- | ----------------- | ---------------- | ----------------- |
| Jupiter        | 77.3950        | 120.1149          | 76.10            | increase weight   |
| Pluto          | 102.8125       | 85.8431           | 55.97            | keep near current |
| Saturn         | 91.0225        | 78.6749           | 55.20            | keep near current |
| Uranus         | 76.6875        | 75.9783           | 50.37            | keep near current |
| Neptune        | 58.4300        | 61.6577           | 44.90            | keep near current |
| Mars           | 48.4725        | 41.7000           | 52.29            | keep near current |
| Mercury        | 12.5125        | 7.6395            | 41.56            | decrease weight   |
| Moon           | 10.5788        | 6.3028            | 37.27            | decrease weight   |
| Sun            | 0.0000         | 0.0000            | 25.00            | keep near current |
| Venus          | 0.0000         | 0.0000            | 25.00            | keep near current |

## Discovered Aspect Weights
| component_name | current_weight | discovered_weight | confidence_score | recommendation    |
| -------------- | -------------- | ----------------- | ---------------- | ----------------- |
| conjunction    | 1.0000         | 1.5723            | 67.12            | increase weight   |
| sextile        | 0.5500         | 0.8548            | 54.75            | increase weight   |
| trine          | 0.7500         | 0.7462            | 47.38            | keep near current |
| opposition     | 0.9000         | 0.6738            | 57.50            | decrease weight   |
| square         | 0.8500         | 0.2029            | 47.83            | remove            |

## Discovered Natal-Target Weights
| component_name | current_weight | discovered_weight | confidence_score | recommendation    |
| -------------- | -------------- | ----------------- | ---------------- | ----------------- |
| Sun            | 1.1000         | 2.0892            | 80.26            | increase weight   |
| MC             | 1.2500         | 1.1878            | 80.00            | keep near current |
| Asc            | 1.2500         | 0.9290            | 71.74            | decrease weight   |
| Moon           | 1.0000         | 0.3940            | 64.62            | remove            |

## Strongest Bullish Contributors
| component_type | component_name | bullish_contribution_score | recommendation    |
| -------------- | -------------- | -------------------------- | ----------------- |
| natal_target   | Sun            | 0.8176                     | increase weight   |
| planet         | Jupiter        | 0.8097                     | increase weight   |
| aspect         | conjunction    | 0.6588                     | increase weight   |
| natal_target   | MC             | 0.4941                     | keep near current |
| aspect         | trine          | 0.3322                     | keep near current |

## Strongest Bearish Contributors
| component_type | component_name | bearish_contribution_score | recommendation    |
| -------------- | -------------- | -------------------------- | ----------------- |
| natal_target   | Sun            | 0.8691                     | increase weight   |
| aspect         | conjunction    | 0.7000                     | increase weight   |
| planet         | Saturn         | 0.5304                     | keep near current |
| planet         | Pluto          | 0.2465                     | keep near current |
| aspect         | opposition     | 0.1669                     | decrease weight   |

## Strongest Reversal Contributors
| component_type | component_name | reversal_contribution_score | recommendation    |
| -------------- | -------------- | --------------------------- | ----------------- |
| natal_target   | Sun            | 0.8691                      | increase weight   |
| aspect         | conjunction    | 0.7000                      | increase weight   |
| planet         | Pluto          | 0.5787                      | keep near current |
| natal_target   | MC             | 0.3228                      | keep near current |
| planet         | Jupiter        | 0.2920                      | increase weight   |

## Strongest Volatility Contributors
| component_type | component_name | volatility_contribution_score | recommendation    |
| -------------- | -------------- | ----------------------------- | ----------------- |
| natal_target   | Sun            | 0.8691                        | increase weight   |
| aspect         | conjunction    | 0.6631                        | increase weight   |
| planet         | Uranus         | 0.5122                        | keep near current |
| natal_target   | MC             | 0.4222                        | keep near current |
| natal_target   | Asc            | 0.3455                        | decrease weight   |

## Notes
- Components with low confidence but non-zero discovered weights should be treated as candidates for manual review rather than immediate automation.
- Sun and Venus currently have low-confidence evidence because the live engine config defines orbital settings for them but no active rules, so their compact recovery features mainly act as placeholders until explicit rule coverage exists.
