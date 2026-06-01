# Regime-Aware Astro Summary

Analysis only. No production config or engine weights were changed.

## Regime Methodology
- Bull Market: price > SMA200 by 5% and btc_return_30d > 5%
- Bear Market: price < SMA200 by 5% and btc_return_30d < -5%
- Sideways: all remaining rows
- High Volatility: btc_vol_30d >= 0.036733 (70th percentile)
- Low Volatility: btc_vol_30d <= 0.023022 (30th percentile)

## Regime Coverage
| regime          | rows | start      | end        |
| --------------- | ---- | ---------- | ---------- |
| Bull Market     | 1458 | 2015-07-08 | 2025-10-09 |
| Bear Market     | 655  | 2015-04-24 | 2026-04-04 |
| Sideways        | 2141 | 2014-09-17 | 2026-05-30 |
| High Volatility | 1267 | 2014-10-17 | 2026-03-06 |
| Low Volatility  | 1267 | 2014-12-15 | 2026-05-30 |

## Strongest Planet Signals By Regime
| regime          | strongest_bullish_planet | strongest_bearish_planet | strongest_reversal_planet | strongest_volatility_planet |
| --------------- | ------------------------ | ------------------------ | ------------------------- | --------------------------- |
| Bull Market     | Pluto                    | Pluto                    | Pluto                     | Uranus                      |
| Bear Market     | Pluto                    | Pluto                    | Pluto                     | Uranus                      |
| Sideways        | Jupiter                  | Pluto                    | Pluto                     | Pluto                       |
| High Volatility | Pluto                    | Saturn                   | Pluto                     | Pluto                       |
| Low Volatility  | Pluto                    | Pluto                    | Pluto                     | Pluto                       |

## Bull vs Bear
### Planets
| component | Bull Market | Bear Market | delta  |
| --------- | ----------- | ----------- | ------ |
| Saturn    | 0.6072      | 0.4252      | 0.1820 |
| Uranus    | 0.8370      | 0.6595      | 0.1775 |
| Pluto     | 0.7047      | 0.6907      | 0.0139 |
| Moon      | 0.1412      | 0.1399      | 0.0013 |
| Sun       | 0.1000      | 0.1000      | 0.0000 |

### Aspects
| component   | Bull Market | Bear Market | delta   |
| ----------- | ----------- | ----------- | ------- |
| sextile     | 0.6100      | 0.3256      | 0.2844  |
| square      | 0.5134      | 0.2327      | 0.2806  |
| trine       | 0.3341      | 0.2973      | 0.0367  |
| opposition  | 0.3366      | 0.4836      | -0.1470 |
| conjunction | 0.6000      | 0.9500      | -0.3500 |

### Natal Targets
| component | Bull Market | Bear Market | delta   |
| --------- | ----------- | ----------- | ------- |
| Asc       | 0.6445      | 0.3677      | 0.2768  |
| MC        | 0.5559      | 0.5559      | 0.0000  |
| Moon      | 0.2856      | 0.3217      | -0.0361 |
| Sun       | 0.7145      | 0.8002      | -0.0857 |

## High Volatility vs Low Volatility
### Planets
| component | High Volatility | Low Volatility | delta  |
| --------- | --------------- | -------------- | ------ |
| Mars      | 0.2951          | 0.2560         | 0.0391 |
| Pluto     | 0.7135          | 0.6833         | 0.0301 |
| Neptune   | 0.4272          | 0.4062         | 0.0210 |
| Mercury   | 0.1648          | 0.1578         | 0.0070 |
| Venus     | 0.1000          | 0.1000         | 0.0000 |

### Aspects
| component   | High Volatility | Low Volatility | delta   |
| ----------- | --------------- | -------------- | ------- |
| square      | 0.5827          | 0.2843         | 0.2984  |
| trine       | 0.4562          | 0.4068         | 0.0494  |
| conjunction | 0.6000          | 0.6000         | 0.0000  |
| opposition  | 0.3435          | 0.3706         | -0.0271 |
| sextile     | 0.3799          | 0.6100         | -0.2301 |

### Natal Targets
| component | High Volatility | Low Volatility | delta   |
| --------- | --------------- | -------------- | ------- |
| Moon      | 0.5500          | 0.2144         | 0.3356  |
| Asc       | 0.4339          | 0.3959         | 0.0379  |
| MC        | 0.5559          | 0.5559         | 0.0000  |
| Sun       | 0.5568          | 0.8002         | -0.2434 |

## Dynamic Weight Recommendations
- Bull Market: increase emphasis on natal_target:Asc, aspect:sextile, aspect:square.
- Bear Market: increase emphasis on aspect:conjunction, planet:Neptune, planet:Mars.
- Sideways: increase emphasis on natal_target:Moon, aspect:opposition.
- High Volatility: increase emphasis on aspect:square, natal_target:Moon, aspect:trine.
- Low Volatility: increase emphasis on aspect:sextile.

## Interpretation
- The regime scores blend current ML feature importance, feature stability, selected-feature support, and regime-specific predictive contribution from future-return correlations.
- Use these results to design regime-specific weights in Astro Engine v4 instead of applying one global weight profile to every market environment.
