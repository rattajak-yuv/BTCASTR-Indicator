# Taxonomy Attribution Engine v1

This report explains the current Forecast Taxonomy v3 states using the live future feature frame, ML probabilities, and astro feature inputs.

## State Summary
| taxonomy_v3 | sample_count | average_astro_momentum | average_ml_probability | typical_momentum_range | typical_probability_range | most_influential_planets | most_influential_aspects |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constructive Drift | 227 | 1.1979 | 0.5045 | 0.66 to 1.79 | 0.46 to 0.54 | mars_signal (+0.08), uranus_signal (+0.07), neptune_signal (+0.03) | square_strength (+0.22), opposition_strength (+0.03), trine_strength (+0.03) |
| High Momentum Expansion | 30 | 0.8947 | 0.6125 | 0.59 to 1.14 | 0.60 to 0.63 | saturn_signal (+0.55), mars_signal (+0.42), uranus_signal (+0.17) | trine_strength (+0.79), opposition_strength (+0.44) |
| Tactical Neutral | 81 | 1.4247 | 0.5966 | 0.23 to 1.54 | 0.58 to 0.61 | saturn_signal (+0.45), jupiter_signal (+0.24), neptune_signal (+0.10) | conjunction_strength (+0.45), sextile_strength (+0.39) |
| Defensive / Weak Trend | 17 | 0.2395 | 0.3449 | 0.10 to 0.48 | 0.27 to 0.40 | saturn_signal (+0.55) | opposition_strength (+0.47) |
| High Volatility Risk | 10 | 1.0437 | 0.4247 | 0.53 to 1.12 | 0.42 to 0.43 | neptune_signal (+0.60), jupiter_signal (+0.19) | sextile_strength (+1.11), conjunction_strength (+0.83), trine_strength (+0.00) |

## Per-State Narrative
### Constructive Drift
- Average astro momentum: `1.20`
- Average ML probability: `50.45%`
- Typical momentum range: `0.66 to 1.79`
- Typical probability range: `0.46 to 0.54`
- Top positive astro features: astro_compression_score (+0.24), compression (+0.24), raw_astro_total_strength (+0.19)
- Top negative astro features: raw_astro_directional_signal (-0.01)
- Most influential planets: mars_signal (+0.08), uranus_signal (+0.07), neptune_signal (+0.03)
- Most influential aspects: square_strength (+0.22), opposition_strength (+0.03), trine_strength (+0.03)

### High Momentum Expansion
- Average astro momentum: `0.89`
- Average ML probability: `61.25%`
- Typical momentum range: `0.59 to 1.14`
- Typical probability range: `0.60 to 0.63`
- Top positive astro features: N/A
- Top negative astro features: raw_astro_total_strength (-1.26), reversal (-1.13), astro_reversal_score (-1.13)
- Most influential planets: saturn_signal (+0.55), mars_signal (+0.42), uranus_signal (+0.17)
- Most influential aspects: trine_strength (+0.79), opposition_strength (+0.44)

### Tactical Neutral
- Average astro momentum: `1.42`
- Average ML probability: `59.66%`
- Typical momentum range: `0.23 to 1.54`
- Typical probability range: `0.58 to 0.61`
- Top positive astro features: trend_start (+0.28), astro_trend_start_score (+0.28), astro_momentum_v2 (+0.28)
- Top negative astro features: astro_compression_score (-0.38), compression (-0.38), astro_bearish_score_smooth (-0.22)
- Most influential planets: saturn_signal (+0.45), jupiter_signal (+0.24), neptune_signal (+0.10)
- Most influential aspects: conjunction_strength (+0.45), sextile_strength (+0.39)

### Defensive / Weak Trend
- Average astro momentum: `0.24`
- Average ML probability: `34.49%`
- Typical momentum range: `0.10 to 0.48`
- Typical probability range: `0.27 to 0.40`
- Top positive astro features: N/A
- Top negative astro features: raw_astro_total_strength (-1.44), bullish (-1.39), astro_bullish_score (-1.39)
- Most influential planets: saturn_signal (+0.55)
- Most influential aspects: opposition_strength (+0.47)

### High Volatility Risk
- Average astro momentum: `1.04`
- Average ML probability: `42.47%`
- Typical momentum range: `0.53 to 1.12`
- Typical probability range: `0.42 to 0.43`
- Top positive astro features: raw_astro_event_count (+1.73), house_activation_strength (+1.54), compression (+1.51)
- Top negative astro features: astro_momentum_v2 (-0.34), astro_momentum_v2_smooth (-0.13), raw_astro_directional_signal (-0.02)
- Most influential planets: neptune_signal (+0.60), jupiter_signal (+0.19)
- Most influential aspects: sextile_strength (+1.11), conjunction_strength (+0.83), trine_strength (+0.00)

## Feature Importance Snapshot
| taxonomy_v3 | feature | feature_family | state_mean | overall_mean | differential | zscore_diff | direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constructive Drift | saturn_signal | planet | -0.4999 | -0.3454 | -0.1545 | -0.2472 | negative |
| Constructive Drift | astro_compression_score | core_astro | 0.4269 | 0.3371 | 0.0898 | 0.2394 | positive |
| Constructive Drift | compression | raw_score | 0.4269 | 0.3371 | 0.0898 | 0.2394 | positive |
| Constructive Drift | square_strength | aspect | 1.7805 | 1.3448 | 0.4357 | 0.2175 | positive |
| Constructive Drift | mc_target_strength | natal_target | 1.2968 | 1.0267 | 0.2701 | 0.2048 | positive |
| Constructive Drift | raw_astro_total_strength | core_astro | 7.6631 | 7.0225 | 0.6406 | 0.1927 | positive |
| Constructive Drift | astro_bearish_score_smooth | core_astro | 0.6152 | 0.5213 | 0.0939 | 0.1837 | positive |
| Constructive Drift | sun_target_strength | natal_target | 0.5132 | 0.3366 | 0.1766 | 0.1824 | positive |
| Constructive Drift | astro_trend_end_score | core_astro | 0.9203 | 0.8145 | 0.1058 | 0.1793 | positive |
| Constructive Drift | trend_end | raw_score | 0.9203 | 0.8145 | 0.1058 | 0.1793 | positive |
| Constructive Drift | astro_bearish_score | core_astro | 0.6155 | 0.5256 | 0.0898 | 0.1689 | positive |
| Constructive Drift | bearish | raw_score | 0.6155 | 0.5256 | 0.0898 | 0.1689 | positive |
| Constructive Drift | house_activation_strength | core_astro | 1.9736 | 1.7381 | 0.2355 | 0.1509 | positive |
| Constructive Drift | raw_astro_event_count | core_astro | 4.8899 | 4.6630 | 0.2269 | 0.1437 | positive |
| Constructive Drift | astro_bullish_score_smooth | core_astro | 1.8131 | 1.6958 | 0.1173 | 0.1258 | positive |
| Constructive Drift | bullish | raw_score | 1.8127 | 1.6945 | 0.1182 | 0.1241 | positive |
| Constructive Drift | astro_bullish_score | core_astro | 1.8127 | 1.6945 | 0.1182 | 0.1241 | positive |
| Constructive Drift | reversal | raw_score | 1.3166 | 1.2146 | 0.1020 | 0.1222 | positive |
| Constructive Drift | astro_reversal_score | core_astro | 1.3166 | 1.2146 | 0.1020 | 0.1222 | positive |
| Constructive Drift | astro_volatility_score | core_astro | 1.3079 | 1.2255 | 0.0824 | 0.1195 | positive |
| Constructive Drift | volatility | raw_score | 1.3079 | 1.2255 | 0.0824 | 0.1195 | positive |
| Constructive Drift | mars_signal | planet | 0.1396 | 0.1120 | 0.0277 | 0.0770 | positive |
| Constructive Drift | uranus_signal | planet | 0.8836 | 0.8258 | 0.0577 | 0.0736 | positive |
| Constructive Drift | asc_target_strength | natal_target | 0.9154 | 0.8281 | 0.0873 | 0.0702 | positive |
| Constructive Drift | trend_start | raw_score | 1.2632 | 1.2106 | 0.0526 | 0.0661 | positive |
| Constructive Drift | astro_trend_start_score | core_astro | 1.2632 | 1.2106 | 0.0526 | 0.0661 | positive |
| Constructive Drift | conjunction_strength | aspect | 2.2106 | 2.2875 | -0.0770 | -0.0343 | negative |
| Constructive Drift | opposition_strength | aspect | 0.2565 | 0.2347 | 0.0218 | 0.0323 | positive |
| Constructive Drift | trine_strength | aspect | 0.9350 | 0.9098 | 0.0252 | 0.0295 | positive |
| Constructive Drift | neptune_signal | planet | 0.2737 | 0.2645 | 0.0092 | 0.0274 | positive |
| Constructive Drift | jupiter_signal | planet | 0.7430 | 0.7081 | 0.0350 | 0.0273 | positive |
| Constructive Drift | astro_momentum_v2 | core_astro | 1.1972 | 1.1688 | 0.0284 | 0.0271 | positive |
| Constructive Drift | astro_momentum_v2_smooth | core_astro | 1.1979 | 1.1744 | 0.0234 | 0.0230 | positive |
| Constructive Drift | raw_astro_directional_signal | core_astro | 1.5401 | 1.5650 | -0.0249 | -0.0127 | negative |
| Constructive Drift | moon_target_strength | natal_target | 0.0365 | 0.0357 | 0.0008 | 0.0055 | positive |
| Constructive Drift | sextile_strength | aspect | 0.5071 | 0.5077 | -0.0006 | -0.0015 | negative |
| Constructive Drift | mercury_signal | planet | 0.0000 | 0.0000 | 0.0000 |  | positive |
| Constructive Drift | moon_signal | planet | 0.0000 | 0.0000 | 0.0000 |  | positive |
| Constructive Drift | pluto_signal | planet | 0.0000 | 0.0000 | 0.0000 |  | positive |
| Constructive Drift | sun_signal | planet | 0.0000 | 0.0000 | 0.0000 |  | positive |
