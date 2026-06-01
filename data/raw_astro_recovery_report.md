# Raw Astro Feature Recovery v1

## Recovered Feature Coverage
- `ml_dataset.csv` columns: 447
- Recovered aggregate columns present: 19 / 19
- Recovered columns: sun_signal, moon_signal, mercury_signal, venus_signal, mars_signal, jupiter_signal, saturn_signal, uranus_signal, neptune_signal, pluto_signal, conjunction_strength, trine_strength, sextile_strength, square_strength, opposition_strength, sun_target_strength, moon_target_strength, asc_target_strength, mc_target_strength

## Before vs After
| stage           | best_horizon_days | selected_raw_aspect_features | selected_planet_features | selected_natal_target_features | balanced_score | return_drawdown_ratio | accuracy |
| --------------- | ----------------- | ---------------------------- | ------------------------ | ------------------------------ | -------------- | --------------------- | -------- |
| before_recovery | 7                 | 0                            | 7                        | 0                              | 490.8219       | 1243.3207             | 0.5598   |
| after_recovery  | 14                | 5                            | 15                       | 4                              | 319.0593       | 788.4418              | 0.5826   |

## Selection Detail
- Before selected raw aspect features: none
- After selected raw aspect features: conjunction_strength, opposition_strength, sextile_strength, square_strength, trine_strength
- Before selected planet features: planet_bullish_Jupiter, planet_bullish_Saturn, planet_reversal_Pluto, planet_trend_end_Pluto, planet_trend_start_Jupiter, planet_trend_start_Pluto, planet_volatility_Pluto
- After selected planet features: jupiter_signal, mars_signal, mercury_signal, moon_signal, neptune_signal, planet_bullish_Jupiter, planet_trend_end_Pluto, planet_trend_start_Jupiter, planet_trend_start_Pluto, planet_volatility_Pluto, pluto_signal, saturn_signal, sun_signal, uranus_signal, venus_signal
- Before selected natal-target features: none
- After selected natal-target features: asc_target_strength, mc_target_strength, moon_target_strength, sun_target_strength

## Interpretation
- The recovery is successful if compact raw aspect and planet aggregates are present in `ml_dataset.csv` and survive into `selected_features.csv`.
- Balanced-score, return/drawdown, and accuracy changes should be interpreted together with the raw-feature retention counts, because the goal is not just performance but preserving raw astro structure for downstream explainability and optimization.
