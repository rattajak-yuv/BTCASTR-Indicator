# Aspect Optimization Report

Aspect-only experimental config was used. Planet weights and natal target weights were left unchanged.

- Production config: `astro_model_config.json`
- Experimental config: `astro_model_config_experimental.json`

## Production vs Aspect Optimized
| stage            | best_horizon_days | balanced_score | return_drawdown_ratio | accuracy | selected_features | selected_aspect_features | trades |
| ---------------- | ----------------- | -------------- | --------------------- | -------- | ----------------- | ------------------------ | ------ |
| production       | 14                | 319.0593       | 788.4418              | 0.5826   | 147               | 5                        | 222    |
| aspect_optimized | 3                 | 150.5895       | 379.1789              | 0.5356   | 152               | 5                        | 244    |

## Aspect Feature Retention
- Production selected aspect features: conjunction_strength, opposition_strength, sextile_strength, square_strength, trine_strength
- Optimized selected aspect features: conjunction_strength, opposition_strength, sextile_strength, square_strength, trine_strength

## Interpretation
- Use the optimized run only as evidence for whether discovered aspect weights improve the ML research pipeline.
- No production config was overwritten in this phase.
