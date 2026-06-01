# Regime-Weighted Astro Engine Report

Production config was left unchanged. The v4 run used a separate regime-aware config.

- Production config: `astro_model_config.json`
- Regime-aware config: `astro_model_config_v4.json`

## Production vs Regime-Aware Astro Engine
| stage           | best_horizon_days | balanced_score | return_drawdown_ratio | accuracy | trades | selected_features |
| --------------- | ----------------- | -------------- | --------------------- | -------- | ------ | ----------------- |
| production      | 14                | 149.9350       | 326.5585              | 0.5513   | 206    | 148               |
| regime_aware_v4 | 7                 | 271.4743       | 677.7616              | 0.5405   | 239    | 153               |

## Interpretation
- Improved over production on balanced score: yes
- Balanced score delta: 121.5394
- Return/drawdown delta: 351.2031
- Accuracy delta: -0.0108
- Trade delta: 33
