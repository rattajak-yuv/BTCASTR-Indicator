# Ensemble Engine V1 Report

Analysis only. `app.py` was not modified.

## Ensemble Horizons
- Horizons used: 7D, 14D, 30D, 60D
- Ensemble signal source: `ml_position_raw` from `ml_predictions.csv`
- Ensemble probability source: weighted average of `ml_prob_up`
- Ensemble confidence: blend of vote strength and probability distance from 0.5

## Horizon Weights
| horizon_days | balanced_score | return_drawdown_ratio | equal_weight | balanced_score_weight | return_dd_weight |
| ------------ | -------------- | --------------------- | ------------ | --------------------- | ---------------- |
| 7            | 271.4743       | 677.7616              | 0.2500       | 0.7071                | 0.7248           |
| 14           | 44.8630        | 103.0349              | 0.2500       | 0.1169                | 0.1102           |
| 30           | 30.9136        | 71.1406               | 0.2500       | 0.0805                | 0.0761           |
| 60           | 36.6770        | 83.1180               | 0.2500       | 0.0955                | 0.0889           |

## Strategy Comparison
| strategy_name                  | strategy_type   | balanced_score | return_drawdown_ratio | max_drawdown | total_return | trades | accuracy |
| ------------------------------ | --------------- | -------------- | --------------------- | ------------ | ------------ | ------ | -------- |
| Regime-Aware V4                | regime_aware_v4 | 271.4743       | 677.7616              | -0.6724      | 455.7202     | 239    | 0.5405   |
| Production                     | production      | 149.9350       | 326.5585              | -0.8744      | 285.5444     | 206    | 0.5513   |
| Balanced Score Weighted Voting | ensemble_v1     | 100.5563       | 236.1068              | -0.7669      | 181.0636     | 224    | 0.5574   |
| Return/DD Weighted Voting      | ensemble_v1     | 100.5563       | 236.1068              | -0.7669      | 181.0636     | 224    | 0.5574   |
| Equal Weight Voting            | ensemble_v1     | 12.9524        | 30.7970               | -0.8701      | 26.7977      | 230    | 0.4536   |

## Portfolio Winners
- Best Trading Portfolio: Regime-Aware V4 (271.4743)
- Best Long-Term Portfolio: Regime-Aware V4 (455.7202)
- Best Combined Portfolio: Regime-Aware V4 (combined rank score 7.00)
