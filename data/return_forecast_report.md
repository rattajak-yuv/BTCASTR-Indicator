# Return Forecast Engine v1

## Best Forecast Horizon

- Best forecast horizon by balanced score: `60D`
- Balanced score: `4.8458`
- Return/drawdown ratio: `15.3584`
- Total return: `8.1836`
- Max drawdown: `-0.5328`
- Trades: `190`
- MAE: `0.3335`
- RMSE: `0.4533`
- Directional accuracy: `0.5213`
- Average confidence score: `0.7123`

## Forecast Horizon Ranking

| strategy_name | balanced_score | return_drawdown_ratio | total_return | max_drawdown | trades | mae | rmse | directional_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Return Forecast 60D | 4.8458 | 15.3584 | 8.1836 | -0.5328 | 190 | 0.3335 | 0.4533 | 0.5213 |
| Return Forecast 30D | 1.9494 | 7.6645 | 3.6937 | -0.4819 | 180 | 0.2161 | 0.2933 | 0.5208 |
| Return Forecast 7D | 0.0431 | 3.4041 | 1.1447 | -0.3363 | 374 | 0.0743 | 0.1029 | 0.5111 |
| Return Forecast 14D | -0.4027 | 1.8476 | 0.7768 | -0.4204 | 310 | 0.1185 | 0.1616 | 0.5114 |

## Portfolio Comparison

| strategy_name | horizon_days | balanced_score | return_drawdown_ratio | total_return | max_drawdown | trades | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Return Forecast 60D | 60 | 4.8458 | 15.3584 | 8.1836 | -0.5328 | 190 | 0.5213 |
| Production | 14 | 149.9350 | 326.5585 | 285.5444 | -0.8744 | 206 | 0.5513 |
| Regime-Aware V4 | 7 | 271.4743 | 677.7616 | 455.7202 | -0.6724 | 239 | 0.5405 |
| Buy & Hold | 60 | 59.3695 | 133.3491 | 111.2118 | -0.8340 | 0 |  |

## Conclusion

- Versus Regime-Aware V4: balanced score delta = `-266.6285`
- Versus Production: balanced score delta = `-145.0891`
- Versus Buy & Hold on the same period: balanced score delta = `-54.5236`
