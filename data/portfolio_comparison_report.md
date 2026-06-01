# Portfolio Framework Split v1

## Best Spot Strategy

- Horizon: `7D`
- Total return: `348.8295`
- Max drawdown: `-0.7428`
- Return/drawdown ratio: `469.5853`
- Balanced score: `197.6511`
- Accuracy: `0.5405`
- Trades: `239`
- State mix: `BTC_100:1682, BTC_50:1588, CASH_100:240`

## Best Long/Short Strategy

- Horizon: `7D`
- Total return: `455.7202`
- Max drawdown: `-0.6724`
- Return/drawdown ratio: `677.7616`
- Balanced score: `271.4743`
- Accuracy: `0.5405`
- Trades: `239`
- State mix: `LONG_100:1682, CASH:1588, SHORT_100:240`

## Portfolio Comparison

| portfolio_type | horizon_days | balanced_score | return_drawdown_ratio | total_return | max_drawdown | accuracy | trades |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Best Spot Strategy | 7 | 197.6511 | 469.5853 | 348.8295 | -0.7428 | 0.5405 | 239 |
| Best Long/Short Strategy | 7 | 271.4743 | 677.7616 | 455.7202 | -0.6724 | 0.5405 | 239 |
| Regime-Aware V4 Baseline | 7 | 271.4743 | 677.7616 |  |  | 0.5405 | 239 |

## Recommendation

- Long/Short Portfolio is better suited for the current Astro Engine V4 because the v4 signal stream converts bearish calls into materially stronger return/risk performance than the no-shorting spot mapping.
