# Model Health Summary

## Horizon Summary
- Best horizon by balanced score: 60D (balanced_score=40.1558, return/drawdown=90.9479)
- Best horizon by return/drawdown ratio: 14D (return/drawdown=91.4544, balanced_score=36.5078)

## Best Threshold Per Horizon
| Horizon | Current Long | Current Short | Best Long | Best Short | Baseline Balanced Score | Best Balanced Score | Gain vs Baseline | Return/DD Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.0000 | 0.5600 | 0.4400 | 0.5200 | 0.3500 | 2.3541 | 74.4018 | 30.6056 | 198.3718 |
| 7.0000 | 0.5700 | 0.4300 | 0.5200 | 0.3500 | 31.8552 | 490.8219 | 14.4079 | 1243.3207 |
| 14.0000 | 0.5800 | 0.4200 | 0.5200 | 0.3200 | 36.5078 | 111.5245 | 2.0548 | 262.3874 |
| 30.0000 | 0.6000 | 0.4000 | 0.5200 | 0.3200 | 22.0007 | 232.7993 | 9.5815 | 554.6451 |
| 60.0000 | 0.6200 | 0.3800 | 0.5600 | 0.3500 | 40.1558 | 80.0474 | 0.9934 | 183.5331 |
| 90.0000 | 0.6300 | 0.3700 | 0.5200 | 0.3500 | 14.7766 | 61.9779 | 3.1943 | 135.4943 |

## Feature Selection Health
- Selected features: 136
- Selected features flagged as noisy: 33

### Top 20 Robust Features
| Feature | Robustness Score | Mean Importance | Horizon Coverage | Dominant Horizon |
| --- | --- | --- | --- | --- |
| astro_compression_score_roll_min_30 | 2674.6875 | 0.0076 | 0.1667 | macro |
| astro_bullish_score_smooth_roll_max_21 | 2621.5656 | 0.0075 | 0.1667 | macro |
| astro_trend_end_score_roll_max_30 | 2553.8917 | 0.0073 | 0.1667 | macro |
| astro_bullish_score_roll_max_21 | 2421.5447 | 0.0069 | 0.1667 | macro |
| astro_trend_start_score_roll_min_21 | 2373.7474 | 0.0068 | 0.1667 | macro |
| astro_bullish_score_roll_min_21 | 2308.5295 | 0.0066 | 0.1667 | macro |
| astro_bearish_score_roll_min_21 | 2252.3004 | 0.0064 | 0.1667 | macro |
| astro_volatility_score_roll_max_21 | 2130.4837 | 0.0061 | 0.1667 | swing |
| astro_momentum_v2_sma_60 | 2084.4563 | 0.0060 | 0.1667 | swing |
| astro_bullish_score_smooth_sma_30 | 2056.3398 | 0.0059 | 0.1667 | macro |
| astro_bearish_score_smooth_roll_max_30 | 2012.0924 | 0.0057 | 0.1667 | macro |
| astro_reversal_score_roll_max_21 | 1991.2522 | 0.0057 | 0.1667 | swing |
| astro_trend_end_score_chg_30 | 1990.8079 | 0.0057 | 0.1667 | swing |
| astro_bullish_score_ema_60 | 1981.6755 | 0.0057 | 0.1667 | macro |
| btc_return_14d | 1900.4575 | 0.0054 | 0.1667 | timing |
| astro_compression_score_ema_30 | 1896.4350 | 0.0054 | 0.1667 | swing |
| astro_bearish_score_smooth_ema_30 | 1895.3987 | 0.0054 | 0.1667 | macro |
| astro_bearish_score_roll_min_5 | 1877.9180 | 0.0054 | 0.1667 | macro |
| astro_bullish_score_sma_3 | 1876.4507 | 0.0054 | 0.1667 | timing |
| astro_volatility_score_roll_max_30 | 1861.5877 | 0.0053 | 0.1667 | swing |

### Top 20 Noisy Features
| Feature | Std Importance | Mean Importance | Robustness Score | Dominant Horizon |
| --- | --- | --- | --- | --- |
| buy_hold_equity | 0.0096 | 0.0175 | 0.8942 | macro |
| strategy_drawdown | 0.0065 | 0.0102 | 0.7652 | macro |
| strategy_equity | 0.0046 | 0.0087 | 0.8665 | macro |
| planet_bullish_Pluto | 0.0039 | 0.0115 | 1.2013 | macro |
| planet_bearish_Pluto | 0.0034 | 0.0076 | 0.9006 | macro |
| buy_hold_drawdown | 0.0032 | 0.0115 | 1.5034 | macro |
| astro_bullish_score_roll_max_30 | 0.0028 | 0.0081 | 1.1362 | macro |
| astro_trend_start_score_roll_max_21 | 0.0028 | 0.0070 | 1.0813 | swing |
| astro_bullish_score_smooth_roll_max_30 | 0.0028 | 0.0077 | 1.1042 | macro |
| astro_bearish_score_smooth_ema_60 | 0.0026 | 0.0077 | 1.1514 | macro |
| astro_bullish_score_smooth_ema_60 | 0.0025 | 0.0065 | 0.9772 | macro |
| astro_bearish_score_ema_60 | 0.0025 | 0.0075 | 1.1965 | macro |
| planet_trend_start_Pluto | 0.0024 | 0.0066 | 1.2199 | macro |
| astro_compression_score_sma_60 | 0.0023 | 0.0073 | 1.2284 | macro |
| astro_bullish_score_smooth_roll_min_30 | 0.0023 | 0.0076 | 1.2617 | macro |
| astro_trend_start_score_roll_max_10 | 0.0021 | 0.0065 | 1.2645 | swing |
| astro_bullish_score_roll_min_30 | 0.0018 | 0.0083 | 1.6521 | macro |
| astro_bearish_score_smooth_roll_min_21 | 0.0018 | 0.0062 | 1.3152 | macro |
| astro_trend_start_score_sma_30 | 0.0018 | 0.0055 | 1.2101 | swing |
| planet_reversal_Pluto | 0.0017 | 0.0062 | 1.5238 | macro |

## Recommended Next Action
- continue threshold tuning
- Reason: Threshold tuning still shows meaningful headroom across horizons, with median balanced-score improvement of 6.39x and max improvement of 30.61x over the baseline thresholds.
