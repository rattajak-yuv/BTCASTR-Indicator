# Forecast Calibration Engine v1

## Summary
- Historical window classes observed: `7`
- Reference taxonomy classes evaluated: `Strong Bull, Bull Expansion, Accumulation, Neutral, Transition, High Risk, Bearish`
- Pairwise class comparisons with enough data: `63`
- Pairwise comparisons marked statistically distinct: `41`
- Underpowered classes excluded from merge recommendations: `Bearish (4 windows), Strong Bull (2 windows)`

## Best Historical Class By Horizon
| window_class | horizon_days | sample_count | average_forward_return | median_forward_return | volatility | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Accumulation | 7 | 85 | 0.0408 | 0.0307 | 0.0943 | 0.5882 |
| Neutral | 14 | 87 | 0.0526 | 0.0421 | 0.1051 | 0.7931 |
| Transition | 30 | 1094 | 0.0898 | 0.0493 | 0.2086 | 0.6453 |

## Calibration By Class
| window_class | horizon_days | sample_count | average_forward_return | median_forward_return | volatility | win_rate | max_gain | max_loss | recommended_taxonomy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accumulation | 7 | 85 | 0.0408 | 0.0307 | 0.0943 | 0.5882 | 0.3889 | -0.1825 | Bull Expansion / Accumulation / Neutral / Transition |
| Neutral | 7 | 87 | 0.0262 | 0.0229 | 0.0681 | 0.6552 | 0.2180 | -0.1445 | Bull Expansion / Accumulation / Neutral / Transition |
| Bull Expansion | 7 | 683 | 0.0223 | 0.0155 | 0.0928 | 0.5739 | 0.3775 | -0.2872 | Bull Expansion / Accumulation / Neutral / Transition |
| Transition | 7 | 1117 | 0.0159 | 0.0066 | 0.0838 | 0.5515 | 0.4036 | -0.3294 | Bull Expansion / Accumulation / Neutral / Transition |
| High Risk | 7 | 553 | -0.0063 | -0.0018 | 0.0780 | 0.4774 | 0.2423 | -0.4525 | High Risk |
| Strong Bull | 7 | 65 | -0.0176 | -0.0223 | 0.0431 | 0.3385 | 0.0664 | -0.1319 | Strong Bull |
| Bearish | 7 | 110 | -0.0324 | -0.0329 | 0.0836 | 0.3091 | 0.1385 | -0.2711 | Bearish |
| Neutral | 14 | 87 | 0.0526 | 0.0421 | 0.1051 | 0.7931 | 0.2839 | -0.1934 | Bull Expansion / Accumulation / Neutral / Transition |
| Accumulation | 14 | 85 | 0.0511 | 0.0367 | 0.1135 | 0.6706 | 0.3512 | -0.1614 | Bull Expansion / Accumulation / Neutral / Transition |
| Transition | 14 | 1110 | 0.0346 | 0.0226 | 0.1194 | 0.6135 | 0.5979 | -0.2998 | Bull Expansion / Accumulation / Neutral / Transition |
| Bull Expansion | 14 | 683 | 0.0346 | 0.0159 | 0.1432 | 0.5373 | 0.4790 | -0.4029 | Bull Expansion / Accumulation / Neutral / Transition |
| High Risk | 14 | 553 | -0.0035 | -0.0031 | 0.1178 | 0.4738 | 0.6587 | -0.4346 | High Risk |
| Strong Bull | 14 | 65 | -0.0248 | -0.0336 | 0.0560 | 0.3077 | 0.1376 | -0.1329 | Strong Bull |
| Bearish | 14 | 110 | -0.0437 | -0.0385 | 0.1148 | 0.3273 | 0.2681 | -0.2765 | Bearish |
| Transition | 30 | 1094 | 0.0898 | 0.0493 | 0.2086 | 0.6453 | 1.2039 | -0.3310 | Bull Expansion / Accumulation / Neutral / Transition |
| Accumulation | 30 | 85 | 0.0644 | 0.0473 | 0.1447 | 0.6588 | 0.5673 | -0.2615 | Bull Expansion / Accumulation / Neutral / Transition |
| Bull Expansion | 30 | 683 | 0.0499 | 0.0137 | 0.2109 | 0.5344 | 0.5894 | -0.4293 | Bull Expansion / Accumulation / Neutral / Transition |
| Neutral | 30 | 87 | 0.0361 | 0.0149 | 0.1657 | 0.5287 | 0.3885 | -0.1898 | Bull Expansion / Accumulation / Neutral / Transition |
| High Risk | 30 | 553 | 0.0107 | -0.0100 | 0.1685 | 0.4720 | 0.5589 | -0.5131 | High Risk |
| Bearish | 30 | 110 | -0.0349 | -0.0452 | 0.1505 | 0.4091 | 0.2621 | -0.2755 | Bearish |
| Strong Bull | 30 | 65 | -0.0358 | -0.0674 | 0.0995 | 0.3538 | 0.1586 | -0.2029 | Strong Bull |

## Pairwise Distinctness
| class_a | class_b | distinct_horizons | tested_horizons | average_p_value | average_effect_size |
| --- | --- | --- | --- | --- | --- |
| Accumulation | Bearish | 3 | 3 | 0.0005 | 0.7723 |
| Neutral | Bearish | 3 | 3 | 0.0015 | 0.6905 |
| Transition | Bearish | 3 | 3 | 0.0005 | 0.6149 |
| Bull Expansion | Bearish | 3 | 3 | 0.0005 | 0.5242 |
| Accumulation | High Risk | 3 | 3 | 0.0028 | 0.4583 |
| Transition | High Risk | 3 | 3 | 0.0005 | 0.3317 |
| High Risk | Bearish | 3 | 3 | 0.0043 | 0.3156 |
| Bull Expansion | High Risk | 3 | 3 | 0.0005 | 0.2737 |
| Strong Bull | Bull Expansion | 3 | 3 | 0.0012 | -0.4319 |
| Strong Bull | Transition | 3 | 3 | 0.0013 | -0.5105 |
| Strong Bull | Neutral | 3 | 3 | 0.0015 | -0.7088 |
| Strong Bull | Accumulation | 3 | 3 | 0.0005 | -0.7843 |
| Neutral | High Risk | 2 | 3 | 0.0666 | 0.3524 |
| Accumulation | Transition | 1 | 3 | 0.1667 | 0.1028 |
| Neutral | Transition | 1 | 3 | 0.1536 | 0.0051 |
| Strong Bull | High Risk | 1 | 3 | 0.1476 | -0.2080 |
| Strong Bull | Bearish | 0 | 3 | 0.4581 | 0.1309 |
| Accumulation | Neutral | 0 | 3 | 0.4638 | 0.1146 |
| Bull Expansion | Neutral | 0 | 3 | 0.4969 | -0.0352 |
| Bull Expansion | Transition | 0 | 3 | 0.3733 | -0.0392 |

## Recommended Calibrated Taxonomy
- `Bearish` -> `Bearish`: kept as a standalone class.
- `Transition` -> `Bull Expansion / Accumulation / Neutral / Transition`: merged with Bull Expansion, Accumulation, Neutral.
- `Neutral` -> `Bull Expansion / Accumulation / Neutral / Transition`: merged with Bull Expansion, Accumulation, Transition.
- `Bull Expansion` -> `Bull Expansion / Accumulation / Neutral / Transition`: merged with Accumulation, Neutral, Transition.
- `Accumulation` -> `Bull Expansion / Accumulation / Neutral / Transition`: merged with Bull Expansion, Neutral, Transition.
- `High Risk` -> `High Risk`: kept as a standalone class.
- `Strong Bull` -> `Strong Bull`: kept as a standalone class.
