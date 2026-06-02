# Forecast Calibration Engine v1

## Summary
- Historical window classes observed: `7`
- Reference taxonomy classes evaluated: `Strong Bull, Bull Expansion, Accumulation, Neutral, Transition, High Risk, Bearish`
- Pairwise class comparisons with enough data: `63`
- Pairwise comparisons marked statistically distinct: `38`
- Underpowered classes excluded from merge recommendations: `Strong Bull (2 windows)`

## Best Historical Class By Horizon
| window_class | horizon_days | sample_count | average_forward_return | median_forward_return | volatility | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Neutral | 7 | 87 | 0.0494 | 0.0284 | 0.0948 | 0.6782 |
| Neutral | 14 | 87 | 0.1046 | 0.0561 | 0.1361 | 0.8161 |
| Neutral | 30 | 87 | 0.2456 | 0.2279 | 0.1997 | 0.8506 |

## Calibration By Class
| window_class | horizon_days | sample_count | average_forward_return | median_forward_return | volatility | win_rate | max_gain | max_loss | recommended_taxonomy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Neutral | 7 | 87 | 0.0494 | 0.0284 | 0.0948 | 0.6782 | 0.2721 | -0.2155 | Neutral |
| Strong Bull | 7 | 141 | 0.0236 | 0.0145 | 0.0752 | 0.5674 | 0.2533 | -0.1319 | Strong Bull |
| Transition | 7 | 1135 | 0.0185 | 0.0115 | 0.0838 | 0.5736 | 0.4036 | -0.2711 | Transition |
| Accumulation | 7 | 91 | 0.0130 | 0.0007 | 0.0888 | 0.5165 | 0.3889 | -0.1458 | Bull Expansion / Accumulation / High Risk / Bearish |
| Bull Expansion | 7 | 564 | 0.0064 | 0.0004 | 0.0886 | 0.5053 | 0.3775 | -0.3294 | Bull Expansion / Accumulation / High Risk / Bearish |
| Bearish | 7 | 156 | -0.0007 | -0.0097 | 0.0764 | 0.4231 | 0.1947 | -0.2092 | Bull Expansion / Accumulation / High Risk / Bearish |
| High Risk | 7 | 528 | -0.0055 | -0.0041 | 0.0846 | 0.4678 | 0.3715 | -0.4525 | Bull Expansion / Accumulation / High Risk / Bearish |
| Neutral | 14 | 87 | 0.1046 | 0.0561 | 0.1361 | 0.8161 | 0.5187 | -0.1281 | Neutral |
| Strong Bull | 14 | 141 | 0.0433 | 0.0114 | 0.1144 | 0.5532 | 0.3463 | -0.1589 | Strong Bull |
| Transition | 14 | 1128 | 0.0362 | 0.0216 | 0.1256 | 0.6117 | 0.6587 | -0.2991 | Transition |
| Bull Expansion | 14 | 564 | 0.0171 | 0.0026 | 0.1341 | 0.5089 | 0.4712 | -0.4029 | Bull Expansion / Accumulation / High Risk / Bearish |
| Accumulation | 14 | 91 | 0.0055 | -0.0250 | 0.1143 | 0.4066 | 0.4790 | -0.1629 | Bull Expansion / Accumulation / High Risk / Bearish |
| Bearish | 14 | 156 | -0.0050 | -0.0065 | 0.0987 | 0.4679 | 0.2681 | -0.2201 | Bull Expansion / Accumulation / High Risk / Bearish |
| High Risk | 14 | 528 | -0.0055 | -0.0022 | 0.1162 | 0.4848 | 0.5844 | -0.4346 | Bull Expansion / Accumulation / High Risk / Bearish |
| Neutral | 30 | 87 | 0.2456 | 0.2279 | 0.1997 | 0.8506 | 0.6715 | -0.0539 | Neutral |
| Strong Bull | 30 | 141 | 0.1052 | 0.0769 | 0.1972 | 0.6312 | 0.5894 | -0.2029 | Strong Bull |
| Transition | 30 | 1112 | 0.0664 | 0.0347 | 0.1964 | 0.5962 | 1.2039 | -0.3310 | Transition |
| Bearish | 30 | 156 | 0.0592 | 0.0498 | 0.1898 | 0.5962 | 0.5424 | -0.2292 | Bull Expansion / Accumulation / High Risk / Bearish |
| Bull Expansion | 30 | 564 | 0.0230 | 0.0010 | 0.2022 | 0.5053 | 0.5673 | -0.4293 | Bull Expansion / Accumulation / High Risk / Bearish |
| High Risk | 30 | 528 | 0.0170 | 0.0023 | 0.1773 | 0.5019 | 0.5589 | -0.5131 | Bull Expansion / Accumulation / High Risk / Bearish |
| Accumulation | 30 | 91 | -0.0078 | -0.0535 | 0.1647 | 0.3626 | 0.5365 | -0.2171 | Bull Expansion / Accumulation / High Risk / Bearish |

## Pairwise Distinctness
| class_a | class_b | distinct_horizons | tested_horizons | average_p_value | average_effect_size |
| --- | --- | --- | --- | --- | --- |
| Neutral | High Risk | 3 | 3 | 0.0005 | 0.9406 |
| Neutral | Bearish | 3 | 3 | 0.0005 | 0.8398 |
| Neutral | Transition | 3 | 3 | 0.0008 | 0.6054 |
| Strong Bull | High Risk | 3 | 3 | 0.0005 | 0.4188 |
| Strong Bull | Bearish | 3 | 3 | 0.0153 | 0.3362 |
| Transition | High Risk | 3 | 3 | 0.0005 | 0.2947 |
| Strong Bull | Neutral | 3 | 3 | 0.0093 | -0.5034 |
| Bull Expansion | Neutral | 3 | 3 | 0.0005 | -0.7439 |
| Accumulation | Neutral | 3 | 3 | 0.0037 | -0.8534 |
| Strong Bull | Accumulation | 2 | 3 | 0.1086 | 0.3560 |
| Strong Bull | Bull Expansion | 2 | 3 | 0.0190 | 0.2693 |
| Transition | Bearish | 2 | 3 | 0.2262 | 0.2013 |
| Accumulation | Transition | 2 | 3 | 0.1911 | -0.2312 |
| Accumulation | Bearish | 1 | 3 | 0.2244 | -0.0338 |
| High Risk | Bearish | 1 | 3 | 0.4939 | -0.0986 |
| Bull Expansion | Transition | 1 | 3 | 0.0038 | -0.1697 |
| Bull Expansion | High Risk | 0 | 3 | 0.2131 | 0.1160 |
| Strong Bull | Transition | 0 | 3 | 0.3473 | 0.1051 |
| Bull Expansion | Accumulation | 0 | 3 | 0.3703 | 0.0566 |
| Accumulation | High Risk | 0 | 3 | 0.2251 | 0.0565 |

## Recommended Calibrated Taxonomy
- `High Risk` -> `Bull Expansion / Accumulation / High Risk / Bearish`: merged with Bull Expansion, Accumulation, Bearish.
- `Bull Expansion` -> `Bull Expansion / Accumulation / High Risk / Bearish`: merged with Accumulation, High Risk, Bearish.
- `Bearish` -> `Bull Expansion / Accumulation / High Risk / Bearish`: merged with Bull Expansion, Accumulation, High Risk.
- `Accumulation` -> `Bull Expansion / Accumulation / High Risk / Bearish`: merged with Bull Expansion, High Risk, Bearish.
- `Neutral` -> `Neutral`: kept as a standalone class.
- `Strong Bull` -> `Strong Bull`: kept as a standalone class.
- `Transition` -> `Transition`: kept as a standalone class.
