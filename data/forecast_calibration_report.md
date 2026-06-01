# Forecast Calibration Engine v1

## Summary
- Historical window classes observed: `7`
- Reference taxonomy classes evaluated: `Strong Bull, Bull Expansion, Accumulation, Neutral, Transition, High Risk, Bearish`
- Pairwise class comparisons with enough data: `63`
- Pairwise comparisons marked statistically distinct: `35`
- Underpowered classes excluded from merge recommendations: `Strong Bull (2 windows)`

## Best Historical Class By Horizon
| window_class | horizon_days | sample_count | average_forward_return | median_forward_return | volatility | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Neutral | 7 | 124 | 0.0349 | 0.0133 | 0.0906 | 0.5484 |
| Neutral | 14 | 124 | 0.0896 | 0.0446 | 0.1347 | 0.7500 |
| Neutral | 30 | 124 | 0.1969 | 0.1866 | 0.2095 | 0.7742 |

## Calibration By Class
| window_class | horizon_days | sample_count | average_forward_return | median_forward_return | volatility | win_rate | max_gain | max_loss | recommended_taxonomy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Neutral | 7 | 124 | 0.0349 | 0.0133 | 0.0906 | 0.5484 | 0.2721 | -0.1767 | Neutral |
| Strong Bull | 7 | 142 | 0.0237 | 0.0149 | 0.0749 | 0.5704 | 0.2533 | -0.1319 | Strong Bull |
| Accumulation | 7 | 54 | 0.0180 | -0.0027 | 0.0931 | 0.5000 | 0.3889 | -0.1457 | Bull Expansion / Accumulation / Transition / High Risk |
| Transition | 7 | 1115 | 0.0180 | 0.0113 | 0.0825 | 0.5821 | 0.4036 | -0.2711 | Bull Expansion / Accumulation / Transition / High Risk |
| Bull Expansion | 7 | 601 | 0.0068 | -0.0001 | 0.0919 | 0.4992 | 0.3775 | -0.3294 | Bull Expansion / Accumulation / Transition / High Risk |
| High Risk | 7 | 474 | -0.0007 | -0.0007 | 0.0854 | 0.4895 | 0.3715 | -0.4525 | Bull Expansion / Accumulation / Transition / High Risk |
| Bearish | 7 | 192 | -0.0104 | -0.0121 | 0.0749 | 0.4062 | 0.1947 | -0.2092 | Bearish |
| Neutral | 14 | 124 | 0.0896 | 0.0446 | 0.1347 | 0.7500 | 0.5187 | -0.1281 | Neutral |
| Strong Bull | 14 | 142 | 0.0424 | 0.0106 | 0.1145 | 0.5493 | 0.3463 | -0.1589 | Strong Bull |
| Transition | 14 | 1108 | 0.0335 | 0.0206 | 0.1222 | 0.6020 | 0.6587 | -0.2991 | Bull Expansion / Accumulation / Transition / High Risk |
| Accumulation | 14 | 54 | 0.0190 | 0.0185 | 0.0914 | 0.5556 | 0.2800 | -0.1860 | Bull Expansion / Accumulation / Transition / High Risk |
| Bull Expansion | 14 | 601 | 0.0160 | -0.0016 | 0.1408 | 0.4992 | 0.4790 | -0.4029 | Bull Expansion / Accumulation / Transition / High Risk |
| High Risk | 14 | 474 | 0.0048 | 0.0027 | 0.1172 | 0.5169 | 0.5844 | -0.4346 | Bull Expansion / Accumulation / Transition / High Risk |
| Bearish | 14 | 192 | -0.0247 | -0.0175 | 0.0959 | 0.4115 | 0.2681 | -0.2201 | Bearish |
| Neutral | 30 | 124 | 0.1969 | 0.1866 | 0.2095 | 0.7742 | 0.6715 | -0.1566 | Neutral |
| Strong Bull | 30 | 142 | 0.1033 | 0.0758 | 0.1978 | 0.6268 | 0.5894 | -0.2029 | Strong Bull |
| Accumulation | 30 | 54 | 0.0845 | 0.0496 | 0.1068 | 0.7963 | 0.5673 | -0.0923 | Bull Expansion / Accumulation / Transition / High Risk |
| Transition | 30 | 1092 | 0.0584 | 0.0247 | 0.1939 | 0.5751 | 1.2039 | -0.3310 | Bull Expansion / Accumulation / Transition / High Risk |
| High Risk | 30 | 474 | 0.0410 | 0.0277 | 0.1757 | 0.5759 | 0.5589 | -0.5131 | Bull Expansion / Accumulation / Transition / High Risk |
| Bull Expansion | 30 | 601 | 0.0210 | -0.0038 | 0.2083 | 0.4875 | 0.5366 | -0.4293 | Bull Expansion / Accumulation / Transition / High Risk |
| Bearish | 30 | 192 | 0.0059 | -0.0460 | 0.1965 | 0.4167 | 0.5415 | -0.2882 | Bearish |

## Pairwise Distinctness
| class_a | class_b | distinct_horizons | tested_horizons | average_p_value | average_effect_size |
| --- | --- | --- | --- | --- | --- |
| Neutral | Bearish | 3 | 3 | 0.0005 | 0.8356 |
| Neutral | High Risk | 3 | 3 | 0.0005 | 0.6531 |
| Strong Bull | Bearish | 3 | 3 | 0.0005 | 0.5287 |
| Neutral | Transition | 3 | 3 | 0.0135 | 0.4544 |
| Accumulation | Bearish | 3 | 3 | 0.0088 | 0.4158 |
| Transition | Bearish | 3 | 3 | 0.0007 | 0.3693 |
| Strong Bull | High Risk | 3 | 3 | 0.0012 | 0.3193 |
| Bull Expansion | Neutral | 3 | 3 | 0.0012 | -0.5579 |
| Transition | High Risk | 2 | 3 | 0.0302 | 0.1848 |
| Strong Bull | Neutral | 2 | 3 | 0.0911 | -0.3240 |
| Accumulation | Neutral | 2 | 3 | 0.0871 | -0.4529 |
| Strong Bull | Bull Expansion | 1 | 3 | 0.0247 | 0.2603 |
| Bull Expansion | Bearish | 1 | 3 | 0.1343 | 0.1924 |
| High Risk | Bearish | 1 | 3 | 0.0683 | 0.1911 |
| Strong Bull | Transition | 1 | 3 | 0.2880 | 0.1242 |
| Bull Expansion | Accumulation | 1 | 3 | 0.4356 | -0.1525 |
| Accumulation | High Risk | 0 | 3 | 0.2022 | 0.1986 |
| Strong Bull | Accumulation | 0 | 3 | 0.4584 | 0.1297 |
| Bull Expansion | High Risk | 0 | 3 | 0.1419 | 0.0224 |
| Accumulation | Transition | 0 | 3 | 0.5622 | 0.0056 |

## Recommended Calibrated Taxonomy
- `Bearish` -> `Bearish`: kept as a standalone class.
- `Transition` -> `Bull Expansion / Accumulation / Transition / High Risk`: merged with Bull Expansion, Accumulation, High Risk.
- `High Risk` -> `Bull Expansion / Accumulation / Transition / High Risk`: merged with Bull Expansion, Accumulation, Transition.
- `Bull Expansion` -> `Bull Expansion / Accumulation / Transition / High Risk`: merged with Accumulation, Transition, High Risk.
- `Accumulation` -> `Bull Expansion / Accumulation / Transition / High Risk`: merged with Bull Expansion, Transition, High Risk.
- `Neutral` -> `Neutral`: kept as a standalone class.
- `Strong Bull` -> `Strong Bull`: kept as a standalone class.
