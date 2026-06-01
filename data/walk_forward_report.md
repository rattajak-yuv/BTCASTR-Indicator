# Walk-Forward Validation Framework v1

## Setup

- Target horizon: `7D`
- Rolling training window: `5` calendar years
- Folds evaluated: `8`

## Out-of-Sample Results By Fold

| fold_id | train_start | train_end | test_year | test_balanced_score | test_return_drawdown_ratio | test_total_return | test_max_drawdown | test_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2014-10-17 | 2018-12-31 | 2019 | -0.2676 | 0.2190 | 0.0744 | -0.3396 | 0.5205 |
| 2 | 2015-01-01 | 2019-12-31 | 2020 | 6.6032 | 23.5004 | 3.1874 | -0.1356 | 0.6202 |
| 3 | 2016-01-01 | 2020-12-31 | 2021 | 0.8070 | 2.6336 | 1.4469 | -0.5494 | 0.5589 |
| 4 | 2017-01-01 | 2021-12-31 | 2022 | -0.8931 | -0.8015 | -0.2982 | -0.3721 | 0.4740 |
| 5 | 2018-01-01 | 2022-12-31 | 2023 | -0.0270 | 0.3013 | 0.0418 | -0.1387 | 0.4521 |
| 6 | 2019-01-01 | 2023-12-31 | 2024 | 0.7782 | 2.3666 | 0.4671 | -0.1974 | 0.5027 |
| 7 | 2020-01-01 | 2024-12-31 | 2025 | -0.9066 | -0.7304 | -0.2291 | -0.3137 | 0.4575 |
| 8 | 2021-01-01 | 2025-12-31 | 2026 | -0.6840 | -0.4571 | -0.1303 | -0.2851 | 0.4800 |

## Summary

- Average out-of-sample balanced score: `0.6763`
- Average out-of-sample return/drawdown ratio: `3.3790`
- Average out-of-sample total return: `0.5700`
- Average out-of-sample max drawdown: `-0.2914`
- Average out-of-sample accuracy: `0.5082`
- Median out-of-sample balanced score: `-0.1473`
- Positive balanced-score windows: `3` / `8`
- Positive return windows: `5` / `8`
- Best test period: `2020` with balanced score `6.6032`
- Worst test period: `2025` with balanced score `-0.9066`
- Stability score: `41.62` / 100

## In-Sample vs Out-of-Sample

- Current full-period in-sample balanced score: `271.4743`
- Average rolling train balanced score: `111431.8486`
- Average rolling test balanced score: `0.6763`
- Current full-period in-sample return/drawdown ratio: `677.7616`
- Average rolling train return/drawdown ratio: `423180.7172`
- Average rolling test return/drawdown ratio: `3.3790`
- Current full-period in-sample accuracy: `0.5405`
- Average rolling train accuracy: `0.7872`
- Average rolling test accuracy: `0.5082`

- Rolling train metrics are optimistic by design because each fold is evaluated on the same data used to fit that fold's model. The more important robustness signal is the yearly unseen test-window behavior.

## Overfit Assessment

- Astro Engine V4 is `likely overfit` based on the current walk-forward evidence.
- Average test balanced score is `0.0000`x the rolling train average and `0.0025`x the current full-period in-sample result.
