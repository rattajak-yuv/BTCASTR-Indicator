# Overfit Attribution Analysis

## Scope

- Target horizon: `7D`
- Rolling training window: `5` calendar years
- Assumption: the current live `ml_dataset.csv` already embeds Raw Astro Recovery and Regime-Aware V4 score generation. This ablation therefore isolates the model-side layers we can truly toggle from the current repo state.

## Stack Summary

| stack_id | stack_name | avg_train_score | avg_test_score | stability_score | overfit_ratio | avg_test_return_drawdown_ratio | avg_test_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | Astro + Feature Selection | 3571.7933 | 1.6900 | 51.6503 | 0.0005 | 6.5027 | 0.5082 |
| A | Astro Engine only | 3500.3084 | 1.5184 | 52.8400 | 0.0004 | 6.4857 | 0.5088 |
| D | Astro + Feature Selection + Threshold Tuning + Signal Layer | 414.8961 | 0.9654 | 53.6064 | 0.0023 | 3.5449 | 0.5082 |
| C | Astro + Feature Selection + Threshold Tuning | 2223.4130 | 0.7879 | 50.2330 | 0.0004 | 3.1741 | 0.5082 |
| E | Astro + Feature Selection + Threshold Tuning + Signal Layer + Regime Layer | 220.5888 | 0.4654 | 55.3386 | 0.0021 | 1.9569 | 0.5082 |
| F | Full Production Stack | 2344.3435 | 0.3794 | 66.6997 | 0.0002 | 1.7378 | 0.5082 |

## Incremental Attribution

| component_added | from_stack | to_stack | delta_avg_test_score | delta_stability_score | delta_overfit_ratio |
| --- | --- | --- | --- | --- | --- |
| Astro + Feature Selection | A | B | 0.1716 | -1.1897 | 0.0000 |
| Threshold Tuning | B | C | -0.9021 | -1.4173 | -0.0001 |
| Signal Layer | C | D | 0.1775 | 3.3734 | 0.0020 |
| Regime Layer | D | E | -0.4999 | 1.7322 | -0.0002 |
| Full Production Stack | E | F | -0.0860 | 11.3611 | -0.0019 |

## Findings

- Component adding the strongest real out-of-sample alpha: `Astro + Feature Selection` with average test score `1.6900` and stability `51.65`.
- Most overfit stack: `Full Production Stack` with overfit ratio `0.000162`.
- Most stable stack: `Full Production Stack` with stability score `66.70`.
- Components adding real OOS alpha: `C->D`.
- Components increasing overfit pressure: `B->C, D->E, E->F`.
- Recommended minimal robust architecture: `A - Astro Engine only` with average test score `1.5184`, stability `52.84`, and overfit ratio `0.000434`.
