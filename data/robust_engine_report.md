# Robust Astro Engine v1

## Scope

- Target horizon: `7D`
- Rolling training window: `5` calendar years
- Assumption: the current live `ml_dataset.csv` already includes the Raw Astro Recovery substrate, so this engine isolates the simplest current model-side architecture: Astro -> Feature Selection -> ML.

## Comparison

| stack_name | avg_test_score | stability_score | overfit_ratio | avg_test_accuracy | avg_test_return_drawdown_ratio | best_test_score | worst_test_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Robust Astro Engine v1 | 1.6900 | 51.6503 | 0.0005 | 0.5082 | 6.5027 | 7.6393 | -0.6704 |
| Current Production Stack | 0.3794 | 66.6997 | 0.0002 | 0.5082 | 1.7378 | 1.9900 | -0.8125 |

## Robust Engine Summary

- Average test score: `1.6900`
- Stability score: `51.65`
- Overfit ratio: `0.000473`
- Accuracy: `0.5082`
- Return/drawdown ratio: `6.5027`

## Production Delta

- Test score delta vs production: `1.3105`
- Stability score delta vs production: `-15.05`
- Overfit ratio delta vs production: `0.000311`
- Accuracy delta vs production: `0.0000`
- Return/drawdown delta vs production: `4.7650`

## Recommendation

- Robust Astro Engine v1 should replace the current production candidate for further research, because it delivers stronger average out-of-sample score, better average return/drawdown, and a healthier overfit ratio with less stack complexity.
