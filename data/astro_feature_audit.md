# Astro Feature Coverage Audit

## Overview
- Dataset columns inspected: 424
- Valid model features: 399
- Selected features: 136
- Broad astro-related valid features: 383
- Astro-related selected features: 127

## Category Counts
| category              | total_features | selected_features | removed_features |
| --------------------- | -------------- | ----------------- | ---------------- |
| raw_aspect_feature    | 5              | 0                 | 5                |
| planet_feature        | 56             | 7                 | 49               |
| natal_target_feature  | 7              | 1                 | 6                |
| house_feature         | 1              | 0                 | 1                |
| composite_astro_score | 314            | 119               | 195              |
| ml_derived_feature    | 16             | 9                 | 7                |

## Retention Summary
- Raw aspect features retained: 0
- Raw aspect features removed: 5
- Planets represented in valid features: jupiter, mars, mercury, moon, neptune, pluto, saturn, uranus
- Planets represented in selected features: jupiter, pluto, saturn
- Aspects represented in valid features: conjunction, opposition, sextile, square, trine
- Aspects represented in selected features: none
- Natal targets represented in valid features: bearish, bullish, compression, momentum, reversal, trend_end, trend_start, volatility
- Natal targets represented in selected features: bearish, bullish, compression, momentum, reversal, trend_end, trend_start, volatility

## Mapping Verification
- Astro valid features covered by current importance mapping: 372
- Astro valid features missing from current importance mapping: 11
- Astro selected features missing from current importance mapping: 1

Current explainability mapping is correct for prefix-based `astro_*`, `planet_*`, and `aspect_count_*` features, but it is incomplete for astro-related standalone and non-prefix composite columns.

### Missing Mapping Features
trend_start, contraction_score, expansion_score, narrative_score, trigger_score, bearish, bullish, compression, reversal, trend_end, volatility

### Missing Mapping Features That Are Currently Selected
trend_start

## Recommended Fixes Before Astro Auto-Optimization v2
- Expand the astro feature detector to include standalone astro components such as `trend_start`, `trend_end`, `bullish`, `bearish`, `reversal`, `volatility`, and `compression`.
- Treat non-prefix astro composites such as `astro_momentum`, `astro_momentum_smooth`, `expansion_score`, `contraction_score`, `narrative_score`, and `trigger_score` as astro features in the explainability layer.
- Split `house_position` out from raw aspect counts so house coverage is reported separately instead of being mixed into aspect buckets.
- Add a mapping audit step to the auto-optimization workflow so it fails loudly when selected astro-related features are not covered by the explainability taxonomy.
- Interpret near-zero aspect importance carefully: the current selected feature set retains zero raw aspect-count features, so weak aspect results are partly a real feature-coverage outcome, not only a grouping bug.
