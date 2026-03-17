# P4 Comparison and 2024 Benchmark Summary

## Scope

This note summarizes the work completed on 2026-03-16 for:

- coefficient comparison in the `p4` inference workflow
- multi-target-aware evaluation
- summary-column upgrades in `p4_0_generate_cv_predictions.py`
- 2024 direct-binding benchmark compound extraction
- Top-K benchmark evaluation on the extracted 2024 compounds

Primary script:

- `/mnt/d/stp_proj_v3/script/p4_0_generate_cv_predictions.py`

## 1. Four-Coefficient Pilot Comparison

The following four coefficient files were compared under matched `p4` pilot settings:

- `K1 + stp2014`
- `K5 + stp2014`
- `K1 + stp2019`
- `K5 + stp2019`

Pilot output directory:

- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316`

Evaluation design:

- same sampled queries across all cases
- `20` molecules per size bin
- random seed `20260316`
- all `4524` human targets retained per query for unbiased ranking evaluation

Main pilot result:

- overall winner in the small pilot: `K1 + stp2014`
- `K5 + stp2014` remained competitive and performed slightly better on some broader cutoffs

Saved files:

- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316/comparison_summary.csv`
- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316/comparison_by_size_bin.csv`
- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316/comparison_summary.md`

## 2. K5 Layout Support Added to p4_0

During the coefficient comparison, `K5 paired_sum` failed because `p4_0_generate_cv_predictions.py` only supported:

- `feature_dim == 6`
- or `feature_dim == 2 * K`

The project `K5 paired_sum` JSON files actually store:

- `intercept`
- `a_shape`
- `a_fp`

which means:

- `feature_dim == 2`

The script was patched to support this layout by using a two-feature proxy from the selected top-K candidate set:

- max selected 3D similarity
- max selected 2D similarity

This patch enabled direct `p4` comparison of the K5 coefficient files.

## 3. Large Comparison: K1 vs K5 under stp2014

After the pilot, a larger matched comparison was run for:

- `K1 + stp2014`
- `K5 + stp2014`

Large comparison output directory:

- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large`

Evaluation design:

- same sampled queries across both cases
- `300` molecules per size bin
- total queries: `2100`
- all `4524` human targets retained per query

### Large-comparison result

The large run overturned the small-pilot result.

Final winner:

- `K5 + stp2014`

Saved comparison files:

- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large/comparison_summary.csv`
- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large/comparison_by_size_bin.csv`
- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large/comparison_summary.md`

### Key metrics

`K5 + stp2014`

- Top-1: `0.01904761904761905`
- Top-5: `0.051904761904761905`
- Top-10: `0.08238095238095237`
- Top-15: `0.1`
- Top-100: `0.3485714285714286`
- MRR: `0.044221609645171654`
- median best true rank: `189.5`

`K1 + stp2014`

- Top-1: `0.011904761904761904`
- Top-5: `0.04238095238095238`
- Top-10: `0.07095238095238095`
- Top-15: `0.09142857142857143`
- Top-100: `0.3352380952380952`
- MRR: `0.03630004822622972`
- median best true rank: `194.5`

### Practical conclusion

Under the current `p4` inference setup, the best overall configuration is:

- `K5 + stp2014`

## 4. Query Summary Upgrade for Multi-Target Evaluation

The query summary logic in `p4_0_generate_cv_predictions.py` was extended.

New columns added:

- `true_targets_top1_count`
- `true_targets_top5_count`
- `true_targets_top10_count`
- `true_targets_top15_count`
- `true_targets_top100_count`
- `top5_hit`
- `top15_hit`

Important point:

- the existing `top1_hit`, `top10_hit`, and `top100_hit` definitions were not changed
- they still depend on `best_true_rank`

Therefore:

- old and new `top1/top10/top100` rates are identical
- the new columns only add count-based multi-target information

## 5. Lightweight Resummarization Support

`p4_0_generate_cv_predictions.py` was extended with:

- `--resummarize-only <existing_parquet>`

Purpose:

- rebuild query summaries and precision sidecars from an existing parquet
- avoid rerunning expensive leave-one-out inference

Because overwriting some existing sidecar files under `/mnt/d` caused permission friction, updated summaries were also written with `_v2` suffixes for inspection.

Examples:

- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large/k1_stp2014_large.query_summary_v2.csv`
- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large/k5_stp2014_large.query_summary_v2.csv`

## 6. Multi-Target Comparison: K1 vs K5

Multi-target-aware metrics were computed from the `_v2` summaries.

Saved files:

- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large/multitarget_comparison_overall.csv`
- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large/multitarget_comparison_by_size_bin.csv`
- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large/multitarget_comparison_pivot.csv`
- `/mnt/d/stp_proj_v3/results/p4_coef_compare_20260316_large/multitarget_comparison_summary.md`

### Multi-target conclusion

`K5 + stp2014` beat `K1 + stp2014` at every tested cutoff for:

- query-level hit rate
- mean true-target count@K
- mean true-target recall@K

Examples:

At `K = 10`

- `K1` mean true-target recall@10: `0.05955681531872008`
- `K5` mean true-target recall@10: `0.06622383084287846`

At `K = 100`

- `K1` mean true-target recall@100: `0.291459821483631`
- `K5` mean true-target recall@100: `0.3024401533211057`

This confirms that `K5 + stp2014` is also better under a multi-target evaluation lens.

## 7. 2024 Direct-Binding Compound Selection

The user asked for compounds satisfying:

- direct binders
- `Ki`, `KD`, `IC50`, or `EC50` < `1 nM`
- confidence score > `3`
- at least two different human targets
- `< 80` heavy atoms
- `SINGLE PROTEIN` or `PROTEIN COMPLEX`
- publication year `2024`

The dataset used for extraction:

- `/mnt/d/stp_proj_v3/features_store/chembl36_stp_training_set.parquet`

The extraction was rerun under the accepted assumption that this parquet already reflects a direct-binding filtered source.

Output directory:

- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_selection_20260316`

Saved files:

- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_selection_20260316/eligible_compounds_all.csv`
- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_selection_20260316/selected_compounds_up_to_500.csv`
- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_selection_20260316/selected_compound_target_rows_up_to_500.csv`
- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_selection_20260316/selection_summary.json`

### Extraction result

Although `500` compounds were requested, only `70` compounds satisfy the full condition set locally.

Selection summary:

- requested compounds: `500`
- eligible compounds: `70`
- selected compounds: `70`
- eligible rows: `146`
- selected rows: `146`

## 8. 2024 Direct-Binding Benchmark with K5 + stp2014

The extracted `70` compounds were then benchmarked with the current best `p4` configuration:

- `K5 + stp2014`

Benchmark output directory:

- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_benchmark_20260316`

Saved files:

- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_benchmark_20260316/predictions_top100.parquet`
- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_benchmark_20260316/query_summary.csv`
- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_benchmark_20260316/size_bin_precision.csv`
- `/mnt/d/stp_proj_v3/results/p4_2024_direct_binding_benchmark_20260316/benchmark_summary.json`

### Benchmark result

- selected compounds: `70`
- output rows: `7000`
- Top-1: `0.014285714285714285`
- Top-5: `0.07142857142857142`
- Top-10: `0.14285714285714285`
- Top-15: `0.17142857142857143`
- Top-100: `0.37142857142857144`

Multi-target summary:

- mean true-target count@10: `0.24285714285714285`
- mean true-target count@15: `0.32857142857142857`
- mean true-target count@100: `1.1428571428571428`
- mean true-target recall@10: `0.0961262282690854`
- mean true-target recall@15: `0.12985638699924412`
- mean true-target recall@100: `0.37142857142857144`

## 9. Simple 2024 Potency Count

Using the same direct-binding source parquet, the count for:

- publication year `2024`
- `Ki`, `KD`, `IC50`, or `EC50`
- activity `< 1 nM`

was also computed.

Result:

- unique compounds: `1011`
- rows: `1123`

Type breakdown:

- `IC50`: `752`
- `Ki`: `340`
- `EC50`: `31`
- `KD`: `0`

## 10. Current Recommended Configuration

For current `p4` work in this repository, the preferred inference setup is:

- coefficient: `K5 paired_sum`
- threshold preset: `stp2014`

Reason:

- best large-sample query-level performance
- best large-sample multi-target performance
- best current practical choice among the evaluated options
