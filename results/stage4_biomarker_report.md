# Stage 4 Biomarker Candidate Report

Main analysis: Dataset 1 subject-level PD/Healthy comparison.

Dataset 2 is only noted as a reference prepared file and is not used in the main biomarker ranking.
Dataset 2 reference status: `results/model_ready_dataset2_prepared.csv` exists.

## Subject-level audit

- Subject count: `252`
- Class distribution: `{0: 64, 1: 188}`
- Acoustic feature count: `752`
- Missing value count: `0`
- Constant feature count: `0`
- Near-constant feature count (`std <= 1e-12`): `35`
- Near-constant features in Top 50: `0`.
- Near-constant features selected by L1 at least once: `18`.
- Main features exclude `id`, `gender`, `class`, and `sex_male`.

## Methods

- Dataset 1 records were aggregated to one row per subject before all statistical tests and models.
- Univariate screening used Mann-Whitney U, Welch t-test, FDR-BH correction, Cohen's d, and Cliff's delta.
- L1 stability selection used subject-level StratifiedKFold; scaling and L1 Logistic were fit only inside training folds.
- ExtraTrees and permutation importance used subject-level StratifiedKFold; permutation importance was computed on held-out folds.
- XGBoost and SHAP are supplemental evidence only; they are not included in the mean-rank formula and do not change Top 10/20/50 membership.
- These features are exploratory biomarker candidates for modeling interpretation and require independent validation.
- Some near-constant features were selected by L1, so L1 stability is treated as one evidence source rather than a standalone final conclusion.
- Permutation importance is sparse (`13` of `752` features have positive mean importance), which may reflect substitution effects among high-dimensional correlated acoustic features.
- The final Top 20 are candidate voice biomarker features, not clinical causal mechanisms.

## Top 10 biomarker candidates

| feature | category | mean_rank | rank_score | q_mannwhitney | cohens_d | cliffs_delta | selection_frequency | permutation_importance_mean | tree_importance_mean | xgboost_importance_mean | shap_mean_abs_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_delta_delta_log_energy | Delta / Delta-delta | 5.66667 | 0.17647 | 7.682e-10 | 0.91226 | 0.59666 | 1.00000 | 0.00833 | 0.00690 | 0.01659 | 0.15090 |
| std_6th_delta_delta | Delta / Delta-delta | 15.50000 | 0.06452 | 1.750e-08 | 0.94325 | 0.53524 | 0.60000 | 0.00833 | 0.00481 | 0.00536 | 0.06874 |
| std_8th_delta | Delta / Delta-delta | 20.66667 | 0.04839 | 1.750e-08 | 0.97263 | 0.53358 | 0.40000 | 0.00769 | 0.00500 | 0.00675 | 0.04442 |
| std_9th_delta_delta | Delta / Delta-delta | 70.58333 | 0.01417 | 1.693e-09 | 1.00852 | 0.57463 | 0.80000 | 0.00000 | 0.00612 | 0.01853 | 0.16342 |
| std_7th_delta_delta | Delta / Delta-delta | 79.41667 | 0.01259 | 6.281e-09 | 0.97751 | 0.55244 | 0.60000 | 0.00000 | 0.00415 | 0.01263 | 0.05110 |
| tqwt_TKEO_std_dec_12 | TQWT | 80.83333 | 0.01237 | 3.179e-08 | -0.94105 | -0.51779 | 0.60000 | 0.00000 | 0.00521 | 0.01127 | 0.00855 |
| tqwt_minValue_dec_12 | TQWT | 85.00000 | 0.01176 | 5.174e-08 | 1.09137 | 0.50482 | 0.40000 | 0.00000 | 0.00631 | 0.00623 | 0.01637 |
| std_9th_delta | Delta / Delta-delta | 85.66667 | 0.01167 | 2.270e-08 | 0.93530 | 0.52859 | 0.40000 | 0.00000 | 0.00470 | 0.01033 | 0.01451 |
| mean_MFCC_2nd_coef | MFCC | 86.16667 | 0.01161 | 5.172e-08 | 1.04153 | 0.50540 | 0.40000 | 0.00000 | 0.00543 | 0.00809 | 0.02309 |
| tqwt_TKEO_std_dec_11 | TQWT | 95.08333 | 0.01052 | 1.716e-07 | -0.92385 | -0.48338 | 0.40000 | 0.00000 | 0.00447 | 0.00310 | 0.00253 |

## Top 20 biomarker candidates

| feature | category | mean_rank | rank_score | q_mannwhitney | cohens_d | cliffs_delta | selection_frequency | permutation_importance_mean | tree_importance_mean | xgboost_importance_mean | shap_mean_abs_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_delta_delta_log_energy | Delta / Delta-delta | 5.66667 | 0.17647 | 7.682e-10 | 0.91226 | 0.59666 | 1.00000 | 0.00833 | 0.00690 | 0.01659 | 0.15090 |
| std_6th_delta_delta | Delta / Delta-delta | 15.50000 | 0.06452 | 1.750e-08 | 0.94325 | 0.53524 | 0.60000 | 0.00833 | 0.00481 | 0.00536 | 0.06874 |
| std_8th_delta | Delta / Delta-delta | 20.66667 | 0.04839 | 1.750e-08 | 0.97263 | 0.53358 | 0.40000 | 0.00769 | 0.00500 | 0.00675 | 0.04442 |
| std_9th_delta_delta | Delta / Delta-delta | 70.58333 | 0.01417 | 1.693e-09 | 1.00852 | 0.57463 | 0.80000 | 0.00000 | 0.00612 | 0.01853 | 0.16342 |
| std_7th_delta_delta | Delta / Delta-delta | 79.41667 | 0.01259 | 6.281e-09 | 0.97751 | 0.55244 | 0.60000 | 0.00000 | 0.00415 | 0.01263 | 0.05110 |
| tqwt_TKEO_std_dec_12 | TQWT | 80.83333 | 0.01237 | 3.179e-08 | -0.94105 | -0.51779 | 0.60000 | 0.00000 | 0.00521 | 0.01127 | 0.00855 |
| tqwt_minValue_dec_12 | TQWT | 85.00000 | 0.01176 | 5.174e-08 | 1.09137 | 0.50482 | 0.40000 | 0.00000 | 0.00631 | 0.00623 | 0.01637 |
| std_9th_delta | Delta / Delta-delta | 85.66667 | 0.01167 | 2.270e-08 | 0.93530 | 0.52859 | 0.40000 | 0.00000 | 0.00470 | 0.01033 | 0.01451 |
| mean_MFCC_2nd_coef | MFCC | 86.16667 | 0.01161 | 5.172e-08 | 1.04153 | 0.50540 | 0.40000 | 0.00000 | 0.00543 | 0.00809 | 0.02309 |
| tqwt_TKEO_std_dec_11 | TQWT | 95.08333 | 0.01052 | 1.716e-07 | -0.92385 | -0.48338 | 0.40000 | 0.00000 | 0.00447 | 0.00310 | 0.00253 |
| std_10th_delta | Delta / Delta-delta | 96.08333 | 0.01041 | 1.100e-07 | 0.86292 | 0.49136 | 0.40000 | 0.00000 | 0.00381 | 0.00473 | 0.02457 |
| tqwt_maxValue_dec_11 | TQWT | 97.41667 | 0.01027 | 9.317e-08 | -1.07366 | -0.49518 | 0.20000 | 0.00000 | 0.00453 | 0.00517 | 0.00470 |
| std_7th_delta | Delta / Delta-delta | 97.66667 | 0.01024 | 2.997e-08 | 0.90159 | 0.52028 | 0.20000 | 0.00000 | 0.00428 | 0.00662 | 0.08221 |
| tqwt_kurtosisValue_dec_36 | TQWT | 101.50000 | 0.00985 | 5.239e-07 | 0.78988 | 0.46509 | 0.40000 | 0.00000 | 0.00435 | 0.00581 | 0.01807 |
| std_6th_delta | Delta / Delta-delta | 101.91667 | 0.00981 | 7.504e-08 | 0.89118 | 0.49867 | 0.20000 | 0.00000 | 0.00444 | 0.00763 | 0.03858 |
| tqwt_kurtosisValue_dec_27 | TQWT | 103.75000 | 0.00964 | 1.979e-05 | -1.02698 | -0.39287 | 0.80000 | 0.00000 | 0.00569 | 0.00130 | 0.00276 |
| tqwt_energy_dec_12 | TQWT | 112.58333 | 0.00888 | 2.934e-06 | -0.84797 | -0.43318 | 0.20000 | 0.00000 | 0.00477 | 0.00045 | 0.00066 |
| DFA | Nonlinear | 118.41667 | 0.00844 | 5.190e-06 | 0.77895 | 0.42138 | 0.60000 | 0.00000 | 0.00195 | 0.00779 | 0.02307 |
| tqwt_entropy_shannon_dec_17 | TQWT | 128.08333 | 0.00781 | 5.917e-06 | -0.77110 | -0.41822 | 0.20000 | 0.00000 | 0.00236 | 0.00227 | 0.00365 |
| tqwt_stdValue_dec_7 | TQWT | 131.66667 | 0.00759 | 4.148e-06 | -0.77837 | -0.42670 | 0.20000 | 0.00000 | 0.00188 | 0.00000 | 0.00000 |

## Top 50 biomarker candidates

| feature | category | mean_rank | rank_score |
| --- | --- | --- | --- |
| std_delta_delta_log_energy | Delta / Delta-delta | 5.66667 | 0.17647 |
| std_6th_delta_delta | Delta / Delta-delta | 15.50000 | 0.06452 |
| std_8th_delta | Delta / Delta-delta | 20.66667 | 0.04839 |
| std_9th_delta_delta | Delta / Delta-delta | 70.58333 | 0.01417 |
| std_7th_delta_delta | Delta / Delta-delta | 79.41667 | 0.01259 |
| tqwt_TKEO_std_dec_12 | TQWT | 80.83333 | 0.01237 |
| tqwt_minValue_dec_12 | TQWT | 85.00000 | 0.01176 |
| std_9th_delta | Delta / Delta-delta | 85.66667 | 0.01167 |
| mean_MFCC_2nd_coef | MFCC | 86.16667 | 0.01161 |
| tqwt_TKEO_std_dec_11 | TQWT | 95.08333 | 0.01052 |
| std_10th_delta | Delta / Delta-delta | 96.08333 | 0.01041 |
| tqwt_maxValue_dec_11 | TQWT | 97.41667 | 0.01027 |
| std_7th_delta | Delta / Delta-delta | 97.66667 | 0.01024 |
| tqwt_kurtosisValue_dec_36 | TQWT | 101.50000 | 0.00985 |
| std_6th_delta | Delta / Delta-delta | 101.91667 | 0.00981 |
| tqwt_kurtosisValue_dec_27 | TQWT | 103.75000 | 0.00964 |
| tqwt_energy_dec_12 | TQWT | 112.58333 | 0.00888 |
| DFA | Nonlinear | 118.41667 | 0.00844 |
| tqwt_entropy_shannon_dec_17 | TQWT | 128.08333 | 0.00781 |
| tqwt_stdValue_dec_7 | TQWT | 131.66667 | 0.00759 |
| tqwt_kurtosisValue_dec_18 | TQWT | 135.58333 | 0.00738 |
| tqwt_kurtosisValue_dec_35 | TQWT | 136.41667 | 0.00733 |
| apq11Shimmer | Shimmer | 137.33333 | 0.00728 |
| tqwt_kurtosisValue_dec_28 | TQWT | 138.33333 | 0.00723 |
| tqwt_kurtosisValue_dec_26 | TQWT | 141.08333 | 0.00709 |
| tqwt_kurtosisValue_dec_20 | TQWT | 141.83333 | 0.00705 |
| std_8th_delta_delta | Delta / Delta-delta | 145.33333 | 0.00688 |
| tqwt_stdValue_dec_12 | TQWT | 145.83333 | 0.00686 |
| tqwt_maxValue_dec_12 | TQWT | 148.58333 | 0.00673 |
| tqwt_entropy_log_dec_12 | TQWT | 149.16667 | 0.00670 |
| tqwt_entropy_shannon_dec_12 | TQWT | 150.00000 | 0.00667 |
| std_10th_delta_delta | Delta / Delta-delta | 152.25000 | 0.00657 |
| tqwt_entropy_log_dec_33 | TQWT | 152.91667 | 0.00654 |
| tqwt_minValue_dec_11 | TQWT | 154.33333 | 0.00648 |
| tqwt_stdValue_dec_11 | TQWT | 154.58333 | 0.00647 |
| std_11th_delta_delta | Delta / Delta-delta | 154.91667 | 0.00646 |
| mean_delta_log_energy | Delta / Delta-delta | 156.41667 | 0.00639 |
| tqwt_maxValue_dec_13 | TQWT | 156.66667 | 0.00638 |
| tqwt_entropy_shannon_dec_13 | TQWT | 156.75000 | 0.00638 |
| tqwt_entropy_shannon_dec_11 | TQWT | 156.83333 | 0.00638 |
| tqwt_entropy_shannon_dec_35 | TQWT | 157.00000 | 0.00637 |
| tqwt_stdValue_dec_13 | TQWT | 157.33333 | 0.00636 |
| std_delta_log_energy | Delta / Delta-delta | 157.75000 | 0.00634 |
| tqwt_entropy_log_dec_11 | TQWT | 157.91667 | 0.00633 |
| tqwt_minValue_dec_13 | TQWT | 158.08333 | 0.00633 |
| tqwt_TKEO_mean_dec_12 | TQWT | 159.08333 | 0.00629 |
| tqwt_energy_dec_15 | TQWT | 160.75000 | 0.00622 |
| tqwt_entropy_log_dec_35 | TQWT | 162.58333 | 0.00615 |
| std_11th_delta | Delta / Delta-delta | 163.91667 | 0.00610 |
| std_12th_delta_delta | Delta / Delta-delta | 165.50000 | 0.00604 |

## Supplemental XGBoost/SHAP evidence

- XGBoost/SHAP available: `True`.
- XGBoost/SHAP import error: ``.
- Top SHAP feature: `std_8th_delta_delta`.
- Top XGBoost importance feature: `tqwt_TKEO_mean_dec_12`.

## Output files

- `results/stage4_subject_level_feature_table_dataset1.csv`
- `results/stage4_feature_audit_dataset1.csv`
- `results/stage4_univariate_statistics_dataset1.csv`
- `results/stage4_l1_stability_selection_dataset1.csv`
- `results/stage4_tree_importance_dataset1.csv`
- `results/stage4_permutation_importance_dataset1.csv`
- `results/stage4_xgboost_importance_dataset1.csv`
- `results/stage4_shap_importance_dataset1.csv`
- `results/stage4_biomarker_rank_summary_dataset1.csv`
- `results/stage4_top10_biomarkers_dataset1.csv`
- `results/stage4_top20_biomarkers_dataset1.csv`
- `results/stage4_top50_biomarkers_dataset1.csv`
- `results/figures/stage4_top20_biomarker_rank.png`
- `results/figures/stage4_top20_effect_size.png`
- `results/figures/stage4_feature_category_counts_top50.png`
- `results/figures/stage4_volcano_plot_dataset1.png`
- `results/figures/stage4_shap_top20_mean_abs.png`

## Guardrails

- The 755 recordings were not treated as independent samples for significance testing.
- Metadata and sex variables were excluded from the main acoustic feature list.
- Final ranking combines univariate, stability, permutation, and tree-based evidence rather than relying only on tree importance.
- Supplemental XGBoost/SHAP evidence is not used by Stage 5 or Stage 6.
- The report uses associative interpretation only.

Univariate features with FDR q < 0.05: `423`