# Stage 5 Dataset 1 PD Two-Cluster Voice Phenotyping

## 1. Stage 5 目标与边界
本阶段只在 Dataset 1 的 PD 受试者内部进行 k=2 无监督聚类，目标是得到基于语音特征的二类潜在表型，并训练一个预测聚类标签的 cluster-label classifier。
所有结论均为探索性辅助分型，不是临床确诊结果。

## 2. 为什么不能做真实运动型/非运动型监督诊断
由于附件数据未提供运动型与非运动型的真实临床标签，本文不直接构建监督分类模型，而是在帕金森病受试者内部进行无监督二类聚类，以获得基于语音特征的潜在分型。随后训练的分类器预测的是聚类标签，而非临床确诊标签。因此，该模型只能作为运动型/非运动型辅助诊断的探索性参考，仍需真实临床症状标签进一步验证。

## 3. 输入文件与样本审计
- 输入 subject-level 文件：`results/stage4_subject_level_feature_table_dataset1.csv`
- 总受试者数：`252`
- PD 受试者数：`188`
- class 分布：`{0: 64, 1: 188}`
- 缺失值总数：`0`
- duplicated id 数：`0`

## 4. 特征集与预处理
使用 Stage 4 的 Top20、Top50 与 all_acoustic 三套特征集。所有候选方案先执行 StandardScaler；PCA 表示均在标准化之后生成。
`id`, `gender`, `class`, `sex_male`, `record_count` 及任何标签或后续生成列均不进入聚类输入。`sex_male` 仅用于事后描述。

## 5. 聚类候选方案完整指标表
| feature_set | representation | clustering_method | cluster_0_size | cluster_1_size | smaller_cluster_ratio | silhouette | calinski_harabasz | davies_bouldin | stability_ari_mean | recommended_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top20_biomarkers | scaled_only | KMeans | 62 | 126 | 0.3298 | 0.2950 | 78.4693 | 1.3274 | 0.9818 | False |
| top20_biomarkers | scaled_only | GaussianMixture | 106 | 82 | 0.4362 | 0.1908 | 50.3604 | 1.7521 | 0.7700 | False |
| top20_biomarkers | scaled_only | AgglomerativeClustering | 133 | 55 | 0.2926 | 0.2972 | 74.9374 | 1.3023 | 0.9099 | False |
| top20_biomarkers | pca_5 | KMeans | 62 | 126 | 0.3298 | 0.3674 | 109.9050 | 1.0711 | 0.9854 | True |
| top20_biomarkers | pca_5 | GaussianMixture | 143 | 45 | 0.2394 | 0.2324 | 30.0584 | 2.1576 | 0.4834 | False |
| top20_biomarkers | pca_5 | AgglomerativeClustering | 135 | 53 | 0.2819 | 0.3713 | 104.7317 | 1.0343 | 0.8616 | False |
| top20_biomarkers | pca_10 | KMeans | 62 | 126 | 0.3298 | 0.3164 | 87.1264 | 1.2437 | 0.9823 | False |
| top20_biomarkers | pca_10 | GaussianMixture | 136 | 52 | 0.2766 | 0.1792 | 38.4521 | 1.8398 | 0.5014 | False |
| top20_biomarkers | pca_10 | AgglomerativeClustering | 134 | 54 | 0.2872 | 0.3187 | 82.5403 | 1.2165 | 0.9035 | False |
| top50_biomarkers | pca_10 | KMeans | 97 | 91 | 0.4840 | 0.2581 | 70.0834 | 1.4360 | 0.9195 | False |
| top50_biomarkers | pca_10 | GaussianMixture | 100 | 88 | 0.4681 | 0.2258 | 58.2064 | 1.5884 | 0.7518 | False |
| top50_biomarkers | pca_10 | AgglomerativeClustering | 138 | 50 | 0.2660 | 0.2612 | 57.8719 | 1.3177 | 0.7248 | False |
| top50_biomarkers | pca_90 | KMeans | 94 | 94 | 0.5000 | 0.2411 | 64.3397 | 1.5190 | 0.9219 | False |
| top50_biomarkers | pca_90 | GaussianMixture | 98 | 90 | 0.4787 | 0.2088 | 53.2439 | 1.6764 | 0.7176 | False |
| top50_biomarkers | pca_90 | AgglomerativeClustering | 135 | 53 | 0.2819 | 0.2409 | 54.3222 | 1.4149 | 0.8634 | False |
| all_acoustic | pca_20 | KMeans | 95 | 93 | 0.4947 | 0.1657 | 31.5744 | 2.1189 | 0.9499 | False |
| all_acoustic | pca_20 | GaussianMixture | 90 | 98 | 0.4787 | 0.1551 | 29.7123 | 2.1797 | 0.6714 | False |
| all_acoustic | pca_20 | AgglomerativeClustering | 105 | 83 | 0.4415 | 0.1505 | 27.1848 | 2.2701 | 0.6855 | False |
| all_acoustic | pca_90 | KMeans | 95 | 93 | 0.4947 | 0.1239 | 24.1713 | 2.5093 | 0.9510 | False |
| all_acoustic | pca_90 | GaussianMixture | 93 | 95 | 0.4947 | 0.1234 | 24.0801 | 2.5137 | 0.6426 | False |
| all_acoustic | pca_90 | AgglomerativeClustering | 98 | 90 | 0.4787 | 0.1064 | 19.2744 | 2.7809 | 0.6413 | False |

## 6. 稳定性分析结果
每个候选方案执行 `50` 次 80% 受试者重采样，重新标准化、PCA 与聚类，并在可比较样本上计算 Adjusted Rand Index。
稳定性均值和标准差已写入 `results/stage5_clustering_metrics.csv`。

## 7. 主推荐方案选择理由
主推荐方案为 `top20_biomarkers / pca_5 / KMeans`。该方案最小簇比例为 0.330，稳定性 ARI 均值为 0.985，在聚类质量、簇均衡性、稳定性与解释性之间取得较好平衡。Top20/Top50 候选优先于 all_acoustic，因为其特征来源于 Stage 4 候选生物标志物，更适合论文解释；all_acoustic 结果作为参考，避免把高维噪声依赖方案作为默认主结论。

选择时没有只看单一指标。若某些候选方案单项 silhouette 或 Calinski-Harabasz 较高，但簇规模过度失衡、稳定性较弱、或依赖更高维声学空间，则不作为主推荐方案。

Selection score 前三候选：

| feature_set | representation | clustering_method | selection_score | smaller_cluster_ratio | silhouette | stability_ari_mean |
| --- | --- | --- | --- | --- | --- | --- |
| top20_biomarkers | pca_5 | KMeans | 1.0221 | 0.3298 | 0.3674 | 0.9854 |
| top20_biomarkers | pca_5 | AgglomerativeClustering | 0.9320 | 0.2819 | 0.3713 | 0.8616 |
| top20_biomarkers | pca_10 | KMeans | 0.9196 | 0.3298 | 0.3164 | 0.9823 |

## 8. 主方案两簇人数与比例
| cluster | n_subjects | ratio |
| --- | --- | --- |
| 0 | 62 | 0.3298 |
| 1 | 126 | 0.6702 |

## 9. 两簇 Top 区分语音特征
| feature | category | zmean_difference_cluster1_minus_cluster0 | cohens_d_cluster1_minus_cluster0 | q_mannwhitney | direction |
| --- | --- | --- | --- | --- | --- |
| std_6th_delta_delta | Delta / Delta-delta | -1.7310 | -2.9627 | 0.0000 | higher_in_cluster_0 |
| std_7th_delta_delta | Delta / Delta-delta | -1.7291 | -2.9531 | 0.0000 | higher_in_cluster_0 |
| std_10th_delta | Delta / Delta-delta | -1.7138 | -2.8779 | 0.0000 | higher_in_cluster_0 |
| std_6th_delta | Delta / Delta-delta | -1.6988 | -2.8081 | 0.0000 | higher_in_cluster_0 |
| std_9th_delta | Delta / Delta-delta | -1.6970 | -2.7997 | 0.0000 | higher_in_cluster_0 |
| std_9th_delta_delta | Delta / Delta-delta | -1.6923 | -2.7785 | 0.0000 | higher_in_cluster_0 |
| std_7th_delta | Delta / Delta-delta | -1.6890 | -2.7638 | 0.0000 | higher_in_cluster_0 |
| std_8th_delta | Delta / Delta-delta | -1.6557 | -2.6235 | 0.0000 | higher_in_cluster_0 |
| std_delta_delta_log_energy | Delta / Delta-delta | -0.9892 | -1.1115 | 0.0000 | higher_in_cluster_0 |
| tqwt_entropy_shannon_dec_17 | TQWT | 0.8584 | 0.9331 | 0.0000 | higher_in_cluster_1 |
| tqwt_maxValue_dec_11 | TQWT | 0.7815 | 0.8358 | 0.0000 | higher_in_cluster_1 |
| tqwt_minValue_dec_12 | TQWT | -0.6519 | -0.6812 | 0.0000 | higher_in_cluster_0 |
| tqwt_stdValue_dec_7 | TQWT | 0.5674 | 0.5856 | 0.0001 | higher_in_cluster_1 |
| tqwt_kurtosisValue_dec_36 | TQWT | -0.4759 | -0.4857 | 0.0020 | higher_in_cluster_0 |
| tqwt_TKEO_std_dec_11 | TQWT | 0.4647 | 0.4736 | 0.0000 | higher_in_cluster_1 |
| tqwt_TKEO_std_dec_12 | TQWT | 0.2925 | 0.2938 | 0.0000 | higher_in_cluster_1 |
| tqwt_energy_dec_12 | TQWT | -0.2400 | -0.2402 | 0.2159 | higher_in_cluster_0 |
| tqwt_kurtosisValue_dec_27 | TQWT | 0.1711 | 0.1707 | 0.8631 | higher_in_cluster_1 |
| DFA | Nonlinear | 0.0909 | 0.0905 | 0.6732 | higher_in_cluster_1 |
| mean_MFCC_2nd_coef | MFCC | 0.0687 | 0.0684 | 0.8592 | higher_in_cluster_1 |

## 10. 特征类别分布与解释
| category | top_feature_count |
| --- | --- |
| Delta / Delta-delta | 9 |
| TQWT | 9 |
| Nonlinear | 1 |
| MFCC | 1 |
Top 区分特征的类别分布用于描述两个潜在语音表型的声学差异来源，不构成临床病因解释。

## 11. 性别分布的事后描述
| cluster | 0 | 1 | All |
| --- | --- | --- | --- |
| 0 | 25 | 37 | 62 |
| 1 | 56 | 70 | 126 |
| All | 81 | 107 | 188 |
`sex_male` 未进入聚类或 cluster-label classifier，只作为事后描述，不能作因果解释。

## 12. cluster-label classifier 结果
| model | fold | accuracy | balanced_accuracy | f1_macro | f1_weighted | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | mean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | std | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SVM-RBF | mean | 0.9839 | 0.9756 | 0.9814 | 0.9837 | 1.0000 |
| SVM-RBF | std | 0.0131 | 0.0199 | 0.0152 | 0.0133 | 0.0000 |
| Random Forest | mean | 0.9627 | 0.9603 | 0.9582 | 0.9627 | 0.9968 |
| Random Forest | std | 0.0270 | 0.0324 | 0.0303 | 0.0271 | 0.0040 |
该分类器学习的是无监督聚类标签，用于将潜在语音表型分型规则模型化；它不是训练自真实临床运动型/非运动型标签，因此不能解释为真实临床症状分类性能。

## 13. 局限性
- 聚类标签来自语音特征内部结构，没有外部临床症状标签验证。
- Mann-Whitney U 与 FDR 是聚类后的描述性比较，不是外部验证假设检验，也不是临床显著性证据。
- cluster 编号本身没有临床含义；任何 potential motor-like voice phenotype 或 potential nonmotor-like voice phenotype 命名都只是解释性命名。
- all_acoustic 高维结果可能受冗余与噪声影响，因此不优先作为论文主解释方案。

## 14. 论文建议表述
建议写作：本文在 PD 受试者内部基于 Stage 4 筛选的语音特征进行 k=2 无监督聚类，得到两个基于语音特征的潜在表型簇，并进一步训练 cluster-label classifier 以复现该探索性分型规则。
不建议写作：将本阶段结果表述为真实临床亚型诊断性能。

## 输出文件
- `results/stage5_pd_subject_clustering_input_audit.csv`
- `results/stage5_clustering_metrics.csv`
- `results/stage5_cluster_assignments_dataset1_pd.csv`
- `results/stage5_cluster_feature_differences.csv`
- `results/stage5_cluster_interpretation_top_features.csv`
- `results/stage5_cluster_label_classifier_metrics.csv`
- `results/stage5_two_cluster_report.md`
- `results/figures/stage5_pca_k2_clusters_top20.png`
- `results/figures/stage5_pca_k2_clusters_top50.png`
- `results/figures/stage5_cluster_size_comparison.png`
- `results/figures/stage5_cluster_feature_difference_top20.png`
