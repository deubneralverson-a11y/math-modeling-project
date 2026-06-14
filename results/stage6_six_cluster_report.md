# Stage 6 Dataset 1 PD Six-Cluster Voice Phenotyping

## 1. Stage 6 目标与边界
本阶段在 PD 受试者内部进行 k=6 无监督聚类，输出基于语音特征的六类潜在表型，并训练 six-cluster-label classifier 预测聚类标签。
六个 cluster 编号没有临床症状含义。

## 2. 为什么不能做真实六类临床症状监督诊断
由于附件数据未提供静止性震颤、运动迟缓、肌肉僵直、疼痛、痴呆、睡眠障碍的真实临床标签，本文不直接构建六类症状监督分类模型，而是在帕金森病受试者内部进行 k=6 无监督聚类，以获得基于语音特征的六类潜在表型。随后训练的分类器预测的是聚类标签，而非临床确诊标签。因此，该模型只能作为六类帕金森病症状辅助诊断的探索性语音表型参考，仍需真实临床症状标签进一步验证。

## 3. 输入文件与样本审计
- Subject-level input: `results/stage4_subject_level_feature_table_dataset1.csv`
- Total subjects: `252`
- PD subjects: `188`
- Class distribution: `{0: 64, 1: 188}`
- Missing values: `0`
- Duplicated id count: `0`
- Stage 5 assignment file: `results/stage5_cluster_assignments_dataset1_pd.csv`

## 4. 特征集与预处理
使用 Top20、Top50 和 all_acoustic 三套特征集，所有表示均先 StandardScaler，再按需要 PCA。`id`, `gender`, `class`, `sex_male`, `record_count`, Stage 5 cluster 和 Stage 6 cluster 不进入聚类或分类器输入。

## 5. k=6 聚类候选方案完整指标表
| feature_set | representation | clustering_method | cluster_sizes | min_cluster_size | min_cluster_ratio | silhouette | davies_bouldin | stability_ari_mean | recommended_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top20_biomarkers | scaled_only | KMeans | 0:57;1:17;2:32;3:6;4:39;5:37 | 6 | 0.0319 | 0.1787 | 1.7744 | 0.6846 | False |
| top20_biomarkers | scaled_only | GaussianMixture | 0:42;1:7;2:51;3:37;4:34;5:17 | 7 | 0.0372 | 0.1329 | 1.7215 | 0.5536 | False |
| top20_biomarkers | scaled_only | AgglomerativeClustering | 0:59;1:61;2:16;3:2;4:39;5:11 | 2 | 0.0106 | 0.1724 | 1.6817 | 0.6094 | False |
| top20_biomarkers | pca_5 | KMeans | 0:38;1:39;2:53;3:6;4:17;5:35 | 6 | 0.0319 | 0.2650 | 1.3416 | 0.7026 | False |
| top20_biomarkers | pca_5 | GaussianMixture | 0:35;1:13;2:5;3:85;4:18;5:32 | 5 | 0.0266 | 0.0926 | 1.7294 | 0.3679 | False |
| top20_biomarkers | pca_5 | AgglomerativeClustering | 0:13;1:55;2:34;3:28;4:25;5:33 | 13 | 0.0691 | 0.1823 | 1.4268 | 0.4582 | True |
| top20_biomarkers | pca_10 | KMeans | 0:30;1:36;2:17;3:47;4:7;5:51 | 7 | 0.0372 | 0.1685 | 1.5510 | 0.6790 | False |
| top20_biomarkers | pca_10 | GaussianMixture | 0:56;1:9;2:40;3:36;4:31;5:16 | 9 | 0.0479 | 0.1256 | 1.9423 | 0.4267 | False |
| top20_biomarkers | pca_10 | AgglomerativeClustering | 0:65;1:56;2:21;3:11;4:33;5:2 | 2 | 0.0106 | 0.1852 | 1.5609 | 0.6154 | False |
| top50_biomarkers | pca_10 | KMeans | 0:36;1:53;2:49;3:37;4:2;5:11 | 2 | 0.0106 | 0.2054 | 1.3959 | 0.7422 | False |
| top50_biomarkers | pca_10 | GaussianMixture | 0:35;1:58;2:2;3:54;4:25;5:14 | 2 | 0.0106 | 0.1989 | 1.4682 | 0.6091 | False |
| top50_biomarkers | pca_10 | AgglomerativeClustering | 0:75;1:14;2:48;3:47;4:2;5:2 | 2 | 0.0106 | 0.2137 | 1.2569 | 0.6926 | False |
| top50_biomarkers | pca_90 | KMeans | 0:67;1:13;2:27;3:2;4:53;5:26 | 2 | 0.0106 | 0.1764 | 1.5333 | 0.7223 | False |
| top50_biomarkers | pca_90 | GaussianMixture | 0:48;1:12;2:39;3:52;4:1;5:36 | 1 | 0.0053 | 0.1747 | 1.4670 | 0.6043 | False |
| top50_biomarkers | pca_90 | AgglomerativeClustering | 0:53;1:42;2:13;3:55;4:2;5:23 | 2 | 0.0106 | 0.1377 | 1.5705 | 0.6423 | False |
| all_acoustic | pca_20 | KMeans | 0:56;1:11;2:28;3:72;4:20;5:1 | 1 | 0.0053 | 0.1598 | 1.5234 | 0.7236 | False |
| all_acoustic | pca_20 | GaussianMixture | 0:6;1:39;2:44;3:25;4:9;5:65 | 6 | 0.0319 | 0.1607 | 1.6931 | 0.4881 | False |
| all_acoustic | pca_20 | AgglomerativeClustering | 0:2;1:17;2:33;3:66;4:16;5:54 | 2 | 0.0106 | 0.1310 | 1.6571 | 0.6380 | False |
| all_acoustic | pca_90 | KMeans | 0:38;1:57;2:12;3:21;4:59;5:1 | 1 | 0.0053 | 0.0864 | 1.8854 | 0.6607 | False |
| all_acoustic | pca_90 | GaussianMixture | 0:14;1:22;2:27;3:60;4:24;5:41 | 14 | 0.0745 | 0.0898 | 2.3991 | 0.4682 | False |
| all_acoustic | pca_90 | AgglomerativeClustering | 0:2;1:65;2:73;3:12;4:25;5:11 | 2 | 0.0106 | 0.0794 | 1.9598 | 0.6113 | False |

## 6. k=2 到 k=8 参考扫描结果
k 扫描只用于说明 k=6 的相对聚类质量和稳定性；k=6 是由题目六类设定驱动的探索性语音表型划分，不是由真实六类临床标签监督得到。
| k | clustering_method | min_cluster_size | silhouette | davies_bouldin | stability_ari_mean |
| --- | --- | --- | --- | --- | --- |
| 2 | KMeans | 62 | 0.3674 | 1.0711 | 0.9897 |
| 2 | GaussianMixture | 45 | 0.2324 | 2.1576 | 0.4425 |
| 2 | AgglomerativeClustering | 53 | 0.3713 | 1.0343 | 0.8677 |
| 3 | KMeans | 54 | 0.2962 | 1.2220 | 0.8274 |
| 3 | GaussianMixture | 40 | 0.2836 | 1.6666 | 0.6023 |
| 3 | AgglomerativeClustering | 13 | 0.3624 | 1.0776 | 0.6452 |
| 4 | KMeans | 8 | 0.3005 | 1.1435 | 0.9264 |
| 4 | GaussianMixture | 16 | 0.2182 | 1.9830 | 0.4807 |
| 4 | AgglomerativeClustering | 13 | 0.2326 | 1.3395 | 0.5259 |
| 5 | KMeans | 7 | 0.2812 | 1.2285 | 0.9405 |
| 5 | GaussianMixture | 7 | 0.1396 | 1.7969 | 0.4046 |
| 5 | AgglomerativeClustering | 13 | 0.1996 | 1.4092 | 0.4819 |
| 6 | KMeans | 6 | 0.2650 | 1.3416 | 0.7026 |
| 6 | GaussianMixture | 5 | 0.0926 | 1.7294 | 0.3679 |
| 6 | AgglomerativeClustering | 13 | 0.1823 | 1.4268 | 0.4582 |
| 7 | KMeans | 6 | 0.2300 | 1.3372 | 0.6490 |
| 7 | GaussianMixture | 4 | 0.0634 | 1.8064 | 0.3704 |
| 7 | AgglomerativeClustering | 2 | 0.1893 | 1.3496 | 0.4934 |
| 8 | KMeans | 2 | 0.2430 | 1.2309 | 0.6556 |
| 8 | GaussianMixture | 2 | 0.1376 | 1.5858 | 0.4011 |
| 8 | AgglomerativeClustering | 2 | 0.1860 | 1.3667 | 0.5200 |

## 7. 稳定性分析结果
每个 k=6 候选和 k 扫描方案均执行 `50` 次 80% PD 受试者重采样，重新标准化、PCA 与聚类，并计算 ARI。
主方案 Consensus KMeans 参考：within-cluster mean `0.614`，between-cluster mean `0.121`，separation `0.494`。

## 8. 主推荐方案选择理由
主推荐方案为 `top20_biomarkers / pca_5 / AgglomerativeClustering`。该方案的最小簇人数为 `13`，最小簇比例为 `0.069`，稳定性 ARI 均值为 `0.458`。选择时同时考虑了簇大小、稳定性、聚类指标、特征集可解释性、是否依赖高维噪声，以及与 Stage 5 二类语音表型的关系。

没有只按单项指标选择方案。若某些候选 silhouette 或 Calinski-Harabasz 较高，但存在小簇、不稳定或高维噪声依赖，则不作为论文主解释方案。

| feature_set | representation | clustering_method | selection_score | min_cluster_size | min_cluster_ratio | silhouette | stability_ari_mean | stage5_nmi | recommended_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top20_biomarkers | pca_5 | AgglomerativeClustering | 0.6314 | 13 | 0.0691 | 0.1823 | 0.4582 | 0.4089 | True |
| all_acoustic | pca_90 | GaussianMixture | 0.1810 | 14 | 0.0745 | 0.0898 | 0.4682 | 0.1160 | False |
| top20_biomarkers | scaled_only | AgglomerativeClustering | -inf | 2 | 0.0106 | 0.1724 | 0.6094 | 0.4695 | False |
| top20_biomarkers | scaled_only | GaussianMixture | -inf | 7 | 0.0372 | 0.1329 | 0.5536 | 0.4426 | False |
| top20_biomarkers | scaled_only | KMeans | -inf | 6 | 0.0319 | 0.1787 | 0.6846 | 0.4614 | False |

## 9. 六个簇的人数与比例
| stage6_cluster | n_subjects | ratio |
| --- | --- | --- |
| 0 | 13 | 0.0691 |
| 1 | 55 | 0.2926 |
| 2 | 34 | 0.1809 |
| 3 | 28 | 0.1489 |
| 4 | 25 | 0.1330 |
| 5 | 33 | 0.1755 |

## 10. 六个簇的声学特征画像
| stage6_cluster | n_subjects | ratio | dominant_feature_categories | tentative_voice_phenotype_label |
| --- | --- | --- | --- | --- |
| 0 | 13 | 0.0691 | TQWT:7; Delta / Delta-delta:2; MFCC:1 | latent_voice_phenotype_A_tqwt_high |
| 1 | 55 | 0.2926 | TQWT:6; Delta / Delta-delta:2; MFCC:1; Nonlinear:1 | latent_voice_phenotype_B_tqwt_high |
| 2 | 34 | 0.1809 | Delta / Delta-delta:5; TQWT:3; Nonlinear:1; MFCC:1 | latent_voice_phenotype_C_dynamic_delta_high |
| 3 | 28 | 0.1489 | Delta / Delta-delta:9; Nonlinear:1 | latent_voice_phenotype_D_dynamic_delta_high |
| 4 | 25 | 0.1330 | Delta / Delta-delta:6; TQWT:4 | latent_voice_phenotype_E_dynamic_delta_high |
| 5 | 33 | 0.1755 | Delta / Delta-delta:9; Nonlinear:1 | latent_voice_phenotype_F_dynamic_delta_low |

## 11. 每个簇的 Top 区分语音特征
| stage6_cluster | feature | category | one_vs_rest_zmean_difference | one_vs_rest_cohens_d | q_kruskal |
| --- | --- | --- | --- | --- | --- |
| 0 | tqwt_TKEO_std_dec_11 | TQWT | 2.7661 | 3.8991 | 0.0000 |
| 0 | tqwt_TKEO_std_dec_12 | TQWT | 2.6527 | 3.5914 | 0.0000 |
| 0 | tqwt_minValue_dec_12 | TQWT | -2.4258 | -3.0516 | 0.0000 |
| 0 | tqwt_maxValue_dec_11 | TQWT | 2.2995 | 2.8037 | 0.0000 |
| 0 | tqwt_stdValue_dec_7 | TQWT | 1.4312 | 1.5576 | 0.0000 |
| 0 | tqwt_energy_dec_12 | TQWT | 1.4193 | 1.5585 | 0.0000 |
| 0 | mean_MFCC_2nd_coef | MFCC | -1.0572 | -1.1891 | 0.0000 |
| 0 | tqwt_entropy_shannon_dec_17 | TQWT | 0.9755 | 0.9901 | 0.0000 |
| 0 | std_7th_delta_delta | Delta / Delta-delta | -0.9082 | -0.7841 | 0.0000 |
| 0 | std_9th_delta | Delta / Delta-delta | -0.9016 | -0.7770 | 0.0000 |
| 1 | mean_MFCC_2nd_coef | MFCC | 1.1909 | 1.3306 | 0.0000 |
| 1 | DFA | Nonlinear | 0.9059 | 0.9795 | 0.0000 |
| 1 | tqwt_TKEO_std_dec_11 | TQWT | -0.8265 | -0.5922 | 0.0000 |
| 1 | tqwt_stdValue_dec_7 | TQWT | -0.8246 | -0.7844 | 0.0000 |
| 1 | tqwt_maxValue_dec_11 | TQWT | -0.7516 | -0.5950 | 0.0000 |
| 1 | tqwt_minValue_dec_12 | TQWT | 0.7250 | 0.5388 | 0.0000 |
| 1 | tqwt_TKEO_std_dec_12 | TQWT | -0.6685 | -0.4141 | 0.0000 |
| 1 | tqwt_entropy_shannon_dec_17 | TQWT | -0.6088 | -0.6162 | 0.0000 |
| 1 | std_6th_delta_delta | Delta / Delta-delta | -0.5409 | -0.5503 | 0.0000 |
| 1 | std_6th_delta | Delta / Delta-delta | -0.5090 | -0.5134 | 0.0000 |
| 2 | tqwt_entropy_shannon_dec_17 | TQWT | 1.1324 | 1.4277 | 0.0000 |
| 2 | DFA | Nonlinear | -1.0380 | -1.3478 | 0.0000 |
| 2 | std_10th_delta | Delta / Delta-delta | -0.7127 | -0.6744 | 0.0000 |
| 2 | std_9th_delta_delta | Delta / Delta-delta | -0.6737 | -0.6492 | 0.0000 |
| 2 | mean_MFCC_2nd_coef | MFCC | -0.5920 | -0.8675 | 0.0000 |
| 2 | std_9th_delta | Delta / Delta-delta | -0.5258 | -0.4947 | 0.0000 |
| 2 | tqwt_stdValue_dec_7 | TQWT | 0.4887 | 0.7175 | 0.0000 |
| 2 | std_8th_delta | Delta / Delta-delta | -0.4685 | -0.4450 | 0.0000 |
| 2 | std_delta_delta_log_energy | Delta / Delta-delta | -0.4493 | -0.4742 | 0.0000 |
| 2 | tqwt_energy_dec_12 | TQWT | -0.4327 | -0.2292 | 0.0000 |
| 3 | std_6th_delta | Delta / Delta-delta | 1.8021 | 2.4342 | 0.0000 |
| 3 | std_6th_delta_delta | Delta / Delta-delta | 1.7974 | 2.4309 | 0.0000 |
| 3 | std_7th_delta | Delta / Delta-delta | 1.7735 | 2.2941 | 0.0000 |
| 3 | std_7th_delta_delta | Delta / Delta-delta | 1.6865 | 2.1147 | 0.0000 |
| 3 | std_9th_delta | Delta / Delta-delta | 1.6745 | 2.0935 | 0.0000 |
| 3 | std_10th_delta | Delta / Delta-delta | 1.5897 | 1.9779 | 0.0000 |
| 3 | std_8th_delta | Delta / Delta-delta | 1.5756 | 1.8959 | 0.0000 |
| 3 | std_9th_delta_delta | Delta / Delta-delta | 1.5605 | 1.8981 | 0.0000 |
| 3 | std_delta_delta_log_energy | Delta / Delta-delta | 1.1474 | 1.2047 | 0.0000 |
| 3 | DFA | Nonlinear | -0.7545 | -0.9204 | 0.0000 |
| 4 | tqwt_maxValue_dec_11 | TQWT | -1.4369 | -1.2168 | 0.0000 |
| 4 | std_9th_delta_delta | Delta / Delta-delta | 1.4131 | 1.5940 | 0.0000 |
| 4 | tqwt_entropy_shannon_dec_17 | TQWT | -1.3559 | -1.3130 | 0.0000 |
| 4 | tqwt_minValue_dec_12 | TQWT | 1.3509 | 1.0945 | 0.0000 |
| 4 | std_10th_delta | Delta / Delta-delta | 1.3316 | 1.5038 | 0.0000 |
| 4 | std_9th_delta | Delta / Delta-delta | 1.3236 | 1.4633 | 0.0000 |
| 4 | std_8th_delta | Delta / Delta-delta | 1.2381 | 1.3383 | 0.0000 |
| 4 | std_7th_delta_delta | Delta / Delta-delta | 1.2035 | 1.3067 | 0.0000 |
| 4 | std_6th_delta_delta | Delta / Delta-delta | 1.1344 | 1.2713 | 0.0000 |
| 4 | tqwt_stdValue_dec_7 | TQWT | -1.1097 | -0.9168 | 0.0000 |
| 5 | std_6th_delta | Delta / Delta-delta | -1.3139 | -1.4057 | 0.0000 |
| 5 | std_6th_delta_delta | Delta / Delta-delta | -1.2907 | -1.3674 | 0.0000 |
| 5 | std_9th_delta | Delta / Delta-delta | -1.2465 | -1.3611 | 0.0000 |
| 5 | std_7th_delta | Delta / Delta-delta | -1.2123 | -1.3106 | 0.0000 |
| 5 | std_8th_delta | Delta / Delta-delta | -1.2073 | -1.3202 | 0.0000 |
| 5 | std_7th_delta_delta | Delta / Delta-delta | -1.2037 | -1.3016 | 0.0000 |
| 5 | std_9th_delta_delta | Delta / Delta-delta | -1.1878 | -1.2701 | 0.0000 |
| 5 | std_10th_delta | Delta / Delta-delta | -1.1062 | -1.1386 | 0.0000 |
| 5 | std_delta_delta_log_energy | Delta / Delta-delta | -0.8849 | -0.9602 | 0.0000 |
| 5 | DFA | Nonlinear | 0.5556 | 0.4201 | 0.0000 |

## 12. 特征类别分布与解释
| category | top_feature_count |
| --- | --- |
| Delta / Delta-delta | 33 |
| TQWT | 20 |
| Nonlinear | 4 |
| MFCC | 3 |
这些类别分布用于描述六类潜在语音表型的声学差异来源，不是临床病因解释。

## 13. Stage 5 二类表型与 Stage 6 六类表型关系
| stage5_cluster | stage6_cluster | count | row_percentage | column_percentage | normalized_mutual_info_stage5_vs_stage6 | collapsed_stage6_to_stage5_majority_ari |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.0000 | 0.0000 | 0.4089 | 0.8128 |
| 0 | 1 | 5 | 0.0806 | 0.0909 | 0.4089 | 0.8128 |
| 0 | 2 | 4 | 0.0645 | 0.1176 | 0.4089 | 0.8128 |
| 0 | 3 | 28 | 0.4516 | 1.0000 | 0.4089 | 0.8128 |
| 0 | 4 | 25 | 0.4032 | 1.0000 | 0.4089 | 0.8128 |
| 0 | 5 | 0 | 0.0000 | 0.0000 | 0.4089 | 0.8128 |
| 1 | 0 | 13 | 0.1032 | 1.0000 | 0.4089 | 0.8128 |
| 1 | 1 | 50 | 0.3968 | 0.9091 | 0.4089 | 0.8128 |
| 1 | 2 | 30 | 0.2381 | 0.8824 | 0.4089 | 0.8128 |
| 1 | 3 | 0 | 0.0000 | 0.0000 | 0.4089 | 0.8128 |
| 1 | 4 | 0 | 0.0000 | 0.0000 | 0.4089 | 0.8128 |
| 1 | 5 | 33 | 0.2619 | 1.0000 | 0.4089 | 0.8128 |
该交叉表用于观察六类潜在表型是否可视为二类潜在语音表型的细分；不能解释为真实临床层级诊断。

## 14. 性别分布的事后描述
| stage6_cluster | 0 | 1 | All |
| --- | --- | --- | --- |
| 0 | 7 | 6 | 13 |
| 1 | 26 | 29 | 55 |
| 2 | 14 | 20 | 34 |
| 3 | 11 | 17 | 28 |
| 4 | 10 | 15 | 25 |
| 5 | 13 | 20 | 33 |
| All | 81 | 107 | 188 |
`sex_male` 未进入 Stage 6 聚类或 six-cluster-label classifier，只用于事后描述，不能作为因果解释。

## 15. six-cluster-label classifier 结果
| model | fold | accuracy | balanced_accuracy | f1_macro | f1_weighted | precision_macro | recall_macro | multiclass_roc_auc_ovr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multinomial Logistic Regression | mean | 0.8777 | 0.8858 | 0.8843 | 0.8765 | 0.9077 | 0.8858 | 0.9844 |
| Multinomial Logistic Regression | std | 0.0462 | 0.0564 | 0.0514 | 0.0483 | 0.0352 | 0.0564 | 0.0118 |
| SVM-RBF | mean | 0.8834 | 0.8718 | 0.8740 | 0.8819 | 0.8989 | 0.8718 | 0.9910 |
| SVM-RBF | std | 0.0567 | 0.0628 | 0.0581 | 0.0576 | 0.0486 | 0.0628 | 0.0055 |
| Random Forest | mean | 0.8780 | 0.8666 | 0.8714 | 0.8754 | 0.8974 | 0.8666 | 0.9842 |
| Random Forest | std | 0.0350 | 0.0451 | 0.0396 | 0.0354 | 0.0318 | 0.0451 | 0.0093 |
该分类器学习的是无监督六类聚类标签，用于将潜在语音表型分型规则模型化；它不是训练自真实静止性震颤、运动迟缓、肌肉僵直、疼痛、痴呆或睡眠障碍临床标签，因此不能解释为真实六类症状诊断性能。

## 16. 局限性
- 六类标签来自语音特征内部结构，没有外部临床症状标签验证。
- Kruskal-Wallis 和 FDR 是聚类后的描述性比较，不是外部临床验证假设检验。
- cluster 编号和 tentative label 均为声学解释命名，不对应具体临床症状。
- k=6 由题目设定驱动，k 扫描只提供参考。

## 17. 论文建议表述
建议写作：本文在 PD 受试者内部基于 Stage 4 筛选的声学特征进行 k=6 无监督聚类，得到六个基于语音特征的潜在表型，并训练 six-cluster-label classifier 复现该探索性分型规则。
不建议写作：将本阶段结果表述为真实六类临床症状诊断性能。

## 输出文件
- `results/stage6_six_cluster_input_audit.csv`
- `results/stage6_six_cluster_metrics.csv`
- `results/stage6_k_scan_metrics.csv`
- `results/stage6_six_cluster_assignments_dataset1_pd.csv`
- `results/stage6_six_cluster_feature_profiles.csv`
- `results/stage6_six_cluster_top_features_by_cluster.csv`
- `results/stage6_six_cluster_profiles.csv`
- `results/stage6_stage5_crosswalk.csv`
- `results/stage6_six_cluster_label_classifier_metrics.csv`
- `results/stage6_six_cluster_label_classifier_confusion_matrices.csv`