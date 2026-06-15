# 六阶段分析与审计 Workflow

本文档记录本项目从原始数据读取、建模、候选生物标志物识别、PD 内部分型到独立逻辑审计的完整流程。所有阶段的核心原则是：样本单位、交叉验证边界、标签边界和论文表述边界必须一致，避免出现“代码可运行但方法边界错误”的问题。

## 0. 总体数据流

Dataset 1 使用 `pd_speech_features.csv`，必须按 `pd.read_csv(path, header=1)` 读取。原始规模为 756 行、755 列；去除 1 条完全重复行后为 755 条录音记录；再按 `id` 聚合为 252 名受试者；其中 PD 受试者为 188 名。

Dataset 2 使用 `ReplicatedAcousticFeatures-ParkinsonDatabase.csv`，按普通表头 `pd.read_csv(path)` 读取。原始规模为 240 行、48 列；每名受试者按 `ID` 分组处理，不能把同一受试者的多条录音当作独立临床样本解释。

主数据流：

```text
Raw Dataset 1
756 records
  -> Stage 2 deduplication
755 records
  -> Stage 4 subject-level aggregation by id
252 subjects
  -> Stage 5/6 PD-only filtering
188 PD subjects
```

所有监督模型和统计比较必须明确区分 record-level 与 subject-level。论文主结论优先使用 subject-level 结果。

## 1. Stage 1：数据读取与原始结构审计

主要职责：

- 确认 Dataset 1 使用 `header=1` 读取。
- 确认 Dataset 2 使用普通表头读取。
- 确认原始形状、元信息列、声学特征数、缺失值、重复行和同一受试者标签一致性。

关键文件：

- 输入：`data/**/pd_speech_features.csv`
- 输入：`data/**/ReplicatedAcousticFeatures-ParkinsonDatabase.csv`
- 当前项目中没有独立 `src/stage1_data_audit.py`；Stage 1/2 读取与 QC 逻辑由 `src/stage2_qc_pipeline.py` 承担。
- 输出参考：`results/data_audit_report.md`

红线：

- Dataset 1 如果没有按 `header=1` 读取，则 Stage 1 至后续所有结果全部失效。
- 任一数据集若存在同一受试者内部标签冲突，必须先修正数据定义，不能继续建模。

## 2. Stage 2：描述统计、性别统一与建模准备

脚本：

- `src/stage2_qc_pipeline.py`

主要职责：

- Dataset 1 新增 `sex_male = gender`。
- Dataset 2 新增 `sex_male = 1 - Gender`。
- 保留原始 `gender/Gender`，不覆盖原始字段。
- Dataset 1 去重后输出 755 行。
- Dataset 2 prepared 文件只新增 `sex_male`，不对 44 个声学特征做全数据标准化。
- 保存记录级和受试者级类别 × 性别交叉表。

关键输出：

- `results/model_ready_dataset1_deduplicated.csv`
- `results/model_ready_dataset2_prepared.csv`
- `results/dataset1_class_sex_male_crosstab_record.csv`
- `results/dataset1_class_sex_male_crosstab_subject.csv`
- `results/dataset2_class_sex_male_crosstab_record.csv`
- `results/dataset2_class_sex_male_crosstab_subject.csv`

红线：

- `model_ready_dataset2_prepared.csv` 不得是全数据 z-score 标准化后的文件。
- `sex_male` 是统一性别变量，不能进入主声学模型，只能用于补充模型、敏感性分析或事后描述。

## 3. Stage 3：PD/Healthy 二分类基线模型

脚本：

- `src/stage3_baseline_models.py`

主要职责：

- 使用 `StratifiedGroupKFold`，若环境不支持则退化为 `GroupKFold` 并记录原因。
- Dataset 1 的 `groups` 使用 `id`。
- Dataset 2 的 `groups` 使用 `ID`。
- 每个 fold 的 train/test subject 必须无重叠。
- 主模型 `acoustic_only` 排除 `id`、`gender/Gender`、`class/Status`、`Recording`、`sex_male`。
- `acoustic_plus_sex` 仅作为补充模型。
- Logistic Regression 与 SVM 使用 `Pipeline(StandardScaler, model)`，标准化只在训练折内学习。
- subject-level 指标通过同一受试者多条录音概率取均值得到。

关键输出：

- `results/stage3_baseline_metrics_subject_level.csv`
- `results/stage3_baseline_metrics_record_level.csv`
- `results/stage3_fold_split_audit.csv`
- `results/stage3_confusion_matrices.csv`
- `results/stage3_baseline_report.md`
- Stage 3 metrics may include `XGBoost (supplemental)` rows; these are supplemental baseline results, not replacements for the primary model interpretation.

红线：

- 禁止普通行级 `train_test_split`。
- 禁止 train/test subject 重叠。
- 禁止在交叉验证外对全数据执行监督建模相关的标准化、PCA 或特征选择。
- 主报告应采用 subject-level acoustic_only 结果，而不是 record-level 结果。

论文口径：

- 可以报告 sensitivity、specificity、balanced accuracy、AUC 等指标。
- 需要说明同一受试者多条录音没有被当作独立受试者解释。

## 4. Stage 4：关键语音生物标志物识别

脚本：

- `src/stage4_biomarker_identification.py`

主要职责：

- 使用 `results/model_ready_dataset1_deduplicated.csv`。
- 先按 `id` 聚合为 subject-level，再做统计检验和特征筛选。
- subject-level 表应为 252 行，类别分布为 `{0:64, 1:188}`。
- 声学特征为 752 个，排除 `id`、`gender`、`class`、`sex_male`。
- Mann-Whitney U、Welch t-test、FDR-BH、Cohen's d、Cliff's delta 均基于 subject-level 数据。
- L1 稳定选择使用 subject-level StratifiedKFold，标准化通过 Pipeline 在训练折内执行。
- ExtraTrees 和 permutation importance 不得把测试折信息用于训练。
- Top20/Top50 来自综合排名，不是单一 tree importance。

关键输出：

- `results/stage4_subject_level_feature_table_dataset1.csv`
- `results/stage4_top20_biomarkers_dataset1.csv`
- `results/stage4_top50_biomarkers_dataset1.csv`
- `results/stage4_biomarker_rank_summary_dataset1.csv`
- `results/stage4_univariate_statistics_dataset1.csv`
- `results/stage4_l1_stability_selection_dataset1.csv`
- `results/stage4_tree_importance_dataset1.csv`
- `results/stage4_permutation_importance_dataset1.csv`
- `results/stage4_xgboost_importance_dataset1.csv`
- `results/stage4_shap_importance_dataset1.csv`
- `results/figures/stage4_shap_top20_mean_abs.png`
- Stage 4 XGBoost/SHAP outputs are supplemental interpretability evidence only; they do not enter the `mean_rank` formula and must not change Top10/Top20/Top50 membership.

红线：

- Stage 4 不得在 755 条记录上做显著性检验或特征筛选；必须在 252 名受试者上进行。
- 元信息列、标签列和 `sex_male` 不得进入主特征集。

论文口径：

- Top20/Top50 是候选语音生物标志物，不是临床因果机制。
- 不能把 FDR 显著性写成独立外部验证证据。

## 5. Stage 5：PD 二类潜在语音表型聚类

脚本：

- `src/stage5_pd_two_cluster_phenotyping.py`

主要职责：

- 使用 `results/stage4_subject_level_feature_table_dataset1.csv`。
- 只保留 `class == 1` 的 188 名 PD 受试者。
- 使用 Stage 4 Top20、Top50 和 all_acoustic 三套特征集。
- 聚类输入排除 `id`、`gender`、`class`、`sex_male`、`record_count` 和任何 cluster label。
- 每个候选方案先 StandardScaler，再按需要 PCA。
- 候选方法至少包括 KMeans、GaussianMixture、AgglomerativeClustering。
- 输出 silhouette、Calinski-Harabasz、Davies-Bouldin、簇人数、最小簇比例、稳定性 ARI。
- 主方案为 `top20_biomarkers / pca_5 / KMeans`。
- 主方案簇人数为 62 和 126。
- cluster-label classifier 只预测 Stage 5 聚类标签，不是临床诊断器。

关键输出：

- `results/stage5_clustering_metrics.csv`
- `results/stage5_cluster_assignments_dataset1_pd.csv`
- `results/stage5_cluster_feature_differences.csv`
- `results/stage5_cluster_label_classifier_metrics.csv`
- `results/stage5_two_cluster_report.md`

红线：

- 不得使用 Healthy 样本参与 PD 分型。
- 不得使用 563/564 条 PD 录音记录作为独立样本聚类。
- 不得把 Stage 5 cluster label 写成真实运动型/非运动型临床标签。
- classifier 不能写成真实临床诊断器。

论文口径：

- 推荐写法：在 PD 受试者内部基于 Stage 4 筛选的语音特征进行 k=2 无监督聚类，得到两个基于语音特征的潜在表型簇，并训练 cluster-label classifier 复现该探索性分型规则。
- 避免写法：模型可诊断真实运动型/非运动型 PD。

## 6. Stage 6：PD 六类潜在语音表型聚类

脚本：

- `src/stage6_pd_six_cluster_phenotyping.py`

主要职责：

- 使用 `results/stage4_subject_level_feature_table_dataset1.csv`。
- 只保留 `class == 1` 的 188 名 PD 受试者。
- 不使用 Healthy 样本，不使用记录级样本。
- 构建 Top20、Top50、all_acoustic 三套特征集，其中 all_acoustic 为 752 个声学特征。
- 聚类输入不得包含 Stage 5 cluster、Stage 6 cluster、`sex_male`、`id`、`gender`、`class`、`record_count`。
- 对 k=6 的所有候选方案输出完整指标。
- 做 k=2 到 k=8 的参考扫描。
- 做 50 次 80% 重采样稳定性分析。
- 主方案为 `top20_biomarkers / pca_5 / AgglomerativeClustering`。
- 六簇人数为 13、55、34、28、25、33；最小簇人数为 13。
- 输出 Stage 5 × Stage 6 交叉表。
- six-cluster-label classifier 只预测 Stage 6 聚类标签。

关键输出：

- `results/stage6_six_cluster_metrics.csv`
- `results/stage6_six_cluster_assignments_dataset1_pd.csv`
- `results/stage6_stage5_crosswalk.csv`
- `results/stage6_six_cluster_label_classifier_metrics.csv`
- `results/stage6_six_cluster_report.md`

红线：

- Stage 6 聚类输入不得使用 Stage 5 cluster label。
- Stage 6 聚类输入不得使用 `sex_male` 或任何真实/派生标签列。
- 六个 cluster 不得直接命名为静止性震颤、运动迟缓、肌肉僵直、疼痛、痴呆、睡眠障碍。
- six-cluster-label classifier 不能解释为真实六类临床症状诊断性能。

特别备注：

- 唯一条件是：Stage 6 主方案选择中使用 Stage 5 NMI 作为辅助推荐指标，需要在报告和论文中明确披露，避免被误解为 Stage 5 标签泄漏进六类聚类输入。
- 该 Stage 5 NMI 只用于候选方案推荐评分和解释 Stage 5/Stage 6 关系，不进入 StandardScaler、PCA 或 k=6 聚类输入矩阵。
- 论文中应明确区分：“Stage 5 cluster 未作为 Stage 6 聚类特征”与“Stage 5 NMI 被用作候选方案选择的辅助稳定/一致性参考”。

论文口径：

- 推荐写法：本文在 PD 受试者内部基于 Stage 4 筛选的声学特征进行 k=6 无监督聚类，得到六个基于语音特征的潜在表型，并训练 six-cluster-label classifier 复现该探索性分型规则。
- 必须补充：Stage 6 的主方案选择综合考虑聚类质量、簇大小、稳定性、特征可解释性和与 Stage 5 二类潜在语音表型的一致性；其中 Stage 5 NMI 仅作为候选方案推荐指标之一，不作为六类聚类输入特征。
- 避免写法：模型可诊断静止性震颤、运动迟缓、肌肉僵直、疼痛、痴呆、睡眠障碍六类真实临床症状。

## 6b. Stage 5b/6b: PD-only feature-set and weight sensitivity

Script:
- `src/stage5b_stage6b_pdonly_sensitivity.py`

Main responsibility:
- Use the sealed Stage 4 subject-level table and retain only `class == 1` PD subjects.
- Build PD-only candidate feature sets and compare them against Stage 4 Top20/Top50 controls.
- Recompute Stage 5b k=2 and Stage 6b k=6 candidate metrics without interpretability bonus.
- Run weight-sensitivity sampling for objective clustering-quality metrics.
- Keep Stage 5b/6b as sensitivity analysis; it must not rewrite Stage 5/6 main assignments or clinical wording.

Key outputs:
- `results/stage5b_pd_only_feature_sets_audit.csv`
- `results/stage6b_pd_only_feature_sets_audit.csv`
- `results/stage5b_pd_only_retained_features.csv`
- `results/stage6b_pd_only_retained_features.csv`
- `results/stage5b_clustering_metrics_no_bonus.csv`
- `results/stage6b_clustering_metrics_no_bonus.csv`
- `results/stage5b_weight_sensitivity.csv`
- `results/stage6b_weight_sensitivity.csv`
- `results/stage5b_candidate_rank_frequency.csv`
- `results/stage6b_candidate_rank_frequency.csv`
- `results/stage5b_stage6b_feature_and_weight_sensitivity_report.md`
- `results/stage5b_stage6b_paper_revision_suggestions.md`
- `results/figures/stage5b_weight_sensitivity_top_candidates.png`
- `results/figures/stage6b_weight_sensitivity_top_candidates.png`
- `results/figures/stage5b_feature_set_metric_comparison.png`
- `results/figures/stage6b_feature_set_metric_comparison.png`
- `results/figures/stage5b_stage6b_original_vs_pdonly_comparison.png`

Red lines:
- Stage 5b/6b must remain PD-only and subject-level.
- Stage 5b/6b must not use Healthy samples, record-level rows, sex variables, IDs, or cluster labels as clustering features.
- Stage 5b/6b outputs are sensitivity evidence, not replacements for the sealed Stage 5/6 main reports unless the paper explicitly decides to revise the main scheme.

## 7. 独立逻辑审计

脚本：

- `src/audit_all_stages_logic.py`

审计原则：

- 只读取代码、原始数据和既有 `results/` 输出。
- 不调用 Stage 1-6 建模函数。
- 不重新训练模型。
- 不改动封版结果。

审计输出：

- `results/audit_all_stages_logic_report.md`
- `results/audit_all_stages_checks.csv`
- `results/audit_feature_leakage_checks.csv`
- `results/audit_subject_lineage_checks.csv`
- `results/audit_metric_consistency_checks.csv`
- `results/audit_static_code_risk_patterns.csv`

当前审计结论：

- 结论：`CONDITIONAL PASS`
- 结构化检查：88 项；87 项 pass，1 项 warning，0 项 fail。
- 无核心红线失败。
- 不需要重跑 Stage 1-6。
- 唯一 warning：Stage 6 使用 `stage5_nmi` 参与候选主方案推荐评分；这不是聚类输入泄漏，但必须在报告和论文中披露。

## 8. 全局红线清单

任何阶段出现以下情况，应判定为相关阶段必须重做：

- Dataset 1 未按 `header=1` 读取。
- 同一受试者多条录音被当作独立受试者用于监督验证或统计显著性解释。
- Stage 3 使用普通行级随机划分。
- Stage 3 train/test subject 有重叠。
- Stage 4 在记录级而非 subject-level 上做特征筛选或显著性检验。
- Stage 5/6 使用 Healthy 样本参与 PD 分型。
- Stage 5/6 使用记录级样本作为独立聚类样本。
- 主模型或主聚类输入包含标签列、ID 列、`sex_male` 或 cluster label。
- Stage 5/6 classifier 被写成真实临床标签分类器。
- 报告或论文将聚类标签表述为真实临床症状标签。

## 9. 论文写作边界

可以写：

- Dataset 1 和 Dataset 2 均按受试者分组处理，避免同一受试者录音泄漏。
- 主分类模型使用 subject-level grouped cross-validation。
- Stage 4 候选特征是 subject-level 统计与模型证据的综合结果。
- Stage 5/6 是 PD 受试者内部的探索性无监督语音表型发现。
- Stage 5/6 classifier 是 cluster-label classifier，用于复现探索性聚类规则。
- Stage 6 主方案选择使用 Stage 5 NMI 作为辅助推荐指标之一，但 Stage 5 label 不进入六类聚类输入。

不能写：

- 多条录音是相互独立的临床样本。
- Stage 4 特征是已验证的临床因果生物标志物。
- Stage 5 两簇等同于真实运动型/非运动型 PD。
- Stage 6 六簇等同于六种真实临床症状诊断。
- cluster-label classifier 的 AUC/accuracy 是真实临床症状诊断性能。
- `sex_male` 或 Stage 5/6 cluster label 进入了主声学模型或主聚类输入。

## 10. 推荐执行顺序

```text
python src/stage2_qc_pipeline.py
python src/stage3_baseline_models.py
python src/stage4_biomarker_identification.py
python src/stage5_pd_two_cluster_phenotyping.py
python src/stage6_pd_six_cluster_phenotyping.py
python src/stage5b_stage6b_pdonly_sensitivity.py
python src/audit_all_stages_logic.py
```

## 11. Supplemental analysis boundary

- Stage 3 may include `XGBoost (supplemental)` as an additional PD/Healthy baseline model under the same grouped cross-validation protocol.
- Stage 4 may include fold-wise XGBoost feature importance and held-out-fold SHAP mean absolute importance as supplemental interpretability evidence.
- Stage 4 XGBoost/SHAP outputs do not enter the `mean_rank` formula and must not change the Top10/Top20/Top50 biomarker candidate membership.
- Stage 5 and Stage 6 must not import, call, or consume XGBoost/SHAP outputs. They remain PD-only unsupervised latent voice phenotype analyses based on Stage 4 selected acoustic features.
- Stage 5b/6b is sensitivity analysis for PD-only feature sets and objective weighting; it does not overwrite sealed Stage 5/6 main outputs by itself.
- The required supplemental outputs are `results/stage4_xgboost_importance_dataset1.csv`, `results/stage4_shap_importance_dataset1.csv`, and `results/figures/stage4_shap_top20_mean_abs.png`.

每次更新任一 Stage 脚本或关键输出后，都应重新运行 `src/audit_all_stages_logic.py`，并以 `results/audit_all_stages_logic_report.md` 的结论作为论文成稿前的逻辑边界检查。
