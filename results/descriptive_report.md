# 第二阶段描述性统计与质量控制报告

本报告由 `src/stage2_qc_pipeline.py` 生成，所有文本使用 UTF-8 写入，CSV 使用 utf-8-sig 写入以兼容 Excel。

## 统一数据配置

```python
DATASET_CONFIG = {
    'dataset1': {'id_col': 'id', 'label_col': 'class', 'gender_col': 'gender', 'metadata_cols': ['id', 'gender', 'class']},
    'dataset2': {'id_col': 'ID', 'label_col': 'Status', 'gender_col': 'Gender', 'recording_col': 'Recording', 'metadata_cols': ['ID', 'Recording', 'Status', 'Gender']},
}
```

## 性别编码统一

- 两个数据集原始性别编码方向不同。
- 为避免解释混乱，本阶段新增统一变量 `sex_male`，不覆盖原始 `gender/Gender`。
- `sex_male=1` 表示男性，`sex_male=0` 表示女性。
- Dataset 1: 保留原始列 `gender`，新增 `sex_male = gender`。
- Dataset 2: 保留原始列 `Gender`，新增 `sex_male = 1 - Gender`。
- 后续跨数据集分析和交叉表均使用 `sex_male`；原始 `gender/Gender` 只用于数据审计和溯源，不进入统一建模特征。
- 主模型应只使用语音特征；`sex_male` 不直接进入主模型，仅用于补充模型或敏感性分析，以检验性别混杂影响。

## Dataset 1

- 源文件: `data/附件 1 parkinson+s+disease+classification/pd_speech_features.csv`
- 读取方式: `pd.read_csv(path, header=1, encoding="utf-8")`
- 原始形状: `(756, 755)`
- 元信息列: `id, gender, class`
- 统一性别变量规则: `sex_male = gender`
- 数值语音特征数量: `752`
- 完全重复行数量: `1`
- 受试者数量: `252`
- 按受试者标签不一致数量: `0`
- 按受试者原始性别不一致数量: `0`
- 按受试者 `sex_male` 不一致数量: `0`
- 记录级类别分布: `{0: 192, 1: 564}`
- 受试者级类别分布: `{0: 64, 1: 188}`
- 记录级 `sex_male` 分布: `{0: 366, 1: 390}`
- 受试者级 `sex_male` 分布: `{0: 122, 1: 130}`

### 输出
- 记录级描述统计: `results/descriptive_statistics_dataset1.csv`
- 受试者聚合描述统计: `results/grouped_subject_statistics_dataset1.csv`
- 记录级类别 × `sex_male` 交叉表: `results/dataset1_class_sex_male_crosstab_record.csv`
- 受试者级类别 × `sex_male` 交叉表: `results/dataset1_class_sex_male_crosstab_subject.csv`

## Dataset 2

- 源文件: `data/附件2 ReplicatedAcousticFeatures-ParkinsonDatabase/ReplicatedAcousticFeatures-ParkinsonDatabase.csv`
- 读取方式: `pd.read_csv(path, encoding="utf-8")`
- 原始形状: `(240, 48)`
- 元信息列: `ID, Recording, Status, Gender`
- 统一性别变量规则: `sex_male = 1 - Gender`
- 数值语音特征数量: `44`
- 完全重复行数量: `0`
- 受试者数量: `80`
- 按受试者标签不一致数量: `0`
- 按受试者原始性别不一致数量: `0`
- 按受试者 `sex_male` 不一致数量: `0`
- 记录级类别分布: `{0: 120, 1: 120}`
- 受试者级类别分布: `{0: 40, 1: 40}`
- 记录级 `sex_male` 分布: `{0: 96, 1: 144}`
- 受试者级 `sex_male` 分布: `{0: 32, 1: 48}`

### 输出
- 记录级描述统计: `results/descriptive_statistics_dataset2.csv`
- 受试者聚合描述统计: `results/grouped_subject_statistics_dataset2.csv`
- 记录级类别 × `sex_male` 交叉表: `results/dataset2_class_sex_male_crosstab_record.csv`
- 受试者级类别 × `sex_male` 交叉表: `results/dataset2_class_sex_male_crosstab_subject.csv`

### Recording 分布检查
- 记录级 `Recording` 分布: `{1: 80, 2: 80, 3: 80}`
- 每名受试者均有且仅有一次 `Recording=1/2/3` 的数量: `80`
- 不满足上述条件的受试者数量: `0`

## Dataset 1 去重说明

- 原始版保留 756 行，用于数据描述。
- 去重版 755 行用于避免完全重复样本对建模产生重复加权。
- 使用去重版后，不再严格满足每个受试者 3 条记录，后续受试者级聚合必须显式说明。
- 去重后受试者记录数分布: `{2: 1, 3: 251}`。
- 去重建模准备版: `results/model_ready_dataset1_deduplicated.csv`。
- Dataset 2 prepared 建模准备版: `results/model_ready_dataset2_prepared.csv`。

## 图像输出

- `results/figures/class_distribution_dataset1.png`
- `results/figures/class_distribution_dataset2.png`
- `results/figures/gender_distribution_dataset1.png`
- `results/figures/gender_distribution_dataset2.png`

## 边界说明

本阶段只生成描述统计、受试者聚合、编码统一、交叉表、去重准备数据和基础分布图，不生成预测结果、降维结果、合成采样结果、无监督分组结果或特征排序结果。
