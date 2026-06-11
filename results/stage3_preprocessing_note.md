# Stage 3 Preprocessing Note

## Dataset 2 `standardized` naming

`results/model_ready_dataset2_standardized.csv` 中的 `standardized` 仅指建模准备阶段的字段格式与性别编码统一，不表示对声学特征做过统计标准化。

具体含义：

- 保留原始字段 `ID`, `Recording`, `Status`, `Gender`。
- 保留原始 44 个声学特征列。
- 新增统一性别变量 `sex_male = 1 - Gender`，使 `sex_male=1` 表示男性，`sex_male=0` 表示女性。
- 没有覆盖原始 `Gender` 列。

## Acoustic feature scaling check

未对 Dataset 2 声学特征做全数据 z-score。

核查依据：

- `src/stage2_qc_pipeline.py` 中 Dataset 2 的建模准备文件由原始数据读取后仅调用 `add_sex_male()`，随后直接写出。
- `results/model_ready_dataset2_standardized.csv` 与原始文件 `data/附件2 ReplicatedAcousticFeatures-ParkinsonDatabase/ReplicatedAcousticFeatures-ParkinsonDatabase.csv` 对比后，44 个声学特征列逐列完全一致，最大绝对差为 `0.0`。
- 该文件额外新增的列只有 `sex_male`。

因此，本阶段无需废弃 `model_ready_dataset2_standardized.csv`，也无需改为重新从原始数据读取。

## Stage 3 scaling policy

第三阶段模型训练中的标准化只发生在 sklearn `Pipeline` 内：

- Logistic Regression: `StandardScaler` + `LogisticRegression`
- SVM-RBF: `StandardScaler` + `SVC`
- Random Forest: 不做标准化，保持树模型基线口径

这避免了在交叉验证外对全数据做 z-score，从而避免数据泄漏。

## Review artifact

第三阶段建模脚本已提交在：

- `src/stage3_baseline_models.py`

该脚本可直接用于审查 Dataset 1 和 Dataset 2 的分组交叉验证、Pipeline 标准化、record-level/subject-level 指标与输出文件生成逻辑。
