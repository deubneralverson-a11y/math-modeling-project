# Descriptive Statistics Report

This second-stage report contains descriptive statistics, subject-level aggregation, basic distribution figures, and the Dataset 1 duplicate-row handling record.

## Unified Dataset Configuration

```python
DATASET_CONFIG = {
    'dataset1': {'id_col': 'id', 'label_col': 'class', 'gender_col': 'gender', 'metadata_cols': ['id', 'gender', 'class']},
    'dataset2': {'id_col': 'ID', 'label_col': 'Status', 'gender_col': 'Gender', 'recording_col': 'Recording', 'metadata_cols': ['ID', 'Recording', 'Status', 'Gender']},
}
```

## Dataset 1

- Source file: `D:\硕士\数学建模\A题-帕金森病症的声音分类识别问题\附件 1 parkinson+s+disease+classification\pd_speech_features.csv`
- Read method: `pd.read_csv(path, header=1)`
- Raw shape used for descriptive statistics: `(756, 755)`
- Metadata columns: `id, gender, class`
- Numeric voice feature count: `752`
- Full duplicate rows: `1`
- Subject count: `252`
- Label consistency issues by subject: `0`
- Gender consistency issues by subject: `0`
- Record-level class distribution: `{'0': 192, '1': 564}`
- Subject-level class distribution: `{'0': 64, '1': 188}`
- Record-level gender distribution: `{'0': 366, '1': 390}`
- Subject-level gender distribution: `{'0': 122, '1': 130}`

### Statistics Outputs
- Record-level descriptive statistics: `results/descriptive_statistics_dataset1.csv`
- Subject-aggregated descriptive statistics: `results/grouped_subject_statistics_dataset1.csv`

## Dataset 2

- Source file: `D:\硕士\数学建模\A题-帕金森病症的声音分类识别问题\附件2 ReplicatedAcousticFeatures-ParkinsonDatabase\ReplicatedAcousticFeatures-ParkinsonDatabase.csv`
- Read method: `pd.read_csv(path)`
- Raw shape used for descriptive statistics: `(240, 48)`
- Metadata columns: `ID, Recording, Status, Gender`
- Numeric voice feature count: `44`
- Full duplicate rows: `0`
- Subject count: `80`
- Label consistency issues by subject: `0`
- Gender consistency issues by subject: `0`
- Record-level class distribution: `{'0': 120, '1': 120}`
- Subject-level class distribution: `{'0': 40, '1': 40}`
- Record-level gender distribution: `{'0': 144, '1': 96}`
- Subject-level gender distribution: `{'0': 48, '1': 32}`

### Statistics Outputs
- Record-level descriptive statistics: `results/descriptive_statistics_dataset2.csv`
- Subject-aggregated descriptive statistics: `results/grouped_subject_statistics_dataset2.csv`

### Recording Distribution Check
- Record-level Recording distribution: `{'1': 80, '2': 80, '3': 80}`
- Subjects with exactly Recording=1/2/3 once each: `80`
- Subjects without exactly Recording=1/2/3 once each: `0`

## Dataset 1 Duplicate Handling

- Original audit row count retained for this stage: `756`
- Full duplicate row count: `1`
- Deduplicated model-preparation row count: `755`
- Deduplicated output: `results/model_ready_dataset1_deduplicated.csv`

## Basic Figure Outputs

- `results/figures/class_distribution_dataset1.png`
- `results/figures/class_distribution_dataset2.png`
- `results/figures/gender_distribution_dataset1.png`
- `results/figures/gender_distribution_dataset2.png`

## Boundary Statement

This stage only created summary tables, subject-level aggregates, duplicate-handling output, and basic distribution figures. It did not create predictive outputs, dimensionality-reduction outputs, synthetic-sampling outputs, unsupervised grouping outputs, or feature-ranking outputs.
