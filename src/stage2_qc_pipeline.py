from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

CSV_ENCODING = "utf-8"
OUTPUT_CSV_ENCODING = "utf-8-sig"

DATASET_CONFIG = {
    "dataset1": {
        "display_name": "Dataset 1",
        "file_name": "pd_speech_features.csv",
        "read_kwargs": {"header": 1, "encoding": CSV_ENCODING},
        "read_method": 'pd.read_csv(path, header=1, encoding="utf-8")',
        "id_col": "id",
        "label_col": "class",
        "gender_col": "gender",
        "recording_col": None,
        "metadata_cols": ["id", "gender", "class"],
        "sex_male_rule": "sex_male = gender",
    },
    "dataset2": {
        "display_name": "Dataset 2",
        "file_name": "ReplicatedAcousticFeatures-ParkinsonDatabase.csv",
        "read_kwargs": {"encoding": CSV_ENCODING},
        "read_method": 'pd.read_csv(path, encoding="utf-8")',
        "id_col": "ID",
        "label_col": "Status",
        "gender_col": "Gender",
        "recording_col": "Recording",
        "metadata_cols": ["ID", "Recording", "Status", "Gender"],
        "sex_male_rule": "sex_male = 1 - Gender",
    },
}

STAT_COLUMNS = ["mean", "std", "median", "min", "max", "IQR"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def find_source_csvs() -> dict[str, Path]:
    paths = {path.name: path for path in DATA_DIR.rglob("*.csv")}
    missing = [
        config["file_name"]
        for config in DATASET_CONFIG.values()
        if config["file_name"] not in paths
    ]
    if missing:
        raise FileNotFoundError(f"Missing source CSV files: {missing}")
    return paths


def add_sex_male(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    cfg = DATASET_CONFIG[dataset_key]
    out = df.copy()
    if dataset_key == "dataset1":
        out["sex_male"] = out[cfg["gender_col"]]
    elif dataset_key == "dataset2":
        out["sex_male"] = 1 - out[cfg["gender_col"]]
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    return out


def feature_columns(df: pd.DataFrame, dataset_key: str) -> list[str]:
    cfg = DATASET_CONFIG[dataset_key]
    excluded = set(cfg["metadata_cols"]) | {"sex_male"}
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded
    ]


def describe_features(
    df: pd.DataFrame,
    features: list[str],
    dataset_key: str,
    scope: str,
    label_value: object = "",
) -> pd.DataFrame:
    numeric = df[features]
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    stats = pd.DataFrame(
        {
            "dataset": dataset_key,
            "scope": scope,
            "label_value": label_value,
            "feature": features,
            "n": numeric.count().reindex(features).astype(int).values,
            "mean": numeric.mean().reindex(features).values,
            "std": numeric.std(ddof=1).reindex(features).values,
            "median": numeric.median().reindex(features).values,
            "min": numeric.min().reindex(features).values,
            "max": numeric.max().reindex(features).values,
            "IQR": (q3 - q1).reindex(features).values,
        }
    )
    return stats[["dataset", "scope", "label_value", "feature", "n"] + STAT_COLUMNS]


def record_level_stats(
    df: pd.DataFrame, dataset_key: str, features: list[str]
) -> pd.DataFrame:
    cfg = DATASET_CONFIG[dataset_key]
    frames = [describe_features(df, features, dataset_key, "record_overall")]
    for label_value, group in df.groupby(cfg["label_col"], dropna=False):
        frames.append(
            describe_features(
                group, features, dataset_key, "record_by_label", label_value
            )
        )
    return pd.concat(frames, ignore_index=True)


def subject_table(
    df: pd.DataFrame, dataset_key: str, features: list[str]
) -> pd.DataFrame:
    cfg = DATASET_CONFIG[dataset_key]
    id_col = cfg["id_col"]
    label_col = cfg["label_col"]
    gender_col = cfg["gender_col"]
    feature_means = df.groupby(id_col, dropna=False)[features].mean().reset_index()
    subject_meta = (
        df.groupby(id_col, dropna=False)
        .agg(
            record_count=(id_col, "size"),
            label_unique_count=(label_col, lambda s: s.nunique(dropna=False)),
            gender_unique_count=(gender_col, lambda s: s.nunique(dropna=False)),
            sex_male_unique_count=("sex_male", lambda s: s.nunique(dropna=False)),
            label_value=(label_col, "first"),
            raw_gender_value=(gender_col, "first"),
            sex_male=("sex_male", "first"),
        )
        .reset_index()
    )
    return subject_meta.merge(feature_means, on=id_col, how="left")


def subject_level_stats(
    subjects: pd.DataFrame, dataset_key: str, features: list[str]
) -> pd.DataFrame:
    frames = [describe_features(subjects, features, dataset_key, "subject_overall")]
    for label_value, group in subjects.groupby("label_value", dropna=False):
        frames.append(
            describe_features(
                group, features, dataset_key, "subject_by_label", label_value
            )
        )
    return pd.concat(frames, ignore_index=True)


def save_crosstab(
    df: pd.DataFrame,
    label_col: str,
    output_path: Path,
    index_name: str,
) -> pd.DataFrame:
    table = pd.crosstab(df[label_col], df["sex_male"], margins=True)
    table.index.name = index_name
    table.columns.name = "sex_male"
    table.to_csv(output_path, encoding=OUTPUT_CSV_ENCODING)
    return table


def save_distribution_figure(
    record_counts: pd.Series,
    subject_counts: pd.Series,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
    panels = [
        ("Record level", record_counts, "#2f80ed"),
        ("Subject level", subject_counts, "#27ae60"),
    ]
    for ax, (panel_title, counts, color) in zip(axes, panels):
        counts = counts.sort_index()
        labels = [str(label) for label in counts.index]
        ax.bar(labels, counts.values, color=color)
        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for index, value in enumerate(counts.values):
            ax.text(index, value, str(int(value)), ha="center", va="bottom")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path)
    plt.close(fig)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, encoding=OUTPUT_CSV_ENCODING, index=False)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    source_paths = find_source_csvs()

    loaded: dict[str, pd.DataFrame] = {}
    subjects_by_dataset: dict[str, pd.DataFrame] = {}
    feature_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    label_conflicts: dict[str, int] = {}
    gender_conflicts: dict[str, int] = {}
    sex_conflicts: dict[str, int] = {}
    crosstab_outputs: list[Path] = []
    report_lines: list[str] = [
        "# 第二阶段描述性统计与质量控制报告",
        "",
        "本报告由 `src/stage2_qc_pipeline.py` 生成，所有文本使用 UTF-8 写入，CSV 使用 utf-8-sig 写入以兼容 Excel。",
        "",
        "## 统一数据配置",
        "",
        "```python",
        "DATASET_CONFIG = {",
        "    'dataset1': {'id_col': 'id', 'label_col': 'class', 'gender_col': 'gender', 'metadata_cols': ['id', 'gender', 'class']},",
        "    'dataset2': {'id_col': 'ID', 'label_col': 'Status', 'gender_col': 'Gender', 'recording_col': 'Recording', 'metadata_cols': ['ID', 'Recording', 'Status', 'Gender']},",
        "}",
        "```",
        "",
        "## 性别编码统一",
        "",
        "- 两个数据集原始性别编码方向不同。",
        "- 为避免解释混乱，本阶段新增统一变量 `sex_male`，不覆盖原始 `gender/Gender`。",
        "- `sex_male=1` 表示男性，`sex_male=0` 表示女性。",
        "- Dataset 1: 保留原始列 `gender`，新增 `sex_male = gender`。",
        "- Dataset 2: 保留原始列 `Gender`，新增 `sex_male = 1 - Gender`。",
        "- 后续跨数据集分析和交叉表均使用 `sex_male`；原始 `gender/Gender` 只用于数据审计和溯源，不进入统一建模特征。",
        "- 主模型应只使用语音特征；`sex_male` 不直接进入主模型，仅用于补充模型或敏感性分析，以检验性别混杂影响。",
        "",
    ]

    for dataset_key, cfg in DATASET_CONFIG.items():
        path = source_paths[cfg["file_name"]]
        df_raw = pd.read_csv(path, **cfg["read_kwargs"])
        df = add_sex_male(df_raw, dataset_key)
        loaded[dataset_key] = df
        features = feature_columns(df, dataset_key)
        feature_counts[dataset_key] = len(features)
        duplicate_counts[dataset_key] = int(df_raw.duplicated().sum())

        record_stats = record_level_stats(df, dataset_key, features)
        subjects = subject_table(df, dataset_key, features)
        subjects_by_dataset[dataset_key] = subjects
        subject_stats = subject_level_stats(subjects, dataset_key, features)

        write_csv(record_stats, RESULTS_DIR / f"descriptive_statistics_{dataset_key}.csv")
        write_csv(
            subject_stats,
            RESULTS_DIR / f"grouped_subject_statistics_{dataset_key}.csv",
        )

        for obsolete in RESULTS_DIR.glob(f"{dataset_key}_class_gender_crosstab_*.csv"):
            obsolete.unlink()
        record_crosstab_path = (
            RESULTS_DIR / f"{dataset_key}_class_sex_male_crosstab_record.csv"
        )
        subject_crosstab_path = (
            RESULTS_DIR / f"{dataset_key}_class_sex_male_crosstab_subject.csv"
        )
        save_crosstab(
            df,
            cfg["label_col"],
            record_crosstab_path,
            cfg["label_col"],
        )
        save_crosstab(
            subjects.rename(columns={"label_value": cfg["label_col"]}),
            cfg["label_col"],
            subject_crosstab_path,
            cfg["label_col"],
        )
        crosstab_outputs.extend([record_crosstab_path, subject_crosstab_path])

        label_conflicts[dataset_key] = int(
            (subjects["label_unique_count"] > 1).sum()
        )
        gender_conflicts[dataset_key] = int(
            (subjects["gender_unique_count"] > 1).sum()
        )
        sex_conflicts[dataset_key] = int(
            (subjects["sex_male_unique_count"] > 1).sum()
        )

        save_distribution_figure(
            df[cfg["label_col"]].value_counts(),
            subjects["label_value"].value_counts(),
            f"{cfg['display_name']} class distribution",
            cfg["label_col"],
            FIGURES_DIR / f"class_distribution_{dataset_key}.png",
        )
        save_distribution_figure(
            df["sex_male"].value_counts(),
            subjects["sex_male"].value_counts(),
            f"{cfg['display_name']} sex_male distribution",
            "sex_male",
            FIGURES_DIR / f"gender_distribution_{dataset_key}.png",
        )

        report_lines.extend(
            [
                f"## {cfg['display_name']}",
                "",
                f"- 源文件: `{rel(path)}`",
                f"- 读取方式: `{cfg['read_method']}`",
                f"- 原始形状: `{df_raw.shape}`",
                f"- 元信息列: `{', '.join(cfg['metadata_cols'])}`",
                f"- 统一性别变量规则: `{cfg['sex_male_rule']}`",
                f"- 数值语音特征数量: `{len(features)}`",
                f"- 完全重复行数量: `{duplicate_counts[dataset_key]}`",
                f"- 受试者数量: `{subjects.shape[0]}`",
                f"- 按受试者标签不一致数量: `{label_conflicts[dataset_key]}`",
                f"- 按受试者原始性别不一致数量: `{gender_conflicts[dataset_key]}`",
                f"- 按受试者 `sex_male` 不一致数量: `{sex_conflicts[dataset_key]}`",
                f"- 记录级类别分布: `{df[cfg['label_col']].value_counts().sort_index().to_dict()}`",
                f"- 受试者级类别分布: `{subjects['label_value'].value_counts().sort_index().to_dict()}`",
                f"- 记录级 `sex_male` 分布: `{df['sex_male'].value_counts().sort_index().to_dict()}`",
                f"- 受试者级 `sex_male` 分布: `{subjects['sex_male'].value_counts().sort_index().to_dict()}`",
                "",
                "### 输出",
                f"- 记录级描述统计: `results/descriptive_statistics_{dataset_key}.csv`",
                f"- 受试者聚合描述统计: `results/grouped_subject_statistics_{dataset_key}.csv`",
                f"- 记录级类别 × `sex_male` 交叉表: `results/{record_crosstab_path.name}`",
                f"- 受试者级类别 × `sex_male` 交叉表: `results/{subject_crosstab_path.name}`",
                "",
            ]
        )

        if cfg["recording_col"]:
            recording_col = cfg["recording_col"]
            recording_counts = df[recording_col].value_counts().sort_index()
            per_subject_recording = df.groupby(cfg["id_col"])[recording_col].apply(
                lambda s: tuple(sorted(s.dropna().tolist()))
            )
            expected = (1, 2, 3)
            report_lines.extend(
                [
                    "### Recording 分布检查",
                    f"- 记录级 `Recording` 分布: `{recording_counts.to_dict()}`",
                    f"- 每名受试者均有且仅有一次 `Recording=1/2/3` 的数量: `{int((per_subject_recording == expected).sum())}`",
                    f"- 不满足上述条件的受试者数量: `{int((per_subject_recording != expected).sum())}`",
                    "",
                ]
            )

    dataset1 = loaded["dataset1"]
    dataset2 = loaded["dataset2"]
    dataset1_dedup = dataset1.drop_duplicates()
    dataset1_dedup_path = RESULTS_DIR / "model_ready_dataset1_deduplicated.csv"
    dataset2_standardized_path = RESULTS_DIR / "model_ready_dataset2_standardized.csv"
    write_csv(dataset1_dedup, dataset1_dedup_path)
    write_csv(dataset2, dataset2_standardized_path)

    d1_dedup_record_counts = (
        dataset1_dedup.groupby(DATASET_CONFIG["dataset1"]["id_col"])
        .size()
        .value_counts()
        .sort_index()
        .to_dict()
    )

    report_lines.extend(
        [
            "## Dataset 1 去重说明",
            "",
            "- 原始版保留 756 行，用于数据描述。",
            "- 去重版 755 行用于避免完全重复样本对建模产生重复加权。",
            "- 使用去重版后，不再严格满足每个受试者 3 条记录，后续受试者级聚合必须显式说明。",
            f"- 去重后受试者记录数分布: `{d1_dedup_record_counts}`。",
            f"- 去重建模准备版: `results/{dataset1_dedup_path.name}`。",
            f"- Dataset 2 标准化建模准备版: `results/{dataset2_standardized_path.name}`。",
            "",
            "## 图像输出",
            "",
            "- `results/figures/class_distribution_dataset1.png`",
            "- `results/figures/class_distribution_dataset2.png`",
            "- `results/figures/gender_distribution_dataset1.png`",
            "- `results/figures/gender_distribution_dataset2.png`",
            "",
            "## 边界说明",
            "",
            "本阶段只生成描述统计、受试者聚合、编码统一、交叉表、去重准备数据和基础分布图，不生成预测结果、降维结果、合成采样结果、无监督分组结果或特征排序结果。",
            "",
        ]
    )

    (RESULTS_DIR / "descriptive_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("Generated stage 2 QC outputs:")
    for path in [
        RESULTS_DIR / "descriptive_statistics_dataset1.csv",
        RESULTS_DIR / "descriptive_statistics_dataset2.csv",
        RESULTS_DIR / "grouped_subject_statistics_dataset1.csv",
        RESULTS_DIR / "grouped_subject_statistics_dataset2.csv",
        *crosstab_outputs,
        dataset1_dedup_path,
        dataset2_standardized_path,
        RESULTS_DIR / "descriptive_report.md",
    ]:
        print(f"- {rel(path)}")


if __name__ == "__main__":
    main()
