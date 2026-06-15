from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
    XGBOOST_IMPORT_ERROR = ""
except ImportError as exc:
    XGBClassifier = None
    HAS_XGBOOST = False
    XGBOOST_IMPORT_ERROR = str(exc)

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP_KFOLD = True
    STRATIFIED_GROUP_KFOLD_ERROR = ""
except ImportError as exc:
    StratifiedGroupKFold = None
    HAS_STRATIFIED_GROUP_KFOLD = False
    STRATIFIED_GROUP_KFOLD_ERROR = str(exc)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_CSV_ENCODING = "utf-8-sig"
RANDOM_STATE = 42
N_SPLITS = 5
POS_LABEL = 1
NEG_LABEL = 0
THRESHOLD = 0.5


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    path: Path
    id_col: str
    label_col: str
    metadata_cols: tuple[str, ...]
    expected_record_count_distribution: dict[int, int] | None = None


DATASET_CONFIGS = (
    DatasetConfig(
        key="dataset1",
        display_name="Dataset 1",
        path=RESULTS_DIR / "model_ready_dataset1_deduplicated.csv",
        id_col="id",
        label_col="class",
        metadata_cols=("id", "gender", "class"),
        expected_record_count_distribution={2: 1, 3: 251},
    ),
    DatasetConfig(
        key="dataset2",
        display_name="Dataset 2",
        path=RESULTS_DIR / "model_ready_dataset2_prepared.csv",
        id_col="ID",
        label_col="Status",
        metadata_cols=("ID", "Recording", "Status", "Gender"),
    ),
)


METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding=OUTPUT_CSV_ENCODING)


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    if not config.path.exists():
        raise FileNotFoundError(f"Missing input file: {config.path}")
    df = pd.read_csv(config.path)
    required = set(config.metadata_cols) | {config.id_col, config.label_col, "sex_male"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{config.display_name} is missing required columns: {missing}")
    if df[config.label_col].isna().any():
        raise ValueError(f"{config.display_name} has missing labels")
    if df[config.id_col].isna().any():
        raise ValueError(f"{config.display_name} has missing subject IDs")
    if df.isna().sum().sum() > 0:
        raise ValueError(f"{config.display_name} contains missing values")
    label_unique_by_subject = df.groupby(config.id_col)[config.label_col].nunique(
        dropna=False
    )
    if int(label_unique_by_subject.max()) != 1:
        raise ValueError(f"{config.display_name} has subjects with inconsistent labels")
    return df


def acoustic_columns(df: pd.DataFrame, config: DatasetConfig) -> list[str]:
    excluded = set(config.metadata_cols) | {"sex_male"}
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded
    ]


def feature_sets(df: pd.DataFrame, config: DatasetConfig) -> dict[str, list[str]]:
    acoustic = acoustic_columns(df, config)
    if not acoustic:
        raise ValueError(f"{config.display_name} has no acoustic feature columns")
    return {
        "acoustic_only": acoustic,
        "acoustic_plus_sex": acoustic + ["sex_male"],
    }


def make_models() -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "SVM-RBF": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost (supplemental)"] = Pipeline(
            steps=[
                (
                    "model",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=5.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                )
            ]
        )
    return models


def make_splitter() -> tuple[Any, str, str]:
    if HAS_STRATIFIED_GROUP_KFOLD:
        return (
            StratifiedGroupKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            ),
            "StratifiedGroupKFold",
            "",
        )
    return (
        GroupKFold(n_splits=N_SPLITS),
        "GroupKFold",
        STRATIFIED_GROUP_KFOLD_ERROR,
    )


def distribution(values: pd.Series | np.ndarray) -> dict[int, int]:
    series = pd.Series(values).astype(int)
    return {
        int(label): int(count)
        for label, count in series.value_counts().sort_index().items()
    }


def subject_label_distribution(
    df: pd.DataFrame, id_col: str, label_col: str
) -> dict[int, int]:
    labels = df.groupby(id_col, dropna=False)[label_col].first()
    return distribution(labels)


def add_split_audit(
    audit_rows: list[dict[str, Any]],
    config: DatasetConfig,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    splitter_name: str,
    fallback_reason: str,
) -> None:
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    train_subjects = set(train_df[config.id_col])
    test_subjects = set(test_df[config.id_col])
    overlap = train_subjects & test_subjects
    if overlap:
        raise ValueError(
            f"{config.display_name} fold {fold} has subject leakage: {sorted(overlap)[:5]}"
        )
    audit_rows.append(
        {
            "dataset": config.key,
            "dataset_display_name": config.display_name,
            "fold": fold,
            "splitter": splitter_name,
            "fallback_reason": fallback_reason,
            "train_subject_count": len(train_subjects),
            "test_subject_count": len(test_subjects),
            "train_record_count": len(train_df),
            "test_record_count": len(test_df),
            "train_record_class_distribution": distribution(train_df[config.label_col]),
            "test_record_class_distribution": distribution(test_df[config.label_col]),
            "train_subject_class_distribution": subject_label_distribution(
                train_df, config.id_col, config.label_col
            ),
            "test_subject_class_distribution": subject_label_distribution(
                test_df, config.id_col, config.label_col
            ),
            "overlap_subject_count": len(overlap),
        }
    )


def positive_probability(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if POS_LABEL not in classes:
            raise ValueError(f"Positive label {POS_LABEL} not found in model classes")
        return model.predict_proba(x_test)[:, classes.index(POS_LABEL)]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Model does not expose predict_proba or decision_function")


def compute_metrics(
    y_true: pd.Series | np.ndarray, y_prob: np.ndarray
) -> tuple[dict[str, float], dict[str, int]]:
    y_true_array = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true_array, y_pred, labels=[NEG_LABEL, POS_LABEL]
    ).ravel()
    roc_auc = (
        float(roc_auc_score(y_true_array, y_prob))
        if len(np.unique(y_true_array)) == 2
        else np.nan
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred)),
        "roc_auc": roc_auc,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(
            precision_score(y_true_array, y_pred, pos_label=POS_LABEL, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_array, y_pred, pos_label=POS_LABEL, zero_division=0)
        ),
        "f1": float(f1_score(y_true_array, y_pred, pos_label=POS_LABEL, zero_division=0)),
    }
    matrix = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics, matrix


def subject_level_predictions(
    df_test: pd.DataFrame,
    config: DatasetConfig,
    record_prob: np.ndarray,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "subject_id": df_test[config.id_col].to_numpy(),
            "y_true": df_test[config.label_col].astype(int).to_numpy(),
            "y_prob": record_prob,
        }
    )
    subject_df = (
        pred_df.groupby("subject_id", dropna=False)
        .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
        .reset_index()
    )
    label_counts = pred_df.groupby("subject_id", dropna=False)["y_true"].nunique(
        dropna=False
    )
    if int(label_counts.max()) != 1:
        raise ValueError(f"{config.display_name} test fold has inconsistent labels")
    return subject_df


def metric_row(
    config: DatasetConfig,
    feature_set: str,
    model_name: str,
    fold: int | str,
    summary: str,
    n_records: int,
    n_subjects: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": config.key,
        "dataset_display_name": config.display_name,
        "feature_set": feature_set,
        "model": model_name,
        "fold": fold,
        "summary": summary,
        "n_records": n_records,
        "n_subjects": n_subjects,
    }
    row.update(metrics)
    return row


def confusion_row(
    config: DatasetConfig,
    feature_set: str,
    model_name: str,
    level: str,
    fold: int | str,
    summary: str,
    matrix: dict[str, int],
) -> dict[str, Any]:
    return {
        "dataset": config.key,
        "dataset_display_name": config.display_name,
        "feature_set": feature_set,
        "model": model_name,
        "level": level,
        "fold": fold,
        "summary": summary,
        **matrix,
    }


def append_summary_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "dataset_display_name", "feature_set", "model"]
    summary_rows: list[dict[str, Any]] = []
    fold_rows = metrics_df[metrics_df["summary"] == "fold"]
    for values, group in fold_rows.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, values, strict=True))
        for summary, values_by_metric in (
            ("mean", group[list(METRIC_COLUMNS)].mean(numeric_only=True)),
            ("std", group[list(METRIC_COLUMNS)].std(ddof=1, numeric_only=True)),
        ):
            row = {
                **base,
                "fold": "",
                "summary": summary,
                "n_records": "",
                "n_subjects": "",
            }
            row.update({metric: values_by_metric[metric] for metric in METRIC_COLUMNS})
            summary_rows.append(row)
    return pd.concat([metrics_df, pd.DataFrame(summary_rows)], ignore_index=True)


def append_confusion_summary_rows(confusion_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "dataset_display_name", "feature_set", "model", "level"]
    matrix_cols = ["tn", "fp", "fn", "tp"]
    summary_rows: list[dict[str, Any]] = []
    fold_rows = confusion_df[confusion_df["summary"] == "fold"]
    for values, group in fold_rows.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, values, strict=True))
        row.update(
            {
                "fold": "",
                "summary": "aggregate",
                **{
                    column: int(group[column].sum())
                    for column in matrix_cols
                },
            }
        )
        summary_rows.append(row)
    return pd.concat([confusion_df, pd.DataFrame(summary_rows)], ignore_index=True)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float) and not pd.isna(value):
                values.append(format(value, floatfmt))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append(values)

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def run_dataset(
    config: DatasetConfig,
    record_metric_rows: list[dict[str, Any]],
    subject_metric_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
    report_lines: list[str],
) -> None:
    df = load_dataset(config)
    features_by_name = feature_sets(df, config)
    models = make_models()
    splitter, splitter_name, fallback_reason = make_splitter()

    y = df[config.label_col].astype(int)
    groups = df[config.id_col]
    splits = list(splitter.split(df, y, groups))
    if len(splits) != N_SPLITS:
        raise ValueError(f"{config.display_name} expected {N_SPLITS} folds")

    record_count_distribution = (
        df.groupby(config.id_col).size().value_counts().sort_index().to_dict()
    )
    if config.expected_record_count_distribution is not None:
        expected = config.expected_record_count_distribution
        if record_count_distribution != expected:
            raise ValueError(
                f"{config.display_name} record count distribution changed: "
                f"{record_count_distribution}, expected {expected}"
            )

    report_lines.extend(
        [
            f"## {config.display_name}",
            "",
            f"- Input: `{rel(config.path)}`",
            f"- Splitter: `{splitter_name}`",
            f"- Subjects: `{df[config.id_col].nunique()}`",
            f"- Records: `{len(df)}`",
            f"- Subject record count distribution: `{record_count_distribution}`",
            f"- Record-level class distribution: `{distribution(df[config.label_col])}`",
            f"- Subject-level class distribution: "
            f"`{subject_label_distribution(df, config.id_col, config.label_col)}`",
            f"- Acoustic feature count: `{len(features_by_name['acoustic_only'])}`",
            "- Feature sets: `acoustic_only` is the main analysis; "
            "`acoustic_plus_sex` is supplemental.",
        ]
    )
    if HAS_XGBOOST:
        report_lines.append(
            "- XGBoost is included as a supplemental baseline and does not replace "
            "the primary baseline interpretation."
        )
    else:
        report_lines.append(
            f"- XGBoost supplemental baseline was skipped because `xgboost` is unavailable: `{XGBOOST_IMPORT_ERROR}`."
        )
    if config.key == "dataset2":
        report_lines.append(
            "- Dataset 2 prepared file only contains metadata cleanup and sex_male "
            "encoding; acoustic features are not globally z-score standardized before "
            "cross-validation. Scaling for Logistic Regression and SVM-RBF is "
            "performed only within sklearn Pipeline in each training fold."
        )
    report_lines.extend(["", "### Fold audit", ""])
    if fallback_reason:
        report_lines.extend(
            [
                f"- StratifiedGroupKFold unavailable; fallback reason: `{fallback_reason}`",
                "",
            ]
        )

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        add_split_audit(
            audit_rows,
            config,
            df,
            train_idx,
            test_idx,
            fold,
            splitter_name,
            fallback_reason,
        )
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        report_lines.append(
            f"- Fold {fold}: train subjects `{train_df[config.id_col].nunique()}`, "
            f"test subjects `{test_df[config.id_col].nunique()}`, "
            f"train records `{len(train_df)}`, test records `{len(test_df)}`, "
            f"test subject classes "
            f"`{subject_label_distribution(test_df, config.id_col, config.label_col)}`"
        )

        for feature_set_name, columns in features_by_name.items():
            x_train = train_df[columns]
            x_test = test_df[columns]
            y_train = train_df[config.label_col].astype(int)
            y_test = test_df[config.label_col].astype(int)

            for model_name, model in models.items():
                model.fit(x_train, y_train)
                record_prob = positive_probability(model, x_test)

                record_metrics, record_matrix = compute_metrics(y_test, record_prob)
                record_metric_rows.append(
                    metric_row(
                        config,
                        feature_set_name,
                        model_name,
                        fold,
                        "fold",
                        len(test_df),
                        test_df[config.id_col].nunique(),
                        record_metrics,
                    )
                )
                confusion_rows.append(
                    confusion_row(
                        config,
                        feature_set_name,
                        model_name,
                        "record",
                        fold,
                        "fold",
                        record_matrix,
                    )
                )

                subject_df = subject_level_predictions(test_df, config, record_prob)
                subject_metrics, subject_matrix = compute_metrics(
                    subject_df["y_true"], subject_df["y_prob"].to_numpy()
                )
                subject_metric_rows.append(
                    metric_row(
                        config,
                        feature_set_name,
                        model_name,
                        fold,
                        "fold",
                        len(test_df),
                        len(subject_df),
                        subject_metrics,
                    )
                )
                confusion_rows.append(
                    confusion_row(
                        config,
                        feature_set_name,
                        model_name,
                        "subject",
                        fold,
                        "fold",
                        subject_matrix,
                    )
                )
    report_lines.append("")


def write_report(
    report_lines: list[str],
    record_metrics: pd.DataFrame,
    subject_metrics: pd.DataFrame,
) -> None:
    report_lines.extend(
        [
            "## Main subject-level results",
            "",
            "The primary model comparison uses `acoustic_only` subject-level mean metrics.",
            "",
        ]
    )
    main = subject_metrics[
        (subject_metrics["summary"] == "mean")
        & (subject_metrics["feature_set"] == "acoustic_only")
    ].copy()
    display_cols = [
        "dataset_display_name",
        "model",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
    ]
    report_lines.append(markdown_table(main[display_cols], floatfmt=".4f"))
    report_lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `results/stage3_baseline_metrics_record_level.csv`",
            "- `results/stage3_baseline_metrics_subject_level.csv`",
            "- `results/stage3_fold_split_audit.csv`",
            "- `results/stage3_confusion_matrices.csv`",
            "- `results/stage3_baseline_report.md`",
            "",
            "## Constraints",
            "",
            "- No row-wise random `train_test_split` was used.",
            "- All train/test splits are grouped by subject ID.",
            "- Standardization is only applied inside sklearn `Pipeline` objects.",
            "- No SMOTE, PCA, feature selection, SHAP, clustering, or large grid search was used.",
            "- XGBoost is reported as a supplemental baseline only.",
            "- `sex_male` is excluded from the main `acoustic_only` models.",
            "",
        ]
    )
    (RESULTS_DIR / "stage3_baseline_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    record_metric_rows: list[dict[str, Any]] = []
    subject_metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    report_lines = [
        "# Stage 3 Baseline PD/Healthy Classification Report",
        "",
        "This report is generated by `src/stage3_baseline_models.py`.",
        "",
        "Positive class: `1 = PD`; negative class: `0 = Healthy`.",
        "",
        "Models: Logistic Regression, SVM-RBF, Random Forest, and optional XGBoost (supplemental).",
        "",
    ]

    for config in DATASET_CONFIGS:
        run_dataset(
            config,
            record_metric_rows,
            subject_metric_rows,
            audit_rows,
            confusion_rows,
            report_lines,
        )

    record_metrics = append_summary_rows(pd.DataFrame(record_metric_rows))
    subject_metrics = append_summary_rows(pd.DataFrame(subject_metric_rows))
    audit = pd.DataFrame(audit_rows)
    confusion = append_confusion_summary_rows(pd.DataFrame(confusion_rows))

    write_csv(
        record_metrics, RESULTS_DIR / "stage3_baseline_metrics_record_level.csv"
    )
    write_csv(
        subject_metrics, RESULTS_DIR / "stage3_baseline_metrics_subject_level.csv"
    )
    write_csv(audit, RESULTS_DIR / "stage3_fold_split_audit.csv")
    write_csv(confusion, RESULTS_DIR / "stage3_confusion_matrices.csv")
    write_report(report_lines, record_metrics, subject_metrics)


if __name__ == "__main__":
    main()
