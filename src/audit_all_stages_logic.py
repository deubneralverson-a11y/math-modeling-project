from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

OUTPUT_CSV_ENCODING = "utf-8-sig"

DATASET1_FILE = "pd_speech_features.csv"
DATASET2_FILE = "ReplicatedAcousticFeatures-ParkinsonDatabase.csv"

CHECK_COLUMNS = [
    "stage",
    "check_name",
    "expected",
    "observed",
    "status",
    "severity",
    "evidence_file",
    "notes",
]

STATIC_KEYWORDS = [
    "fit_transform",
    "StandardScaler(",
    "PCA(",
    "SelectKBest",
    "SMOTE",
    "train_test_split",
    "cross_val_score",
    "GridSearchCV",
    "LogisticRegressionCV",
    "permutation_importance",
    "GroupKFold",
    "StratifiedGroupKFold",
    ".groupby(",
    "id",
    "ID",
    "Recording",
    "sex_male",
    "gender",
    "Gender",
    "class",
    "Status",
    "cluster",
    "stage5_cluster",
    "stage6_cluster",
]

FORBIDDEN_FEATURE_COLUMNS = {
    "id",
    "ID",
    "gender",
    "Gender",
    "class",
    "Status",
    "Recording",
    "sex_male",
    "record_count",
    "cluster",
    "stage5_cluster",
    "stage6_cluster",
}

CLINICAL_SYMPTOM_TERMS = [
    "resting tremor",
    "bradykinesia",
    "rigidity",
    "pain",
    "dementia",
    "sleep disorder",
    "静止性震颤",
    "运动迟缓",
    "肌肉僵直",
    "疼痛",
    "痴呆",
    "睡眠障碍",
]


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.static_rows: list[dict[str, Any]] = []

    def add_check(
        self,
        stage: str,
        check_name: str,
        expected: Any,
        observed: Any,
        status: str,
        severity: str,
        evidence_file: str,
        notes: str = "",
    ) -> None:
        if status not in {"pass", "warning", "fail"}:
            raise ValueError(f"Invalid check status: {status}")
        self.rows.append(
            {
                "stage": stage,
                "check_name": check_name,
                "expected": stringify(expected),
                "observed": stringify(observed),
                "status": status,
                "severity": severity,
                "evidence_file": evidence_file,
                "notes": notes,
            }
        )


def stringify(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple, set)):
        return str(value)
    return "" if value is None else str(value)


def rel(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return path_obj.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def find_data_file(file_name: str) -> Path | None:
    matches = sorted(DATA_DIR.rglob(file_name))
    return matches[0] if matches else None


def value_counts_dict(series: pd.Series) -> dict[Any, int]:
    out: dict[Any, int] = {}
    for key, value in series.value_counts(dropna=False).sort_index().items():
        if pd.isna(key):
            out["nan"] = int(value)
        elif isinstance(key, (np.integer, int)):
            out[int(key)] = int(value)
        elif isinstance(key, (np.floating, float)) and float(key).is_integer():
            out[int(key)] = int(value)
        else:
            out[key] = int(value)
    return out


def normalize_id_set(series: pd.Series) -> set[str]:
    return set(series.astype(str).tolist())


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def parse_dictish(value: Any) -> Any:
    if pd.isna(value):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return value


def check_equal(
    audit: Audit,
    stage: str,
    check_name: str,
    expected: Any,
    observed: Any,
    severity: str,
    evidence_file: str,
    notes: str = "",
) -> None:
    audit.add_check(
        stage,
        check_name,
        expected,
        observed,
        "pass" if observed == expected else "fail",
        severity,
        evidence_file,
        notes,
    )


def check_true(
    audit: Audit,
    stage: str,
    check_name: str,
    condition: bool,
    expected: str,
    observed: Any,
    severity: str,
    evidence_file: str,
    fail_status: str = "fail",
    notes: str = "",
) -> None:
    audit.add_check(
        stage,
        check_name,
        expected,
        observed,
        "pass" if condition else fail_status,
        severity,
        evidence_file,
        notes,
    )


def numeric_feature_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded
    ]


def audit_stage1_and_stage2(audit: Audit) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    dataset1_path = find_data_file(DATASET1_FILE)
    dataset2_path = find_data_file(DATASET2_FILE)

    check_true(
        audit,
        "Stage 1",
        "Dataset 1 source file exists",
        dataset1_path is not None,
        DATASET1_FILE,
        rel(dataset1_path) if dataset1_path else "missing",
        "critical",
        "data",
    )
    check_true(
        audit,
        "Stage 1",
        "Dataset 2 source file exists",
        dataset2_path is not None,
        DATASET2_FILE,
        rel(dataset2_path) if dataset2_path else "missing",
        "critical",
        "data",
    )
    if dataset1_path is None or dataset2_path is None:
        return out

    stage2_path = SRC_DIR / "stage2_qc_pipeline.py"
    stage2_text = safe_read_text(stage2_path)
    has_header_1 = bool(re.search(r"header\s*['\"]?\s*:\s*1|header\s*=\s*1", stage2_text))
    check_true(
        audit,
        "Stage 1",
        "Dataset 1 code uses header=1",
        has_header_1,
        "pd.read_csv(..., header=1)",
        "header=1 found" if has_header_1 else "header=1 not found",
        "critical",
        rel(stage2_path),
        notes="No standalone stage1_data_audit.py exists; stage2_qc_pipeline.py is the source of truth.",
    )

    raw1 = read_csv(dataset1_path, header=1)
    raw2 = read_csv(dataset2_path)
    out["raw1"] = raw1
    out["raw2"] = raw2

    check_equal(audit, "Stage 1", "Dataset 1 raw shape", (756, 755), raw1.shape, "critical", rel(dataset1_path))
    check_equal(audit, "Stage 1", "Dataset 2 raw shape", (240, 48), raw2.shape, "critical", rel(dataset2_path))
    check_equal(audit, "Stage 1", "Dataset 1 metadata columns", ["id", "gender", "class"], [c for c in ["id", "gender", "class"] if c in raw1.columns], "high", rel(dataset1_path))
    check_equal(audit, "Stage 1", "Dataset 2 metadata columns", ["ID", "Recording", "Status", "Gender"], [c for c in ["ID", "Recording", "Status", "Gender"] if c in raw2.columns], "high", rel(dataset2_path))

    d1_features = numeric_feature_columns(raw1, {"id", "gender", "class"})
    d2_features = numeric_feature_columns(raw2, {"ID", "Recording", "Status", "Gender"})
    check_equal(audit, "Stage 1", "Dataset 1 acoustic feature count", 752, len(d1_features), "critical", rel(dataset1_path))
    check_equal(audit, "Stage 1", "Dataset 2 acoustic feature count", 44, len(d2_features), "critical", rel(dataset2_path))
    check_equal(audit, "Stage 1", "Dataset 1 fully duplicated row count", 1, int(raw1.duplicated().sum()), "high", rel(dataset1_path))
    check_equal(audit, "Stage 1", "Dataset 1 missing values", 0, int(raw1.isna().sum().sum()), "high", rel(dataset1_path))
    check_equal(audit, "Stage 1", "Dataset 2 missing values", 0, int(raw2.isna().sum().sum()), "high", rel(dataset2_path))
    check_equal(audit, "Stage 1", "Dataset 1 subject label conflicts", 0, int((raw1.groupby("id")["class"].nunique(dropna=False) > 1).sum()), "critical", rel(dataset1_path))
    check_equal(audit, "Stage 1", "Dataset 2 subject label conflicts", 0, int((raw2.groupby("ID")["Status"].nunique(dropna=False) > 1).sum()), "critical", rel(dataset2_path))

    d1_ready_path = RESULTS_DIR / "model_ready_dataset1_deduplicated.csv"
    d2_ready_path = RESULTS_DIR / "model_ready_dataset2_prepared.csv"
    d1_ready = read_csv(d1_ready_path)
    d2_ready = read_csv(d2_ready_path)
    out["d1_ready"] = d1_ready
    out["d2_ready"] = d2_ready

    check_equal(audit, "Stage 2", "Dataset 1 ready row count after deduplication", 755, d1_ready.shape[0], "critical", rel(d1_ready_path))
    check_equal(audit, "Stage 2", "Dataset 1 ready column count", 756, d1_ready.shape[1], "high", rel(d1_ready_path))
    check_equal(audit, "Stage 2", "Dataset 2 prepared shape", (240, 49), d2_ready.shape, "high", rel(d2_ready_path))
    check_true(audit, "Stage 2", "Dataset 1 sex_male equals gender", bool((d1_ready["sex_male"] == d1_ready["gender"]).all()), "all rows sex_male == gender", "ok" if (d1_ready["sex_male"] == d1_ready["gender"]).all() else "mismatch", "high", rel(d1_ready_path))
    check_true(audit, "Stage 2", "Dataset 2 sex_male equals 1 - Gender", bool((d2_ready["sex_male"] == 1 - d2_ready["Gender"]).all()), "all rows sex_male == 1 - Gender", "ok" if (d2_ready["sex_male"] == 1 - d2_ready["Gender"]).all() else "mismatch", "high", rel(d2_ready_path))
    check_true(audit, "Stage 2", "Original gender columns retained", {"gender"}.issubset(d1_ready.columns) and {"Gender"}.issubset(d2_ready.columns), "gender and Gender retained", f"dataset1 gender={'gender' in d1_ready.columns}; dataset2 Gender={'Gender' in d2_ready.columns}", "medium", "results")
    check_equal(audit, "Stage 2", "Dataset 1 subject record count distribution", {2: 1, 3: 251}, value_counts_dict(d1_ready.groupby("id").size()), "critical", rel(d1_ready_path))

    raw2_features = numeric_feature_columns(raw2, {"ID", "Recording", "Status", "Gender"})
    d2_ready_features = numeric_feature_columns(d2_ready, {"ID", "Recording", "Status", "Gender", "sex_male"})
    same_feature_columns = raw2_features == d2_ready_features
    same_feature_values = bool(np.allclose(raw2[raw2_features].to_numpy(dtype=float), d2_ready[d2_ready_features].to_numpy(dtype=float))) if same_feature_columns else False
    check_true(
        audit,
        "Stage 2",
        "Dataset 2 prepared acoustic values unchanged",
        same_feature_columns and same_feature_values,
        "only sex_male added; acoustic columns unchanged",
        f"same_columns={same_feature_columns}; same_values={same_feature_values}",
        "critical",
        f"{rel(dataset2_path)}; {rel(d2_ready_path)}",
    )

    for path in [
        RESULTS_DIR / "dataset1_class_sex_male_crosstab_record.csv",
        RESULTS_DIR / "dataset1_class_sex_male_crosstab_subject.csv",
        RESULTS_DIR / "dataset2_class_sex_male_crosstab_record.csv",
        RESULTS_DIR / "dataset2_class_sex_male_crosstab_subject.csv",
    ]:
        check_true(audit, "Stage 2", f"{path.name} exists", path.exists(), "file exists", rel(path) if path.exists() else "missing", "medium", rel(path))

    return out


def audit_stage3(audit: Audit) -> None:
    code_path = SRC_DIR / "stage3_baseline_models.py"
    code = safe_read_text(code_path)
    fold_path = RESULTS_DIR / "stage3_fold_split_audit.csv"
    metrics_path = RESULTS_DIR / "stage3_baseline_metrics_subject_level.csv"
    folds = read_csv(fold_path)
    metrics = read_csv(metrics_path)

    check_true(audit, "Stage 3", "Uses StratifiedGroupKFold or GroupKFold", "StratifiedGroupKFold" in code and "GroupKFold" in code, "grouped CV code present", "present" if "GroupKFold" in code else "missing", "critical", rel(code_path))
    check_true(audit, "Stage 3", "No executable train_test_split import/call", "from sklearn.model_selection import train_test_split" not in code and "train_test_split(" not in code, "no row-level train_test_split call", "no call" if "train_test_split(" not in code else "call found", "critical", rel(code_path))
    check_equal(audit, "Stage 3", "Fold split audit overlap count", 0, int(folds["overlap_subject_count"].max()), "critical", rel(fold_path))
    check_equal(audit, "Stage 3", "Fold splitter names", ["StratifiedGroupKFold"], sorted(folds["splitter"].dropna().unique().tolist()), "high", rel(fold_path))
    check_true(audit, "Stage 3", "Pipeline scaler for LR/SVM", "Pipeline" in code and "StandardScaler()" in code, "StandardScaler inside sklearn Pipeline", "present", "critical", rel(code_path))
    check_true(audit, "Stage 3", "acoustic_only excludes sex_male", 'excluded = set(config.metadata_cols) | {"sex_male"}' in code, "metadata and sex_male excluded", "exclusion expression found" if 'excluded = set(config.metadata_cols) | {"sex_male"}' in code else "not found", "critical", rel(code_path))

    required_metric_cols = {"balanced_accuracy", "roc_auc", "sensitivity", "specificity"}
    check_true(audit, "Stage 3", "Subject-level metrics include required clinical metrics", required_metric_cols.issubset(metrics.columns), sorted(required_metric_cols), sorted(set(metrics.columns) & required_metric_cols), "medium", rel(metrics_path))
    check_true(audit, "Stage 3", "Main subject-level acoustic_only results exist", ((metrics["feature_set"] == "acoustic_only") & (metrics["summary"] == "mean")).any(), "summary rows for acoustic_only", int(((metrics["feature_set"] == "acoustic_only") & (metrics["summary"] == "mean")).sum()), "high", rel(metrics_path))
    verify_metric_summary(audit, "Stage 3", metrics, ["dataset", "feature_set", "model"], "summary", "fold", ["accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "recall", "f1"], ddof=1, evidence_file=rel(metrics_path))


def audit_stage4(audit: Audit, tables: dict[str, pd.DataFrame]) -> None:
    code_path = SRC_DIR / "stage4_biomarker_identification.py"
    code = safe_read_text(code_path)
    subject_path = RESULTS_DIR / "stage4_subject_level_feature_table_dataset1.csv"
    top20_path = RESULTS_DIR / "stage4_top20_biomarkers_dataset1.csv"
    top50_path = RESULTS_DIR / "stage4_top50_biomarkers_dataset1.csv"
    xgboost_path = RESULTS_DIR / "stage4_xgboost_importance_dataset1.csv"
    shap_path = RESULTS_DIR / "stage4_shap_importance_dataset1.csv"
    subject = read_csv(subject_path)
    top20 = read_csv(top20_path)
    top50 = read_csv(top50_path)
    xgboost_importance = read_csv(xgboost_path)
    shap_importance = read_csv(shap_path)
    tables["stage4_subject"] = subject
    tables["stage4_top20"] = top20
    tables["stage4_top50"] = top50

    check_true(audit, "Stage 4", "Uses Dataset 1 deduplicated ready file", "model_ready_dataset1_deduplicated.csv" in code, "ready Dataset 1 input", "found" if "model_ready_dataset1_deduplicated.csv" in code else "missing", "critical", rel(code_path))
    check_true(audit, "Stage 4", "Code aggregates by id before statistics", 'df.groupby("id"' in code and "build_subject_table" in code, "subject-level groupby before feature tests", "found", "critical", rel(code_path))
    check_equal(audit, "Stage 4", "Subject-level row count", 252, subject.shape[0], "critical", rel(subject_path))
    check_equal(audit, "Stage 4", "Subject-level class distribution", {0: 64, 1: 188}, value_counts_dict(subject["class"]), "critical", rel(subject_path))
    features = numeric_feature_columns(subject, {"id", "gender", "class", "sex_male", "record_count"})
    check_equal(audit, "Stage 4", "Subject-level acoustic feature count", 752, len(features), "critical", rel(subject_path))
    check_true(audit, "Stage 4", "Top20 row count", top20.shape[0] == 20, 20, top20.shape[0], "high", rel(top20_path))
    check_true(audit, "Stage 4", "Top50 row count", top50.shape[0] == 50, 50, top50.shape[0], "high", rel(top50_path))
    forbidden_top = sorted((set(top20["feature"]) | set(top50["feature"])) & FORBIDDEN_FEATURE_COLUMNS)
    check_equal(audit, "Stage 4", "Top20/Top50 forbidden metadata features", [], forbidden_top, "critical", f"{rel(top20_path)}; {rel(top50_path)}")
    forbidden_xgb_shap = sorted((set(xgboost_importance["feature"]) | set(shap_importance["feature"])) & FORBIDDEN_FEATURE_COLUMNS)
    check_equal(audit, "Stage 4", "XGBoost/SHAP forbidden metadata features", [], forbidden_xgb_shap, "critical", f"{rel(xgboost_path)}; {rel(shap_path)}")
    check_true(audit, "Stage 4", "SHAP supplemental columns present", {"shap_mean_abs_mean", "rank_shap_mean_abs"}.issubset(shap_importance.columns), ["shap_mean_abs_mean", "rank_shap_mean_abs"], sorted(set(shap_importance.columns) & {"shap_mean_abs_mean", "rank_shap_mean_abs"}), "medium", rel(shap_path))
    check_true(audit, "Stage 4", "L1 scaling inside Pipeline", "Pipeline" in code and "LogisticRegressionCV" in code and "StandardScaler()" in code, "Pipeline(StandardScaler, LogisticRegressionCV)", "found", "critical", rel(code_path))
    check_true(audit, "Stage 4", "Permutation importance uses held-out fold function", "heldout_permutation_importance" in code and "x[test_idx]" in code, "held-out fold permutation", "found", "high", rel(code_path))
    check_true(audit, "Stage 4", "Report guards candidate biomarkers from causal interpretation", "not clinical causal mechanisms" in code or "require independent validation" in code, "guardrail wording present", "present" if "clinical causal" in code or "independent validation" in code else "not found", "medium", rel(code_path))


def audit_stage5(audit: Audit, tables: dict[str, pd.DataFrame]) -> None:
    code_path = SRC_DIR / "stage5_pd_two_cluster_phenotyping.py"
    code = safe_read_text(code_path)
    metrics_path = RESULTS_DIR / "stage5_clustering_metrics.csv"
    assign_path = RESULTS_DIR / "stage5_cluster_assignments_dataset1_pd.csv"
    input_audit_path = RESULTS_DIR / "stage5_pd_subject_clustering_input_audit.csv"
    classifier_path = RESULTS_DIR / "stage5_cluster_label_classifier_metrics.csv"
    report_path = RESULTS_DIR / "stage5_two_cluster_report.md"
    metrics = read_csv(metrics_path)
    assignments = read_csv(assign_path)
    input_audit = read_csv(input_audit_path)
    classifier = read_csv(classifier_path)
    report = safe_read_text(report_path)
    tables["stage5_assign"] = assignments

    check_true(audit, "Stage 5", "Uses Stage 4 subject-level feature table", "stage4_subject_level_feature_table_dataset1.csv" in code, "Stage 4 subject table", "found", "critical", rel(code_path))
    check_equal(audit, "Stage 5", "PD-only assignment row count", 188, assignments.shape[0], "critical", rel(assign_path))
    check_equal(audit, "Stage 5", "Assignments class distribution", {1: 188}, value_counts_dict(assignments["class"]), "critical", rel(assign_path))
    check_equal(audit, "Stage 5", "Assignment duplicated id count", 0, int(assignments["id"].duplicated().sum()), "critical", rel(assign_path))
    check_equal(audit, "Stage 5", "Input audit acoustic feature count", 752, int(input_audit.loc[0, "acoustic_feature_count"]), "critical", rel(input_audit_path))
    check_equal(audit, "Stage 5", "Top20 feature count", 20, int(input_audit.loc[0, "top20_feature_count"]), "high", rel(input_audit_path))
    check_equal(audit, "Stage 5", "Top50 feature count", 50, int(input_audit.loc[0, "top50_feature_count"]), "high", rel(input_audit_path))
    check_equal(audit, "Stage 5", "Recommended flag count", 1, int(metrics["recommended_flag"].sum()), "critical", rel(metrics_path))
    recommended = metrics.loc[metrics["recommended_flag"] == True].iloc[0]
    check_equal(audit, "Stage 5", "Recommended scheme", ("top20_biomarkers", "pca_5", "KMeans"), (recommended["feature_set"], recommended["representation"], recommended["clustering_method"]), "critical", rel(metrics_path))
    check_equal(audit, "Stage 5", "Recommended cluster sizes", {0: 62, 1: 126}, value_counts_dict(assignments["cluster"]), "critical", rel(assign_path))
    check_true(audit, "Stage 5", "Candidate count covers feature sets and methods", metrics.shape[0] == 21 and set(metrics["clustering_method"]) == {"KMeans", "GaussianMixture", "AgglomerativeClustering"}, "21 candidates with 3 methods", f"rows={metrics.shape[0]}; methods={sorted(metrics['clustering_method'].unique())}", "medium", rel(metrics_path))
    check_true(audit, "Stage 5", "Cluster input excludes forbidden columns by code", "is_forbidden_feature_column" in code and "METADATA_COLUMNS" in code, "metadata and label token exclusion", "found", "critical", rel(code_path))
    check_true(audit, "Stage 5", "Cluster-label classifier uses selected acoustic features", "run_cluster_label_classifier" in code and "pd_subjects[features]" in code, "features only; labels are cluster labels", "found", "high", rel(code_path))
    no_xgb_shap = re.search(r"\bfrom\s+xgboost\b|\bimport\s+xgboost\b|\bXGBClassifier\b|\bfrom\s+shap\b|\bimport\s+shap\b|\bshap\.", code, flags=re.IGNORECASE) is None
    check_true(audit, "Stage 5", "No XGBoost/SHAP import or call", no_xgb_shap, "no xgboost/shap tokens", "absent" if no_xgb_shap else "found", "critical", rel(code_path))
    verify_metric_summary(audit, "Stage 5", classifier, ["model"], "fold", "fold", ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc"], ddof=0, evidence_file=rel(classifier_path), summary_column_is_fold=True)
    audit_report_boundary(audit, "Stage 5", report, rel(report_path), expected_classifier_phrase="cluster-label classifier")


def audit_stage6(audit: Audit, tables: dict[str, pd.DataFrame]) -> None:
    code_path = SRC_DIR / "stage6_pd_six_cluster_phenotyping.py"
    code = safe_read_text(code_path)
    metrics_path = RESULTS_DIR / "stage6_six_cluster_metrics.csv"
    assign_path = RESULTS_DIR / "stage6_six_cluster_assignments_dataset1_pd.csv"
    input_audit_path = RESULTS_DIR / "stage6_six_cluster_input_audit.csv"
    crosswalk_path = RESULTS_DIR / "stage6_stage5_crosswalk.csv"
    classifier_path = RESULTS_DIR / "stage6_six_cluster_label_classifier_metrics.csv"
    k_scan_path = RESULTS_DIR / "stage6_k_scan_metrics.csv"
    report_path = RESULTS_DIR / "stage6_six_cluster_report.md"
    metrics = read_csv(metrics_path)
    assignments = read_csv(assign_path)
    input_audit = read_csv(input_audit_path)
    crosswalk = read_csv(crosswalk_path)
    classifier = read_csv(classifier_path)
    k_scan = read_csv(k_scan_path)
    report = safe_read_text(report_path)
    tables["stage6_assign"] = assignments

    check_true(audit, "Stage 6", "Uses Stage 4 subject-level feature table", "stage4_subject_level_feature_table_dataset1.csv" in code, "Stage 4 subject table", "found", "critical", rel(code_path))
    check_equal(audit, "Stage 6", "PD-only assignment row count", 188, assignments.shape[0], "critical", rel(assign_path))
    check_equal(audit, "Stage 6", "Assignments class distribution", {1: 188}, value_counts_dict(assignments["class"]), "critical", rel(assign_path))
    check_equal(audit, "Stage 6", "Assignment duplicated id count", 0, int(assignments["id"].duplicated().sum()), "critical", rel(assign_path))
    check_equal(audit, "Stage 6", "Input audit acoustic feature count", 752, int(input_audit.loc[0, "acoustic_feature_count"]), "critical", rel(input_audit_path))
    check_equal(audit, "Stage 6", "Stage 5 overlap subject count", 188, int(input_audit.loc[0, "stage5_label_overlap_subject_count"]), "high", rel(input_audit_path))
    check_equal(audit, "Stage 6", "Recommended flag count", 1, int(metrics["recommended_flag"].sum()), "critical", rel(metrics_path))
    recommended = metrics.loc[metrics["recommended_flag"] == True].iloc[0]
    check_equal(audit, "Stage 6", "Recommended scheme", ("top20_biomarkers", "pca_5", "AgglomerativeClustering"), (recommended["feature_set"], recommended["representation"], recommended["clustering_method"]), "critical", rel(metrics_path))
    check_equal(audit, "Stage 6", "Recommended cluster sizes", {0: 13, 1: 55, 2: 34, 3: 28, 4: 25, 5: 33}, value_counts_dict(assignments["stage6_cluster"]), "critical", rel(assign_path))
    check_equal(audit, "Stage 6", "Minimum cluster size", 13, int(assignments["stage6_cluster"].value_counts().min()), "high", rel(assign_path))
    check_true(audit, "Stage 6", "k=2 to k=8 reference scan exists", sorted(k_scan["k"].unique().tolist()) == list(range(2, 9)), "k values 2..8", sorted(k_scan["k"].unique().tolist()), "medium", rel(k_scan_path))
    check_true(audit, "Stage 6", "50-run stability analysis recorded", int(metrics["stability_n_runs"].min()) == 50 and int(k_scan["stability_n_runs"].min()) == 50, "stability_n_runs == 50", f"k6={metrics['stability_n_runs'].unique().tolist()}; scan={k_scan['stability_n_runs'].unique().tolist()}", "medium", f"{rel(metrics_path)}; {rel(k_scan_path)}")
    check_true(audit, "Stage 6", "Cluster input excludes forbidden columns by code", "is_forbidden_feature_column" in code and "METADATA_COLUMNS" in code, "metadata and label token exclusion", "found", "critical", rel(code_path))
    no_xgb_shap = re.search(r"\bfrom\s+xgboost\b|\bimport\s+xgboost\b|\bXGBClassifier\b|\bfrom\s+shap\b|\bimport\s+shap\b|\bshap\.", code, flags=re.IGNORECASE) is None
    check_true(audit, "Stage 6", "No XGBoost/SHAP import or call", no_xgb_shap, "no xgboost/shap tokens", "absent" if no_xgb_shap else "found", "critical", rel(code_path))
    stage5_selection_note = "stage5_nmi" in code and "selection_score" in code
    audit.add_check(
        "Stage 6",
        "Stage 5 cluster use limited to alignment/crosswalk/recommendation scoring",
        "not used as clustering input",
        "stage5_nmi appears in recommendation scoring" if stage5_selection_note else "no selection scoring use found",
        "warning" if stage5_selection_note else "pass",
        "medium",
        rel(code_path),
        "This is not clustering input leakage, but it should be disclosed because Stage 5 labels influence candidate recommendation.",
    )
    verify_crosswalk(audit, assignments, crosswalk, rel(crosswalk_path))
    verify_metric_summary(audit, "Stage 6", classifier, ["model"], "fold", "fold", ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "multiclass_roc_auc_ovr"], ddof=0, evidence_file=rel(classifier_path), summary_column_is_fold=True)
    audit_report_boundary(audit, "Stage 6", report, rel(report_path), expected_classifier_phrase="six-cluster-label classifier")


def audit_cross_stage(audit: Audit, tables: dict[str, pd.DataFrame]) -> None:
    raw1 = tables["raw1"]
    d1_ready = tables["d1_ready"]
    subject = tables["stage4_subject"]
    stage5 = tables["stage5_assign"]
    stage6 = tables["stage6_assign"]
    top20 = tables["stage4_top20"]
    top50 = tables["stage4_top50"]

    lineage = {
        "raw_dataset1_rows": int(raw1.shape[0]),
        "deduplicated_rows": int(d1_ready.shape[0]),
        "subject_level_rows": int(subject.shape[0]),
        "pd_subject_rows": int((subject["class"] == 1).sum()),
        "stage5_assignment_rows": int(stage5.shape[0]),
        "stage6_assignment_rows": int(stage6.shape[0]),
    }
    expected = {
        "raw_dataset1_rows": 756,
        "deduplicated_rows": 755,
        "subject_level_rows": 252,
        "pd_subject_rows": 188,
        "stage5_assignment_rows": 188,
        "stage6_assignment_rows": 188,
    }
    check_equal(audit, "Cross-stage", "Dataset 1 lineage", expected, lineage, "critical", "data/results")
    check_equal(audit, "Cross-stage", "Stage 5 and Stage 6 id set equality", True, normalize_id_set(stage5["id"]) == normalize_id_set(stage6["id"]), "critical", "results")

    top20_features = top20["feature"].astype(str).tolist()
    top50_features = top50["feature"].astype(str).tolist()
    check_equal(audit, "Cross-stage", "Top20 equals first 20 of Top50", top20_features, top50_features[:20], "high", "results")
    missing_top_features = sorted((set(top20_features) | set(top50_features)) - set(subject.columns))
    check_equal(audit, "Cross-stage", "Top20/Top50 exist in subject table", [], missing_top_features, "critical", "results")
    all_acoustic = numeric_feature_columns(subject, {"id", "gender", "class", "sex_male", "record_count"})
    check_equal(audit, "Cross-stage", "All-acoustic feature count", 752, len(all_acoustic), "critical", rel(RESULTS_DIR / "stage4_subject_level_feature_table_dataset1.csv"))


def verify_metric_summary(
    audit: Audit,
    stage: str,
    metrics: pd.DataFrame,
    group_cols: list[str],
    summary_col: str,
    fold_marker_col: str,
    metric_cols: list[str],
    ddof: int,
    evidence_file: str,
    summary_column_is_fold: bool = False,
) -> None:
    mismatches: list[str] = []
    for group_values, group in metrics.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        if summary_column_is_fold:
            fold_rows = group[~group[summary_col].astype(str).isin(["mean", "std"])]
            mean_rows = group[group[summary_col].astype(str) == "mean"]
            std_rows = group[group[summary_col].astype(str) == "std"]
        else:
            fold_rows = group[group[summary_col].astype(str) == fold_marker_col]
            mean_rows = group[group[summary_col].astype(str) == "mean"]
            std_rows = group[group[summary_col].astype(str) == "std"]
        if mean_rows.empty or std_rows.empty:
            mismatches.append(f"{group_values}: missing mean/std")
            continue
        for metric in metric_cols:
            observed_mean = float(mean_rows.iloc[0][metric])
            observed_std = float(std_rows.iloc[0][metric])
            expected_mean = float(fold_rows[metric].mean(skipna=True))
            expected_std = float(fold_rows[metric].std(ddof=ddof, skipna=True))
            if not floats_close(observed_mean, expected_mean) or not floats_close(observed_std, expected_std):
                mismatches.append(
                    f"{group_values}/{metric}: mean {observed_mean} vs {expected_mean}; std {observed_std} vs {expected_std}"
                )
    check_equal(
        audit,
        stage,
        "Metric summary rows equal fold recomputation",
        [],
        mismatches[:10],
        "high",
        evidence_file,
        notes=f"Checked {len(metric_cols)} metrics per group; ddof={ddof}.",
    )


def floats_close(left: float, right: float, tol: float = 1e-10) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return abs(left - right) <= tol


def verify_crosswalk(audit: Audit, assignments: pd.DataFrame, crosswalk: pd.DataFrame, evidence_file: str) -> None:
    observed = (
        assignments.groupby(["stage5_cluster", "stage6_cluster"])
        .size()
        .reset_index(name="count")
    )
    merged = crosswalk[["stage5_cluster", "stage6_cluster", "count"]].merge(
        observed,
        on=["stage5_cluster", "stage6_cluster"],
        how="outer",
        suffixes=("_file", "_recomputed"),
    ).fillna(0)
    mismatches = merged.loc[merged["count_file"].astype(int) != merged["count_recomputed"].astype(int)]
    check_equal(
        audit,
        "Stage 6",
        "Stage 5 x Stage 6 crosswalk counts",
        0,
        int(mismatches.shape[0]),
        "high",
        evidence_file,
    )


def audit_report_boundary(
    audit: Audit,
    stage: str,
    report: str,
    evidence_file: str,
    expected_classifier_phrase: str,
) -> None:
    lower = report.lower()
    has_classifier_phrase = expected_classifier_phrase.lower() in lower
    check_true(
        audit,
        stage,
        "Report names classifier as cluster-label classifier",
        has_classifier_phrase,
        expected_classifier_phrase,
        "found" if has_classifier_phrase else "missing",
        "high",
        evidence_file,
        fail_status="warning",
    )
    direct_symptom_terms = [term for term in CLINICAL_SYMPTOM_TERMS if term.lower() in lower]
    guarded = any(
        marker in report
        for marker in [
            "not a clinical",
            "not clinical",
            "not a clinical symptom diagnosis",
            "cannot",
            "不能",
            "不是",
            "不对应",
            "不直接",
            "而非",
        ]
    )
    status = "pass" if not direct_symptom_terms or guarded else "fail"
    audit.add_check(
        stage,
        "Report avoids treating clusters as true clinical labels",
        "no unguarded clinical symptom labels",
        direct_symptom_terms if direct_symptom_terms else "none found",
        status,
        "critical",
        evidence_file,
        "Clinical terms are acceptable only when explicitly negated or guarded.",
    )
    replacement_count = report.count("\ufffd")
    audit.add_check(
        stage,
        "Report UTF-8 replacement character check",
        "0 replacement characters",
        replacement_count,
        "warning" if replacement_count else "pass",
        "low",
        evidence_file,
        "A warning here indicates display/encoding risk, not a modeling redline.",
    )


def classify_static_match(file_path: Path, line_no: int, keyword: str, line: str) -> tuple[str, str, str]:
    normalized = line.strip()
    file_name = file_path.name
    risk = "safe"
    needs = "no"
    explanation = "Pattern is used for configuration, reporting, grouping, or explicit exclusion."

    if keyword in {"SMOTE", "SelectKBest", "cross_val_score", "GridSearchCV"}:
        if keyword in normalized and not normalized.startswith("- No"):
            risk = "warning"
            explanation = f"{keyword} appears; verify it is not used outside grouped CV."
    if keyword == "train_test_split":
        if "No row-wise random" in normalized or normalized.startswith("- No"):
            risk = "safe"
            explanation = "Only appears in report guardrail text stating it was not used."
        else:
            risk = "fail"
            needs = "yes"
            explanation = "Row-level train_test_split would violate the Stage 3 subject grouping redline."
    if keyword in {"StandardScaler(", "PCA(", "fit_transform"}:
        if file_name in {"stage3_baseline_models.py", "stage4_biomarker_identification.py"}:
            if "Pipeline" in normalized or "scaler" in normalized or keyword == "PCA(":
                risk = "safe"
                explanation = "Supervised-model scaling is inside sklearn Pipeline or no PCA is used in Stage 3/4."
            else:
                risk = "warning"
                explanation = "Verify fitting is restricted to training folds."
        elif file_name in {"stage5_pd_two_cluster_phenotyping.py", "stage6_pd_six_cluster_phenotyping.py"}:
            risk = "warning"
            explanation = "Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded."
        else:
            risk = "warning"
            explanation = "Preprocessing fit pattern needs context review."
    if keyword == "LogisticRegressionCV":
        risk = "safe" if "Pipeline" in safe_nearby_text(file_path, line_no, radius=12) else "warning"
        explanation = "Stage 4 L1 stability selection uses LogisticRegressionCV inside a fold-level Pipeline." if risk == "safe" else "Verify LogisticRegressionCV is inside a fold-level Pipeline."
    if keyword == "permutation_importance":
        risk = "safe"
        explanation = "Project implements held-out permutation importance rather than fitting on all data."
    if keyword in {"GroupKFold", "StratifiedGroupKFold"}:
        risk = "safe"
        explanation = "Grouped splitters protect subject-level train/test separation."
    if keyword in {"sex_male", "gender", "Gender", "class", "Status", "Recording", "id", "ID", "cluster", "stage5_cluster", "stage6_cluster"}:
        nearby = safe_nearby_text(file_path, line_no, radius=3).lower()
        if any(token in nearby for token in ["excluded", "not enter", "不进入", "metadata", "required", "post", "crosstab", "assignments", "audit"]):
            risk = "safe"
            explanation = "Metadata/label column appears in validation, exclusion, audit, assignment, or post-hoc description."
        elif file_name == "stage6_pd_six_cluster_phenotyping.py" and keyword == "stage5_cluster":
            risk = "warning"
            explanation = "Stage 5 labels are aligned for crosswalk and recommendation scoring; audit verifies they are not clustering inputs."
        elif keyword in {"class", "Status", "cluster", "stage5_cluster", "stage6_cluster"} and any(token in nearby for token in ["y =", "labels", "label"]):
            risk = "warning"
            explanation = "Label appears as a target or derived assignment; verify it is not part of feature matrices."
        else:
            risk = "safe"
            explanation = "Metadata/label token occurrence is not evidence of feature leakage by itself."
    return risk, explanation, needs


def safe_nearby_text(file_path: Path, line_no: int, radius: int) -> str:
    lines = safe_read_text(file_path).splitlines()
    start = max(0, line_no - radius - 1)
    end = min(len(lines), line_no + radius)
    return "\n".join(lines[start:end])


def audit_static_patterns(audit: Audit) -> None:
    for file_path in sorted(SRC_DIR.glob("*.py")):
        if file_path.name == Path(__file__).name:
            continue
        lines = safe_read_text(file_path).splitlines()
        for line_no, line in enumerate(lines, start=1):
            for keyword in STATIC_KEYWORDS:
                if keyword in line:
                    risk, explanation, needs = classify_static_match(file_path, line_no, keyword, line)
                    audit.static_rows.append(
                        {
                            "file": rel(file_path),
                            "line": line_no,
                            "keyword": keyword,
                            "code_snippet": line.strip(),
                            "risk_level": risk,
                            "explanation": explanation,
                            "needs_modification": needs,
                        }
                    )
    fail_count = sum(1 for row in audit.static_rows if row["risk_level"] == "fail")
    warning_count = sum(1 for row in audit.static_rows if row["risk_level"] == "warning")
    audit.add_check(
        "Static code audit",
        "High-risk pattern scan fail count",
        0,
        fail_count,
        "pass" if fail_count == 0 else "fail",
        "critical",
        "results/audit_static_code_risk_patterns.csv",
        f"warning_count={warning_count}; warnings require explanation but are not automatic redlines.",
    )


def write_outputs(audit: Audit) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    checks = pd.DataFrame(audit.rows, columns=CHECK_COLUMNS)
    static = pd.DataFrame(audit.static_rows)

    checks_path = RESULTS_DIR / "audit_all_stages_checks.csv"
    static_path = RESULTS_DIR / "audit_static_code_risk_patterns.csv"
    feature_path = RESULTS_DIR / "audit_feature_leakage_checks.csv"
    lineage_path = RESULTS_DIR / "audit_subject_lineage_checks.csv"
    metric_path = RESULTS_DIR / "audit_metric_consistency_checks.csv"
    report_path = RESULTS_DIR / "audit_all_stages_logic_report.md"

    checks.to_csv(checks_path, index=False, encoding=OUTPUT_CSV_ENCODING)
    static.to_csv(static_path, index=False, encoding=OUTPUT_CSV_ENCODING)

    feature_mask = checks["check_name"].str.contains(
        "feature|forbidden|sex_male|metadata|classifier|cluster input|Report avoids",
        case=False,
        regex=True,
    ) | checks["stage"].eq("Static code audit")
    checks.loc[feature_mask].to_csv(feature_path, index=False, encoding=OUTPUT_CSV_ENCODING)

    lineage_mask = checks["check_name"].str.contains(
        "lineage|row count|subject|PD-only|id set|shape|class distribution|record count",
        case=False,
        regex=True,
    )
    checks.loc[lineage_mask].to_csv(lineage_path, index=False, encoding=OUTPUT_CSV_ENCODING)

    metric_mask = checks["check_name"].str.contains(
        "Metric|metrics|Recommended|cluster sizes|crosswalk|stability|k=2",
        case=False,
        regex=True,
    )
    checks.loc[metric_mask].to_csv(metric_path, index=False, encoding=OUTPUT_CSV_ENCODING)

    report_path.write_text(build_report(checks, static), encoding="utf-8")


def build_report(checks: pd.DataFrame, static: pd.DataFrame) -> str:
    critical_fails = checks[(checks["status"] == "fail") & (checks["severity"].isin(["critical", "high"]))]
    all_fails = checks[checks["status"] == "fail"]
    warnings = checks[checks["status"] == "warning"]
    static_warnings = static[static["risk_level"] == "warning"] if not static.empty else pd.DataFrame()
    static_fails = static[static["risk_level"] == "fail"] if not static.empty else pd.DataFrame()

    if not all_fails.empty or not static_fails.empty:
        conclusion = "FAIL"
        rerun = "At least one redline or high-severity check failed; rerun the affected stage after the minimal fix."
    elif not warnings.empty or not static_warnings.empty:
        conclusion = "CONDITIONAL PASS"
        rerun = "No core stage rerun is required based on this audit; address warnings as documentation or boundary clarifications unless new evidence changes them."
    else:
        conclusion = "PASS"
        rerun = "No stage rerun is required."

    by_status = checks["status"].value_counts().to_dict()
    by_stage = checks.groupby(["stage", "status"]).size().unstack(fill_value=0).reset_index()

    lines = [
        "# Six-Stage Logic Audit Report",
        "",
        f"Final conclusion: `{conclusion}`",
        "",
        "## Rerun Decision",
        "",
        rerun,
        "",
        "## Check Summary",
        "",
        f"- Total checks: `{checks.shape[0]}`",
        f"- Status counts: `{by_status}`",
        f"- Static code findings: `{static.shape[0]}` rows; warnings `{0 if static.empty else int((static['risk_level'] == 'warning').sum())}`, fails `{0 if static.empty else int((static['risk_level'] == 'fail').sum())}`",
        "",
        markdown_table(by_stage),
        "",
        "## Redline Failures",
        "",
    ]
    if critical_fails.empty and static_fails.empty:
        lines.append("No critical/high redline failures were found.")
    else:
        fail_cols = ["stage", "check_name", "expected", "observed", "severity", "evidence_file", "notes"]
        if not critical_fails.empty:
            lines.append(markdown_table(critical_fails[fail_cols]))
        if not static_fails.empty:
            lines.append(markdown_table(static_fails[["file", "line", "keyword", "risk_level", "explanation", "needs_modification"]].head(50)))

    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    warning_cols = ["stage", "check_name", "observed", "severity", "evidence_file", "notes"]
    if warnings.empty and static_warnings.empty:
        lines.append("No warnings were recorded.")
    else:
        if not warnings.empty:
            lines.append(markdown_table(warnings[warning_cols].head(80)))
        if not static_warnings.empty:
            lines.extend(
                [
                    "",
                    "Static pattern warnings are explainability items, not automatic redlines:",
                    "",
                    markdown_table(static_warnings[["file", "line", "keyword", "risk_level", "explanation", "needs_modification"]].head(80)),
                ]
            )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `results/audit_all_stages_logic_report.md`",
            "- `results/audit_all_stages_checks.csv`",
            "- `results/audit_feature_leakage_checks.csv`",
            "- `results/audit_subject_lineage_checks.csv`",
            "- `results/audit_metric_consistency_checks.csv`",
            "- `results/audit_static_code_risk_patterns.csv`",
            "",
            "## Minimal Fix Guidance",
            "",
        ]
    )
    if conclusion == "FAIL":
        lines.append("Fix only the failed redline rows above, then rerun the affected stage and this independent audit.")
    elif conclusion == "CONDITIONAL PASS":
        lines.append("Keep the sealed modeling outputs intact. Recommended follow-up is limited to clarifying warnings, especially documentation around Stage 6 candidate recommendation scoring and any report encoding/display issues.")
    else:
        lines.append("No fixes are required by this audit.")
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = list(df.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                text = str(value)
                text = text.replace("\n", " ").replace("|", "\\|")
                values.append(text)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    audit = Audit()
    tables = audit_stage1_and_stage2(audit)
    audit_stage3(audit)
    audit_stage4(audit, tables)
    audit_stage5(audit, tables)
    audit_stage6(audit, tables)
    audit_cross_stage(audit, tables)
    audit_static_patterns(audit)
    write_outputs(audit)


if __name__ == "__main__":
    main()
