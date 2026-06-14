from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUBJECT_TABLE_PATH = RESULTS_DIR / "stage4_subject_level_feature_table_dataset1.csv"
TOP20_PATH = RESULTS_DIR / "stage4_top20_biomarkers_dataset1.csv"
TOP50_PATH = RESULTS_DIR / "stage4_top50_biomarkers_dataset1.csv"
RANK_SUMMARY_PATH = RESULTS_DIR / "stage4_biomarker_rank_summary_dataset1.csv"
STAGE5_ASSIGNMENTS_PATH = RESULTS_DIR / "stage5_cluster_assignments_dataset1_pd.csv"
STAGE5_DIFFERENCES_PATH = RESULTS_DIR / "stage5_cluster_feature_differences.csv"

OUTPUT_CSV_ENCODING = "utf-8-sig"
RANDOM_STATE = 42
EXPECTED_SUBJECTS = 252
EXPECTED_PD_SUBJECTS = 188
EXPECTED_ACOUSTIC_FEATURES = 752
PD_LABEL = 1
MAIN_K = 6
STABILITY_RUNS = 50
STABILITY_SAMPLE_FRACTION = 0.8

METADATA_COLUMNS = {"id", "gender", "class", "sex_male", "record_count"}
LABEL_COLUMN_TOKENS = ("cluster", "label", "prediction", "phenotype")
METHODS = ["KMeans", "GaussianMixture", "AgglomerativeClustering"]
FEATURE_SET_REPRESENTATIONS = {
    "top20_biomarkers": ["scaled_only", "pca_5", "pca_10"],
    "top50_biomarkers": ["pca_10", "pca_90"],
    "all_acoustic": ["pca_20", "pca_90"],
}
SAFE_LABEL_SUFFIXES = ["A", "B", "C", "D", "E", "F"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding=OUTPUT_CSV_ENCODING)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")


def is_forbidden_feature_column(column: str) -> bool:
    name = column.lower()
    return column in METADATA_COLUMNS or any(token in name for token in LABEL_COLUMN_TOKENS)


def categorize_feature(feature: str) -> str:
    name = feature.lower()
    if "tqwt" in name:
        return "TQWT"
    if any(token in name for token in ["wavelet", "_lt_", "ed_", "ed2_", "entropy"]):
        return "Wavelet"
    if "jitter" in name:
        return "Jitter"
    if "shimmer" in name or "shim_" in name or "apq" in name:
        return "Shimmer"
    if "hnr" in name or "harmonicity" in name or "harmtonoise" in name:
        return "HNR / Harmonicity"
    if "intensity" in name:
        return "Intensity"
    if (
        name.startswith("f")
        and len(name) == 2
        and name[1].isdigit()
        or name.startswith("b")
        and len(name) == 2
        and name[1].isdigit()
    ):
        return "Formant / Bandwidth"
    if "mfcc" in name:
        return "MFCC"
    if "delta_delta" in name or "delta" in name:
        return "Delta / Delta-delta"
    if any(token in name for token in ["rpde", "dfa", "ppe", "gne"]):
        return "Nonlinear"
    return "Other"


def load_inputs() -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_paths = [
        SUBJECT_TABLE_PATH,
        TOP20_PATH,
        TOP50_PATH,
        RANK_SUMMARY_PATH,
        STAGE5_ASSIGNMENTS_PATH,
        STAGE5_DIFFERENCES_PATH,
    ]
    for path in required_paths:
        require_file(path)

    subject_table = pd.read_csv(SUBJECT_TABLE_PATH)
    top20 = pd.read_csv(TOP20_PATH)
    top50 = pd.read_csv(TOP50_PATH)
    rank_summary = pd.read_csv(RANK_SUMMARY_PATH)
    stage5_assignments = pd.read_csv(STAGE5_ASSIGNMENTS_PATH)
    stage5_differences = pd.read_csv(STAGE5_DIFFERENCES_PATH)

    required_subject_columns = {"id", "class", "sex_male"}
    missing = sorted(required_subject_columns - set(subject_table.columns))
    if missing:
        raise ValueError(f"Subject-level table missing required columns: {missing}")
    if subject_table.shape[0] != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} subjects, found {subject_table.shape[0]}")
    duplicated_id_count = int(subject_table["id"].duplicated().sum())
    if duplicated_id_count:
        raise ValueError(f"Subject-level table must have one row per subject; duplicated ids={duplicated_id_count}")

    for label, table, expected_rows in [("Top20", top20, 20), ("Top50", top50, 50)]:
        if "feature" not in table.columns:
            raise ValueError(f"{label} file is missing the required 'feature' column")
        if table.shape[0] != expected_rows:
            raise ValueError(f"{label} expected {expected_rows} rows, found {table.shape[0]}")

    required_stage5_columns = {"id", "cluster"}
    missing_stage5 = sorted(required_stage5_columns - set(stage5_assignments.columns))
    if missing_stage5:
        raise ValueError(f"Stage 5 assignments missing required columns: {missing_stage5}")

    top20_features = top20["feature"].astype(str).tolist()
    top50_features = top50["feature"].astype(str).tolist()
    missing_top20 = sorted(set(top20_features) - set(subject_table.columns))
    missing_top50 = sorted(set(top50_features) - set(subject_table.columns))
    if missing_top20 or missing_top50:
        raise ValueError(
            "Stage 4 biomarker features missing from subject-level table: "
            f"top20={missing_top20}, top50={missing_top50}"
        )

    return subject_table, top20_features, top50_features, rank_summary, stage5_assignments, stage5_differences


def acoustic_columns(subject_table: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in subject_table.select_dtypes(include="number").columns
        if not is_forbidden_feature_column(column)
    ]
    if len(columns) != EXPECTED_ACOUSTIC_FEATURES:
        raise ValueError(f"Expected {EXPECTED_ACOUSTIC_FEATURES} acoustic features, found {len(columns)}")
    return columns


def validate_pd_subjects(subject_table: pd.DataFrame, stage5_assignments: pd.DataFrame) -> pd.DataFrame:
    pd_subjects = subject_table.loc[subject_table["class"] == PD_LABEL].copy()
    if pd_subjects.shape[0] != EXPECTED_PD_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_PD_SUBJECTS} PD subjects, found {pd_subjects.shape[0]}")
    overlap = set(pd_subjects["id"]).intersection(set(stage5_assignments["id"]))
    if len(overlap) != EXPECTED_PD_SUBJECTS:
        raise ValueError(f"Stage 5 assignments overlap {len(overlap)} PD subjects, expected {EXPECTED_PD_SUBJECTS}")
    return pd_subjects


def write_input_audit(
    subject_table: pd.DataFrame,
    pd_subjects: pd.DataFrame,
    all_features: list[str],
    top20_features: list[str],
    top50_features: list[str],
    stage5_assignments: pd.DataFrame,
) -> None:
    excluded_columns = [
        column if column in subject_table.columns else f"{column} (not present)"
        for column in ["id", "gender", "class", "sex_male", "record_count"]
    ]
    overlap = len(set(pd_subjects["id"]).intersection(set(stage5_assignments["id"])))
    audit = pd.DataFrame(
        [
            {
                "input_file": rel(SUBJECT_TABLE_PATH),
                "total_subjects_before_filter": int(subject_table.shape[0]),
                "pd_subjects_after_filter": int(pd_subjects.shape[0]),
                "class_distribution_before_filter": str(
                    {int(k): int(v) for k, v in subject_table["class"].value_counts().sort_index().items()}
                ),
                "class_distribution_after_filter": str(
                    {int(k): int(v) for k, v in pd_subjects["class"].value_counts().sort_index().items()}
                ),
                "missing_value_count": int(subject_table.isna().sum().sum()),
                "duplicated_id_count": int(subject_table["id"].duplicated().sum()),
                "acoustic_feature_count": int(len(all_features)),
                "excluded_columns": "; ".join(excluded_columns + ["stage5_cluster", "stage6_cluster"]),
                "feature_sets_used": "top20_biomarkers; top50_biomarkers; all_acoustic",
                "top20_feature_count": int(len(top20_features)),
                "top50_feature_count": int(len(top50_features)),
                "all_acoustic_feature_count": int(len(all_features)),
                "stage5_assignment_file_used": rel(STAGE5_ASSIGNMENTS_PATH),
                "stage5_label_overlap_subject_count": int(overlap),
            }
        ]
    )
    write_csv(audit, RESULTS_DIR / "stage6_six_cluster_input_audit.csv")


def build_representation(data: pd.DataFrame, representation: str) -> tuple[np.ndarray, int, float]:
    scaled = StandardScaler().fit_transform(data.to_numpy(dtype=float))
    if representation == "scaled_only":
        return scaled, scaled.shape[1], np.nan
    if representation == "pca_90":
        pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    elif representation.startswith("pca_"):
        pca = PCA(n_components=int(representation.split("_", maxsplit=1)[1]), random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    projected = pca.fit_transform(scaled)
    return projected, projected.shape[1], float(np.sum(pca.explained_variance_ratio_))


def cluster_labels(matrix: np.ndarray, method: str, k: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    if method == "KMeans":
        return KMeans(n_clusters=k, n_init=100, random_state=random_state).fit_predict(matrix)
    if method == "GaussianMixture":
        return GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state,
        ).fit_predict(matrix)
    if method == "AgglomerativeClustering":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(matrix)
    raise ValueError(f"Unknown clustering method: {method}")


def clustering_quality(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, str]:
    if np.unique(labels).size < 2:
        return np.nan, np.nan, np.nan, "single cluster produced"
    try:
        return (
            float(silhouette_score(matrix, labels)),
            float(calinski_harabasz_score(matrix, labels)),
            float(davies_bouldin_score(matrix, labels)),
            "",
        )
    except ValueError as exc:
        return np.nan, np.nan, np.nan, f"metric calculation failed: {exc}"


def stability_analysis(
    data: pd.DataFrame,
    representation: str,
    method: str,
    k: int,
    full_labels: np.ndarray,
    n_runs: int = STABILITY_RUNS,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE + k)
    n_subjects = data.shape[0]
    sample_size = int(round(n_subjects * STABILITY_SAMPLE_FRACTION))
    scores: list[float] = []
    for run_index in range(n_runs):
        sample_index = np.sort(rng.choice(n_subjects, size=sample_size, replace=False))
        sample_matrix, _, _ = build_representation(data.iloc[sample_index], representation)
        sample_labels = cluster_labels(sample_matrix, method, k, random_state=RANDOM_STATE + run_index + 1)
        scores.append(float(adjusted_rand_score(full_labels[sample_index], sample_labels)))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def cluster_size_stats(labels: np.ndarray, k: int) -> dict[str, Any]:
    counts = pd.Series(labels).value_counts().reindex(range(k), fill_value=0).astype(int)
    min_size = int(counts.min())
    max_size = int(counts.max())
    min_ratio = float(min_size / len(labels))
    max_ratio = float(max_size / len(labels))
    return {
        "cluster_sizes": ";".join(f"{cluster}:{int(size)}" for cluster, size in counts.items()),
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "min_cluster_ratio": min_ratio,
        "max_cluster_ratio": max_ratio,
        "size_imbalance_ratio": float(max_size / min_size) if min_size > 0 else np.inf,
    }


def run_clustering_candidates(
    pd_subjects: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    k: int = MAIN_K,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    artifacts: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    for feature_set, features in feature_sets.items():
        data = pd_subjects[features]
        for representation in FEATURE_SET_REPRESENTATIONS[feature_set]:
            matrix, n_components, explained_variance = build_representation(data, representation)
            for method in METHODS:
                labels = cluster_labels(matrix, method, k)
                size_stats = cluster_size_stats(labels, k)
                silhouette, calinski, davies, metric_warning = clustering_quality(matrix, labels)
                stability_mean, stability_std = stability_analysis(data, representation, method, k, labels)

                warnings = []
                if metric_warning:
                    warnings.append(metric_warning)
                if size_stats["min_cluster_size"] < 8:
                    warnings.append("min_cluster_size < 8; not eligible for recommendation")
                if size_stats["min_cluster_ratio"] < 0.05:
                    warnings.append("min_cluster_ratio < 0.05; not eligible for recommendation")

                key = (feature_set, representation, method, k)
                rows.append(
                    {
                        "feature_set": feature_set,
                        "representation": representation,
                        "clustering_method": method,
                        "k": int(k),
                        "n_subjects": int(len(labels)),
                        "n_features_original": int(len(features)),
                        "n_components_after_pca": int(n_components),
                        "pca_explained_variance_sum": explained_variance,
                        **size_stats,
                        "silhouette": silhouette,
                        "calinski_harabasz": calinski,
                        "davies_bouldin": davies,
                        "stability_ari_mean": stability_mean,
                        "stability_ari_std": stability_std,
                        "stability_n_runs": STABILITY_RUNS,
                        "recommended_flag": False,
                        "exclusion_or_warning": "; ".join(warnings),
                    }
                )
                artifacts[key] = {
                    "labels": labels,
                    "matrix": matrix,
                    "features": features,
                    "n_components_after_pca": n_components,
                    "pca_explained_variance_sum": explained_variance,
                }
    return pd.DataFrame(rows), artifacts


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    valid = series.astype(float).replace([np.inf, -np.inf], np.nan)
    min_value = valid.min(skipna=True)
    max_value = valid.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.5, index=series.index)
    scaled = (valid - min_value) / (max_value - min_value)
    if not higher_is_better:
        scaled = 1 - scaled
    return scaled.fillna(0.0)


def stage5_alignment_score(labels: np.ndarray, stage5_labels: np.ndarray) -> float:
    nmi = normalized_mutual_info_score(stage5_labels, labels)
    return float(nmi)


def choose_recommended(
    metrics: pd.DataFrame,
    artifacts: dict[tuple[str, str, str, int], dict[str, Any]],
    stage5_labels: np.ndarray,
) -> tuple[pd.DataFrame, tuple[str, str, str, int]]:
    out = metrics.copy()
    alignments = []
    for _, row in out.iterrows():
        key = (row["feature_set"], row["representation"], row["clustering_method"], int(row["k"]))
        alignments.append(stage5_alignment_score(artifacts[key]["labels"], stage5_labels))
    out["stage5_nmi"] = alignments

    eligible = (out["min_cluster_size"] >= 8) & (out["min_cluster_ratio"] >= 0.05)
    if not eligible.any():
        raise ValueError("No k=6 candidate is eligible after min cluster constraints")

    balance = 1 - (out["max_cluster_ratio"] - out["min_cluster_ratio"]).clip(lower=0)
    score = (
        0.20 * minmax(out["silhouette"], True)
        + 0.12 * minmax(out["calinski_harabasz"], True)
        + 0.13 * minmax(out["davies_bouldin"], False)
        + 0.30 * minmax(out["stability_ari_mean"], True)
        + 0.15 * balance
        + 0.10 * minmax(out["stage5_nmi"], True)
    )
    score += out["feature_set"].map(
        {"top20_biomarkers": 0.04, "top50_biomarkers": 0.08, "all_acoustic": -0.03}
    )
    score = score.where(eligible, -np.inf)
    out["selection_score"] = score
    best_index = int(out["selection_score"].idxmax())
    best_score = float(out.loc[best_index, "selection_score"])

    interpretable_close = out[
        (out["feature_set"].isin(["top20_biomarkers", "top50_biomarkers"]))
        & eligible
        & (out["selection_score"] >= best_score - 0.08)
    ]
    if not interpretable_close.empty and out.loc[best_index, "feature_set"] == "all_acoustic":
        best_index = int(interpretable_close["selection_score"].idxmax())

    top50_close = out[
        (out["feature_set"] == "top50_biomarkers")
        & eligible
        & (out["selection_score"] >= float(out.loc[best_index, "selection_score"]) - 0.03)
    ]
    if not top50_close.empty and out.loc[best_index, "feature_set"] == "top20_biomarkers":
        best_index = int(top50_close["selection_score"].idxmax())

    out["recommended_flag"] = False
    out.loc[best_index, "recommended_flag"] = True
    key = (
        str(out.loc[best_index, "feature_set"]),
        str(out.loc[best_index, "representation"]),
        str(out.loc[best_index, "clustering_method"]),
        int(out.loc[best_index, "k"]),
    )
    return out, key


def run_k_scan(
    pd_subjects: pd.DataFrame,
    features: list[str],
    feature_set: str,
    representation: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = pd_subjects[features]
    for k in range(2, 9):
        matrix, n_components, explained_variance = build_representation(data, representation)
        for method in METHODS:
            labels = cluster_labels(matrix, method, k)
            size_stats = cluster_size_stats(labels, k)
            silhouette, calinski, davies, metric_warning = clustering_quality(matrix, labels)
            stability_mean, stability_std = stability_analysis(data, representation, method, k, labels)
            rows.append(
                {
                    "feature_set": feature_set,
                    "representation": representation,
                    "clustering_method": method,
                    "k": int(k),
                    "n_subjects": int(len(labels)),
                    "n_features_original": int(len(features)),
                    "n_components_after_pca": int(n_components),
                    "pca_explained_variance_sum": explained_variance,
                    **size_stats,
                    "silhouette": silhouette,
                    "calinski_harabasz": calinski,
                    "davies_bouldin": davies,
                    "stability_ari_mean": stability_mean,
                    "stability_ari_std": stability_std,
                    "stability_n_runs": STABILITY_RUNS,
                    "exclusion_or_warning": metric_warning,
                }
            )
    k_scan = pd.DataFrame(rows)
    write_csv(k_scan, RESULTS_DIR / "stage6_k_scan_metrics.csv")
    return k_scan


def consensus_kmeans_summary(data: pd.DataFrame, representation: str, labels: np.ndarray) -> dict[str, float]:
    n_subjects = data.shape[0]
    coassociation = np.zeros((n_subjects, n_subjects), dtype=float)
    coobserved = np.zeros((n_subjects, n_subjects), dtype=float)
    rng = np.random.default_rng(RANDOM_STATE + 600)
    sample_size = int(round(n_subjects * STABILITY_SAMPLE_FRACTION))
    for run_index in range(STABILITY_RUNS):
        sample_index = np.sort(rng.choice(n_subjects, size=sample_size, replace=False))
        sample_matrix, _, _ = build_representation(data.iloc[sample_index], representation)
        sample_labels = cluster_labels(sample_matrix, "KMeans", MAIN_K, random_state=RANDOM_STATE + run_index + 1)
        for i_pos, i_subject in enumerate(sample_index):
            same = sample_labels == sample_labels[i_pos]
            coassociation[i_subject, sample_index[same]] += 1
            coobserved[i_subject, sample_index] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.divide(coassociation, coobserved, out=np.full_like(coassociation, np.nan), where=coobserved > 0)
    same_mask = labels[:, None] == labels[None, :]
    diff_mask = labels[:, None] != labels[None, :]
    return {
        "consensus_within_cluster_mean": float(np.nanmean(consensus[same_mask])),
        "consensus_between_cluster_mean": float(np.nanmean(consensus[diff_mask])),
        "consensus_separation": float(np.nanmean(consensus[same_mask]) - np.nanmean(consensus[diff_mask])),
    }


def pca_2d_coordinates(data: pd.DataFrame) -> np.ndarray:
    scaled = StandardScaler().fit_transform(data.to_numpy(dtype=float))
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)


def aligned_stage5(stage5_assignments: pd.DataFrame, pd_subjects: pd.DataFrame) -> pd.Series:
    stage5 = stage5_assignments[["id", "cluster"]].rename(columns={"cluster": "stage5_cluster"})
    merged = pd_subjects[["id"]].merge(stage5, on="id", how="left")
    if merged["stage5_cluster"].isna().any():
        raise ValueError("Missing Stage 5 cluster labels after aligning to PD subject order")
    return merged["stage5_cluster"].astype(int)


def write_assignments(
    pd_subjects: pd.DataFrame,
    stage5_labels: pd.Series,
    recommended_key: tuple[str, str, str, int],
    artifact: dict[str, Any],
) -> pd.DataFrame:
    feature_set, representation, method, _ = recommended_key
    labels = np.asarray(artifact["labels"], dtype=int)
    counts = pd.Series(labels).value_counts().to_dict()
    coords = pca_2d_coordinates(pd_subjects[artifact["features"]])
    assignments = pd.DataFrame(
        {
            "id": pd_subjects["id"].to_numpy(),
            "class": pd_subjects["class"].to_numpy(),
            "sex_male": pd_subjects["sex_male"].to_numpy(),
            "stage5_cluster": stage5_labels.to_numpy(),
            "stage6_cluster": labels,
            "stage6_cluster_size": [int(counts[label]) for label in labels],
            "stage6_cluster_ratio": [float(counts[label] / len(labels)) for label in labels],
            "feature_set": feature_set,
            "representation": representation,
            "clustering_method": method,
            "pca_1": coords[:, 0],
            "pca_2": coords[:, 1],
        }
    )
    write_csv(assignments, RESULTS_DIR / "stage6_six_cluster_assignments_dataset1_pd.csv")
    return assignments


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return adjusted
    valid_p = p[valid]
    order = np.argsort(valid_p)
    ordered_p = valid_p[order]
    n = len(ordered_p)
    ordered_q = ordered_p * n / np.arange(1, n + 1)
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    ordered_q = np.clip(ordered_q, 0, 1)
    q = np.empty_like(ordered_q)
    q[order] = ordered_q
    adjusted[valid] = q
    return adjusted


def cohens_d(values_1: np.ndarray, values_0: np.ndarray) -> float:
    n_1 = len(values_1)
    n_0 = len(values_0)
    if n_1 < 2 or n_0 < 2:
        return np.nan
    var_1 = np.var(values_1, ddof=1)
    var_0 = np.var(values_0, ddof=1)
    pooled = ((n_1 - 1) * var_1 + (n_0 - 1) * var_0) / (n_1 + n_0 - 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(values_1) - np.mean(values_0)) / np.sqrt(pooled))


def build_feature_profiles(
    pd_subjects: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd_subjects[features]
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(raw.to_numpy(dtype=float)),
        columns=features,
        index=raw.index,
    )
    rows: list[dict[str, Any]] = []
    for feature in features:
        row: dict[str, Any] = {"feature": feature, "category": categorize_feature(feature)}
        cluster_values = []
        for cluster in range(MAIN_K):
            values = raw.loc[labels == cluster, feature].to_numpy(dtype=float)
            z_values = scaled.loc[labels == cluster, feature].to_numpy(dtype=float)
            row[f"cluster_{cluster}_mean"] = float(np.mean(values))
            row[f"cluster_{cluster}_zmean"] = float(np.mean(z_values))
            cluster_values.append(values)
        zmeans = np.array([row[f"cluster_{cluster}_zmean"] for cluster in range(MAIN_K)], dtype=float)
        try:
            h_stat, p_value = kruskal(*cluster_values)
        except ValueError:
            h_stat, p_value = np.nan, np.nan
        row["max_abs_zmean_difference"] = float(np.max(zmeans) - np.min(zmeans))
        row["kruskal_h"] = float(h_stat)
        row["p_kruskal"] = float(p_value)
        rows.append(row)

    profiles = pd.DataFrame(rows)
    profiles["q_kruskal"] = fdr_bh(profiles["p_kruskal"].to_numpy())
    ordered_columns = (
        ["feature", "category"]
        + [f"cluster_{cluster}_mean" for cluster in range(MAIN_K)]
        + [f"cluster_{cluster}_zmean" for cluster in range(MAIN_K)]
        + ["max_abs_zmean_difference", "kruskal_h", "p_kruskal", "q_kruskal"]
    )
    profiles = profiles[ordered_columns].sort_values(
        ["max_abs_zmean_difference", "q_kruskal"],
        ascending=[False, True],
    ).reset_index(drop=True)

    top_rows: list[dict[str, Any]] = []
    for cluster in range(MAIN_K):
        other_mask = labels != cluster
        cluster_mask = labels == cluster
        for _, row in profiles.iterrows():
            feature = str(row["feature"])
            zmean = float(row[f"cluster_{cluster}_zmean"])
            other_zmean = float(np.mean([row[f"cluster_{other}_zmean"] for other in range(MAIN_K) if other != cluster]))
            contrast = zmean - other_zmean
            d_value = cohens_d(
                raw.loc[cluster_mask, feature].to_numpy(dtype=float),
                raw.loc[other_mask, feature].to_numpy(dtype=float),
            )
            top_rows.append(
                {
                    "stage6_cluster": cluster,
                    "feature": feature,
                    "category": row["category"],
                    "cluster_zmean": zmean,
                    "other_clusters_mean_zmean": other_zmean,
                    "one_vs_rest_zmean_difference": contrast,
                    "one_vs_rest_cohens_d": d_value,
                    "abs_one_vs_rest_zmean_difference": abs(contrast),
                    "abs_one_vs_rest_cohens_d": abs(d_value) if np.isfinite(d_value) else np.nan,
                    "q_kruskal": row["q_kruskal"],
                    "direction": "positive" if contrast >= 0 else "negative",
                }
            )
    top_by_cluster = (
        pd.DataFrame(top_rows)
        .sort_values(
            ["stage6_cluster", "abs_one_vs_rest_zmean_difference", "abs_one_vs_rest_cohens_d", "q_kruskal"],
            ascending=[True, False, False, True],
        )
        .groupby("stage6_cluster", group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )

    write_csv(profiles, RESULTS_DIR / "stage6_six_cluster_feature_profiles.csv")
    write_csv(top_by_cluster, RESULTS_DIR / "stage6_six_cluster_top_features_by_cluster.csv")
    return profiles, top_by_cluster


def category_phrase(category: str) -> str:
    mapping = {
        "Delta / Delta-delta": "dynamic_delta",
        "TQWT": "tqwt",
        "MFCC": "mfcc",
        "Shimmer": "shimmer",
        "Jitter": "jitter",
        "HNR / Harmonicity": "harmonicity",
        "Nonlinear": "nonlinear",
        "Intensity": "intensity",
        "Formant / Bandwidth": "formant_bandwidth",
        "Wavelet": "wavelet",
        "Other": "mixed_acoustic",
    }
    return mapping.get(category, "mixed_acoustic")


def build_cluster_profiles(
    assignments: pd.DataFrame,
    top_by_cluster: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster in range(MAIN_K):
        cluster_assignments = assignments.loc[assignments["stage6_cluster"] == cluster]
        cluster_features = top_by_cluster.loc[top_by_cluster["stage6_cluster"] == cluster]
        category_counts = cluster_features["category"].value_counts()
        dominant_categories = "; ".join(f"{cat}:{int(count)}" for cat, count in category_counts.items())
        positives = cluster_features.loc[cluster_features["one_vs_rest_zmean_difference"] >= 0].head(5)
        negatives = cluster_features.loc[cluster_features["one_vs_rest_zmean_difference"] < 0].head(5)
        if positives.empty:
            positives = cluster_features.sort_values("one_vs_rest_zmean_difference", ascending=False).head(5)
        if negatives.empty:
            negatives = cluster_features.sort_values("one_vs_rest_zmean_difference", ascending=True).head(5)
        dominant_category = category_counts.index[0] if not category_counts.empty else "Other"
        direction = "high" if positives["one_vs_rest_zmean_difference"].mean() >= abs(negatives["one_vs_rest_zmean_difference"].mean()) else "low"
        label = f"latent_voice_phenotype_{SAFE_LABEL_SUFFIXES[cluster]}_{category_phrase(str(dominant_category))}_{direction}"
        rows.append(
            {
                "stage6_cluster": cluster,
                "n_subjects": int(cluster_assignments.shape[0]),
                "ratio": float(cluster_assignments.shape[0] / assignments.shape[0]),
                "dominant_feature_categories": dominant_categories,
                "top_positive_features": "; ".join(positives["feature"].astype(str).tolist()),
                "top_negative_features": "; ".join(negatives["feature"].astype(str).tolist()),
                "stage5_cluster_distribution": str(
                    {int(k): int(v) for k, v in cluster_assignments["stage5_cluster"].value_counts().sort_index().items()}
                ),
                "sex_male_distribution": str(
                    {int(k): int(v) for k, v in cluster_assignments["sex_male"].value_counts().sort_index().items()}
                ),
                "tentative_voice_phenotype_label": label,
                "interpretation_note": "Exploratory acoustic phenotype label; not a clinical symptom diagnosis.",
            }
        )
    profiles = pd.DataFrame(rows)
    write_csv(profiles, RESULTS_DIR / "stage6_six_cluster_profiles.csv")
    return profiles


def stage5_crosswalk(assignments: pd.DataFrame) -> pd.DataFrame:
    crosstab = pd.crosstab(assignments["stage5_cluster"], assignments["stage6_cluster"])
    nmi = float(normalized_mutual_info_score(assignments["stage5_cluster"], assignments["stage6_cluster"]))
    ari = float(adjusted_rand_score(assignments["stage5_cluster"], assignments["stage6_cluster"]))
    majority_map = {}
    for stage6_cluster in sorted(assignments["stage6_cluster"].unique()):
        subset = assignments.loc[assignments["stage6_cluster"] == stage6_cluster, "stage5_cluster"]
        majority_map[int(stage6_cluster)] = int(subset.value_counts().idxmax())
    collapsed = assignments["stage6_cluster"].map(majority_map)
    collapsed_ari = float(adjusted_rand_score(assignments["stage5_cluster"], collapsed))

    rows: list[dict[str, Any]] = []
    for stage5_cluster in crosstab.index:
        for stage6_cluster in crosstab.columns:
            count = int(crosstab.loc[stage5_cluster, stage6_cluster])
            row_total = int(crosstab.loc[stage5_cluster].sum())
            col_total = int(crosstab[stage6_cluster].sum())
            rows.append(
                {
                    "stage5_cluster": int(stage5_cluster),
                    "stage6_cluster": int(stage6_cluster),
                    "count": count,
                    "row_percentage": float(count / row_total) if row_total else 0.0,
                    "column_percentage": float(count / col_total) if col_total else 0.0,
                    "adjusted_rand_index_stage5_vs_stage6": ari,
                    "normalized_mutual_info_stage5_vs_stage6": nmi,
                    "collapsed_stage6_to_stage5_majority_ari": collapsed_ari,
                }
            )
    crosswalk = pd.DataFrame(rows)
    write_csv(crosswalk, RESULTS_DIR / "stage6_stage5_crosswalk.csv")
    return crosswalk


def run_classifier(pd_subjects: pd.DataFrame, features: list[str], labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd_subjects[features].to_numpy(dtype=float)
    y = labels.astype(int)
    min_cluster_size = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_cluster_size)
    if n_splits < 2:
        raise ValueError("six-cluster-label classifier requires at least 2 samples per class")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    models: dict[str, Any] = {
        "Multinomial Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVM-RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    metric_rows: list[dict[str, Any]] = []
    cm_rows: list[dict[str, Any]] = []
    labels_order = list(range(MAIN_K))

    for model_name, model in models.items():
        fold_rows: list[dict[str, Any]] = []
        for fold_index, (train_index, test_index) in enumerate(cv.split(x, y), start=1):
            model.fit(x[train_index], y[train_index])
            predictions = model.predict(x[test_index])
            try:
                probabilities = model.predict_proba(x[test_index])
                if probabilities.shape[1] == MAIN_K and set(np.unique(y[test_index])) == set(labels_order):
                    auc = float(roc_auc_score(y[test_index], probabilities, multi_class="ovr", labels=labels_order))
                else:
                    auc = np.nan
            except (AttributeError, ValueError):
                auc = np.nan
            row = {
                "model": model_name,
                "fold": fold_index,
                "n_splits": n_splits,
                "accuracy": float(accuracy_score(y[test_index], predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_index], predictions)),
                "f1_macro": float(f1_score(y[test_index], predictions, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y[test_index], predictions, average="weighted", zero_division=0)),
                "precision_macro": float(precision_score(y[test_index], predictions, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y[test_index], predictions, average="macro", zero_division=0)),
                "multiclass_roc_auc_ovr": auc,
            }
            metric_rows.append(row)
            fold_rows.append(row)
            matrix = confusion_matrix(y[test_index], predictions, labels=labels_order)
            for true_label in labels_order:
                for predicted_label in labels_order:
                    cm_rows.append(
                        {
                            "model": model_name,
                            "fold": fold_index,
                            "true_stage6_cluster": true_label,
                            "predicted_stage6_cluster": predicted_label,
                            "count": int(matrix[true_label, predicted_label]),
                        }
                    )
        fold_metrics = pd.DataFrame(fold_rows)
        for summary_name, reducer in [("mean", np.nanmean), ("std", np.nanstd)]:
            metric_rows.append(
                {
                    "model": model_name,
                    "fold": summary_name,
                    "n_splits": n_splits,
                    "accuracy": float(reducer(fold_metrics["accuracy"])),
                    "balanced_accuracy": float(reducer(fold_metrics["balanced_accuracy"])),
                    "f1_macro": float(reducer(fold_metrics["f1_macro"])),
                    "f1_weighted": float(reducer(fold_metrics["f1_weighted"])),
                    "precision_macro": float(reducer(fold_metrics["precision_macro"])),
                    "recall_macro": float(reducer(fold_metrics["recall_macro"])),
                    "multiclass_roc_auc_ovr": float(reducer(fold_metrics["multiclass_roc_auc_ovr"])),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(cm_rows)
    write_csv(metrics, RESULTS_DIR / "stage6_six_cluster_label_classifier_metrics.csv")
    write_csv(confusion, RESULTS_DIR / "stage6_six_cluster_label_classifier_confusion_matrices.csv")
    return metrics, confusion


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df.loc[:, columns].copy()
    table.columns = [str(column) for column in table.columns]
    if max_rows is not None:
        table = table.head(max_rows)
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, separator, *rows])


def plot_pca_clusters(assignments: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in sorted(assignments["stage6_cluster"].unique()):
        subset = assignments.loc[assignments["stage6_cluster"] == cluster]
        ax.scatter(subset["pca_1"], subset["pca_2"], s=44, alpha=0.85, label=f"Cluster {cluster}")
    ax.set_title("Stage 6 Main k=6 Latent Voice Phenotype Clusters")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(title="Stage 6 cluster", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage6_pca_k6_clusters_main.png", dpi=220)
    plt.close(fig)


def plot_cluster_size(assignments: pd.DataFrame) -> None:
    counts = assignments["stage6_cluster"].value_counts().sort_index()
    ratios = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#4c78a8")
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{ratio:.1%}", ha="center", va="bottom")
    ax.set_title("Stage 6 Cluster Size Distribution")
    ax.set_xlabel("Stage 6 cluster")
    ax.set_ylabel("Number of PD subjects")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage6_cluster_size_distribution.png", dpi=220)
    plt.close(fig)


def plot_candidate_metrics(metrics: pd.DataFrame) -> None:
    plot_data = metrics.copy()
    plot_data["candidate"] = (
        plot_data["feature_set"] + "\n" + plot_data["representation"] + "\n" + plot_data["clustering_method"]
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, metric, title in [
        (axes[0], "silhouette", "Silhouette"),
        (axes[1], "davies_bouldin", "Davies-Bouldin"),
        (axes[2], "stability_ari_mean", "Stability ARI Mean"),
    ]:
        colors = np.where(plot_data["recommended_flag"], "#d55e00", "#4c78a8")
        ax.bar(np.arange(plot_data.shape[0]), plot_data[metric], color=colors)
        ax.set_ylabel(title)
        ax.set_title(f"Stage 6 Candidate {title}")
    axes[-1].set_xticks(np.arange(plot_data.shape[0]))
    axes[-1].set_xticklabels(plot_data["candidate"], rotation=75, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage6_candidate_metrics_comparison.png", dpi=220)
    plt.close(fig)


def plot_feature_heatmap(feature_profiles: pd.DataFrame) -> None:
    top = feature_profiles.head(20)
    matrix = top[[f"cluster_{cluster}_zmean" for cluster in range(MAIN_K)]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(top.shape[0]))
    ax.set_yticklabels(top["feature"], fontsize=7)
    ax.set_xticks(np.arange(MAIN_K))
    ax.set_xticklabels([f"Cluster {cluster}" for cluster in range(MAIN_K)])
    ax.set_title("Stage 6 Cluster z-Mean Heatmap for Top Features")
    ax.set_xlabel("Stage 6 cluster")
    ax.set_ylabel("Feature")
    fig.colorbar(image, ax=ax, label="Cluster z-mean")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage6_cluster_feature_heatmap.png", dpi=220)
    plt.close(fig)


def plot_crosswalk_heatmap(crosswalk: pd.DataFrame) -> None:
    pivot = crosswalk.pivot(index="stage5_cluster", columns="stage6_cluster", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"Stage 6 {column}" for column in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"Stage 5 {index}" for index in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", color="#111111")
    ax.set_title("Stage 5 Two-Cluster vs Stage 6 Six-Cluster Crosswalk")
    ax.set_xlabel("Stage 6 cluster")
    ax.set_ylabel("Stage 5 cluster")
    fig.colorbar(image, ax=ax, label="Subject count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage6_stage5_crosswalk_heatmap.png", dpi=220)
    plt.close(fig)


def plot_top_features_by_cluster(top_by_cluster: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    for cluster, ax in enumerate(axes):
        data = top_by_cluster.loc[top_by_cluster["stage6_cluster"] == cluster].copy()
        data = data.sort_values("one_vs_rest_zmean_difference")
        colors = np.where(data["one_vs_rest_zmean_difference"] >= 0, "#2f80ed", "#e07a5f")
        ax.barh(data["feature"], data["one_vs_rest_zmean_difference"], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlabel("One-vs-rest z-mean difference")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Stage 6 Top Distinguishing Features by Cluster", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage6_top_features_by_cluster.png", dpi=220)
    plt.close(fig)


def report_selection_reason(metrics: pd.DataFrame, recommended: pd.Series) -> str:
    top = metrics.sort_values("selection_score", ascending=False).head(5)
    text = (
        f"主推荐方案为 `{recommended['feature_set']} / {recommended['representation']} / "
        f"{recommended['clustering_method']}`。该方案的最小簇人数为 `{int(recommended['min_cluster_size'])}`，"
        f"最小簇比例为 `{recommended['min_cluster_ratio']:.3f}`，稳定性 ARI 均值为 "
        f"`{recommended['stability_ari_mean']:.3f}`。选择时同时考虑了簇大小、稳定性、聚类指标、"
        "特征集可解释性、是否依赖高维噪声，以及与 Stage 5 二类语音表型的关系。\n\n"
        "没有只按单项指标选择方案。若某些候选 silhouette 或 Calinski-Harabasz 较高，但存在小簇、"
        "不稳定或高维噪声依赖，则不作为论文主解释方案。\n\n"
    )
    return text + markdown_table(
        top,
        [
            "feature_set",
            "representation",
            "clustering_method",
            "selection_score",
            "min_cluster_size",
            "min_cluster_ratio",
            "silhouette",
            "stability_ari_mean",
            "stage5_nmi",
            "recommended_flag",
        ],
    )


def write_report(
    subject_table: pd.DataFrame,
    pd_subjects: pd.DataFrame,
    metrics: pd.DataFrame,
    k_scan: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_profiles: pd.DataFrame,
    top_by_cluster: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    crosswalk: pd.DataFrame,
    classifier_metrics: pd.DataFrame,
    consensus: dict[str, float],
) -> None:
    recommended = metrics.loc[metrics["recommended_flag"]].iloc[0]
    classifier_summary = classifier_metrics[classifier_metrics["fold"].isin(["mean", "std"])]
    cluster_size_table = (
        assignments.groupby("stage6_cluster")
        .size()
        .reset_index(name="n_subjects")
    )
    cluster_size_table["ratio"] = cluster_size_table["n_subjects"] / assignments.shape[0]
    sex_table = pd.crosstab(assignments["stage6_cluster"], assignments["sex_male"], margins=True)
    category_counts = top_by_cluster["category"].value_counts().reset_index()
    category_counts.columns = ["category", "top_feature_count"]
    boundary_text = (
        "由于附件数据未提供静止性震颤、运动迟缓、肌肉僵直、疼痛、痴呆、睡眠障碍的真实临床标签，"
        "本文不直接构建六类症状监督分类模型，而是在帕金森病受试者内部进行 k=6 无监督聚类，"
        "以获得基于语音特征的六类潜在表型。随后训练的分类器预测的是聚类标签，而非临床确诊标签。"
        "因此，该模型只能作为六类帕金森病症状辅助诊断的探索性语音表型参考，仍需真实临床症状标签进一步验证。"
    )
    report = [
        "# Stage 6 Dataset 1 PD Six-Cluster Voice Phenotyping",
        "",
        "## 1. Stage 6 目标与边界",
        "本阶段在 PD 受试者内部进行 k=6 无监督聚类，输出基于语音特征的六类潜在表型，并训练 six-cluster-label classifier 预测聚类标签。",
        "六个 cluster 编号没有临床症状含义。",
        "",
        "## 2. 为什么不能做真实六类临床症状监督诊断",
        boundary_text,
        "",
        "## 3. 输入文件与样本审计",
        f"- Subject-level input: `{rel(SUBJECT_TABLE_PATH)}`",
        f"- Total subjects: `{subject_table.shape[0]}`",
        f"- PD subjects: `{pd_subjects.shape[0]}`",
        f"- Class distribution: `{subject_table['class'].value_counts().sort_index().to_dict()}`",
        f"- Missing values: `{int(subject_table.isna().sum().sum())}`",
        f"- Duplicated id count: `{int(subject_table['id'].duplicated().sum())}`",
        f"- Stage 5 assignment file: `{rel(STAGE5_ASSIGNMENTS_PATH)}`",
        "",
        "## 4. 特征集与预处理",
        "使用 Top20、Top50 和 all_acoustic 三套特征集，所有表示均先 StandardScaler，再按需要 PCA。`id`, `gender`, `class`, `sex_male`, `record_count`, Stage 5 cluster 和 Stage 6 cluster 不进入聚类或分类器输入。",
        "",
        "## 5. k=6 聚类候选方案完整指标表",
        markdown_table(
            metrics,
            [
                "feature_set",
                "representation",
                "clustering_method",
                "cluster_sizes",
                "min_cluster_size",
                "min_cluster_ratio",
                "silhouette",
                "davies_bouldin",
                "stability_ari_mean",
                "recommended_flag",
            ],
        ),
        "",
        "## 6. k=2 到 k=8 参考扫描结果",
        "k 扫描只用于说明 k=6 的相对聚类质量和稳定性；k=6 是由题目六类设定驱动的探索性语音表型划分，不是由真实六类临床标签监督得到。",
        markdown_table(
            k_scan,
            [
                "k",
                "clustering_method",
                "min_cluster_size",
                "silhouette",
                "davies_bouldin",
                "stability_ari_mean",
            ],
            max_rows=21,
        ),
        "",
        "## 7. 稳定性分析结果",
        f"每个 k=6 候选和 k 扫描方案均执行 `{STABILITY_RUNS}` 次 80% PD 受试者重采样，重新标准化、PCA 与聚类，并计算 ARI。",
        f"主方案 Consensus KMeans 参考：within-cluster mean `{consensus['consensus_within_cluster_mean']:.3f}`，between-cluster mean `{consensus['consensus_between_cluster_mean']:.3f}`，separation `{consensus['consensus_separation']:.3f}`。",
        "",
        "## 8. 主推荐方案选择理由",
        report_selection_reason(metrics, recommended),
        "",
        "## 9. 六个簇的人数与比例",
        markdown_table(cluster_size_table, ["stage6_cluster", "n_subjects", "ratio"]),
        "",
        "## 10. 六个簇的声学特征画像",
        markdown_table(
            cluster_profiles,
            [
                "stage6_cluster",
                "n_subjects",
                "ratio",
                "dominant_feature_categories",
                "tentative_voice_phenotype_label",
            ],
        ),
        "",
        "## 11. 每个簇的 Top 区分语音特征",
        markdown_table(
            top_by_cluster,
            [
                "stage6_cluster",
                "feature",
                "category",
                "one_vs_rest_zmean_difference",
                "one_vs_rest_cohens_d",
                "q_kruskal",
            ],
            max_rows=60,
        ),
        "",
        "## 12. 特征类别分布与解释",
        markdown_table(category_counts, ["category", "top_feature_count"]),
        "这些类别分布用于描述六类潜在语音表型的声学差异来源，不是临床病因解释。",
        "",
        "## 13. Stage 5 二类表型与 Stage 6 六类表型关系",
        markdown_table(
            crosswalk,
            [
                "stage5_cluster",
                "stage6_cluster",
                "count",
                "row_percentage",
                "column_percentage",
                "normalized_mutual_info_stage5_vs_stage6",
                "collapsed_stage6_to_stage5_majority_ari",
            ],
        ),
        "该交叉表用于观察六类潜在表型是否可视为二类潜在语音表型的细分；不能解释为真实临床层级诊断。",
        "",
        "## 14. 性别分布的事后描述",
        markdown_table(sex_table.reset_index(), list(sex_table.reset_index().columns)),
        "`sex_male` 未进入 Stage 6 聚类或 six-cluster-label classifier，只用于事后描述，不能作为因果解释。",
        "",
        "## 15. six-cluster-label classifier 结果",
        markdown_table(
            classifier_summary,
            [
                "model",
                "fold",
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
                "precision_macro",
                "recall_macro",
                "multiclass_roc_auc_ovr",
            ],
        ),
        "该分类器学习的是无监督六类聚类标签，用于将潜在语音表型分型规则模型化；它不是训练自真实静止性震颤、运动迟缓、肌肉僵直、疼痛、痴呆或睡眠障碍临床标签，因此不能解释为真实六类症状诊断性能。",
        "",
        "## 16. 局限性",
        "- 六类标签来自语音特征内部结构，没有外部临床症状标签验证。",
        "- Kruskal-Wallis 和 FDR 是聚类后的描述性比较，不是外部临床验证假设检验。",
        "- cluster 编号和 tentative label 均为声学解释命名，不对应具体临床症状。",
        "- k=6 由题目设定驱动，k 扫描只提供参考。",
        "",
        "## 17. 论文建议表述",
        "建议写作：本文在 PD 受试者内部基于 Stage 4 筛选的声学特征进行 k=6 无监督聚类，得到六个基于语音特征的潜在表型，并训练 six-cluster-label classifier 复现该探索性分型规则。",
        "不建议写作：将本阶段结果表述为真实六类临床症状诊断性能。",
        "",
        "## 输出文件",
        "- `results/stage6_six_cluster_input_audit.csv`",
        "- `results/stage6_six_cluster_metrics.csv`",
        "- `results/stage6_k_scan_metrics.csv`",
        "- `results/stage6_six_cluster_assignments_dataset1_pd.csv`",
        "- `results/stage6_six_cluster_feature_profiles.csv`",
        "- `results/stage6_six_cluster_top_features_by_cluster.csv`",
        "- `results/stage6_six_cluster_profiles.csv`",
        "- `results/stage6_stage5_crosswalk.csv`",
        "- `results/stage6_six_cluster_label_classifier_metrics.csv`",
        "- `results/stage6_six_cluster_label_classifier_confusion_matrices.csv`",
    ]
    (RESULTS_DIR / "stage6_six_cluster_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    subject_table, top20_features, top50_features, _, stage5_assignments, _ = load_inputs()
    all_features = acoustic_columns(subject_table)
    pd_subjects = validate_pd_subjects(subject_table, stage5_assignments)
    write_input_audit(subject_table, pd_subjects, all_features, top20_features, top50_features, stage5_assignments)

    feature_sets = {
        "top20_biomarkers": top20_features,
        "top50_biomarkers": top50_features,
        "all_acoustic": all_features,
    }
    stage5_labels = aligned_stage5(stage5_assignments, pd_subjects)
    metrics, artifacts = run_clustering_candidates(pd_subjects, feature_sets, MAIN_K)
    metrics, recommended_key = choose_recommended(metrics, artifacts, stage5_labels.to_numpy())
    write_csv(metrics, RESULTS_DIR / "stage6_six_cluster_metrics.csv")

    recommended_artifact = artifacts[recommended_key]
    recommended_labels = np.asarray(recommended_artifact["labels"], dtype=int)
    assignments = write_assignments(pd_subjects, stage5_labels, recommended_key, recommended_artifact)
    k_scan = run_k_scan(
        pd_subjects,
        recommended_artifact["features"],
        recommended_key[0],
        recommended_key[1],
    )
    consensus = consensus_kmeans_summary(
        pd_subjects[recommended_artifact["features"]],
        recommended_key[1],
        recommended_labels,
    )

    feature_profiles, top_by_cluster = build_feature_profiles(
        pd_subjects,
        recommended_labels,
        recommended_artifact["features"],
    )
    cluster_profiles = build_cluster_profiles(assignments, top_by_cluster)
    crosswalk = stage5_crosswalk(assignments)
    classifier_metrics, _ = run_classifier(
        pd_subjects,
        recommended_artifact["features"],
        recommended_labels,
    )

    plot_pca_clusters(assignments)
    plot_cluster_size(assignments)
    plot_candidate_metrics(metrics)
    plot_feature_heatmap(feature_profiles)
    plot_crosswalk_heatmap(crosswalk)
    plot_top_features_by_cluster(top_by_cluster)
    write_report(
        subject_table,
        pd_subjects,
        metrics,
        k_scan,
        assignments,
        feature_profiles,
        top_by_cluster,
        cluster_profiles,
        crosswalk,
        classifier_metrics,
        consensus,
    )


if __name__ == "__main__":
    main()
