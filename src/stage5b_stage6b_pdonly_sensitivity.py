from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, SparsePCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUBJECT_TABLE_PATH = RESULTS_DIR / "stage4_subject_level_feature_table_dataset1.csv"
TOP20_PATH = RESULTS_DIR / "stage4_top20_biomarkers_dataset1.csv"
TOP50_PATH = RESULTS_DIR / "stage4_top50_biomarkers_dataset1.csv"
STAGE5_METRICS_PATH = RESULTS_DIR / "stage5_clustering_metrics.csv"
STAGE6_METRICS_PATH = RESULTS_DIR / "stage6_six_cluster_metrics.csv"
STAGE5_ASSIGNMENTS_PATH = RESULTS_DIR / "stage5_cluster_assignments_dataset1_pd.csv"
STAGE6_ASSIGNMENTS_PATH = RESULTS_DIR / "stage6_six_cluster_assignments_dataset1_pd.csv"

OUTPUT_CSV_ENCODING = "utf-8-sig"
RANDOM_STATE = 42
PD_LABEL = 1
EXPECTED_SUBJECTS = 252
EXPECTED_PD_SUBJECTS = 188
EXPECTED_ACOUSTIC_FEATURES = 752
STABILITY_RUNS = 50
STABILITY_SAMPLE_FRACTION = 0.8
WEIGHT_SENSITIVITY_RUNS = 10_000

METADATA_COLUMNS = {"id", "gender", "class", "sex_male", "record_count"}
LABEL_COLUMN_TOKENS = ("cluster", "label", "prediction", "phenotype")
METHODS = ["KMeans", "GaussianMixture", "AgglomerativeClustering"]
OBJECTIVE_METRICS = [
    "silhouette_norm",
    "calinski_harabasz_norm",
    "davies_bouldin_norm",
    "stability_ari_norm",
    "balance_norm",
]


@dataclass(frozen=True)
class FeatureScheme:
    name: str
    source_features: list[str]
    is_sparse_pca: bool = False
    sparse_components: int | None = None
    selection_rule: str = ""


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding=OUTPUT_CSV_ENCODING)


def require_inputs() -> None:
    for path in [
        SUBJECT_TABLE_PATH,
        TOP20_PATH,
        TOP50_PATH,
        STAGE5_METRICS_PATH,
        STAGE6_METRICS_PATH,
        STAGE5_ASSIGNMENTS_PATH,
        STAGE6_ASSIGNMENTS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")


def is_forbidden_feature_column(column: str) -> bool:
    name = column.lower()
    return column in METADATA_COLUMNS or any(token in name for token in LABEL_COLUMN_TOKENS)


def acoustic_columns(subject_table: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in subject_table.select_dtypes(include="number").columns
        if not is_forbidden_feature_column(column)
    ]
    if len(columns) != EXPECTED_ACOUSTIC_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_ACOUSTIC_FEATURES} acoustic features after metadata "
            f"exclusion, found {len(columns)}"
        )
    forbidden = sorted(column for column in columns if is_forbidden_feature_column(column))
    if forbidden:
        raise ValueError(f"Forbidden columns entered acoustic feature list: {forbidden}")
    return columns


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    require_inputs()
    subject_table = pd.read_csv(SUBJECT_TABLE_PATH)
    top20_table = pd.read_csv(TOP20_PATH)
    top50_table = pd.read_csv(TOP50_PATH)

    required_columns = {"id", "class", "sex_male", "record_count"}
    missing = sorted(required_columns - set(subject_table.columns))
    if missing:
        raise ValueError(f"Subject table missing required columns: {missing}")
    if subject_table.shape[0] != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} subjects, found {subject_table.shape[0]}")
    if int(subject_table["id"].duplicated().sum()):
        raise ValueError("Subject table must contain one row per subject")
    if int(subject_table.isna().sum().sum()):
        raise ValueError("Subject table contains missing values; Stage 5b/6b expects sealed Stage 4 table")

    for label, table, expected_rows in [("Top20", top20_table, 20), ("Top50", top50_table, 50)]:
        if "feature" not in table.columns:
            raise ValueError(f"{label} file missing required 'feature' column")
        if table.shape[0] != expected_rows:
            raise ValueError(f"{label} expected {expected_rows} rows, found {table.shape[0]}")

    pd_subjects = subject_table.loc[subject_table["class"] == PD_LABEL].copy()
    if pd_subjects.shape[0] != EXPECTED_PD_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_PD_SUBJECTS} PD subjects, found {pd_subjects.shape[0]}")
    if set(pd_subjects["class"].unique()) != {PD_LABEL}:
        raise ValueError("Healthy samples entered PD-only table")

    all_features = acoustic_columns(subject_table)
    top20_features = top20_table["feature"].astype(str).tolist()
    top50_features = top50_table["feature"].astype(str).tolist()
    for label, features in [("Top20", top20_features), ("Top50", top50_features)]:
        missing_features = sorted(set(features) - set(all_features))
        if missing_features:
            raise ValueError(f"{label} features not present in acoustic feature list: {missing_features}")

    return subject_table, pd_subjects, all_features, top20_features, top50_features


def pd_iqr_mad(pd_subjects: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    data = pd_subjects[features]
    iqr = (data.quantile(0.75) - data.quantile(0.25)).astype(float)
    mad = (data - data.median()).abs().median().astype(float)
    return iqr, mad


def connected_components_from_corr(corr: np.ndarray, threshold: float) -> list[list[int]]:
    seen = np.zeros(corr.shape[0], dtype=bool)
    components: list[list[int]] = []
    for index in range(corr.shape[0]):
        if seen[index]:
            continue
        stack = [index]
        seen[index] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = np.where((corr[current] > threshold) & (~seen))[0]
            for neighbor in neighbors:
                seen[int(neighbor)] = True
                stack.append(int(neighbor))
        components.append(component)
    return components


def corr_reduce_features(
    pd_subjects: pd.DataFrame,
    features: list[str],
    threshold: float,
    feature_set_name: str,
    iqr: pd.Series,
    mad: pd.Series,
) -> tuple[list[str], pd.DataFrame]:
    corr = pd_subjects[features].corr().abs().to_numpy()
    components = connected_components_from_corr(corr, threshold)
    retained_rows: list[dict[str, Any]] = []
    retained_features: list[str] = []

    for component_id, component in enumerate(components, start=1):
        component_features = [features[index] for index in component]
        retained = sorted(
            component_features,
            key=lambda feature: (float(iqr[feature]), float(mad[feature]), feature),
            reverse=True,
        )[0]
        retained_features.append(retained)
        retained_rows.append(
            {
                "feature_set": feature_set_name,
                "feature": retained,
                "source_feature_count": len(features),
                "threshold_abs_r": threshold,
                "component_id": component_id,
                "component_size": len(component_features),
                "component_features": ";".join(sorted(component_features)),
                "pd_iqr": float(iqr[retained]),
                "pd_mad": float(mad[retained]),
                "selection_rule": "PD-only correlation components; retain largest PD IQR, tie-break by MAD then feature name",
            }
        )

    return retained_features, pd.DataFrame(retained_rows)


def top_dispersion_features(
    features: list[str],
    scores: pd.Series,
    n_features: int,
    feature_set_name: str,
    iqr: pd.Series,
    mad: pd.Series,
) -> tuple[list[str], pd.DataFrame]:
    ordered = sorted(features, key=lambda feature: (float(scores[feature]), feature), reverse=True)
    retained = ordered[:n_features]
    rows = [
        {
            "feature_set": feature_set_name,
            "feature": feature,
            "rank": rank,
            "pd_iqr": float(iqr[feature]),
            "pd_mad": float(mad[feature]),
            "selection_rule": f"PD-only top {n_features} by {'MAD' if 'mad' in feature_set_name else 'IQR'}",
        }
        for rank, feature in enumerate(retained, start=1)
    ]
    return retained, pd.DataFrame(rows)


def build_feature_schemes(
    pd_subjects: pd.DataFrame,
    all_features: list[str],
    top20_features: list[str],
    top50_features: list[str],
) -> tuple[dict[str, FeatureScheme], pd.DataFrame, pd.DataFrame]:
    iqr, mad = pd_iqr_mad(pd_subjects, all_features)
    retained_tables: list[pd.DataFrame] = []

    corr_all_090, details = corr_reduce_features(
        pd_subjects, all_features, 0.90, "pd_corr_reduced_all_090", iqr, mad
    )
    retained_tables.append(details)
    corr_all_095, details = corr_reduce_features(
        pd_subjects, all_features, 0.95, "pd_corr_reduced_all_095", iqr, mad
    )
    retained_tables.append(details)
    corr_top50_090, details = corr_reduce_features(
        pd_subjects, top50_features, 0.90, "pd_corr_reduced_top50_090", iqr, mad
    )
    retained_tables.append(details)

    high_iqr_50, details = top_dispersion_features(
        all_features, iqr, 50, "pd_high_iqr_top50", iqr, mad
    )
    retained_tables.append(details)
    high_iqr_100, details = top_dispersion_features(
        all_features, iqr, 100, "pd_high_iqr_top100", iqr, mad
    )
    retained_tables.append(details)
    high_mad_50, details = top_dispersion_features(
        all_features, mad, 50, "pd_high_mad_top50", iqr, mad
    )
    retained_tables.append(details)
    high_mad_100, details = top_dispersion_features(
        all_features, mad, 100, "pd_high_mad_top100", iqr, mad
    )
    retained_tables.append(details)

    schemes = {
        "top20_biomarkers": FeatureScheme(
            "top20_biomarkers",
            top20_features,
            selection_rule="封版第 4 阶段 Top20；仅作对照；不加评分奖励",
        ),
        "top50_biomarkers": FeatureScheme(
            "top50_biomarkers",
            top50_features,
            selection_rule="封版第 4 阶段 Top50；仅作对照；不加评分奖励",
        ),
        "all_acoustic": FeatureScheme(
            "all_acoustic",
            all_features,
            selection_rule="全部 752 个声学特征；聚类前必须先做 PCA",
        ),
        "pd_corr_reduced_all_090": FeatureScheme(
            "pd_corr_reduced_all_090",
            corr_all_090,
            selection_rule="基于 PD-only 样本，从全部声学特征出发按 |r| > 0.90 做相关性去冗余",
        ),
        "pd_corr_reduced_all_095": FeatureScheme(
            "pd_corr_reduced_all_095",
            corr_all_095,
            selection_rule="基于 PD-only 样本，从全部声学特征出发按 |r| > 0.95 做相关性去冗余",
        ),
        "pd_corr_reduced_top50_090": FeatureScheme(
            "pd_corr_reduced_top50_090",
            corr_top50_090,
            selection_rule="基于 PD-only 样本，对第 4 阶段 Top50 按 |r| > 0.90 做相关性去冗余",
        ),
        "pd_high_iqr_top50": FeatureScheme(
            "pd_high_iqr_top50",
            high_iqr_50,
            selection_rule="基于 PD-only 样本，在 752 个声学特征中保留 IQR 最大的 50 个",
        ),
        "pd_high_iqr_top100": FeatureScheme(
            "pd_high_iqr_top100",
            high_iqr_100,
            selection_rule="基于 PD-only 样本，在 752 个声学特征中保留 IQR 最大的 100 个",
        ),
        "pd_high_mad_top50": FeatureScheme(
            "pd_high_mad_top50",
            high_mad_50,
            selection_rule="基于 PD-only 样本，在 752 个声学特征中保留 MAD 最大的 50 个",
        ),
        "pd_high_mad_top100": FeatureScheme(
            "pd_high_mad_top100",
            high_mad_100,
            selection_rule="基于 PD-only 样本，在 752 个声学特征中保留 MAD 最大的 100 个",
        ),
        "pd_sparse_pca_10": FeatureScheme(
            "pd_sparse_pca_10",
            all_features,
            is_sparse_pca=True,
            sparse_components=10,
            selection_rule="对 752 个 PD-only 声学特征先标准化，再提取 10 维 SparsePCA 表示",
        ),
        "pd_sparse_pca_20": FeatureScheme(
            "pd_sparse_pca_20",
            all_features,
            is_sparse_pca=True,
            sparse_components=20,
            selection_rule="对 752 个 PD-only 声学特征先标准化，再提取 20 维 SparsePCA 表示",
        ),
    }

    audit_rows = []
    for scheme in schemes.values():
        if scheme.is_sparse_pca:
            representations = [f"sparse_pca_{scheme.sparse_components}"]
            n_components = scheme.sparse_components
        elif len(scheme.source_features) <= 50:
            representations = ["scaled_only", "pca_5", "pca_10"]
            n_components = np.nan
        else:
            representations = ["pca_10", "pca_20", "pca_90"]
            n_components = np.nan
        audit_rows.append(
            {
                "feature_set": scheme.name,
                "n_subjects_used_for_selection": EXPECTED_PD_SUBJECTS,
                "selection_population": "Dataset 1 PD only (class == 1)",
                "healthy_subjects_used": 0,
                "record_level_samples_used": 0,
                "n_features_original": len(scheme.source_features),
                "is_sparse_pca_representation": bool(scheme.is_sparse_pca),
                "sparse_pca_components": n_components,
                "representations": ";".join(representations),
                "selection_rule": scheme.selection_rule,
                "forbidden_columns_excluded": ";".join(
                    ["id", "gender", "class", "sex_male", "record_count", "Stage 5 cluster", "Stage 6 cluster"]
                ),
            }
        )

    retained_detail = pd.concat(retained_tables, ignore_index=True)
    return schemes, pd.DataFrame(audit_rows), retained_detail


def representations_for_scheme(scheme: FeatureScheme) -> list[str]:
    if scheme.is_sparse_pca:
        return [f"sparse_pca_{scheme.sparse_components}"]
    if len(scheme.source_features) <= 50:
        return ["scaled_only", "pca_5", "pca_10"]
    return ["pca_10", "pca_20", "pca_90"]


def build_representation(
    data: pd.DataFrame,
    scheme: FeatureScheme,
    representation: str,
) -> tuple[np.ndarray, int, float]:
    scaled = StandardScaler().fit_transform(data[scheme.source_features].to_numpy(dtype=float))
    if scheme.is_sparse_pca:
        n_components = int(scheme.sparse_components or 0)
        model = SparsePCA(
            n_components=n_components,
            random_state=RANDOM_STATE,
            ridge_alpha=0.01,
            max_iter=200,
            tol=1e-4,
            method="cd",
            n_jobs=-1,
        )
        projected = model.fit_transform(scaled)
        return projected, projected.shape[1], np.nan
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
        return KMeans(n_clusters=k, n_init=50, random_state=random_state).fit_predict(matrix)
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


def cluster_size_stats(labels: np.ndarray, k: int) -> dict[str, Any]:
    counts = pd.Series(labels).value_counts().reindex(range(k), fill_value=0).astype(int)
    min_size = int(counts.min())
    max_size = int(counts.max())
    min_ratio = float(min_size / len(labels))
    max_ratio = float(max_size / len(labels))
    smaller_ratio = float(min_size / len(labels))
    return {
        "cluster_sizes": ";".join(f"{cluster}:{int(size)}" for cluster, size in counts.items()),
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "min_cluster_ratio": min_ratio,
        "max_cluster_ratio": max_ratio,
        "size_imbalance_ratio": float(max_size / min_size) if min_size > 0 else np.inf,
        "smaller_cluster_ratio": smaller_ratio,
    }


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
    pd_subjects: pd.DataFrame,
    scheme: FeatureScheme,
    representation: str,
    method: str,
    k: int,
    full_labels: np.ndarray,
    full_matrix: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE + k)
    n_subjects = pd_subjects.shape[0]
    sample_size = int(round(n_subjects * STABILITY_SAMPLE_FRACTION))
    scores: list[float] = []
    for run_index in range(STABILITY_RUNS):
        sample_index = np.sort(rng.choice(n_subjects, size=sample_size, replace=False))
        if scheme.is_sparse_pca:
            sample_matrix = full_matrix[sample_index]
        else:
            sample_data = pd_subjects.iloc[sample_index]
            sample_matrix, _, _ = build_representation(sample_data, scheme, representation)
        sample_labels = cluster_labels(sample_matrix, method, k, RANDOM_STATE + run_index + 1)
        scores.append(float(adjusted_rand_score(full_labels[sample_index], sample_labels)))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def run_clustering_stage(
    pd_subjects: pd.DataFrame,
    schemes: dict[str, FeatureScheme],
    k: int,
    stage: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme in schemes.values():
        for representation in representations_for_scheme(scheme):
            matrix, n_components, explained_variance = build_representation(pd_subjects, scheme, representation)
            for method in METHODS:
                labels = cluster_labels(matrix, method, k)
                size_stats = cluster_size_stats(labels, k)
                silhouette, calinski, davies, metric_warning = clustering_quality(matrix, labels)
                stability_mean, stability_std = stability_analysis(
                    pd_subjects, scheme, representation, method, k, labels, matrix
                )
                warnings = []
                if metric_warning:
                    warnings.append(metric_warning)
                if stage == "stage5b" and size_stats["smaller_cluster_ratio"] < 0.10:
                    warnings.append("smaller_cluster_ratio < 0.10; not eligible for recommendation")
                if stage == "stage6b":
                    if size_stats["min_cluster_size"] < 8:
                        warnings.append("min_cluster_size < 8; not eligible for recommendation")
                    if size_stats["min_cluster_ratio"] < 0.05:
                        warnings.append("min_cluster_ratio < 0.05; not eligible for recommendation")

                rows.append(
                    {
                        "feature_set": scheme.name,
                        "representation": representation,
                        "method": method,
                        "k": int(k),
                        "candidate_id": f"{scheme.name}|{representation}|{method}|k{k}",
                        "n_subjects": int(len(labels)),
                        "n_features_original": int(len(scheme.source_features)),
                        "n_components_after_transform": int(n_components),
                        "pca_explained_variance_sum": explained_variance,
                        **size_stats,
                        "silhouette": silhouette,
                        "calinski_harabasz": calinski,
                        "davies_bouldin": davies,
                        "stability_ari_mean": stability_mean,
                        "stability_ari_std": stability_std,
                        "stability_n_runs": STABILITY_RUNS,
                        "eligible_for_recommendation": len(warnings) == 0,
                        "exclusion_or_warning": "; ".join(warnings),
                    }
                )
    return pd.DataFrame(rows)


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


def add_normalized_metrics(metrics: pd.DataFrame, k: int, stage: str) -> pd.DataFrame:
    out = metrics.copy()
    if stage == "stage5b":
        eligible = out["smaller_cluster_ratio"] >= 0.10
    else:
        eligible = (out["min_cluster_size"] >= 8) & (out["min_cluster_ratio"] >= 0.05)
    out["eligible_for_recommendation"] = eligible
    out["silhouette_norm"] = np.nan
    out["calinski_harabasz_norm"] = np.nan
    out["davies_bouldin_norm"] = np.nan
    out["stability_ari_norm"] = np.nan
    out["balance_norm"] = np.nan

    eligible_metrics = out.loc[eligible]
    if eligible_metrics.empty:
        raise ValueError(f"No eligible candidates for {stage}")

    out.loc[eligible, "silhouette_norm"] = minmax(eligible_metrics["silhouette"], True)
    out.loc[eligible, "calinski_harabasz_norm"] = minmax(
        eligible_metrics["calinski_harabasz"], True
    )
    out.loc[eligible, "davies_bouldin_norm"] = minmax(eligible_metrics["davies_bouldin"], False)
    out.loc[eligible, "stability_ari_norm"] = minmax(eligible_metrics["stability_ari_mean"], True)
    ideal_min_ratio = 1 / k
    out.loc[eligible, "balance_norm"] = (
        eligible_metrics["min_cluster_ratio"].astype(float) / ideal_min_ratio
    ).clip(upper=1.0)
    return out


def weight_sensitivity(
    metrics: pd.DataFrame,
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = metrics.loc[metrics["eligible_for_recommendation"]].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError(f"No eligible candidates available for {stage} sensitivity analysis")

    values = eligible[OBJECTIVE_METRICS].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_STATE + (5 if stage == "stage5b" else 6))
    weights = rng.dirichlet(np.ones(len(OBJECTIVE_METRICS)), size=WEIGHT_SENSITIVITY_RUNS)
    scores = values @ weights.T
    ranks = np.argsort(-scores, axis=0)
    top1_indices = ranks[0, :]
    top3_indices = ranks[:3, :]

    sensitivity_rows = []
    for run_index in range(WEIGHT_SENSITIVITY_RUNS):
        winner = eligible.iloc[int(top1_indices[run_index])]
        sensitivity_rows.append(
            {
                "run": run_index + 1,
                "weight_silhouette": float(weights[run_index, 0]),
                "weight_calinski_harabasz": float(weights[run_index, 1]),
                "weight_davies_bouldin": float(weights[run_index, 2]),
                "weight_stability_ari": float(weights[run_index, 3]),
                "weight_balance": float(weights[run_index, 4]),
                "top1_candidate_id": winner["candidate_id"],
                "top1_feature_set": winner["feature_set"],
                "top1_representation": winner["representation"],
                "top1_method": winner["method"],
                "top1_score": float(scores[int(top1_indices[run_index]), run_index]),
            }
        )

    top1_counts = np.bincount(top1_indices, minlength=eligible.shape[0])
    top3_counts = np.bincount(top3_indices.reshape(-1), minlength=eligible.shape[0])
    rank_frequency = eligible[
        [
            "candidate_id",
            "feature_set",
            "representation",
            "method",
            "k",
            "n_features_original",
            "n_components_after_transform",
            "cluster_sizes",
            "min_cluster_size",
            "min_cluster_ratio",
            "smaller_cluster_ratio",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "stability_ari_mean",
            *OBJECTIVE_METRICS,
        ]
    ].copy()
    rank_frequency["top1_count"] = top1_counts.astype(int)
    rank_frequency["top1_frequency"] = top1_counts / WEIGHT_SENSITIVITY_RUNS
    rank_frequency["top3_count"] = top3_counts.astype(int)
    rank_frequency["top3_frequency"] = top3_counts / WEIGHT_SENSITIVITY_RUNS
    rank_frequency = rank_frequency.sort_values(
        ["top1_frequency", "top3_frequency", "stability_ari_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return pd.DataFrame(sensitivity_rows), rank_frequency


def recommendation_label(rank_frequency: pd.DataFrame) -> str:
    top_frequency = float(rank_frequency.iloc[0]["top1_frequency"])
    if top_frequency >= 0.60:
        return "稳健主推荐方案"
    if top_frequency >= 0.30:
        return "相对推荐方案"
    return "多个可接受方案"


def find_original_candidate(stage: str, rank_frequency: pd.DataFrame) -> pd.DataFrame:
    metrics_path = STAGE5_METRICS_PATH if stage == "stage5b" else STAGE6_METRICS_PATH
    old = pd.read_csv(metrics_path)
    old_method_col = "clustering_method" if "clustering_method" in old.columns else "method"
    rec = old.loc[old["recommended_flag"].astype(str).str.lower().eq("true")]
    if rec.empty:
        return pd.DataFrame()
    row = rec.iloc[0]
    k = 2 if stage == "stage5b" else 6
    representation = str(row["representation"])
    method = str(row[old_method_col])
    feature_set = str(row["feature_set"])
    candidate_id = f"{feature_set}|{representation}|{method}|k{k}"
    return rank_frequency.loc[rank_frequency["candidate_id"] == candidate_id].copy()


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df.loc[:, columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, separator, *rows])


def markdown_table_cn(
    df: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
    max_rows: int | None = None,
) -> str:
    table = df.loc[:, columns].copy()
    table = table.rename(columns=labels)
    if max_rows is not None:
        table = table.head(max_rows)
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, separator, *rows])


def plot_weight_sensitivity(rank_frequency: pd.DataFrame, stage: str, output_path: Path) -> None:
    plot_data = rank_frequency.head(12).copy()
    plot_data["candidate"] = (
        plot_data["feature_set"] + "\n" + plot_data["representation"] + "\n" + plot_data["method"]
    )
    x = np.arange(plot_data.shape[0])
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.38
    ax.bar(x - width / 2, plot_data["top1_frequency"], width, label="Top-1 frequency", color="#4c78a8")
    ax.bar(x + width / 2, plot_data["top3_frequency"], width, label="Top-3 frequency", color="#f58518")
    ax.axhline(0.60, color="#b22222", linestyle="--", linewidth=1, label="Robust threshold")
    ax.axhline(0.30, color="#777777", linestyle=":", linewidth=1, label="Relative threshold")
    ax.set_title(f"{stage.upper()} Weight Sensitivity Top Candidates")
    ax.set_ylabel("Frequency across 10,000 Dirichlet weights")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["candidate"], rotation=70, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, stage: str, output_path: Path) -> None:
    plot_data = metrics.loc[metrics["eligible_for_recommendation"]].copy()
    summary = (
        plot_data.groupby("feature_set", as_index=False)
        .agg(
            silhouette=("silhouette", "mean"),
            stability_ari_mean=("stability_ari_mean", "mean"),
            balance_norm=("balance_norm", "mean"),
        )
        .sort_values("stability_ari_mean", ascending=False)
    )
    x = np.arange(summary.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, metric, title, color in [
        (axes[0], "silhouette", "Mean silhouette", "#4c78a8"),
        (axes[1], "stability_ari_mean", "Mean stability ARI", "#54a24b"),
        (axes[2], "balance_norm", "Mean balance norm", "#f58518"),
    ]:
        ax.bar(x, summary[metric], color=color)
        ax.set_ylabel(title)
        ax.set_title(f"{stage.upper()} Eligible Feature Set {title}")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(summary["feature_set"], rotation=65, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_original_vs_pdonly(
    stage5_rank: pd.DataFrame,
    stage6_rank: pd.DataFrame,
    output_path: Path,
) -> None:
    rows = []
    for stage, stage_key, rank in [
        ("Stage 5b", "stage5b", stage5_rank),
        ("Stage 6b", "stage6b", stage6_rank),
    ]:
        original = find_original_candidate(stage_key, rank)
        if not original.empty:
            rows.append(
                {
                    "stage": stage,
                    "candidate_type": "Original Stage main scheme",
                    "frequency": float(original.iloc[0]["top1_frequency"]),
                    "label": original.iloc[0]["feature_set"],
                }
            )
        rows.append(
            {
                "stage": stage,
                "candidate_type": "Best PD-only no-bonus scheme",
                "frequency": float(rank.iloc[0]["top1_frequency"]),
                "label": rank.iloc[0]["feature_set"],
            }
        )
    plot_data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = plot_data["stage"] + "\n" + plot_data["candidate_type"]
    colors = ["#4c78a8" if "Original" in value else "#f58518" for value in plot_data["candidate_type"]]
    ax.bar(np.arange(plot_data.shape[0]), plot_data["frequency"], color=colors)
    ax.axhline(0.60, color="#b22222", linestyle="--", linewidth=1)
    ax.axhline(0.30, color="#777777", linestyle=":", linewidth=1)
    ax.set_xticks(np.arange(plot_data.shape[0]))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Top-1 frequency")
    ax.set_title("Original Stage 5/6 Main Schemes vs PD-only No-bonus Sensitivity")
    for index, row in plot_data.iterrows():
        ax.text(index, row["frequency"], f"{row['frequency']:.1%}\n{row['label']}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def robustness_sentence(stage: str, rank_frequency: pd.DataFrame) -> str:
    original = find_original_candidate(stage, rank_frequency)
    stage_label = "第 5 阶段" if stage == "stage5b" else "第 6 阶段"
    if original.empty:
        return f"{stage_label} 原主方案未能在 5b/6b 候选集中匹配，不能判定其在无奖励敏感性分析下仍稳健。"
    row = original.iloc[0]
    frequency = float(row["top1_frequency"])
    if frequency >= 0.60:
        result = "仍可视为稳健"
    elif frequency >= 0.30:
        result = "只能视为相对稳健，应同时报告其他可接受方案"
    else:
        result = "不再稳健，不应继续单独称为最佳方案"
    return (
        f"{stage_label} 原主方案 `{row['feature_set']} / {row['representation']} / {row['method']}` "
        f"在无奖励权重敏感性分析中的第一名频率为 {frequency:.1%}，{result}。"
    )


def write_report(
    stage5_metrics: pd.DataFrame,
    stage6_metrics: pd.DataFrame,
    stage5_rank: pd.DataFrame,
    stage6_rank: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    stage5_label = recommendation_label(stage5_rank)
    stage6_label = recommendation_label(stage6_rank)
    report = [
        "# 第 5b/6b 阶段 PD-only 特征扩展与权重敏感性报告",
        "",
        "## 1. 方法学动机",
        "Healthy/PD 判别特征是为区分病例与健康对照而排序的，可能主要反映疾病有无、性别或全局声学差异；PD 内部分型需要捕捉 188 名 PD 受试者内部的异质性。因此第 5b/6b 阶段将特征筛选限定在 PD-only 样本内，并保留第 4 阶段 Top20/Top50 仅作为对照。",
        "",
        "## 2. 新增 PD-only 特征方案",
        markdown_table_cn(
            audit,
            [
                "feature_set",
                "n_features_original",
                "is_sparse_pca_representation",
                "sparse_pca_components",
                "representations",
                "selection_rule",
            ],
            {
                "feature_set": "特征方案",
                "n_features_original": "原始特征数",
                "is_sparse_pca_representation": "是否为 SparsePCA 表示",
                "sparse_pca_components": "SparsePCA 维数",
                "representations": "聚类表示",
                "selection_rule": "筛选规则",
            },
        ),
        "",
        "## 3. 取消人为奖励",
        "第 5b/6b 阶段不对 Top20、Top50 或第 4 阶段 biomarker 特征集加解释性奖励；第 6b 阶段不使用第 5 阶段 NMI 参与主推荐评分；图像好看程度不进入任何评分。解释性只在论文讨论中单独说明。",
        "",
        "## 4. 第 5b 阶段全部候选方案客观指标",
        markdown_table_cn(
            stage5_metrics,
            [
                "feature_set",
                "representation",
                "method",
                "cluster_sizes",
                "smaller_cluster_ratio",
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
                "stability_ari_mean",
                "eligible_for_recommendation",
            ],
            {
                "feature_set": "特征方案",
                "representation": "表示方式",
                "method": "聚类方法",
                "cluster_sizes": "簇人数",
                "smaller_cluster_ratio": "较小簇比例",
                "silhouette": "轮廓系数",
                "calinski_harabasz": "Calinski-Harabasz",
                "davies_bouldin": "Davies-Bouldin",
                "stability_ari_mean": "稳定性 ARI 均值",
                "eligible_for_recommendation": "是否可推荐",
            },
        ),
        "",
        "## 5. 第 6b 阶段全部候选方案客观指标",
        markdown_table_cn(
            stage6_metrics,
            [
                "feature_set",
                "representation",
                "method",
                "cluster_sizes",
                "min_cluster_size",
                "min_cluster_ratio",
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
                "stability_ari_mean",
                "eligible_for_recommendation",
            ],
            {
                "feature_set": "特征方案",
                "representation": "表示方式",
                "method": "聚类方法",
                "cluster_sizes": "簇人数",
                "min_cluster_size": "最小簇人数",
                "min_cluster_ratio": "最小簇比例",
                "silhouette": "轮廓系数",
                "calinski_harabasz": "Calinski-Harabasz",
                "davies_bouldin": "Davies-Bouldin",
                "stability_ari_mean": "稳定性 ARI 均值",
                "eligible_for_recommendation": "是否可推荐",
            },
        ),
        "",
        "## 6. 权重敏感性分析方法",
        f"对通过硬约束过滤的候选方案，先对 silhouette、Calinski-Harabasz、反向 Davies-Bouldin、stability ARI 和 balance 五个指标做 min-max 归一化。随后从 Dirichlet(1,1,1,1,1) 随机生成 {WEIGHT_SENSITIVITY_RUNS} 组非负且和为 1 的权重，逐次计算综合分并记录第一名和前三名频率。稳定性分析使用 50 次 80% PD 受试者重采样；SparsePCA 方案在已拟合的 SparsePCA 表示空间内重聚类以避免重复拟合高耗时表示。",
        "",
        "## 7. 第 5b 阶段第一名与前三名频率",
        f"判定：{stage5_label}。",
        markdown_table_cn(
            stage5_rank,
            [
                "feature_set",
                "representation",
                "method",
                "top1_frequency",
                "top3_frequency",
                "silhouette",
                "davies_bouldin",
                "stability_ari_mean",
                "smaller_cluster_ratio",
            ],
            {
                "feature_set": "特征方案",
                "representation": "表示方式",
                "method": "聚类方法",
                "top1_frequency": "第一名频率",
                "top3_frequency": "前三名频率",
                "silhouette": "轮廓系数",
                "davies_bouldin": "Davies-Bouldin",
                "stability_ari_mean": "稳定性 ARI 均值",
                "smaller_cluster_ratio": "较小簇比例",
            },
            max_rows=20,
        ),
        "",
        "## 8. 第 6b 阶段第一名与前三名频率",
        f"判定：{stage6_label}。",
        markdown_table_cn(
            stage6_rank,
            [
                "feature_set",
                "representation",
                "method",
                "top1_frequency",
                "top3_frequency",
                "min_cluster_size",
                "min_cluster_ratio",
                "silhouette",
                "davies_bouldin",
                "stability_ari_mean",
            ],
            {
                "feature_set": "特征方案",
                "representation": "表示方式",
                "method": "聚类方法",
                "top1_frequency": "第一名频率",
                "top3_frequency": "前三名频率",
                "min_cluster_size": "最小簇人数",
                "min_cluster_ratio": "最小簇比例",
                "silhouette": "轮廓系数",
                "davies_bouldin": "Davies-Bouldin",
                "stability_ari_mean": "稳定性 ARI 均值",
            },
            max_rows=20,
        ),
        "",
        "## 9. 原第 5/6 阶段主方案稳健性",
        robustness_sentence("stage5b", stage5_rank),
        robustness_sentence("stage6b", stage6_rank),
        "",
        "## 10. 论文表述建议",
        "若原方案第一名频率低于 60%，论文中不应继续写作唯一“最佳方案”，应改为“在若干客观指标和权重设定下表现较好的候选方案”或“多个可接受候选方案”。权重敏感性不是外部临床标签验证，只能说明聚类推荐对主观权重的稳健性。",
        "",
        "## 11. 最终建议",
        f"第 5b 阶段：{stage5_label}，首位候选为 `{stage5_rank.iloc[0]['feature_set']} / {stage5_rank.iloc[0]['representation']} / {stage5_rank.iloc[0]['method']}`，第一名频率 {float(stage5_rank.iloc[0]['top1_frequency']):.1%}。",
        f"第 6b 阶段：{stage6_label}，首位候选为 `{stage6_rank.iloc[0]['feature_set']} / {stage6_rank.iloc[0]['representation']} / {stage6_rank.iloc[0]['method']}`，第一名频率 {float(stage6_rank.iloc[0]['top1_frequency']):.1%}。",
        "所有结论仅为 PD 受试者内部语音特征聚类的稳健性补充，不写作真实临床亚型或六症状诊断。",
    ]
    (RESULTS_DIR / "stage5b_stage6b_feature_and_weight_sensitivity_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def write_revision_suggestions(stage5_rank: pd.DataFrame, stage6_rank: pd.DataFrame) -> None:
    stage5_original = find_original_candidate("stage5b", stage5_rank)
    stage6_original = find_original_candidate("stage6b", stage6_rank)

    def recommendation(stage_name: str, original: pd.DataFrame, rank: pd.DataFrame) -> list[str]:
        if original.empty:
            return [
                f"- {stage_name}：原主方案未匹配到 5b/6b 候选，建议改为报告多个候选方案。",
            ]
        freq = float(original.iloc[0]["top1_frequency"])
        top_freq = float(rank.iloc[0]["top1_frequency"])
        if freq >= 0.60:
            decision = "可保留原主方案，但需说明其经过无奖励权重敏感性分析。"
        elif freq >= 0.30:
            decision = "建议保留为相对推荐方案，并同步报告前 2-3 个可接受方案。"
        else:
            decision = "不建议继续称原方案为最佳，应改为候选方案或多方案报告。"
        return [
            f"- {stage_name}：原主方案第一名频率 {freq:.1%}；当前最高候选第一名频率 {top_freq:.1%}。{decision}",
        ]

    lines = [
        "# 第 5b/6b 阶段论文修改建议",
        "",
        "## 1. 是否仍推荐原主方案",
        *recommendation("第（4）问 / 第 5 阶段", stage5_original, stage5_rank),
        *recommendation("第（5）问 / 第 6 阶段", stage6_original, stage6_rank),
        "",
        "## 2. 是否应弱化“最佳方案”",
        "- 若对应阶段最高第一名频率低于 60%，将“最佳方案”改为“相对推荐候选方案”或“多个可接受候选方案”。",
        "- 不把 Top20/Top50 的可解释性写成评分优势，只能在讨论部分解释其论文可读性。",
        "",
        "## 3. 需要改弱的句子类型",
        "- 将“该方案为最优/最佳聚类方案”改为“该方案在本研究设定的客观指标下表现较好”。",
        "- 将“证明存在真实亚型”改为“提示 PD 受试者语音特征中存在可探索的内部异质性”。",
        "- 将“六类症状诊断模型”改为“k=6 潜在语音表型聚类模型”。",
        "- 避免把语音聚类簇写成已由外部临床标签确认的运动相关亚型或六症状诊断类别。",
        "",
        "## 4. 新增表格和图",
        "- 表：第 5b/6b 阶段候选方案客观指标表。",
        "- 表：第 5b/6b 阶段权重敏感性第一名和前三名频率表。",
        "- 图：`stage5b_weight_sensitivity_top_candidates.png` 和 `stage6b_weight_sensitivity_top_candidates.png`。",
        "- 图：`stage5b_feature_set_metric_comparison.png` 和 `stage6b_feature_set_metric_comparison.png`。",
        "- 图：`stage5b_stage6b_original_vs_pdonly_comparison.png`。",
        "",
        "## 5. 放置位置",
        "- 正文保留简短方法说明、首位候选频率和最终推荐判断。",
        "- 完整 96 个候选方案指标、相关性去冗余保留清单和 10000 组权重敏感性明细建议放附录。",
    ]
    (RESULTS_DIR / "stage5b_stage6b_paper_revision_suggestions.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def verify_outputs(stage5_metrics: pd.DataFrame, stage6_metrics: pd.DataFrame) -> None:
    for stage, metrics, k in [("Stage 5b", stage5_metrics, 2), ("Stage 6b", stage6_metrics, 6)]:
        if metrics.shape[0] != 96:
            raise ValueError(f"{stage} expected 96 candidates, found {metrics.shape[0]}")
        if set(metrics["n_subjects"].unique()) != {EXPECTED_PD_SUBJECTS}:
            raise ValueError(f"{stage} contains a non-PD subject count")
        if set(metrics["k"].unique()) != {k}:
            raise ValueError(f"{stage} contains wrong k values")
    forbidden_text = " ".join(stage5_metrics["candidate_id"].tolist() + stage6_metrics["candidate_id"].tolist()).lower()
    for token in ["sex_male", "record_count", "stage5_cluster", "stage6_cluster"]:
        if token in forbidden_text:
            raise ValueError(f"Forbidden token appeared in candidate ids: {token}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    _, pd_subjects, all_features, top20_features, top50_features = load_inputs()
    schemes, audit, retained_detail = build_feature_schemes(
        pd_subjects, all_features, top20_features, top50_features
    )

    write_csv(audit, RESULTS_DIR / "stage5b_pd_only_feature_sets_audit.csv")
    write_csv(audit, RESULTS_DIR / "stage6b_pd_only_feature_sets_audit.csv")
    write_csv(retained_detail, RESULTS_DIR / "stage5b_pd_only_retained_features.csv")
    write_csv(retained_detail, RESULTS_DIR / "stage6b_pd_only_retained_features.csv")

    stage5_metrics = add_normalized_metrics(
        run_clustering_stage(pd_subjects, schemes, 2, "stage5b"), 2, "stage5b"
    )
    stage6_metrics = add_normalized_metrics(
        run_clustering_stage(pd_subjects, schemes, 6, "stage6b"), 6, "stage6b"
    )
    verify_outputs(stage5_metrics, stage6_metrics)

    stage5_sensitivity, stage5_rank = weight_sensitivity(stage5_metrics, "stage5b")
    stage6_sensitivity, stage6_rank = weight_sensitivity(stage6_metrics, "stage6b")

    write_csv(stage5_metrics, RESULTS_DIR / "stage5b_clustering_metrics_no_bonus.csv")
    write_csv(stage6_metrics, RESULTS_DIR / "stage6b_clustering_metrics_no_bonus.csv")
    write_csv(stage5_sensitivity, RESULTS_DIR / "stage5b_weight_sensitivity.csv")
    write_csv(stage6_sensitivity, RESULTS_DIR / "stage6b_weight_sensitivity.csv")
    write_csv(stage5_rank, RESULTS_DIR / "stage5b_candidate_rank_frequency.csv")
    write_csv(stage6_rank, RESULTS_DIR / "stage6b_candidate_rank_frequency.csv")

    plot_weight_sensitivity(
        stage5_rank,
        "stage5b",
        FIGURES_DIR / "stage5b_weight_sensitivity_top_candidates.png",
    )
    plot_weight_sensitivity(
        stage6_rank,
        "stage6b",
        FIGURES_DIR / "stage6b_weight_sensitivity_top_candidates.png",
    )
    plot_metric_comparison(
        stage5_metrics,
        "stage5b",
        FIGURES_DIR / "stage5b_feature_set_metric_comparison.png",
    )
    plot_metric_comparison(
        stage6_metrics,
        "stage6b",
        FIGURES_DIR / "stage6b_feature_set_metric_comparison.png",
    )
    plot_original_vs_pdonly(
        stage5_rank,
        stage6_rank,
        FIGURES_DIR / "stage5b_stage6b_original_vs_pdonly_comparison.png",
    )

    write_report(stage5_metrics, stage6_metrics, stage5_rank, stage6_rank, audit)
    write_revision_suggestions(stage5_rank, stage6_rank)

    print("Stage 5b/6b PD-only sensitivity analysis complete.")
    print(f"Stage 5b top candidate: {stage5_rank.iloc[0]['candidate_id']} ({stage5_rank.iloc[0]['top1_frequency']:.1%})")
    print(f"Stage 6b top candidate: {stage6_rank.iloc[0]['candidate_id']} ({stage6_rank.iloc[0]['top1_frequency']:.1%})")


if __name__ == "__main__":
    main()
