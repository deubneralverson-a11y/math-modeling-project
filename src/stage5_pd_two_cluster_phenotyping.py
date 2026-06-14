from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUBJECT_TABLE_PATH = RESULTS_DIR / "stage4_subject_level_feature_table_dataset1.csv"
TOP20_PATH = RESULTS_DIR / "stage4_top20_biomarkers_dataset1.csv"
TOP50_PATH = RESULTS_DIR / "stage4_top50_biomarkers_dataset1.csv"

OUTPUT_CSV_ENCODING = "utf-8-sig"
RANDOM_STATE = 42
PD_LABEL = 1
EXPECTED_SUBJECTS = 252
EXPECTED_PD_SUBJECTS = 188
EXPECTED_ACOUSTIC_FEATURES = 752
STABILITY_RUNS = 50
STABILITY_SAMPLE_FRACTION = 0.8

METADATA_COLUMNS = {
    "id",
    "gender",
    "class",
    "sex_male",
    "record_count",
}
LABEL_COLUMN_TOKENS = ("cluster", "label", "prediction", "phenotype")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding=OUTPUT_CSV_ENCODING)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Stage 4 input file: {path}")


def load_inputs() -> tuple[pd.DataFrame, list[str], list[str]]:
    for path in [SUBJECT_TABLE_PATH, TOP20_PATH, TOP50_PATH]:
        require_file(path)

    subject_table = pd.read_csv(SUBJECT_TABLE_PATH)
    top20 = pd.read_csv(TOP20_PATH)
    top50 = pd.read_csv(TOP50_PATH)

    required_subject_columns = {"id", "class", "sex_male"}
    missing_subject_columns = sorted(required_subject_columns - set(subject_table.columns))
    if missing_subject_columns:
        raise ValueError(f"Subject table missing required columns: {missing_subject_columns}")

    if subject_table.shape[0] != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} subject rows, found {subject_table.shape[0]}"
        )
    duplicated_id_count = int(subject_table["id"].duplicated().sum())
    if duplicated_id_count:
        raise ValueError(
            f"Subject table must contain one row per subject; duplicated ids: {duplicated_id_count}"
        )

    for label, table, expected_rows in [
        ("Top20", top20, 20),
        ("Top50", top50, 50),
    ]:
        if "feature" not in table.columns:
            raise ValueError(f"{label} input is missing the required 'feature' column")
        if table.shape[0] != expected_rows:
            raise ValueError(f"{label} input expected {expected_rows} rows, found {table.shape[0]}")

    top20_features = top20["feature"].astype(str).tolist()
    top50_features = top50["feature"].astype(str).tolist()
    missing_top20 = sorted(set(top20_features) - set(subject_table.columns))
    missing_top50 = sorted(set(top50_features) - set(subject_table.columns))
    if missing_top20 or missing_top50:
        raise ValueError(
            "Stage 4 biomarker features missing from subject-level table: "
            f"top20={missing_top20}, top50={missing_top50}"
        )

    return subject_table, top20_features, top50_features


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
            f"Expected {EXPECTED_ACOUSTIC_FEATURES} all-acoustic features after metadata "
            f"exclusion, found {len(columns)}"
        )
    forbidden = sorted(set(columns) & METADATA_COLUMNS)
    if forbidden:
        raise ValueError(f"Forbidden metadata columns entered acoustic features: {forbidden}")
    return columns


def validate_pd_subjects(subject_table: pd.DataFrame) -> pd.DataFrame:
    before_counts = subject_table["class"].value_counts().sort_index().to_dict()
    if int(sum(before_counts.values())) != EXPECTED_SUBJECTS:
        raise ValueError("Class distribution does not sum to the expected subject count")

    pd_subjects = subject_table.loc[subject_table["class"] == PD_LABEL].copy()
    if pd_subjects.shape[0] != EXPECTED_PD_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_PD_SUBJECTS} PD subjects, found {pd_subjects.shape[0]}"
        )
    return pd_subjects


def write_input_audit(
    subject_table: pd.DataFrame,
    pd_subjects: pd.DataFrame,
    all_features: list[str],
    top20_features: list[str],
    top50_features: list[str],
) -> None:
    excluded_columns = [
        column if column in subject_table.columns else f"{column} (not present)"
        for column in ["id", "gender", "class", "sex_male", "record_count"]
    ]
    audit = pd.DataFrame(
        [
            {
                "input_file": rel(SUBJECT_TABLE_PATH),
                "total_subjects_before_filter": int(subject_table.shape[0]),
                "pd_subjects_after_filter": int(pd_subjects.shape[0]),
                "class_distribution_before_filter": str(
                    {
                        int(k): int(v)
                        for k, v in subject_table["class"]
                        .value_counts()
                        .sort_index()
                        .items()
                    }
                ),
                "class_distribution_after_filter": str(
                    {
                        int(k): int(v)
                        for k, v in pd_subjects["class"].value_counts().sort_index().items()
                    }
                ),
                "missing_value_count": int(subject_table.isna().sum().sum()),
                "duplicated_id_count": int(subject_table["id"].duplicated().sum()),
                "acoustic_feature_count": int(len(all_features)),
                "excluded_columns": "; ".join(excluded_columns),
                "feature_sets_used": "top20_biomarkers; top50_biomarkers; all_acoustic",
                "top20_feature_count": int(len(top20_features)),
                "top50_feature_count": int(len(top50_features)),
                "all_acoustic_feature_count": int(len(all_features)),
            }
        ]
    )
    write_csv(audit, RESULTS_DIR / "stage5_pd_subject_clustering_input_audit.csv")


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


def build_representation(
    data: pd.DataFrame, representation: str
) -> tuple[np.ndarray, int, float]:
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


def cluster_labels(
    matrix: np.ndarray, method: str, random_state: int = RANDOM_STATE
) -> np.ndarray:
    if method == "KMeans":
        return KMeans(
            n_clusters=2,
            n_init=50,
            random_state=random_state,
        ).fit_predict(matrix)
    if method == "GaussianMixture":
        return GaussianMixture(
            n_components=2,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state,
        ).fit_predict(matrix)
    if method == "AgglomerativeClustering":
        return AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(matrix)
    raise ValueError(f"Unknown clustering method: {method}")


def clustering_quality(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, str]:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return np.nan, np.nan, np.nan, "single cluster produced"
    try:
        silhouette = float(silhouette_score(matrix, labels))
        calinski = float(calinski_harabasz_score(matrix, labels))
        davies = float(davies_bouldin_score(matrix, labels))
    except ValueError as exc:
        return np.nan, np.nan, np.nan, f"metric calculation failed: {exc}"
    return silhouette, calinski, davies, ""


def stability_analysis(
    data: pd.DataFrame,
    representation: str,
    method: str,
    full_labels: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    n_subjects = data.shape[0]
    sample_size = int(round(n_subjects * STABILITY_SAMPLE_FRACTION))
    scores: list[float] = []

    for run_index in range(STABILITY_RUNS):
        sample_index = np.sort(rng.choice(n_subjects, size=sample_size, replace=False))
        sample_data = data.iloc[sample_index]
        sample_matrix, _, _ = build_representation(sample_data, representation)
        sample_labels = cluster_labels(
            sample_matrix,
            method,
            random_state=RANDOM_STATE + run_index + 1,
        )
        scores.append(float(adjusted_rand_score(full_labels[sample_index], sample_labels)))

    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def run_clustering_candidates(
    pd_subjects: pd.DataFrame, feature_sets: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    representations = {
        "top20_biomarkers": ["scaled_only", "pca_5", "pca_10"],
        "top50_biomarkers": ["pca_10", "pca_90"],
        "all_acoustic": ["pca_20", "pca_90"],
    }
    methods = ["KMeans", "GaussianMixture", "AgglomerativeClustering"]

    rows: list[dict[str, Any]] = []
    candidate_artifacts: dict[tuple[str, str, str], dict[str, Any]] = {}

    for feature_set, features in feature_sets.items():
        data = pd_subjects[features]
        for representation in representations[feature_set]:
            matrix, n_components, explained_variance = build_representation(data, representation)
            for method in methods:
                labels = cluster_labels(matrix, method)
                counts = pd.Series(labels).value_counts().reindex([0, 1], fill_value=0)
                cluster_0_size = int(counts.loc[0])
                cluster_1_size = int(counts.loc[1])
                smaller_cluster_size = int(min(cluster_0_size, cluster_1_size))
                smaller_cluster_ratio = float(smaller_cluster_size / len(labels))

                silhouette, calinski, davies, metric_warning = clustering_quality(matrix, labels)
                stability_mean, stability_std = stability_analysis(
                    data,
                    representation,
                    method,
                    labels,
                )

                warnings = []
                if metric_warning:
                    warnings.append(metric_warning)
                if smaller_cluster_ratio < 0.10:
                    warnings.append("smaller_cluster_ratio < 0.10; not eligible for recommendation")

                key = (feature_set, representation, method)
                rows.append(
                    {
                        "feature_set": feature_set,
                        "representation": representation,
                        "clustering_method": method,
                        "n_subjects": int(len(labels)),
                        "n_features_original": int(len(features)),
                        "n_components_after_pca": int(n_components),
                        "pca_explained_variance_sum": explained_variance,
                        "cluster_0_size": cluster_0_size,
                        "cluster_1_size": cluster_1_size,
                        "smaller_cluster_size": smaller_cluster_size,
                        "smaller_cluster_ratio": smaller_cluster_ratio,
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
                candidate_artifacts[key] = {
                    "labels": labels,
                    "matrix": matrix,
                    "features": features,
                    "n_components_after_pca": n_components,
                    "pca_explained_variance_sum": explained_variance,
                }

    metrics = pd.DataFrame(rows)
    return metrics, candidate_artifacts


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = series.astype(float)
    valid = values.replace([np.inf, -np.inf], np.nan)
    min_value = valid.min(skipna=True)
    max_value = valid.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.5, index=series.index)
    scaled = (valid - min_value) / (max_value - min_value)
    if not higher_is_better:
        scaled = 1 - scaled
    return scaled.fillna(0.0)


def choose_recommended(metrics: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str, str]]:
    out = metrics.copy()
    eligible = out["smaller_cluster_ratio"] >= 0.10
    if not eligible.any():
        raise ValueError("No clustering candidate has smaller_cluster_ratio >= 0.10")

    balance = 1 - (0.5 - out["smaller_cluster_ratio"]).abs() / 0.5
    score = (
        0.25 * minmax(out["silhouette"], higher_is_better=True)
        + 0.15 * minmax(out["calinski_harabasz"], higher_is_better=True)
        + 0.15 * minmax(out["davies_bouldin"], higher_is_better=False)
        + 0.30 * minmax(out["stability_ari_mean"], higher_is_better=True)
        + 0.15 * balance
    )
    score += out["feature_set"].map(
        {
            "top20_biomarkers": 0.08,
            "top50_biomarkers": 0.04,
            "all_acoustic": -0.02,
        }
    )
    score = score.where(eligible, -np.inf)
    out["selection_score"] = score

    best_index = int(out["selection_score"].idxmax())
    best_score = float(out.loc[best_index, "selection_score"])

    top20_close = out[
        (out["feature_set"] == "top20_biomarkers")
        & eligible
        & (out["selection_score"] >= best_score - 0.03)
    ]
    if not top20_close.empty:
        best_index = int(top20_close["selection_score"].idxmax())
    else:
        interpretable_close = out[
            (out["feature_set"].isin(["top20_biomarkers", "top50_biomarkers"]))
            & eligible
            & (out["selection_score"] >= best_score - 0.08)
        ]
        if not interpretable_close.empty and out.loc[best_index, "feature_set"] == "all_acoustic":
            best_index = int(interpretable_close["selection_score"].idxmax())

    out["recommended_flag"] = False
    out.loc[best_index, "recommended_flag"] = True
    key = (
        str(out.loc[best_index, "feature_set"]),
        str(out.loc[best_index, "representation"]),
        str(out.loc[best_index, "clustering_method"]),
    )
    return out, key


def pca_2d_coordinates(data: pd.DataFrame) -> np.ndarray:
    scaled = StandardScaler().fit_transform(data.to_numpy(dtype=float))
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)


def write_cluster_assignments(
    pd_subjects: pd.DataFrame,
    recommended_key: tuple[str, str, str],
    recommended_artifact: dict[str, Any],
) -> pd.DataFrame:
    feature_set, representation, method = recommended_key
    labels = np.asarray(recommended_artifact["labels"], dtype=int)
    counts = pd.Series(labels).value_counts().to_dict()
    coords = pca_2d_coordinates(pd_subjects[recommended_artifact["features"]])
    assignments = pd.DataFrame(
        {
            "id": pd_subjects["id"].to_numpy(),
            "class": pd_subjects["class"].to_numpy(),
            "sex_male": pd_subjects["sex_male"].to_numpy(),
            "cluster": labels,
            "cluster_size": [int(counts[label]) for label in labels],
            "feature_set": feature_set,
            "representation": representation,
            "clustering_method": method,
            "pca_1": coords[:, 0],
            "pca_2": coords[:, 1],
        }
    )
    write_csv(assignments, RESULTS_DIR / "stage5_cluster_assignments_dataset1_pd.csv")
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


def cluster_feature_differences(
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
    cluster_0_mask = labels == 0
    cluster_1_mask = labels == 1

    for feature in features:
        raw_0 = raw.loc[cluster_0_mask, feature].to_numpy(dtype=float)
        raw_1 = raw.loc[cluster_1_mask, feature].to_numpy(dtype=float)
        z_0 = scaled.loc[cluster_0_mask, feature].to_numpy(dtype=float)
        z_1 = scaled.loc[cluster_1_mask, feature].to_numpy(dtype=float)

        try:
            u_stat, p_value = mannwhitneyu(raw_0, raw_1, alternative="two-sided")
        except ValueError:
            u_stat, p_value = np.nan, np.nan

        mean_difference = float(np.mean(raw_1) - np.mean(raw_0))
        z_difference = float(np.mean(z_1) - np.mean(z_0))
        d_value = cohens_d(raw_1, raw_0)

        rows.append(
            {
                "feature": feature,
                "category": categorize_feature(feature),
                "cluster_0_mean": float(np.mean(raw_0)),
                "cluster_1_mean": float(np.mean(raw_1)),
                "mean_difference_cluster1_minus_cluster0": mean_difference,
                "cluster_0_zmean": float(np.mean(z_0)),
                "cluster_1_zmean": float(np.mean(z_1)),
                "zmean_difference_cluster1_minus_cluster0": z_difference,
                "cohens_d_cluster1_minus_cluster0": d_value,
                "mannwhitney_u": float(u_stat),
                "p_mannwhitney": float(p_value),
                "abs_cohens_d": float(abs(d_value)) if np.isfinite(d_value) else np.nan,
                "abs_zmean_difference": float(abs(z_difference)),
                "direction": (
                    "higher_in_cluster_1"
                    if z_difference > 0
                    else "higher_in_cluster_0"
                    if z_difference < 0
                    else "no_difference"
                ),
            }
        )

    differences = pd.DataFrame(rows)
    differences["q_mannwhitney"] = fdr_bh(differences["p_mannwhitney"].to_numpy())
    ordered_columns = [
        "feature",
        "category",
        "cluster_0_mean",
        "cluster_1_mean",
        "mean_difference_cluster1_minus_cluster0",
        "cluster_0_zmean",
        "cluster_1_zmean",
        "zmean_difference_cluster1_minus_cluster0",
        "cohens_d_cluster1_minus_cluster0",
        "mannwhitney_u",
        "p_mannwhitney",
        "q_mannwhitney",
        "abs_cohens_d",
        "abs_zmean_difference",
        "direction",
    ]
    differences = differences[ordered_columns]
    differences = differences.sort_values(
        ["abs_zmean_difference", "abs_cohens_d", "q_mannwhitney"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    top_features = differences.head(20).copy()

    write_csv(differences, RESULTS_DIR / "stage5_cluster_feature_differences.csv")
    write_csv(top_features, RESULTS_DIR / "stage5_cluster_interpretation_top_features.csv")
    return differences, top_features


def plot_pca_clusters(
    pd_subjects: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    coords = pca_2d_coordinates(pd_subjects[features])
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=42,
            alpha=0.82,
            label=f"Cluster {cluster}",
        )
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(title="Main cluster label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_cluster_sizes(metrics: pd.DataFrame) -> None:
    plot_data = metrics.copy()
    plot_data["candidate"] = (
        plot_data["feature_set"]
        + "\n"
        + plot_data["representation"]
        + "\n"
        + plot_data["clustering_method"]
    )
    colors = np.where(plot_data["recommended_flag"], "#d55e00", "#4c78a8")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(np.arange(plot_data.shape[0]), plot_data["smaller_cluster_ratio"], color=colors)
    ax.axhline(0.10, color="#cc3311", linestyle="--", linewidth=1.2, label="0.10 threshold")
    ax.set_xticks(np.arange(plot_data.shape[0]))
    ax.set_xticklabels(plot_data["candidate"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Smaller cluster ratio")
    ax.set_title("Stage 5 Candidate Cluster Size Balance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage5_cluster_size_comparison.png", dpi=220)
    plt.close(fig)


def plot_feature_difference_top20(top_features: pd.DataFrame) -> None:
    plot_data = top_features.sort_values("cohens_d_cluster1_minus_cluster0")
    colors = np.where(
        plot_data["cohens_d_cluster1_minus_cluster0"] >= 0,
        "#2f80ed",
        "#e07a5f",
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        plot_data["feature"],
        plot_data["cohens_d_cluster1_minus_cluster0"],
        color=colors,
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Cohen's d (cluster 1 minus cluster 0)")
    ax.set_ylabel("Feature")
    ax.set_title("Stage 5 Top 20 Cluster-Distinguishing Voice Features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage5_cluster_feature_difference_top20.png", dpi=220)
    plt.close(fig)


def run_cluster_label_classifier(
    pd_subjects: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    x = pd_subjects[features].to_numpy(dtype=float)
    y = labels.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
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
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        fold_rows: list[dict[str, Any]] = []
        for fold_index, (train_index, test_index) in enumerate(cv.split(x, y), start=1):
            model.fit(x[train_index], y[train_index])
            predictions = model.predict(x[test_index])
            try:
                probabilities = model.predict_proba(x[test_index])[:, 1]
                auc = float(roc_auc_score(y[test_index], probabilities))
            except (AttributeError, ValueError):
                auc = np.nan

            fold_row = {
                "model": model_name,
                "fold": fold_index,
                "accuracy": float(accuracy_score(y[test_index], predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y[test_index], predictions)
                ),
                "f1_macro": float(f1_score(y[test_index], predictions, average="macro")),
                "f1_weighted": float(
                    f1_score(y[test_index], predictions, average="weighted")
                ),
                "roc_auc": auc,
            }
            rows.append(fold_row)
            fold_rows.append(fold_row)

        fold_metrics = pd.DataFrame(fold_rows)
        for summary_name, reducer in [("mean", np.nanmean), ("std", np.nanstd)]:
            rows.append(
                {
                    "model": model_name,
                    "fold": summary_name,
                    "accuracy": float(reducer(fold_metrics["accuracy"])),
                    "balanced_accuracy": float(
                        reducer(fold_metrics["balanced_accuracy"])
                    ),
                    "f1_macro": float(reducer(fold_metrics["f1_macro"])),
                    "f1_weighted": float(reducer(fold_metrics["f1_weighted"])),
                    "roc_auc": float(reducer(fold_metrics["roc_auc"])),
                }
            )

    classifier_metrics = pd.DataFrame(rows)
    write_csv(
        classifier_metrics,
        RESULTS_DIR / "stage5_cluster_label_classifier_metrics.csv",
    )
    return classifier_metrics


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df.loc[:, columns].copy()
    table.columns = [str(column) for column in table.columns]
    if max_rows is not None:
        table = table.head(max_rows)
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def report_selection_reason(metrics: pd.DataFrame, recommended: pd.Series) -> str:
    high_metric = metrics.sort_values("selection_score", ascending=False).head(3)
    return (
        f"主推荐方案为 `{recommended['feature_set']} / {recommended['representation']} / "
        f"{recommended['clustering_method']}`。该方案最小簇比例为 "
        f"{recommended['smaller_cluster_ratio']:.3f}，稳定性 ARI 均值为 "
        f"{recommended['stability_ari_mean']:.3f}，在聚类质量、簇均衡性、稳定性与解释性之间取得"
        "较好平衡。Top20/Top50 候选优先于 all_acoustic，因为其特征来源于 Stage 4 候选生物标志物，"
        "更适合论文解释；all_acoustic 结果作为参考，避免把高维噪声依赖方案作为默认主结论。\n\n"
        "选择时没有只看单一指标。若某些候选方案单项 silhouette 或 Calinski-Harabasz 较高，"
        "但簇规模过度失衡、稳定性较弱、或依赖更高维声学空间，则不作为主推荐方案。\n\n"
        "Selection score 前三候选：\n\n"
        + markdown_table(
            high_metric,
            [
                "feature_set",
                "representation",
                "clustering_method",
                "selection_score",
                "smaller_cluster_ratio",
                "silhouette",
                "stability_ari_mean",
            ],
        )
    )


def write_report(
    subject_table: pd.DataFrame,
    pd_subjects: pd.DataFrame,
    metrics: pd.DataFrame,
    assignments: pd.DataFrame,
    top_features: pd.DataFrame,
    classifier_metrics: pd.DataFrame,
) -> None:
    recommended = metrics.loc[metrics["recommended_flag"]].iloc[0]
    cluster_counts = assignments["cluster"].value_counts().sort_index()
    cluster_ratio = (cluster_counts / cluster_counts.sum()).round(4)
    sex_table = pd.crosstab(assignments["cluster"], assignments["sex_male"], margins=True)
    category_counts = top_features["category"].value_counts().reset_index()
    category_counts.columns = ["category", "top_feature_count"]
    classifier_summary = classifier_metrics[classifier_metrics["fold"].isin(["mean", "std"])]

    required_boundary_text = (
        "由于附件数据未提供运动型与非运动型的真实临床标签，本文不直接构建监督分类模型，"
        "而是在帕金森病受试者内部进行无监督二类聚类，以获得基于语音特征的潜在分型。"
        "随后训练的分类器预测的是聚类标签，而非临床确诊标签。因此，该模型只能作为"
        "运动型/非运动型辅助诊断的探索性参考，仍需真实临床症状标签进一步验证。"
    )

    report = [
        "# Stage 5 Dataset 1 PD Two-Cluster Voice Phenotyping",
        "",
        "## 1. Stage 5 目标与边界",
        "本阶段只在 Dataset 1 的 PD 受试者内部进行 k=2 无监督聚类，目标是得到基于语音特征的二类潜在表型，并训练一个预测聚类标签的 cluster-label classifier。",
        "所有结论均为探索性辅助分型，不是临床确诊结果。",
        "",
        "## 2. 为什么不能做真实运动型/非运动型监督诊断",
        required_boundary_text,
        "",
        "## 3. 输入文件与样本审计",
        f"- 输入 subject-level 文件：`{rel(SUBJECT_TABLE_PATH)}`",
        f"- 总受试者数：`{subject_table.shape[0]}`",
        f"- PD 受试者数：`{pd_subjects.shape[0]}`",
        f"- class 分布：`{subject_table['class'].value_counts().sort_index().to_dict()}`",
        f"- 缺失值总数：`{int(subject_table.isna().sum().sum())}`",
        f"- duplicated id 数：`{int(subject_table['id'].duplicated().sum())}`",
        "",
        "## 4. 特征集与预处理",
        "使用 Stage 4 的 Top20、Top50 与 all_acoustic 三套特征集。所有候选方案先执行 StandardScaler；PCA 表示均在标准化之后生成。",
        "`id`, `gender`, `class`, `sex_male`, `record_count` 及任何标签或后续生成列均不进入聚类输入。`sex_male` 仅用于事后描述。",
        "",
        "## 5. 聚类候选方案完整指标表",
        markdown_table(
            metrics,
            [
                "feature_set",
                "representation",
                "clustering_method",
                "cluster_0_size",
                "cluster_1_size",
                "smaller_cluster_ratio",
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
                "stability_ari_mean",
                "recommended_flag",
            ],
        ),
        "",
        "## 6. 稳定性分析结果",
        f"每个候选方案执行 `{STABILITY_RUNS}` 次 80% 受试者重采样，重新标准化、PCA 与聚类，并在可比较样本上计算 Adjusted Rand Index。",
        "稳定性均值和标准差已写入 `results/stage5_clustering_metrics.csv`。",
        "",
        "## 7. 主推荐方案选择理由",
        report_selection_reason(metrics, recommended),
        "",
        "## 8. 主方案两簇人数与比例",
        markdown_table(
            pd.DataFrame(
                {
                    "cluster": cluster_counts.index,
                    "n_subjects": cluster_counts.values,
                    "ratio": cluster_ratio.values,
                }
            ),
            ["cluster", "n_subjects", "ratio"],
        ),
        "",
        "## 9. 两簇 Top 区分语音特征",
        markdown_table(
            top_features,
            [
                "feature",
                "category",
                "zmean_difference_cluster1_minus_cluster0",
                "cohens_d_cluster1_minus_cluster0",
                "q_mannwhitney",
                "direction",
            ],
        ),
        "",
        "## 10. 特征类别分布与解释",
        markdown_table(category_counts, ["category", "top_feature_count"]),
        "Top 区分特征的类别分布用于描述两个潜在语音表型的声学差异来源，不构成临床病因解释。",
        "",
        "## 11. 性别分布的事后描述",
        markdown_table(sex_table.reset_index(), list(sex_table.reset_index().columns)),
        "`sex_male` 未进入聚类或 cluster-label classifier，只作为事后描述，不能作因果解释。",
        "",
        "## 12. cluster-label classifier 结果",
        markdown_table(
            classifier_summary,
            [
                "model",
                "fold",
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
                "roc_auc",
            ],
        ),
        "该分类器学习的是无监督聚类标签，用于将潜在语音表型分型规则模型化；它不是训练自真实临床运动型/非运动型标签，因此不能解释为真实临床症状分类性能。",
        "",
        "## 13. 局限性",
        "- 聚类标签来自语音特征内部结构，没有外部临床症状标签验证。",
        "- Mann-Whitney U 与 FDR 是聚类后的描述性比较，不是外部验证假设检验，也不是临床显著性证据。",
        "- cluster 编号本身没有临床含义；任何 potential motor-like voice phenotype 或 potential nonmotor-like voice phenotype 命名都只是解释性命名。",
        "- all_acoustic 高维结果可能受冗余与噪声影响，因此不优先作为论文主解释方案。",
        "",
        "## 14. 论文建议表述",
        "建议写作：本文在 PD 受试者内部基于 Stage 4 筛选的语音特征进行 k=2 无监督聚类，得到两个基于语音特征的潜在表型簇，并进一步训练 cluster-label classifier 以复现该探索性分型规则。",
        "不建议写作：将本阶段结果表述为真实临床亚型诊断性能。",
        "",
        "## 输出文件",
        "- `results/stage5_pd_subject_clustering_input_audit.csv`",
        "- `results/stage5_clustering_metrics.csv`",
        "- `results/stage5_cluster_assignments_dataset1_pd.csv`",
        "- `results/stage5_cluster_feature_differences.csv`",
        "- `results/stage5_cluster_interpretation_top_features.csv`",
        "- `results/stage5_cluster_label_classifier_metrics.csv`",
        "- `results/stage5_two_cluster_report.md`",
        "- `results/figures/stage5_pca_k2_clusters_top20.png`",
        "- `results/figures/stage5_pca_k2_clusters_top50.png`",
        "- `results/figures/stage5_cluster_size_comparison.png`",
        "- `results/figures/stage5_cluster_feature_difference_top20.png`",
    ]
    (RESULTS_DIR / "stage5_two_cluster_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    subject_table, top20_features, top50_features = load_inputs()
    all_features = acoustic_columns(subject_table)
    pd_subjects = validate_pd_subjects(subject_table)
    write_input_audit(
        subject_table,
        pd_subjects,
        all_features,
        top20_features,
        top50_features,
    )

    feature_sets = {
        "top20_biomarkers": top20_features,
        "top50_biomarkers": top50_features,
        "all_acoustic": all_features,
    }
    metrics, artifacts = run_clustering_candidates(pd_subjects, feature_sets)
    metrics, recommended_key = choose_recommended(metrics)
    write_csv(metrics, RESULTS_DIR / "stage5_clustering_metrics.csv")

    recommended_artifact = artifacts[recommended_key]
    labels = np.asarray(recommended_artifact["labels"], dtype=int)
    assignments = write_cluster_assignments(
        pd_subjects,
        recommended_key,
        recommended_artifact,
    )

    differences, top_features = cluster_feature_differences(
        pd_subjects,
        labels,
        recommended_artifact["features"],
    )
    classifier_metrics = run_cluster_label_classifier(
        pd_subjects,
        recommended_artifact["features"],
        labels,
    )

    plot_pca_clusters(
        pd_subjects,
        top20_features,
        labels,
        FIGURES_DIR / "stage5_pca_k2_clusters_top20.png",
        "Stage 5 Main Cluster Labels Projected on Top20 Biomarker PCA",
    )
    plot_pca_clusters(
        pd_subjects,
        top50_features,
        labels,
        FIGURES_DIR / "stage5_pca_k2_clusters_top50.png",
        "Stage 5 Main Cluster Labels Projected on Top50 Biomarker PCA",
    )
    plot_cluster_sizes(metrics)
    plot_feature_difference_top20(top_features)

    write_report(
        subject_table,
        pd_subjects,
        metrics,
        assignments,
        top_features,
        classifier_metrics,
    )


if __name__ == "__main__":
    main()
