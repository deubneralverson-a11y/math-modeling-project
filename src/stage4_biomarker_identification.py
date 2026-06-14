from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATASET1_PATH = RESULTS_DIR / "model_ready_dataset1_deduplicated.csv"
DATASET2_REFERENCE_PATH = RESULTS_DIR / "model_ready_dataset2_prepared.csv"
OUTPUT_CSV_ENCODING = "utf-8-sig"

RANDOM_STATE = 42
N_SPLITS = 5
POS_LABEL = 1
NEG_LABEL = 0
NEAR_CONSTANT_STD_THRESHOLD = 1e-12
L1_CS = [0.01, 0.03, 0.1, 0.3, 1.0]
COEF_EPSILON = 1e-8
N_ESTIMATORS = 500
PERMUTATION_REPEATS = 1

METADATA_COLUMNS = {"id", "gender", "class", "sex_male"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding=OUTPUT_CSV_ENCODING)


def load_dataset1() -> pd.DataFrame:
    if not DATASET1_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {DATASET1_PATH}")
    df = pd.read_csv(DATASET1_PATH)
    required = {"id", "gender", "class", "sex_male"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset 1 is missing required columns: {missing}")
    if df.isna().sum().sum() > 0:
        raise ValueError("Dataset 1 contains missing values before aggregation")
    if int(df.groupby("id")["class"].nunique(dropna=False).max()) != 1:
        raise ValueError("Dataset 1 has subjects with inconsistent class labels")
    if int(df.groupby("id")["sex_male"].nunique(dropna=False).max()) != 1:
        raise ValueError("Dataset 1 has subjects with inconsistent sex_male values")
    return df


def acoustic_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in METADATA_COLUMNS
    ]
    if len(columns) != 752:
        raise ValueError(f"Expected 752 acoustic features, found {len(columns)}")
    forbidden = METADATA_COLUMNS & set(columns)
    if forbidden:
        raise ValueError(f"Forbidden metadata columns in acoustic features: {forbidden}")
    return columns


def build_subject_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    feature_means = df.groupby("id", dropna=False)[features].mean().reset_index()
    subject_meta = (
        df.groupby("id", dropna=False)
        .agg(
            class_label=("class", "first"),
            sex_male=("sex_male", "first"),
            record_count=("id", "size"),
        )
        .reset_index()
        .rename(columns={"class_label": "class"})
    )
    subject_table = subject_meta.merge(feature_means, on="id", how="left")
    if subject_table.shape[0] != 252:
        raise ValueError(f"Expected 252 subjects, found {subject_table.shape[0]}")
    return subject_table


def audit_subject_table(
    subject_table: pd.DataFrame, features: list[str]
) -> dict[str, Any]:
    feature_audit = build_feature_audit_table(subject_table, features)
    return {
        "subject_count": int(subject_table.shape[0]),
        "class_distribution": {
            int(k): int(v)
            for k, v in subject_table["class"].value_counts().sort_index().items()
        },
        "acoustic_feature_count": len(features),
        "missing_value_count": int(subject_table.isna().sum().sum()),
        "constant_feature_count": int(feature_audit["is_constant"].sum()),
        "near_constant_feature_count": int(
            feature_audit["is_near_constant"].sum()
        ),
        "near_constant_features": feature_audit.loc[
            feature_audit["is_near_constant"], "feature"
        ].tolist(),
    }


def build_feature_audit_table(
    subject_table: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    feature_values = subject_table[features]
    stds = feature_values.std(ddof=0)
    unique_counts = feature_values.nunique(dropna=False)
    missing_counts = feature_values.isna().sum()
    audit = pd.DataFrame(
        {
            "feature": features,
            "category": [categorize_feature(feature) for feature in features],
            "subject_level_std": stds.reindex(features).to_numpy(),
            "subject_level_unique_count": unique_counts.reindex(features).to_numpy(),
            "subject_level_missing_count": missing_counts.reindex(features).to_numpy(),
        }
    )
    audit["is_constant"] = audit["subject_level_unique_count"] <= 1
    audit["is_near_constant"] = (
        audit["subject_level_std"] <= NEAR_CONSTANT_STD_THRESHOLD
    )
    return audit


def merge_feature_audit(df: pd.DataFrame, feature_audit: pd.DataFrame) -> pd.DataFrame:
    audit_cols = [
        "feature",
        "subject_level_std",
        "subject_level_unique_count",
        "subject_level_missing_count",
        "is_constant",
        "is_near_constant",
    ]
    return df.merge(feature_audit[audit_cols], on="feature", how="left")


def add_q_value_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["q_mannwhitney", "q_welch"]:
        if column in out.columns:
            out[f"{column}_scientific"] = out[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3e}"
            )
    return out


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


def cohens_d(pd_values: np.ndarray, healthy_values: np.ndarray) -> float:
    n_pd = len(pd_values)
    n_healthy = len(healthy_values)
    if n_pd < 2 or n_healthy < 2:
        return np.nan
    var_pd = np.var(pd_values, ddof=1)
    var_healthy = np.var(healthy_values, ddof=1)
    pooled = ((n_pd - 1) * var_pd + (n_healthy - 1) * var_healthy) / (
        n_pd + n_healthy - 2
    )
    if pooled <= 0:
        return 0.0
    return float((np.mean(pd_values) - np.mean(healthy_values)) / np.sqrt(pooled))


def cliffs_delta(pd_values: np.ndarray, healthy_values: np.ndarray) -> float:
    comparisons = pd_values[:, None] - healthy_values[None, :]
    greater = np.sum(comparisons > 0)
    lesser = np.sum(comparisons < 0)
    return float((greater - lesser) / comparisons.size)


def run_univariate_statistics(
    subject_table: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    y = subject_table["class"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = subject_table[feature].to_numpy(dtype=float)
        pd_values = values[y == POS_LABEL]
        healthy_values = values[y == NEG_LABEL]
        try:
            u_stat, mw_p = mannwhitneyu(
                pd_values, healthy_values, alternative="two-sided"
            )
        except ValueError:
            u_stat, mw_p = np.nan, np.nan
        try:
            t_stat, t_p = ttest_ind(
                pd_values, healthy_values, equal_var=False, nan_policy="omit"
            )
        except ValueError:
            t_stat, t_p = np.nan, np.nan
        d = cohens_d(pd_values, healthy_values)
        delta = cliffs_delta(pd_values, healthy_values)
        rows.append(
            {
                "feature": feature,
                "category": categorize_feature(feature),
                "mean_pd": float(np.mean(pd_values)),
                "mean_healthy": float(np.mean(healthy_values)),
                "median_pd": float(np.median(pd_values)),
                "median_healthy": float(np.median(healthy_values)),
                "mannwhitney_u": float(u_stat),
                "p_mannwhitney": float(mw_p),
                "welch_t": float(t_stat),
                "p_welch": float(t_p),
                "cohens_d": d,
                "abs_cohens_d": abs(d),
                "cliffs_delta": delta,
                "abs_cliffs_delta": abs(delta),
            }
        )
    stats = pd.DataFrame(rows)
    stats["q_mannwhitney"] = fdr_bh(stats["p_mannwhitney"].to_numpy())
    stats["q_welch"] = fdr_bh(stats["p_welch"].to_numpy())
    return stats.sort_values(["q_mannwhitney", "p_mannwhitney"], na_position="last")


def stratified_folds(y: np.ndarray) -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def run_l1_stability(subject_table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = subject_table[features]
    y = subject_table["class"].astype(int).to_numpy()
    selected_counts = pd.Series(0, index=features, dtype=int)
    abs_coef_sums = pd.Series(0.0, index=features, dtype=float)

    for fold, (train_idx, _) in enumerate(stratified_folds(y).split(x, y), start=1):
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegressionCV(
                        Cs=L1_CS,
                        cv=3,
                        penalty="l1",
                        solver="liblinear",
                        scoring="balanced_accuracy",
                        max_iter=5000,
                        random_state=RANDOM_STATE + fold,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        pipeline.fit(x.iloc[train_idx], y[train_idx])
        coefs = np.abs(pipeline.named_steps["model"].coef_.ravel())
        selected = coefs > COEF_EPSILON
        selected_counts += selected.astype(int)
        abs_coef_sums += coefs

    out = pd.DataFrame(
        {
            "feature": features,
            "category": [categorize_feature(feature) for feature in features],
            "selected_folds": selected_counts.to_numpy(),
            "selection_frequency": selected_counts.to_numpy() / N_SPLITS,
            "mean_abs_l1_coefficient": abs_coef_sums.to_numpy() / N_SPLITS,
        }
    )
    return out.sort_values(
        ["selection_frequency", "mean_abs_l1_coefficient"],
        ascending=[False, False],
    )


def run_tree_and_permutation(
    subject_table: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = subject_table[features].to_numpy(dtype=float)
    y = subject_table["class"].astype(int).to_numpy()
    tree_fold_importances: list[np.ndarray] = []
    permutation_fold_importances: list[np.ndarray] = []
    permutation_fold_stds: list[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(stratified_folds(y).split(x, y), start=1):
        model = ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE + fold,
            n_jobs=1,
        )
        model.fit(x[train_idx], y[train_idx])
        tree_fold_importances.append(model.feature_importances_)

        permutation_mean, permutation_std = heldout_permutation_importance(
            model,
            x[test_idx],
            y[test_idx],
            random_state=RANDOM_STATE + fold,
        )
        permutation_fold_importances.append(permutation_mean)
        permutation_fold_stds.append(permutation_std)

    tree_array = np.vstack(tree_fold_importances)
    permutation_array = np.vstack(permutation_fold_importances)
    permutation_std_array = np.vstack(permutation_fold_stds)

    tree_df = pd.DataFrame(
        {
            "feature": features,
            "category": [categorize_feature(feature) for feature in features],
            "tree_importance_mean": tree_array.mean(axis=0),
            "tree_importance_std": tree_array.std(axis=0, ddof=1),
        }
    ).sort_values("tree_importance_mean", ascending=False)

    permutation_df = pd.DataFrame(
        {
            "feature": features,
            "category": [categorize_feature(feature) for feature in features],
            "permutation_importance_mean": permutation_array.mean(axis=0),
            "permutation_importance_std_across_folds": permutation_array.std(
                axis=0, ddof=1
            ),
            "mean_within_fold_permutation_std": permutation_std_array.mean(axis=0),
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    return tree_df, permutation_df


def heldout_permutation_importance(
    model: ExtraTreesClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    baseline = balanced_accuracy_score(y_test, model.predict(x_test))
    importances = np.zeros((x_test.shape[1], PERMUTATION_REPEATS), dtype=float)
    for feature_index in range(x_test.shape[1]):
        for repeat in range(PERMUTATION_REPEATS):
            x_permuted = x_test.copy()
            x_permuted[:, feature_index] = rng.permutation(x_permuted[:, feature_index])
            score = balanced_accuracy_score(y_test, model.predict(x_permuted))
            importances[feature_index, repeat] = baseline - score
    ddof = 1 if PERMUTATION_REPEATS > 1 else 0
    return importances.mean(axis=1), importances.std(axis=1, ddof=ddof)


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


def rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(ascending=False, method="average", na_option="bottom")


def rank_asc(series: pd.Series) -> pd.Series:
    return series.rank(ascending=True, method="average", na_option="bottom")


def build_rank_summary(
    univariate: pd.DataFrame,
    l1: pd.DataFrame,
    tree: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        univariate[
            [
                "feature",
                "category",
                "p_mannwhitney",
                "q_mannwhitney",
                "cohens_d",
                "abs_cohens_d",
                "cliffs_delta",
                "abs_cliffs_delta",
                "q_mannwhitney_scientific",
                "subject_level_std",
                "subject_level_unique_count",
                "is_constant",
                "is_near_constant",
            ]
        ]
        .merge(
            l1[
                [
                    "feature",
                    "selected_folds",
                    "selection_frequency",
                    "mean_abs_l1_coefficient",
                    "l1_selected_any",
                    "l1_selected_near_constant",
                ]
            ],
            on="feature",
            how="left",
        )
        .merge(
            tree[["feature", "tree_importance_mean", "tree_importance_std"]],
            on="feature",
            how="left",
        )
        .merge(
            permutation[
                [
                    "feature",
                    "permutation_importance_mean",
                    "permutation_importance_std_across_folds",
                    "permutation_importance_positive",
                ]
            ],
            on="feature",
            how="left",
        )
    )
    summary["rank_q_mannwhitney"] = rank_asc(summary["q_mannwhitney"])
    summary["rank_abs_cohens_d"] = rank_desc(summary["abs_cohens_d"])
    summary["rank_abs_cliffs_delta"] = rank_desc(summary["abs_cliffs_delta"])
    summary["rank_selection_frequency"] = rank_desc(summary["selection_frequency"])
    summary["rank_permutation_importance"] = rank_desc(
        summary["permutation_importance_mean"]
    )
    summary["rank_tree_importance"] = rank_desc(summary["tree_importance_mean"])
    rank_cols = [
        "rank_q_mannwhitney",
        "rank_abs_cohens_d",
        "rank_abs_cliffs_delta",
        "rank_selection_frequency",
        "rank_permutation_importance",
        "rank_tree_importance",
    ]
    summary["mean_rank"] = summary[rank_cols].mean(axis=1)
    summary["rank_score"] = 1.0 / summary["mean_rank"]
    summary = summary.sort_values(["mean_rank", "feature"]).reset_index(drop=True)
    summary["included_in_top10"] = False
    summary["included_in_top20"] = False
    summary["included_in_top50"] = False
    summary.loc[summary.index[:10], "included_in_top10"] = True
    summary.loc[summary.index[:20], "included_in_top20"] = True
    summary.loc[summary.index[:50], "included_in_top50"] = True
    near_top50 = summary.loc[
        summary["included_in_top50"] & summary["is_near_constant"], "feature"
    ].tolist()
    if near_top50:
        raise ValueError(f"Near-constant features entered Top 50: {near_top50}")
    return summary


def save_top20_rank_plot(summary: pd.DataFrame) -> None:
    top20 = summary.head(20).sort_values("mean_rank", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    ax.barh(top20["feature"], top20["rank_score"], color="#2f80ed")
    ax.set_xlabel("Rank score (1 / mean rank)")
    ax.set_ylabel("Feature")
    ax.set_title("Stage 4 Top 20 Biomarker Candidates")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage4_top20_biomarker_rank.png")
    plt.close(fig)


def save_top20_effect_size_plot(summary: pd.DataFrame) -> None:
    top20 = summary.head(20).copy()
    top20["effect_direction"] = np.where(top20["cohens_d"] >= 0, "PD higher", "Healthy higher")
    colors = top20["effect_direction"].map(
        {"PD higher": "#d62728", "Healthy higher": "#1f77b4"}
    )
    top20 = top20.sort_values("cohens_d")
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    ax.barh(top20["feature"], top20["cohens_d"], color=colors.loc[top20.index])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Cohen's d (PD - Healthy)")
    ax.set_ylabel("Feature")
    ax.set_title("Stage 4 Top 20 Effect Sizes")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage4_top20_effect_size.png")
    plt.close(fig)


def save_top50_category_plot(summary: pd.DataFrame) -> None:
    counts = summary.head(50)["category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.barh(counts.index, counts.values, color="#27ae60")
    ax.set_xlabel("Count in Top 50")
    ax.set_ylabel("Feature category")
    ax.set_title("Stage 4 Feature Category Counts in Top 50")
    for index, value in enumerate(counts.values):
        ax.text(value, index, f" {int(value)}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage4_feature_category_counts_top50.png")
    plt.close(fig)


def save_volcano_plot(univariate: pd.DataFrame) -> None:
    plot_df = univariate.copy()
    min_positive_q = plot_df.loc[
        plot_df["q_mannwhitney"] > 0, "q_mannwhitney"
    ].min()
    if pd.isna(min_positive_q):
        min_positive_q = 1e-300
    plot_df["safe_q"] = plot_df["q_mannwhitney"].replace(0, min_positive_q)
    plot_df["neg_log10_q"] = -np.log10(plot_df["safe_q"])
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    ax.scatter(
        plot_df["cohens_d"],
        plot_df["neg_log10_q"],
        s=18,
        alpha=0.75,
        color="#4b5563",
        edgecolors="none",
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Cohen's d (PD - Healthy)")
    ax.set_ylabel("-log10(FDR q-value)")
    ax.set_title("Stage 4 Dataset 1 Volcano Plot")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage4_volcano_plot_dataset1.png")
    plt.close(fig)


def write_report(
    audit: dict[str, Any],
    summary: pd.DataFrame,
    univariate: pd.DataFrame,
    l1: pd.DataFrame,
) -> None:
    dataset2_ref = (
        f"`{rel(DATASET2_REFERENCE_PATH)}` exists"
        if DATASET2_REFERENCE_PATH.exists()
        else f"`{rel(DATASET2_REFERENCE_PATH)}` not found"
    )
    top20_cols = [
        "feature",
        "category",
        "mean_rank",
        "rank_score",
        "q_mannwhitney",
        "cohens_d",
        "cliffs_delta",
        "selection_frequency",
        "permutation_importance_mean",
        "tree_importance_mean",
    ]
    top10 = summary.head(10)[top20_cols].copy()
    top20 = summary.head(20)[top20_cols].copy()
    top50 = summary.head(50)[["feature", "category", "mean_rank", "rank_score"]].copy()
    near_constant_features = set(audit["near_constant_features"])
    near_constant_in_top50 = sorted(near_constant_features & set(summary.head(50)["feature"]))
    near_constant_l1_selected = l1[
        l1["feature"].isin(near_constant_features) & (l1["selected_folds"] > 0)
    ]
    nonzero_permutation_count = int(
        (summary["permutation_importance_mean"] > 0).sum()
    )
    lines = [
        "# Stage 4 Biomarker Candidate Report",
        "",
        "Main analysis: Dataset 1 subject-level PD/Healthy comparison.",
        "",
        "Dataset 2 is only noted as a reference prepared file and is not used in the main biomarker ranking.",
        f"Dataset 2 reference status: {dataset2_ref}.",
        "",
        "## Subject-level audit",
        "",
        f"- Subject count: `{audit['subject_count']}`",
        f"- Class distribution: `{audit['class_distribution']}`",
        f"- Acoustic feature count: `{audit['acoustic_feature_count']}`",
        f"- Missing value count: `{audit['missing_value_count']}`",
        f"- Constant feature count: `{audit['constant_feature_count']}`",
        f"- Near-constant feature count (`std <= {NEAR_CONSTANT_STD_THRESHOLD}`): `{audit['near_constant_feature_count']}`",
        f"- Near-constant features in Top 50: `{len(near_constant_in_top50)}`.",
        f"- Near-constant features selected by L1 at least once: `{len(near_constant_l1_selected)}`.",
        "- Main features exclude `id`, `gender`, `class`, and `sex_male`.",
        "",
        "## Methods",
        "",
        "- Dataset 1 records were aggregated to one row per subject before all statistical tests and models.",
        "- Univariate screening used Mann-Whitney U, Welch t-test, FDR-BH correction, Cohen's d, and Cliff's delta.",
        "- L1 stability selection used subject-level StratifiedKFold; scaling and L1 Logistic were fit only inside training folds.",
        "- ExtraTrees and permutation importance used subject-level StratifiedKFold; permutation importance was computed on held-out folds.",
        "- SHAP was not run because it is optional and unavailable in the current project environment.",
        "- These features are exploratory biomarker candidates for modeling interpretation and require independent validation.",
        "- Some near-constant features were selected by L1, so L1 stability is treated as one evidence source rather than a standalone final conclusion.",
        f"- Permutation importance is sparse (`{nonzero_permutation_count}` of `{summary.shape[0]}` features have positive mean importance), which may reflect substitution effects among high-dimensional correlated acoustic features.",
        "- The final Top 20 are candidate voice biomarker features, not clinical causal mechanisms.",
        "",
        "## Top 10 biomarker candidates",
        "",
        markdown_table(top10, floatfmt=".5f"),
        "",
        "## Top 20 biomarker candidates",
        "",
        markdown_table(top20, floatfmt=".5f"),
        "",
        "## Top 50 biomarker candidates",
        "",
        markdown_table(top50, floatfmt=".5f"),
        "",
        "## Output files",
        "",
        "- `results/stage4_subject_level_feature_table_dataset1.csv`",
        "- `results/stage4_feature_audit_dataset1.csv`",
        "- `results/stage4_univariate_statistics_dataset1.csv`",
        "- `results/stage4_l1_stability_selection_dataset1.csv`",
        "- `results/stage4_tree_importance_dataset1.csv`",
        "- `results/stage4_permutation_importance_dataset1.csv`",
        "- `results/stage4_biomarker_rank_summary_dataset1.csv`",
        "- `results/stage4_top10_biomarkers_dataset1.csv`",
        "- `results/stage4_top20_biomarkers_dataset1.csv`",
        "- `results/stage4_top50_biomarkers_dataset1.csv`",
        "- `results/figures/stage4_top20_biomarker_rank.png`",
        "- `results/figures/stage4_top20_effect_size.png`",
        "- `results/figures/stage4_feature_category_counts_top50.png`",
        "- `results/figures/stage4_volcano_plot_dataset1.png`",
        "",
        "## Guardrails",
        "",
        "- The 755 recordings were not treated as independent samples for significance testing.",
        "- Metadata and sex variables were excluded from the main acoustic feature list.",
        "- Final ranking combines univariate, stability, permutation, and tree-based evidence rather than relying only on tree importance.",
        "- The report uses associative interpretation only.",
        "",
        f"Univariate features with FDR q < 0.05: `{int((univariate['q_mannwhitney'] < 0.05).sum())}`",
    ]
    (RESULTS_DIR / "stage4_biomarker_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float) and not pd.isna(value):
                if column.startswith("q_"):
                    values.append(format(value, ".3e"))
                else:
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


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    df = load_dataset1()
    features = acoustic_columns(df)
    subject_table = build_subject_table(df, features)
    audit = audit_subject_table(subject_table, features)
    feature_audit = build_feature_audit_table(subject_table, features)

    univariate = run_univariate_statistics(subject_table, features)
    univariate = add_q_value_display_columns(
        merge_feature_audit(univariate, feature_audit)
    )
    l1 = merge_feature_audit(run_l1_stability(subject_table, features), feature_audit)
    l1["l1_selected_any"] = l1["selected_folds"] > 0
    l1["l1_selected_near_constant"] = l1["l1_selected_any"] & l1["is_near_constant"]
    tree, permutation = run_tree_and_permutation(subject_table, features)
    tree = merge_feature_audit(tree, feature_audit)
    permutation = merge_feature_audit(permutation, feature_audit)
    permutation["permutation_importance_positive"] = (
        permutation["permutation_importance_mean"] > 0
    )
    summary = build_rank_summary(univariate, l1, tree, permutation)
    top20 = summary.head(20).copy()
    top10 = summary.head(10).copy()
    top50 = summary.head(50).copy()

    write_csv(
        subject_table,
        RESULTS_DIR / "stage4_subject_level_feature_table_dataset1.csv",
    )
    write_csv(
        feature_audit,
        RESULTS_DIR / "stage4_feature_audit_dataset1.csv",
    )
    write_csv(
        univariate,
        RESULTS_DIR / "stage4_univariate_statistics_dataset1.csv",
    )
    write_csv(
        l1,
        RESULTS_DIR / "stage4_l1_stability_selection_dataset1.csv",
    )
    write_csv(
        tree,
        RESULTS_DIR / "stage4_tree_importance_dataset1.csv",
    )
    write_csv(
        permutation,
        RESULTS_DIR / "stage4_permutation_importance_dataset1.csv",
    )
    write_csv(
        summary,
        RESULTS_DIR / "stage4_biomarker_rank_summary_dataset1.csv",
    )
    write_csv(
        top10,
        RESULTS_DIR / "stage4_top10_biomarkers_dataset1.csv",
    )
    write_csv(
        top20,
        RESULTS_DIR / "stage4_top20_biomarkers_dataset1.csv",
    )
    write_csv(
        top50,
        RESULTS_DIR / "stage4_top50_biomarkers_dataset1.csv",
    )

    save_top20_rank_plot(summary)
    save_top20_effect_size_plot(summary)
    save_top50_category_plot(summary)
    save_volcano_plot(univariate)
    write_report(audit, summary, univariate, l1)


if __name__ == "__main__":
    main()
