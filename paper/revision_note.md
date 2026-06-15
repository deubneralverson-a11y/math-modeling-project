# Revision Note

## Image Explanation Additions

- Added explanatory paragraphs after the three biomarker figures in `sections/05_biomarker.tex`:
  - Top20 candidate feature ranking.
  - Top20 effect size plot.
  - Top50 feature-category count plot.
- Added explanatory paragraphs after the two two-cluster phenotype figures in `sections/07_two_cluster.tex`:
  - PCA visualization of the k=2 latent voice phenotypes.
  - Cohen's \(d\) feature-difference plot.
- Added explanatory paragraphs after the six-cluster phenotype figures in `sections/08_six_cluster.tex`:
  - PCA visualization of the k=6 latent voice phenotypes.
  - Candidate metrics comparison.
  - Cluster size distribution.
  - Cluster feature heatmap.
  - Top features by cluster.
  - Two-cluster/six-cluster crosswalk heatmap.

## Reference Updates

- Removed the unverified local review-PDF reference.
- Expanded references to 15 checked classic sources covering:
  - Parkinson's disease clinical background.
  - PD voice and dysphonia monitoring.
  - Dataset 1 and Dataset 2 source papers.
  - Repeated voice-recording variable selection.
  - scikit-learn, SVM, Random Forest, FDR-BH, LASSO.
  - Silhouette, Calinski-Harabasz, Davies-Bouldin, and Adjusted Rand Index.

## Citation Additions

- Added clinical and voice-background citations in `sections/01_problem.tex`.
- Added Dataset 1/Dataset 2 source citations in `sections/03_data.tex`.
- Added PD voice machine-learning, SVM, Random Forest, and scikit-learn citations in `sections/04_model_diagnosis.tex`.
- Added FDR-BH, LASSO, and PD voice-feature citations in `sections/05_biomarker.tex`.
- Added clustering metric citations in `sections/07_two_cluster.tex` and `sections/08_six_cluster.tex`.
- Converted in-text citation markers to superscript numbered references via `\upcite{...}`.

## Unresolved Items

- None.

## Compile Status

- Static checks passed for unresolved-marker text, forbidden overclaims, unverified review reference text, and newly added figure paths.
- Latest `main.pdf` regeneration completed successfully with two XeLaTeX passes. The generated PDF has 83 pages.
