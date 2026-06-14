# Six-Stage Logic Audit Report

Final conclusion: `CONDITIONAL PASS`

## Rerun Decision

No core stage rerun is required based on this audit; address warnings as documentation or boundary clarifications unless new evidence changes them.

## Check Summary

- Total checks: `88`
- Status counts: `{'pass': 87, 'warning': 1}`
- Static code findings: `861` rows; warnings `205`, fails `0`

| stage | pass | warning |
| --- | --- | --- |
| Cross-stage | 5 | 0 |
| Stage 1 | 14 | 0 |
| Stage 2 | 12 | 0 |
| Stage 3 | 9 | 0 |
| Stage 4 | 11 | 0 |
| Stage 5 | 17 | 0 |
| Stage 6 | 18 | 1 |
| Static code audit | 1 | 0 |

## Redline Failures

No critical/high redline failures were found.

## Warnings

| stage | check_name | observed | severity | evidence_file | notes |
| --- | --- | --- | --- | --- | --- |
| Stage 6 | Stage 5 cluster use limited to alignment/crosswalk/recommendation scoring | stage5_nmi appears in recommendation scoring | medium | src/stage6_pd_six_cluster_phenotyping.py | This is not clustering input leakage, but it should be disclosed because Stage 5 labels influence candidate recommendation. |

Static pattern warnings are explainability items, not automatic redlines:

| file | line | keyword | risk_level | explanation | needs_modification |
| --- | --- | --- | --- | --- | --- |
| src/stage2_qc_pipeline.py | 308 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage2_qc_pipeline.py | 310 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 239 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 240 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 241 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 244 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 254 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 255 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 256 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 257 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 470 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 471 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 511 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage3_baseline_models.py | 626 | SMOTE | warning | SMOTE appears; verify it is not used outside grouped CV. | no |
| src/stage4_biomarker_identification.py | 61 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage4_biomarker_identification.py | 86 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage4_biomarker_identification.py | 91 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage4_biomarker_identification.py | 210 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage4_biomarker_identification.py | 260 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage4_biomarker_identification.py | 308 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 59 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 147 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 236 | fit_transform | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 236 | StandardScaler( | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 241 | PCA( | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 243 | PCA( | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 247 | fit_transform | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 251 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 269 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 272 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 275 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 300 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 328 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 330 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 331 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 332 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 333 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 335 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 346 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 347 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 354 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 446 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 452 | fit_transform | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 452 | StandardScaler( | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 453 | fit_transform | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 453 | PCA( | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 470 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 471 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 474 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 515 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 522 | fit_transform | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 522 | StandardScaler( | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 527 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 528 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 531 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 601 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 610 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 611 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 617 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 622 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 639 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 643 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 665 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 669 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 669 | stage5_cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 673 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 673 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 684 | StandardScaler( | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 697 | StandardScaler( | warning | Unsupervised PD-only clustering fits scaler/PCA on the PD subject subset; this is acceptable only because it is not supervised CV and forbidden columns are excluded. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 760 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 762 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 763 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 763 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 763 | stage5_cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 765 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 841 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 841 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 915 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 915 | cluster | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |
| src/stage5_pd_two_cluster_phenotyping.py | 917 | class | warning | Label appears as a target or derived assignment; verify it is not part of feature matrices. | no |

## Output Files

- `results/audit_all_stages_logic_report.md`
- `results/audit_all_stages_checks.csv`
- `results/audit_feature_leakage_checks.csv`
- `results/audit_subject_lineage_checks.csv`
- `results/audit_metric_consistency_checks.csv`
- `results/audit_static_code_risk_patterns.csv`

## Minimal Fix Guidance

Keep the sealed modeling outputs intact. Recommended follow-up is limited to clarifying warnings, especially documentation around Stage 6 candidate recommendation scoring and any report encoding/display issues.