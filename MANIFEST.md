# Manifest

| Manuscript item | Package file(s) | Primary generating code |
| --- | --- | --- |
| Figure 1 cohort flow | `output/figures/manuscript_ready/Figure1_cohort_flow.png` | `code/phenotype_atlas_v4.py` |
| Figure 2 K=2 atlas | `output/figures/manuscript_ready/Figure2_K2_atlas.png` | `code/k2_characterisation.py` |
| Figure 3 K=3 atlas | `output/figures/manuscript_ready/Figure3_K3_atlas.png` | `code/k3_analysis.py` |
| Figure 4 severity dependence | `output/figures/manuscript_ready/Figure4_severity_independence.png` and `output/provenance/reviewer_round2_metrics.json` | `code/build_reviewer_round2_assets.py` |
| K=2 summary statistics | `output/tables/k2_phenotype_summary.csv` | `code/k2_characterisation.py` |
| K=3 summary statistics | `output/tables/k3_phenotype_summary.csv` | `code/k3_analysis.py` |
| K=4 pairwise distinguishability | `output/tables/pairwise_full.csv`, `output/tables/pairwise_hoa.csv` | `code/phenotype_atlas_v4.py` + `code/pairwise_phenotype_analysis.py` |
| Supplementary cluster-number metrics | `output/tables/k_comparison_extended.csv`, `output/tables/weight_sensitivity.csv`, `output/tables/silhouette_comparison.csv`, `output/tables/centroid_concordance.json`, `output/tables/effect_sizes.csv`, `output/tables/diagnostic_distribution.csv` | `code/supplementary_analyses.py` |
| Final supplementary document | `supplementary_materials/BMC_Ophthalmology_Supplementary_Material_FINAL_round2.docx` | `code/create_supplementary_materials.py` + `code/build_clean_supplementary_docx.py` |

## Restricted-but-documented steps
- `code/phenotype_atlas_v4.py` generates `output/tables/assignments.csv` locally when raw data are supplied.
- That file is needed only as an intermediate for the K=4 pairwise phenotype analysis and is excluded from this package because it carries patient-level linkage fields.
- `code/run_current_submission_workflow.py` removes that file after the downstream steps finish.
