# Corneal Aberration Atlas — Transparency Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19582389.svg)](https://doi.org/10.5281/zenodo.19582389)

Peer-review transparency package for the manuscript submitted to BMC Ophthalmology.

## Source manuscript
- File: `A label-free machine learning atlas of corneal aberration cluster.docx`
- Last modified: `2026-04-14T23:31:00`
- Title: "A label-free machine learning atlas of corneal aberration clusters from dual-surface Zernike coefficients: a retrospective cross-sectional study of 3,533 eyes"

## Included here
- the current analysis and asset-generation scripts under `code/`
- de-identified aggregate output tables under `output/tables/`
- manuscript-ready main figures under `output/figures/manuscript_ready/`
- final supplementary figures, table images, and DOCX under `supplementary_materials/`
- data schema, sharing boundaries, manuscript-alignment notes, and declaration text under `csv/` and `docs/`

## Intentionally excluded
- raw Pentacam export files containing names and birth dates
- patient-level derived outputs such as `assignments.csv`
- the restricted hospital-record sex linkage file used only for descriptive reporting

## Re-run instructions
1. Put the approved raw CSV exports into `csv/`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python code/run_current_submission_workflow.py`.

## Important note
The package is current-manuscript focused. It keeps only the scripts and outputs required to explain the submitted analyses, figures, and supplementary material, while documenting restricted items separately so that old branches do not create confusion.
