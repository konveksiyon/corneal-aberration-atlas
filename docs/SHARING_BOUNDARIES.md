# Sharing Boundaries

## Safe to share from this package
- analysis code under `code/`
- aggregate, de-identified result tables under `output/tables/`
- figure files and supplementary material assets
- schema files and textual documentation

## Do not share publicly
- raw exports placed into `csv/`
- any hospital-record linkage file used to obtain sex
- `output/tables/assignments.csv` if it is created during a local rerun
- any ad hoc patient-level exports generated outside this package

## Why `assignments.csv` is restricted
The intermediate assignment file contains `patient_id` and `exam_key` values derived from name, date of birth, eye, exam date, and exam time. Even though it is analytically useful for pairwise phenotype comparisons, it should be treated as confidential and must not be included in a public transparency package.

## Minimal public-sharing checklist
1. Confirm `csv/` contains no raw data.
2. Confirm `output/tables/assignments.csv` does not exist.
3. Confirm no extra files with patient names were added manually.
4. Share only the package contents listed in `MANIFEST.md`.
