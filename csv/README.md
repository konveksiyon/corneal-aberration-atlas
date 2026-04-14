# Raw Data Placeholder

This folder is intentionally distributed without patient-level raw data.

Why the raw files are absent:
- the original Pentacam exports contain direct identifiers such as names and birth dates
- the manuscript also references a restricted sex variable obtained separately from hospital medical records
- these materials should not be placed in a public package

To rerun the workflow locally after approved access:
1. place the original device exports in this folder as `ZERNIKE-WFA.CSV`, `BADisplay-LOAD.CSV`, and `INDEX-LOAD.CSV`
2. install dependencies from `../requirements.txt`
3. run `python ../code/run_current_submission_workflow.py`

Column inventories and the minimum required fields are documented under `schema/`.
