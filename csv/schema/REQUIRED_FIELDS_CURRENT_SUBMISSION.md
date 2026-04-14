# Required Fields For The Current Submission Workflow

The shareable package does not contain patient-level raw data. To rerun the code locally after ethics-approved access is granted, place the original exports in `csv/` using the filenames below.

## 1. `ZERNIKE-WFA.CSV`
Required identifier and cohort fields:
- `Last Name:`
- `First Name:`
- `D.o.Birth:`
- `Exam Date:`
- `Exam Time:`
- `Exam Eye:`
- `Error:`

Required analysis fields:
- `RMS (CF):`
- `RMS HOA (CF):`
- all corneal front `(CF)` Zernike coefficients from orders 2 to 6
- all corneal back `(CB)` Zernike coefficients from orders 2 to 6

## 2. `BADisplay-LOAD.CSV`
Required fields:
- identifier fields listed above
- `BAD D:`
- `Pachy Min.:` when available

## 3. `INDEX-LOAD.CSV`
Required fields:
- identifier fields listed above
- one pachymetry field used to derive CCT:
  - `Thinnest Pachy:`, or
  - `Pachy Min`, or
  - `D0mm Pachy:`
- `ISV:`
- `IVA:`
- `KI:`
- `CKI:`
- `IHA:`
- `IHD:`
- `K Max (Front):`

## Restricted non-export variable used in the manuscript
- Biological sex was described in the manuscript but was retrieved separately from hospital medical records, not from the standard Pentacam exports.
- That linked hospital-record variable is intentionally not distributed in this package.
