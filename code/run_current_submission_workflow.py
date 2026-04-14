from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
FIG_DIR = ROOT / "output" / "figures"
MANUSCRIPT_DIR = FIG_DIR / "manuscript_ready"
TABLE_DIR = ROOT / "output" / "tables"

SCRIPT_ORDER = [
    "phenotype_atlas_v4.py",
    "k2_characterisation.py",
    "k3_analysis.py",
    "pairwise_phenotype_analysis.py",
    "supplementary_analyses.py",
    "build_reviewer_round2_assets.py",
    "create_supplementary_materials.py",
    "build_clean_supplementary_docx.py",
]


def run_script(script_name: str) -> None:
    script_path = CODE_DIR / script_name
    print(f"Running {script_name} ...")
    subprocess.run([sys.executable, str(script_path)], cwd=ROOT, check=True)


def copy_alias(src_name: str, dst_name: str) -> None:
    src = FIG_DIR / src_name
    dst = MANUSCRIPT_DIR / dst_name
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    for script_name in SCRIPT_ORDER:
        run_script(script_name)

    copy_alias("fig01_cohort_flow.png", "Figure1_cohort_flow.png")
    copy_alias("fig_K2_full_atlas.png", "Figure2_K2_atlas.png")
    copy_alias("fig_K3_full_atlas.png", "Figure3_K3_atlas.png")
    copy_alias("Figure4_severity_independence.png", "Figure4_severity_independence.png")

    restricted_assignments = TABLE_DIR / "assignments.csv"
    if restricted_assignments.exists():
        restricted_assignments.unlink()
        print("Removed restricted output: output/tables/assignments.csv")

    print("Current submission workflow complete.")


if __name__ == "__main__":
    main()
