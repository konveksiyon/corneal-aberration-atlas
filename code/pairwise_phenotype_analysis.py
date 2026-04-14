#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full Atlas vs HOA Atlas: Pairwise Phenotype Similarity Analysis
================================================================
Statistically compares all phenotype pairs in each atlas to identify
genuine separations versus artificial splits.
"""
import hashlib

import sys, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
TBL_DIR = BASE / "output" / "tables"

# Load assignments
assign = pd.read_csv(TBL_DIR / "assignments.csv")

# Load clinical data from pipeline
CSV_DIR = BASE / "csv"

def load(name):
    df = pd.read_csv(CSV_DIR / name, sep=";", encoding="latin-1",
                     on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.rstrip(":")
    return df

df_bad = load("BADisplay-LOAD.CSV")
df_idx = load("INDEX-LOAD.CSV")

# Build exam_key for merging
_SUBJ_FIELDS = ["Last Name", "First Name", "D.o.Birth"]
_EXAM_FIELDS = ["Exam Eye", "Exam Date", "Exam Time"]

for d in [df_bad, df_idx]:
    for c in _SUBJ_FIELDS + _EXAM_FIELDS:
        d[c] = d[c].astype(str).str.strip()
    d["patient_id"] = d[_SUBJ_FIELDS].apply(
        lambda r: hashlib.sha256("_".join(r).encode()).hexdigest()[:16], axis=1)
    d["exam_key"] = d["patient_id"] + "_" + d["Exam Eye"] + "_" + d["Exam Date"] + "_" + d["Exam Time"]
    d.drop_duplicates(subset="exam_key", keep="first", inplace=True)

# Merge clinical variables
clin_vars = ["ISV","IVA","KI","CKI","IHA","IHD","BAD D","K Max (Front)"]

bad_cols = ["exam_key", "BAD D"]
if "Pachy Min." in df_bad.columns:
    bad_cols.append("Pachy Min.")
df_bad_m = df_bad[df_bad["exam_key"].isin(assign["exam_key"])][bad_cols].copy()
df_bad_m["BAD D"] = pd.to_numeric(df_bad_m["BAD D"], errors="coerce")
if "Pachy Min." in df_bad_m.columns:
    df_bad_m["CCT"] = pd.to_numeric(df_bad_m["Pachy Min."], errors="coerce")

idx_cols = ["exam_key"] + [c for c in clin_vars if c in df_idx.columns and c != "BAD D"]
if "Thinnest Pachy" in df_idx.columns:
    idx_cols.append("Thinnest Pachy")
df_idx_m = df_idx[df_idx["exam_key"].isin(assign["exam_key"])][idx_cols].copy()
for c in idx_cols[1:]:
    df_idx_m[c] = pd.to_numeric(df_idx_m[c], errors="coerce")

df = assign.merge(df_bad_m, on="exam_key", how="left")
df = df.merge(df_idx_m, on="exam_key", how="left", suffixes=("","_idx"))

if "CCT" not in df.columns:
    if "Thinnest Pachy" in df.columns:
        df["CCT"] = df["Thinnest Pachy"]

all_vars = ["ISV","IVA","KI","CKI","IHA","IHD","BAD D","K Max (Front)","CCT"]
all_vars = [v for v in all_vars if v in df.columns]

for v in all_vars:
    df[v] = pd.to_numeric(df[v], errors="coerce")

print("=" * 80)
print("PAIRWISE PHENOTYPE SIMILARITY ANALYSIS")
print("=" * 80)

def pairwise_analysis(df, label_col, atlas_name, all_vars):
    """Statistical comparison for each phenotype pair."""
    K = df[label_col].nunique()
    labels = sorted(df[label_col].unique())

    print(f"\n{'='*70}")
    print(f"  {atlas_name} ATLAS (K={K})")
    print(f"{'='*70}")

    # 1) Per-phenotype summary
    print(f"\n  --- Phenotype Summary ---")
    for k in labels:
        mk = df[label_col] == k
        n = mk.sum()
        print(f"  P{k}: N={n} ({n/len(df)*100:.1f}%)")
        for v in all_vars:
            vals = df.loc[mk, v].dropna()
            if len(vals) > 0:
                print(f"    {v:20s}: median={vals.median():.2f}  IQR=[{vals.quantile(.25):.2f}-{vals.quantile(.75):.2f}]  mean={vals.mean():.2f}±{vals.std():.2f}")

    # 2) All pairwise comparisons
    print(f"\n  --- Pairwise Mann-Whitney U Tests ---")
    pair_results = []

    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            pi, pj = labels[i], labels[j]
            print(f"\n  *** P{pi} vs P{pj} ***")

            n_sig = 0
            n_nonsig = 0
            nonsig_vars = []
            sig_vars = []
            effect_sizes = []

            for v in all_vars:
                vi = df.loc[df[label_col]==pi, v].dropna().values
                vj = df.loc[df[label_col]==pj, v].dropna().values

                if len(vi) < 5 or len(vj) < 5:
                    continue

                # Mann-Whitney U
                u_stat, p_val = stats.mannwhitneyu(vi, vj, alternative='two-sided')

                # Effect size: rank-biserial correlation
                n1, n2 = len(vi), len(vj)
                r_rb = 1 - (2*u_stat) / (n1*n2)

                # Cohen's d
                pooled_sd = np.sqrt(((n1-1)*np.std(vi,ddof=1)**2 + (n2-1)*np.std(vj,ddof=1)**2) / (n1+n2-2))
                cohens_d = (np.mean(vi) - np.mean(vj)) / (pooled_sd + 1e-10)

                # Median difference
                med_diff = np.median(vi) - np.median(vj)

                # Overlap coefficient (percentage overlap of distributions)
                min_val = min(vi.min(), vj.min())
                max_val = max(vi.max(), vj.max())
                bins = np.linspace(min_val, max_val, 50)
                h1, _ = np.histogram(vi, bins=bins, density=True)
                h2, _ = np.histogram(vj, bins=bins, density=True)
                overlap = np.sum(np.minimum(h1, h2)) * (bins[1]-bins[0])

                is_sig = p_val < 0.001  # Bonferroni-like strict threshold
                if is_sig:
                    n_sig += 1
                    sig_vars.append(v)
                else:
                    n_nonsig += 1
                    nonsig_vars.append(v)

                effect_sizes.append(abs(cohens_d))

                sig_mark = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                effect_label = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"

                print(f"    {v:20s}: med_diff={med_diff:+8.2f}  Cohen's d={cohens_d:+6.3f} ({effect_label:10s})  overlap={overlap:.1%}  p={p_val:.2e} {sig_mark}")

            mean_effect = np.mean(effect_sizes) if effect_sizes else 0
            max_effect = np.max(effect_sizes) if effect_sizes else 0

            pair_results.append({
                "pair": f"P{pi}-P{pj}",
                "n_sig": n_sig,
                "n_nonsig": n_nonsig,
                "nonsig_vars": ", ".join(nonsig_vars) if nonsig_vars else "none",
                "mean_cohens_d": round(mean_effect, 3),
                "max_cohens_d": round(max_effect, 3),
                "interpretation": "Artificial split" if mean_effect < 0.3 else "Weak separation" if mean_effect < 0.5 else "Moderate separation" if mean_effect < 0.8 else "Strong separation"
            })

            print(f"\n    SUMMARY: {n_sig}/{n_sig+n_nonsig} variables significantly different")
            print(f"    Ortalama |Cohen's d| = {mean_effect:.3f}")
            print(f"    Maksimum |Cohen's d| = {max_effect:.3f}")
            print(f"    --> YORUM: {pair_results[-1]['interpretation']}")

    # 3) Summary table
    print(f"\n  === {atlas_name} ATLAS - PAIRWISE COMPARISON SUMMARY ===")
    pdf = pd.DataFrame(pair_results)
    print(pdf.to_string(index=False))

    return pdf

# FULL ATLAS analizi
full_results = pairwise_analysis(df, "pheno_full", "FULL", all_vars)

# HOA ATLAS analizi
hoa_results = pairwise_analysis(df, "pheno_hoa", "HOA", all_vars)

# Comparative results
print("\n" + "=" * 80)
print("  COMPARATIVE RESULT: FULL vs HOA")
print("=" * 80)

print("\n  FULL Atlas pairwise similarities (ranked by Cohen's d):")
for _, row in full_results.sort_values("mean_cohens_d").iterrows():
    print(f"    {row['pair']:10s}: mean |d| = {row['mean_cohens_d']:.3f}  -> {row['interpretation']}")

print("\n  HOA Atlas pairwise similarities (ranked by Cohen's d):")
for _, row in hoa_results.sort_values("mean_cohens_d").iterrows():
    print(f"    {row['pair']:10s}: mean |d| = {row['mean_cohens_d']:.3f}  -> {row['interpretation']}")

# Most similar pairs
full_most_similar = full_results.loc[full_results["mean_cohens_d"].idxmin()]
hoa_most_similar = hoa_results.loc[hoa_results["mean_cohens_d"].idxmin()]

print(f"\n  FULL Atlas most similar pair: {full_most_similar['pair']} (mean |d| = {full_most_similar['mean_cohens_d']:.3f})")
print(f"  HOA Atlas most similar pair:  {hoa_most_similar['pair']} (mean |d| = {hoa_most_similar['mean_cohens_d']:.3f})")

print("\n  RESULT:")
print("  " + "-"*60)
if full_most_similar['mean_cohens_d'] < 0.3:
    print(f"  FULL Atlas {full_most_similar['pair']} shows ARTIFICIAL SPLIT")
    print(f"  (mean effect size negligible-small: {full_most_similar['mean_cohens_d']:.3f})")
else:
    print(f"  FULL Atlas {full_most_similar['pair']} shows weak but present separation")
    print(f"  (mean effect size: {full_most_similar['mean_cohens_d']:.3f})")

if hoa_most_similar['mean_cohens_d'] < 0.3:
    print(f"  HOA Atlas {hoa_most_similar['pair']} shows ARTIFICIAL SPLIT")
    print(f"  (mean effect size negligible-small: {hoa_most_similar['mean_cohens_d']:.3f})")

# Save results
full_results.to_csv(TBL_DIR / "pairwise_full.csv", index=False)
hoa_results.to_csv(TBL_DIR / "pairwise_hoa.csv", index=False)
print(f"\n  Results saved: {TBL_DIR / 'pairwise_full.csv'}")
print(f"                       {TBL_DIR / 'pairwise_hoa.csv'}")
