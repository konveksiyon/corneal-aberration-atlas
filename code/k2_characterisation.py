#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K=2 Phenotype Characterisation
================================
Produces the full clinical profile of the K=2 clusters.
"""
import hashlib

import sys, warnings, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
CSV_DIR = BASE / "csv"
OUT_DIR = BASE / "output"; OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUT_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TBL_DIR = OUT_DIR / "tables"; TBL_DIR.mkdir(exist_ok=True)

EPS = 1e-10
N_INIT = 20
np.random.seed(42)

# ==============================================================
# DATA LOADING (same pipeline as v4)
# ==============================================================
print("=" * 70)
print("K=2 PHENOTYPE CHARACTERISATION")
print("=" * 70)

def load(name):
    df = pd.read_csv(CSV_DIR / name, sep=";", encoding="latin-1",
                     on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.rstrip(":")
    return df

df_z   = load("ZERNIKE-WFA.CSV")
df_bad = load("BADisplay-LOAD.CSV")
df_idx = load("INDEX-LOAD.CSV")

_SUBJ_FIELDS = ["Last Name", "First Name", "D.o.Birth"]
_EXAM_FIELDS = ["Exam Eye", "Exam Date", "Exam Time"]

def make_key(df):
    """Build hashed patient and exam identifiers from device-export fields."""
    for c in _SUBJ_FIELDS + _EXAM_FIELDS:
        df[c] = df[c].astype(str).str.strip()
    df["patient_id"] = df[_SUBJ_FIELDS].apply(
        lambda r: hashlib.sha256("_".join(r).encode()).hexdigest()[:16], axis=1)
    df["exam_key"] = df["patient_id"] + "_" + df["Exam Eye"] + "_" + df["Exam Date"] + "_" + df["Exam Time"]
    return df

for d in [df_z, df_bad, df_idx]:
    make_key(d)
    d.drop_duplicates(subset="exam_key", keep="first", inplace=True)

keys_3 = set(df_z["exam_key"]) & set(df_bad["exam_key"]) & set(df_idx["exam_key"])
df = df_z[df_z["exam_key"].isin(keys_3)].copy()
df["Error"] = pd.to_numeric(df["Error"], errors="coerce").fillna(99)
df = df[df["Error"] == 0].copy()

cf_cols_all = [c for c in df.columns if "(CF)" in c and c.startswith("Z ")]
def parse_nm(col):
    parts = col.replace("(CF)","").replace("(CB)","").replace("(Cornea)","").strip().split()
    return int(parts[1]), int(parts[2])

cf_info = [(c, *parse_nm(c)) for c in cf_cols_all]
full_cols = [c for c, n, m in cf_info if 2 <= n <= 6]
hoa_cols  = [c for c, n, m in cf_info if 3 <= n <= 6]

for c in full_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for rc in ["RMS (CF)", "RMS HOA (CF)", "RMS LOA (CF)"]:
    if rc in df.columns:
        df[rc] = pd.to_numeric(df[rc], errors="coerce")

# BAD-D merge
bad_merge_cols = ["exam_key", "BAD D"]
if "Pachy Min." in df_bad.columns:
    bad_merge_cols.append("Pachy Min.")
df_bad_m = df_bad[df_bad["exam_key"].isin(df["exam_key"])][bad_merge_cols].copy()
df_bad_m["BAD D"] = pd.to_numeric(df_bad_m["BAD D"], errors="coerce")
if "Pachy Min." in df_bad_m.columns:
    df_bad_m["CCT"] = pd.to_numeric(df_bad_m["Pachy Min."], errors="coerce")
df = df.merge(df_bad_m, on="exam_key", how="left")

# INDEX merge
idx_merge_cols = ["exam_key"]
for c in ["ISV","IVA","KI","CKI","IHA","IHD","K Max (Front)","Thinnest Pachy"]:
    if c in df_idx.columns:
        idx_merge_cols.append(c)
df_idx_m = df_idx[df_idx["exam_key"].isin(df["exam_key"])][idx_merge_cols].copy()
for c in idx_merge_cols[1:]:
    df_idx_m[c] = pd.to_numeric(df_idx_m[c], errors="coerce")
df = df.merge(df_idx_m, on="exam_key", how="left", suffixes=("","_idx"))

if "CCT" not in df.columns:
    if "Thinnest Pachy" in df.columns: df["CCT"] = df["Thinnest Pachy"]
    elif "Thinnest Pachy_idx" in df.columns: df["CCT"] = df["Thinnest Pachy_idx"]

# Filters
pachy_col = "CCT"
zernike_ok = df[full_cols].notna().all(axis=1)
bad_ok = df["BAD D"].notna()
cct_ok = df[pachy_col].notna() & (df[pachy_col] > 100)
df = df[zernike_ok & bad_ok & cct_ok].copy()
df = df[df["Exam Eye"].str.contains("Right", na=False)].copy()
df["Exam Date Parsed"] = pd.to_datetime(df["Exam Date"], format="mixed", dayfirst=False, errors="coerce")
df = df.sort_values(["patient_id", "Exam Date Parsed"])
df = df.drop_duplicates(subset="patient_id", keep="first").copy()
df["dob"] = pd.to_datetime(df[_SUBJ_FIELDS[2]], format="mixed", dayfirst=False, errors="coerce")
df["age"] = (df["Exam Date Parsed"] - df["dob"]).dt.days / 365.25
df = df[(df["age"] >= 18) & (df["age"] <= 120)].copy()

N = len(df)
print(f"\nFinal cohort: {N} eyes")

# ==============================================================
# BLOCK NORMALISATION + K=2 CLUSTERING
# ==============================================================
def get_block(info, order):
    return [c for c, n, m in info if n == order]

def block_normalize(data, info, orders):
    blocks = []
    for o in orders:
        cols = get_block(info, o)
        if not cols: continue
        b = data[cols].values.astype(float)
        norms = np.linalg.norm(b, axis=1, keepdims=True) + EPS
        blocks.append(b / norms)
    z = np.hstack(blocks)
    g = np.linalg.norm(z, axis=1, keepdims=True) + EPS
    return z / g

u_full = block_normalize(df, cf_info, [2,3,4,5,6])
u_hoa  = block_normalize(df, cf_info, [3,4,5,6])
raw_full = df[full_cols].values.astype(float)
rms_total = df["RMS (CF)"].values
rms_hoa   = df["RMS HOA (CF)"].values

def sph_kmeans(X, K, n_init=N_INIT, max_iter=300, rs=42):
    best_in = np.inf; best_l = best_c = None
    for i in range(n_init):
        rng = np.random.RandomState(rs + i)
        c = X[rng.choice(len(X), K, replace=False)].copy()
        c /= (np.linalg.norm(c, axis=1, keepdims=True) + EPS)
        for _ in range(max_iter):
            l = np.argmax(X @ c.T, axis=1)
            nc = np.zeros_like(c)
            for k in range(K):
                m = X[l == k]
                if len(m) > 0: nc[k] = m.mean(0)
            nc /= (np.linalg.norm(nc, axis=1, keepdims=True) + EPS)
            if np.allclose(c, nc, atol=1e-6): break
            c = nc
        sims = np.array([X[j] @ c[l[j]] for j in range(len(X))])
        iner = np.sum(1 - sims)
        if iner < best_in:
            best_in = iner; best_l = l; best_c = c
    return best_l, best_c

print("\n[1] K=2 clustering (FULL)...")
labels_f2, centers_f2 = sph_kmeans(u_full, 2, n_init=N_INIT)

print("[2] K=2 clustering (HOA)...")
labels_h2, centers_h2 = sph_kmeans(u_hoa, 2, n_init=N_INIT)

# Sort clusters by severity (lower BAD-D first)
for labels, name in [(labels_f2, "FULL"), (labels_h2, "HOA")]:
    bad_means = [df.loc[labels==k, "BAD D"].median() for k in range(2)]
    if bad_means[0] > bad_means[1]:
        # Swap labels
        labels_new = labels.copy()
        labels_new[labels==0] = 1
        labels_new[labels==1] = 0
        if name == "FULL":
            labels_f2 = labels_new
            centers_f2 = centers_f2[[1,0]]
        else:
            labels_h2 = labels_new
            centers_h2 = centers_h2[[1,0]]

# ==============================================================
# DETAILED CHARACTERISATION
# ==============================================================
clin_vars = ["ISV","IVA","KI","CKI","IHA","IHD","BAD D","K Max (Front)","CCT"]
clin_vars = [v for v in clin_vars if v in df.columns]
for v in clin_vars:
    df[v] = pd.to_numeric(df[v], errors="coerce")

def characterise_k2(labels, rms_t, rms_h, atlas_name):
    print(f"\n{'='*70}")
    print(f"  K=2 {atlas_name} ATLAS — DETAILED CHARACTERISATION")
    print(f"{'='*70}")

    for k in range(2):
        mk = labels == k
        n = mk.sum()
        pct = n/len(labels)*100

        rt = rms_t[mk]; rt = rt[~np.isnan(rt)]
        rh = rms_h[mk]; rh = rh[~np.isnan(rh)]

        # BAD-D classification
        bad_vals = df.loc[mk, "BAD D"].dropna()
        n_normal = (bad_vals < 1.6).sum()
        n_suspect = ((bad_vals >= 1.6) & (bad_vals <= 2.6)).sum()
        n_abnormal = (bad_vals > 2.6).sum()

        print(f"\n  *** PHENOTYPE {k} (n={n}, {pct:.1f}%) ***")
        print(f"  {'─'*50}")

        # RMS
        print(f"  Total RMS:  {np.median(rt):.2f} [{np.percentile(rt,25):.2f}-{np.percentile(rt,75):.2f}] μm  (mean {np.mean(rt):.2f}±{np.std(rt):.2f})")
        print(f"  HOA RMS:    {np.median(rh):.2f} [{np.percentile(rh,25):.2f}-{np.percentile(rh,75):.2f}] μm  (mean {np.mean(rh):.2f}±{np.std(rh):.2f})")

        # Clinical variables
        for v in clin_vars:
            vals = df.loc[mk, v].dropna()
            print(f"  {v:20s}: {vals.median():.2f} [{vals.quantile(.25):.2f}-{vals.quantile(.75):.2f}]  (mean {vals.mean():.2f}±{vals.std():.2f})")

        # BAD-D categories
        print(f"\n  BAD-D Classification:")
        print(f"    Normal  (<1.6):  {n_normal:4d} ({n_normal/len(bad_vals)*100:.1f}%)")
        print(f"    Suspect (1.6-2.6): {n_suspect:4d} ({n_suspect/len(bad_vals)*100:.1f}%)")
        print(f"    Abnormal(>2.6):  {n_abnormal:4d} ({n_abnormal/len(bad_vals)*100:.1f}%)")

        # Age
        ages = df.loc[mk, "age"].dropna()
        print(f"\n  Yas: {ages.median():.1f} [{ages.quantile(.25):.1f}-{ages.quantile(.75):.1f}]")

    # Mann-Whitney between the two
    print(f"\n  {'─'*50}")
    print(f"  P0 vs P1 — Mann-Whitney U + Cohen's d:")
    for v in ["RMS_total", "RMS_HOA"] + clin_vars + ["age"]:
        if v == "RMS_total":
            v0 = rms_t[labels==0]; v0 = v0[~np.isnan(v0)]
            v1 = rms_t[labels==1]; v1 = v1[~np.isnan(v1)]
            vname = "Total RMS"
        elif v == "RMS_HOA":
            v0 = rms_h[labels==0]; v0 = v0[~np.isnan(v0)]
            v1 = rms_h[labels==1]; v1 = v1[~np.isnan(v1)]
            vname = "HOA RMS"
        elif v == "age":
            v0 = df.loc[labels==0, "age"].dropna().values
            v1 = df.loc[labels==1, "age"].dropna().values
            vname = "Age"
        else:
            v0 = df.loc[labels==0, v].dropna().values
            v1 = df.loc[labels==1, v].dropna().values
            vname = v

        u, p = stats.mannwhitneyu(v0, v1, alternative='two-sided')
        pooled_sd = np.sqrt(((len(v0)-1)*np.std(v0,ddof=1)**2 + (len(v1)-1)*np.std(v1,ddof=1)**2) / (len(v0)+len(v1)-2))
        d = (np.mean(v0) - np.mean(v1)) / (pooled_sd + 1e-10)

        sig = "***" if p < 0.001 else "ns"
        eff = "negligible" if abs(d)<0.2 else "small" if abs(d)<0.5 else "medium" if abs(d)<0.8 else "large"
        print(f"    {vname:20s}: d={d:+.3f} ({eff:10s})  p={p:.2e} {sig}")

characterise_k2(labels_f2, rms_total, rms_hoa, "FULL")
characterise_k2(labels_h2, rms_total, rms_hoa, "HOA")

# ==============================================================
# K=2 ATLAS FIGURE
# ==============================================================
print("\n[3] K=2 Atlas figure generation...")

def zern_R(n, m, rho):
    ma = abs(m); R = np.zeros_like(rho, dtype=float)
    for s in range(int((n-ma)/2)+1):
        num = ((-1)**s)*math.factorial(n-s)
        den = math.factorial(s)*math.factorial(int((n+ma)/2)-s)*math.factorial(int((n-ma)/2)-s)
        R += (num/den)*rho**(n-2*s)
    return R

def zern_Z(n, m, rho, theta):
    Rv = zern_R(n, abs(m), rho)
    return Rv*np.cos(m*theta) if m>=0 else Rv*np.sin(abs(m)*theta)

def recon_wf(coeffs, cinfo, g=100):
    x = np.linspace(-1,1,g); y = np.linspace(-1,1,g)
    X, Y = np.meshgrid(x, y); rho = np.sqrt(X**2+Y**2); th = np.arctan2(Y, X)
    mk = rho<=1; wf = np.zeros_like(rho)
    for coeff, (_,n,m) in zip(coeffs, cinfo):
        wf += coeff * zern_Z(n, m, rho, th)
    wf[~mk] = np.nan
    return wf

def plot_wf(ax, wf, title=""):
    v = np.nanmax(np.abs(wf)); v = max(v, .01)
    im = ax.imshow(wf, extent=[-1,1,-1,1], origin="lower", cmap="RdBu_r",
              vmin=-v, vmax=v, interpolation="bilinear")
    th = np.linspace(0,2*np.pi,100)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=1)
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    return im

# Full atlas K=2
ci_full = [(col,n,m) for col,n,m in cf_info if 2<=n<=6]
cn_full = [col for col,n,m in ci_full]

fig = plt.figure(figsize=(12, 14))
gs = GridSpec(4, 2, figure=fig, hspace=.35, wspace=.3)

for k in range(2):
    mk = labels_f2==k; nk = mk.sum(); pct = nk/len(labels_f2)*100
    bad_med = df.loc[mk, "BAD D"].median()

    # BAD-D categories
    bad_vals = df.loc[mk, "BAD D"].dropna()
    n_norm = (bad_vals < 1.6).sum()
    n_abn = (bad_vals > 2.6).sum()

    cs = centers_f2[k] / (np.linalg.norm(centers_f2[k])+EPS)
    wf = recon_wf(cs, ci_full)
    ax = fig.add_subplot(gs[0,k])
    plot_wf(ax, wf, f"P{k}: Mean Centroid\n(n={nk}, {pct:.1f}%)\nBAD-D={bad_med:.2f}")

    # Medoid
    sims = u_full[mk] @ centers_f2[k]
    med_l = np.argmax(sims)
    med_g = np.where(mk)[0][med_l]
    raw_m = df.iloc[med_g][cn_full].values.astype(float)
    wf_m = recon_wf(raw_m, ci_full)
    ax2 = fig.add_subplot(gs[1,k])
    plot_wf(ax2, wf_m, f"Medoid\nBAD-D={df.iloc[med_g]['BAD D']:.2f}")

    # Low and high severity exemplars
    rk = rms_total[mk]; vr = ~np.isnan(rk)
    if vr.sum() > 4:
        si = np.argsort(rk[vr])
        ll = np.where(vr)[0][si[len(si)//10]]
        hh = np.where(vr)[0][si[-len(si)//10-1]]
        lg = np.where(mk)[0][ll]; hg = np.where(mk)[0][hh]

        wfl = recon_wf(df.iloc[lg][cn_full].values.astype(float), ci_full)
        wfh = recon_wf(df.iloc[hg][cn_full].values.astype(float), ci_full)
        ax3 = fig.add_subplot(gs[2,k])
        plot_wf(ax3, wfl, f"Low RMS ({rk[ll]:.1f}μm)\nBAD-D={df.iloc[lg]['BAD D']:.2f}")
        ax4 = fig.add_subplot(gs[3,k])
        plot_wf(ax4, wfh, f"High RMS ({rk[hh]:.1f}μm)\nBAD-D={df.iloc[hg]['BAD D']:.2f}")

for y, txt in [(.88,"A. Mean Centroid"),(.65,"B. Medoid"),(.42,"C. Low Severity"),(.19,"D. High Severity")]:
    fig.text(.02, y, txt, fontsize=10, fontweight="bold", va="center", rotation=90)
fig.suptitle("Full Atlas K=2: Normal-Mild (P0) vs Ectatic-Severe (P1)", fontsize=14, fontweight="bold", y=.98)
fig.savefig(FIG_DIR/"fig_K2_full_atlas.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: fig_K2_full_atlas.png")

# ==============================================================
# K=2 CLINICAL BOX PLOTS
# ==============================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
cols_k2 = plt.cm.Set1([0.0, 0.9])  # Blue vs Red

for i, v in enumerate(clin_vars):
    r, c = divmod(i, 3)
    ax = axes[r, c]
    d = [df.loc[labels_f2==k, v].dropna() for k in range(2)]

    bp = ax.boxplot(d, labels=["P0\n(Normal-Mild)", "P1\n(Ectatic-Severe)"],
                    patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], cols_k2):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Mann-Whitney
    v0 = d[0].values; v1 = d[1].values
    u, p = stats.mannwhitneyu(v0, v1, alternative='two-sided')
    pooled_sd = np.sqrt(((len(v0)-1)*np.std(v0,ddof=1)**2 + (len(v1)-1)*np.std(v1,ddof=1)**2)/(len(v0)+len(v1)-2))
    cd = abs((np.mean(v0)-np.mean(v1))/(pooled_sd+1e-10))

    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.set_title(f"{v}\n{p_str}  |d| = {cd:.2f}", fontsize=10)
    ax.grid(True, alpha=0.2)

fig.suptitle("K=2 Full Atlas: Clinical Variable Distributions", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_K2_clinical_boxplots.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: fig_K2_clinical_boxplots.png")

# ==============================================================
# K=2 BAD-D DISTRIBUTION OVERLAP
# ==============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BAD-D histogram
for k, (col, label) in enumerate(zip(cols_k2, ["P0 (Normal-Mild)", "P1 (Ectatic-Severe)"])):
    vals = df.loc[labels_f2==k, "BAD D"].dropna()
    axes[0].hist(vals, bins=50, alpha=0.6, color=col, label=label, density=True)
axes[0].axvline(1.6, color="orange", ls="--", lw=2, label="Normal/Suspect (1.6)")
axes[0].axvline(2.6, color="red", ls="--", lw=2, label="Suspect/Abnormal (2.6)")
axes[0].set_xlabel("BAD-D"); axes[0].set_ylabel("Density")
axes[0].set_title("BAD-D Distribution by K=2 Phenotype")
axes[0].legend(fontsize=9)

# Total RMS histogram
for k, (col, label) in enumerate(zip(cols_k2, ["P0 (Normal-Mild)", "P1 (Ectatic-Severe)"])):
    vals = rms_total[labels_f2==k]
    vals = vals[~np.isnan(vals)]
    axes[1].hist(vals, bins=50, alpha=0.6, color=col, label=label, density=True)
axes[1].set_xlabel("Total RMS (μm)"); axes[1].set_ylabel("Density")
axes[1].set_title("Total RMS Distribution by K=2 Phenotype")
axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig(FIG_DIR/"fig_K2_distributions.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: fig_K2_distributions.png")

# ==============================================================
# SAVE K=2 SUMMARY TABLE
# ==============================================================
rows = []
for k in range(2):
    mk = labels_f2 == k
    rt = rms_total[mk]; rt = rt[~np.isnan(rt)]
    rh = rms_hoa[mk]; rh = rh[~np.isnan(rh)]
    bad = df.loc[mk, "BAD D"].dropna()

    rows.append({
        "Phenotype": k,
        "N": mk.sum(),
        "Pct": f"{mk.sum()/N*100:.1f}%",
        "Label": "Normal-Mild" if k==0 else "Ectatic-Severe",
        "Total_RMS_median": f"{np.median(rt):.2f}",
        "Total_RMS_IQR": f"[{np.percentile(rt,25):.2f}-{np.percentile(rt,75):.2f}]",
        "HOA_RMS_median": f"{np.median(rh):.2f}",
        "HOA_RMS_IQR": f"[{np.percentile(rh,25):.2f}-{np.percentile(rh,75):.2f}]",
        "BAD_D_median": f"{bad.median():.2f}",
        "BAD_D_IQR": f"[{bad.quantile(.25):.2f}-{bad.quantile(.75):.2f}]",
        "Kmax_median": f"{df.loc[mk,'K Max (Front)'].median():.2f}",
        "CCT_median": f"{df.loc[mk,'CCT'].median():.0f}",
        "Pct_Normal": f"{(bad<1.6).sum()/(~bad.isna()).sum()*100:.1f}%",
        "Pct_Suspect": f"{((bad>=1.6)&(bad<=2.6)).sum()/(~bad.isna()).sum()*100:.1f}%",
        "Pct_Abnormal": f"{(bad>2.6).sum()/(~bad.isna()).sum()*100:.1f}%",
    })

df_k2 = pd.DataFrame(rows)
df_k2.to_csv(TBL_DIR / "k2_phenotype_summary.csv", index=False)
print(f"\n  K=2 summary saved: {TBL_DIR / 'k2_phenotype_summary.csv'}")
print(df_k2.to_string(index=False))

print("\n\nDONE.")
