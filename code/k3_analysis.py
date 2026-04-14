#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K=3 Analysis: Is the third group subclinical keratoconus?
==========================================================
"""
import hashlib
import sys, warnings, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
CSV_DIR = BASE / "csv"
FIG_DIR = BASE / "output" / "figures"; FIG_DIR.mkdir(exist_ok=True)
TBL_DIR = BASE / "output" / "tables"; TBL_DIR.mkdir(exist_ok=True)
EPS = 1e-10; N_INIT = 20
np.random.seed(42)

# ── DATA LOADING (same pipeline) ──
def load(name):
    df = pd.read_csv(CSV_DIR / name, sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.rstrip(":")
    return df

df_z = load("ZERNIKE-WFA.CSV"); df_bad = load("BADisplay-LOAD.CSV"); df_idx = load("INDEX-LOAD.CSV")

_SUBJ_FIELDS = ["Last Name", "First Name", "D.o.Birth"]
_EXAM_FIELDS = ["Exam Eye", "Exam Date", "Exam Time"]

def make_key(df):
    """Build hashed patient and exam identifiers from device-export fields."""
    for c in _SUBJ_FIELDS + _EXAM_FIELDS:
        df[c] = df[c].astype(str).str.strip()
    df["patient_id"] = df[_SUBJ_FIELDS].apply(
        lambda r: hashlib.sha256("_".join(r).encode()).hexdigest()[:16], axis=1)
    df["exam_key"] = df["patient_id"]+"_"+df["Exam Eye"]+"_"+df["Exam Date"]+"_"+df["Exam Time"]
    return df

for d in [df_z, df_bad, df_idx]:
    make_key(d); d.drop_duplicates(subset="exam_key", keep="first", inplace=True)

keys_3 = set(df_z["exam_key"]) & set(df_bad["exam_key"]) & set(df_idx["exam_key"])
df = df_z[df_z["exam_key"].isin(keys_3)].copy()
df["Error"] = pd.to_numeric(df["Error"], errors="coerce").fillna(99)
df = df[df["Error"]==0].copy()

cf_cols_all = [c for c in df.columns if "(CF)" in c and c.startswith("Z ")]
def parse_nm(col):
    parts = col.replace("(CF)","").replace("(CB)","").replace("(Cornea)","").strip().split()
    return int(parts[1]), int(parts[2])
cf_info = [(c, *parse_nm(c)) for c in cf_cols_all]
full_cols = [c for c,n,m in cf_info if 2<=n<=6]
hoa_cols = [c for c,n,m in cf_info if 3<=n<=6]

for c in full_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
for rc in ["RMS (CF)","RMS HOA (CF)","RMS LOA (CF)"]:
    if rc in df.columns: df[rc] = pd.to_numeric(df[rc], errors="coerce")

bad_merge = ["exam_key","BAD D"]
if "Pachy Min." in df_bad.columns: bad_merge.append("Pachy Min.")
df_bad_m = df_bad[df_bad["exam_key"].isin(df["exam_key"])][bad_merge].copy()
df_bad_m["BAD D"] = pd.to_numeric(df_bad_m["BAD D"], errors="coerce")
if "Pachy Min." in df_bad_m.columns: df_bad_m["CCT"] = pd.to_numeric(df_bad_m["Pachy Min."], errors="coerce")
df = df.merge(df_bad_m, on="exam_key", how="left")

idx_cols = ["exam_key"]+[c for c in ["ISV","IVA","KI","CKI","IHA","IHD","K Max (Front)","Thinnest Pachy"] if c in df_idx.columns]
df_idx_m = df_idx[df_idx["exam_key"].isin(df["exam_key"])][idx_cols].copy()
for c in idx_cols[1:]: df_idx_m[c] = pd.to_numeric(df_idx_m[c], errors="coerce")
df = df.merge(df_idx_m, on="exam_key", how="left", suffixes=("","_idx"))
if "CCT" not in df.columns:
    if "Thinnest Pachy" in df.columns: df["CCT"] = df["Thinnest Pachy"]

zernike_ok = df[full_cols].notna().all(axis=1)
bad_ok = df["BAD D"].notna()
cct_ok = df["CCT"].notna() & (df["CCT"]>100)
df = df[zernike_ok & bad_ok & cct_ok].copy()
df = df[df["Exam Eye"].str.contains("Right", na=False)].copy()
df["Exam Date Parsed"] = pd.to_datetime(df["Exam Date"], format="mixed", dayfirst=False, errors="coerce")
df = df.sort_values(["patient_id","Exam Date Parsed"])
df = df.drop_duplicates(subset="patient_id", keep="first").copy()
df["dob"] = pd.to_datetime(df[_SUBJ_FIELDS[2]], format="mixed", dayfirst=False, errors="coerce")
df["age"] = (df["Exam Date Parsed"]-df["dob"]).dt.days/365.25
df = df[(df["age"]>=18)&(df["age"]<=120)].copy()
N = len(df)
print(f"Cohort: {N}")

# ── BLOCK NORM + CLUSTERING ──
def get_block(info, order): return [c for c,n,m in info if n==order]
def block_normalize(data, info, orders):
    blocks = []
    for o in orders:
        cols = get_block(info, o)
        if not cols: continue
        b = data[cols].values.astype(float)
        norms = np.linalg.norm(b, axis=1, keepdims=True)+EPS
        blocks.append(b/norms)
    z = np.hstack(blocks)
    g = np.linalg.norm(z, axis=1, keepdims=True)+EPS
    return z/g

u_full = block_normalize(df, cf_info, [2,3,4,5,6])
rms_total = df["RMS (CF)"].values
rms_hoa = df["RMS HOA (CF)"].values

def sph_kmeans(X, K, n_init=N_INIT, max_iter=300, rs=42):
    best_in = np.inf; best_l = best_c = None
    for i in range(n_init):
        rng = np.random.RandomState(rs+i)
        c = X[rng.choice(len(X), K, replace=False)].copy()
        c /= (np.linalg.norm(c, axis=1, keepdims=True)+EPS)
        for _ in range(max_iter):
            l = np.argmax(X@c.T, axis=1)
            nc = np.zeros_like(c)
            for k in range(K):
                m = X[l==k]
                if len(m)>0: nc[k] = m.mean(0)
            nc /= (np.linalg.norm(nc, axis=1, keepdims=True)+EPS)
            if np.allclose(c, nc, atol=1e-6): break
            c = nc
        sims = np.array([X[j]@c[l[j]] for j in range(len(X))])
        iner = np.sum(1-sims)
        if iner < best_in: best_in=iner; best_l=l; best_c=c
    return best_l, best_c

from sklearn.metrics import adjusted_rand_score
def boot_stability(X, K, nb=100, rs=42):
    ref_l, _ = sph_kmeans(X, K, n_init=10, rs=rs)
    rng = np.random.RandomState(rs); aris = []
    for b in range(nb):
        idx = rng.choice(len(X), len(X), replace=True)
        bl, _ = sph_kmeans(X[idx], K, n_init=5, rs=rs+b+1)
        aris.append(adjusted_rand_score(ref_l[idx], bl))
    return np.mean(aris), np.std(aris)

# ── K=2, K=3, K=4 COMPARISON ──
print("\n" + "="*70)
print("K=2, K=3, K=4 COMPARISON")
print("="*70)

clin_vars = [v for v in ["ISV","IVA","KI","CKI","IHA","IHD","BAD D","K Max (Front)","CCT"] if v in df.columns]
for v in clin_vars: df[v] = pd.to_numeric(df[v], errors="coerce")

results = {}
for K in [2, 3, 4]:
    print(f"\n--- K={K} ---")
    labels, centers = sph_kmeans(u_full, K, n_init=N_INIT)
    sil = silhouette_score(u_full, labels, metric="cosine")
    stab_m, stab_s = boot_stability(u_full, K, nb=100)

    # Sort by BAD-D median
    bad_meds = [df.loc[labels==k, "BAD D"].median() for k in range(K)]
    order = np.argsort(bad_meds)
    new_labels = np.zeros_like(labels)
    new_centers = np.zeros_like(centers)
    for new_k, old_k in enumerate(order):
        new_labels[labels==old_k] = new_k
        new_centers[new_k] = centers[old_k]
    labels = new_labels; centers = new_centers

    conf = np.array([u_full[i]@centers[labels[i]] for i in range(N)])
    results[K] = {"labels": labels, "centers": centers, "sil": sil, "stab": stab_m, "stab_sd": stab_s, "conf": conf}

    print(f"  Silhouette: {sil:.4f}, Stability: {stab_m:.4f}±{stab_s:.4f}")
    for k in range(K):
        mk = labels==k; n=mk.sum()
        bad = df.loc[mk,"BAD D"].dropna()
        rt = rms_total[mk]; rt=rt[~np.isnan(rt)]
        rh = rms_hoa[mk]; rh=rh[~np.isnan(rh)]
        n_norm = (bad<1.6).sum(); n_susp = ((bad>=1.6)&(bad<=2.6)).sum(); n_abn = (bad>2.6).sum()
        print(f"  P{k}: n={n} ({n/N*100:.1f}%) | BAD-D={bad.median():.2f} | TotalRMS={np.median(rt):.2f} | HOARMS={np.median(rh):.2f} | Kmax={df.loc[mk,'K Max (Front)'].median():.2f} | CCT={df.loc[mk,'CCT'].median():.0f} | Norm={n_norm/len(bad)*100:.0f}% Susp={n_susp/len(bad)*100:.0f}% Abn={n_abn/len(bad)*100:.0f}%")

# ── K=3 PAIRWISE ANALYSIS ──
print("\n" + "="*70)
print("K=3 PAIRWISE ANALYSIS")
print("="*70)

labels3 = results[3]["labels"]
centers3 = results[3]["centers"]

pair_results_k3 = []
for i in range(3):
    for j in range(i+1, 3):
        print(f"\n  P{i} vs P{j}:")
        n_sig = 0; n_ns = 0; ds = []; ns_vars = []
        for v in clin_vars:
            vi = df.loc[labels3==i, v].dropna().values
            vj = df.loc[labels3==j, v].dropna().values
            if len(vi)<5 or len(vj)<5: continue
            u, p = stats.mannwhitneyu(vi, vj, alternative='two-sided')
            pooled = np.sqrt(((len(vi)-1)*np.std(vi,ddof=1)**2+(len(vj)-1)*np.std(vj,ddof=1)**2)/(len(vi)+len(vj)-2))
            d = (np.mean(vi)-np.mean(vj))/(pooled+1e-10)
            ds.append(abs(d))
            sig = "***" if p<0.001 else "ns"
            eff = "negligible" if abs(d)<0.2 else "small" if abs(d)<0.5 else "medium" if abs(d)<0.8 else "large"
            if p<0.001: n_sig+=1
            else: n_ns+=1; ns_vars.append(v)
            print(f"    {v:20s}: d={d:+.3f} ({eff:10s}) p={p:.2e} {sig}")
        mean_d = np.mean(ds)
        interp = "Artificial split" if mean_d<0.3 else "Weak separation" if mean_d<0.5 else "Moderate separation" if mean_d<0.8 else "Strong separation"
        pair_results_k3.append({"pair": f"P{i}-P{j}", "mean_d": mean_d, "n_sig": n_sig, "n_ns": n_ns,
                                "ns_vars": ", ".join(ns_vars) if ns_vars else "none", "interp": interp})
        print(f"    ==> Mean |d|={mean_d:.3f}, Sig={n_sig}/9, NS={n_ns}/9 -> {interp}")

print("\nK=3 OZET:")
for r in pair_results_k3:
    print(f"  {r['pair']}: mean|d|={r['mean_d']:.3f} ({r['interp']}), NS vars: {r['ns_vars']}")

# ── K=3 ATLAS FIGURE ──
print("\n[FIGURE] K=3 Atlas...")
ci_full = [(col,n,m) for col,n,m in cf_info if 2<=n<=6]
cn_full = [col for col,n,m in ci_full]

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
    ax.imshow(wf, extent=[-1,1,-1,1], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="bilinear")
    th = np.linspace(0,2*np.pi,100); ax.plot(np.cos(th), np.sin(th), "k-", lw=1)
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

fig = plt.figure(figsize=(15, 14))
gs = GridSpec(4, 3, figure=fig, hspace=.35, wspace=.3)
group_names = ["Normal-Mild", "Intermediate\n(Suspect/Transitional)", "Ectatic-Severe"]

for k in range(3):
    mk = labels3==k; nk = mk.sum(); pct = nk/N*100
    bad_med = df.loc[mk,"BAD D"].median()
    rms_med = np.nanmedian(rms_total[mk])
    hoa_med = np.nanmedian(rms_hoa[mk])
    bad_vals = df.loc[mk,"BAD D"].dropna()
    n_norm = (bad_vals<1.6).sum(); n_susp = ((bad_vals>=1.6)&(bad_vals<=2.6)).sum(); n_abn = (bad_vals>2.6).sum()

    cs = centers3[k]/(np.linalg.norm(centers3[k])+EPS)
    wf = recon_wf(cs, ci_full)
    ax = fig.add_subplot(gs[0,k])
    plot_wf(ax, wf, f"P{k}: {group_names[k]}\n(n={nk}, {pct:.1f}%)\nBAD-D={bad_med:.2f}, RMS={rms_med:.2f}")

    sims = u_full[mk]@centers3[k]; med_l = np.argmax(sims); med_g = np.where(mk)[0][med_l]
    raw_m = df.iloc[med_g][cn_full].values.astype(float)
    ax2 = fig.add_subplot(gs[1,k])
    plot_wf(ax2, recon_wf(raw_m, ci_full), f"Medoid\nBAD-D={df.iloc[med_g]['BAD D']:.2f}")

    rk = rms_total[mk]; vr = ~np.isnan(rk)
    if vr.sum()>4:
        si = np.argsort(rk[vr])
        ll = np.where(vr)[0][si[len(si)//10]]; hh = np.where(vr)[0][si[-len(si)//10-1]]
        lg = np.where(mk)[0][ll]; hg = np.where(mk)[0][hh]
        ax3 = fig.add_subplot(gs[2,k])
        plot_wf(ax3, recon_wf(df.iloc[lg][cn_full].values.astype(float), ci_full),
                f"Low ({rk[ll]:.1f}um)\nBAD-D={df.iloc[lg]['BAD D']:.2f}")
        ax4 = fig.add_subplot(gs[3,k])
        plot_wf(ax4, recon_wf(df.iloc[hg][cn_full].values.astype(float), ci_full),
                f"High ({rk[hh]:.1f}um)\nBAD-D={df.iloc[hg]['BAD D']:.2f}")

for y, txt in [(.88,"A. Centroid"),(.65,"B. Medoid"),(.42,"C. Low RMS"),(.19,"D. High RMS")]:
    fig.text(.02, y, txt, fontsize=10, fontweight="bold", va="center", rotation=90)
fig.suptitle("Full Atlas K=3: Normal-Mild / Intermediate / Ectatic-Severe", fontsize=14, fontweight="bold", y=.98)
fig.savefig(FIG_DIR/"fig_K3_full_atlas.png", dpi=200, bbox_inches="tight"); plt.close()
print("  -> fig_K3_full_atlas.png")

# ── K=3 CLINICAL BOX PLOTS ──
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
cols3 = ["#2196F3", "#FF9800", "#F44336"]
for i, v in enumerate(clin_vars):
    r, c = divmod(i, 3); ax = axes[r,c]
    d = [df.loc[labels3==k, v].dropna() for k in range(3)]
    bp = ax.boxplot(d, labels=["P0\nNormal", "P1\nIntermediate", "P2\nEctatic"],
                    patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], cols3):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    _, kw_p = stats.kruskal(*[x.values for x in d if len(x)>1])
    ax.set_title(f"{v}\nKW p={'<0.001' if kw_p<0.001 else f'{kw_p:.3f}'}", fontsize=10)
    ax.grid(True, alpha=0.2)
fig.suptitle("K=3 Clinical Variables", fontsize=14, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR/"fig_K3_clinical_boxplots.png", dpi=200, bbox_inches="tight"); plt.close()
print("  -> fig_K3_clinical_boxplots.png")

# ── K=3 BAD-D DISTRIBUTION ──
fig, ax = plt.subplots(figsize=(10, 5))
for k, (col, label) in enumerate(zip(cols3, ["P0 Normal-Mild","P1 Intermediate","P2 Ectatic-Severe"])):
    vals = df.loc[labels3==k,"BAD D"].dropna()
    ax.hist(vals, bins=60, alpha=0.5, color=col, label=f"{label} (n={(labels3==k).sum()})", density=True)
ax.axvline(1.6, color="orange", ls="--", lw=2, label="Normal/Suspect (1.6)")
ax.axvline(2.6, color="red", ls="--", lw=2, label="Suspect/Abnormal (2.6)")
ax.set_xlabel("BAD-D"); ax.set_ylabel("Density"); ax.legend(fontsize=9)
ax.set_title("BAD-D Distribution by K=3 Phenotype")
fig.tight_layout(); fig.savefig(FIG_DIR/"fig_K3_bad_distribution.png", dpi=200); plt.close()
print("  -> fig_K3_bad_distribution.png")

# ── COMPREHENSIVE K COMPARISON TABLE ──
print("\n" + "="*70)
print("K=2 vs K=3 vs K=4: FINAL COMPARISON")
print("="*70)

# Centroid similarity for each K
for K in [2,3,4]:
    c = results[K]["centers"]
    sims = []
    for i in range(K):
        for j in range(i+1,K):
            s = np.dot(c[i],c[j])/(np.linalg.norm(c[i])*np.linalg.norm(c[j])+EPS)
            sims.append(s)
    print(f"\n  K={K}: max_centroid_sim={max(sims) if sims else 0:.4f}, mean={np.mean(sims) if sims else 0:.4f}")
    print(f"        silhouette={results[K]['sil']:.4f}, stability={results[K]['stab']:.4f}±{results[K]['stab_sd']:.4f}")

# Save K=3 summary
rows = []
for k in range(3):
    mk = labels3==k
    rt = rms_total[mk]; rt=rt[~np.isnan(rt)]
    rh = rms_hoa[mk]; rh=rh[~np.isnan(rh)]
    bad = df.loc[mk,"BAD D"].dropna()
    rows.append({
        "Phenotype": k, "N": mk.sum(), "Pct": f"{mk.sum()/N*100:.1f}%",
        "Label": ["Normal-Mild","Intermediate","Ectatic-Severe"][k],
        "Total_RMS": f"{np.median(rt):.2f} [{np.percentile(rt,25):.2f}-{np.percentile(rt,75):.2f}]",
        "HOA_RMS": f"{np.median(rh):.2f} [{np.percentile(rh,25):.2f}-{np.percentile(rh,75):.2f}]",
        "BAD_D": f"{bad.median():.2f} [{bad.quantile(.25):.2f}-{bad.quantile(.75):.2f}]",
        "Kmax": f"{df.loc[mk,'K Max (Front)'].median():.2f}",
        "CCT": f"{df.loc[mk,'CCT'].median():.0f}",
        "Pct_Normal": f"{(bad<1.6).sum()/len(bad)*100:.1f}%",
        "Pct_Suspect": f"{((bad>=1.6)&(bad<=2.6)).sum()/len(bad)*100:.1f}%",
        "Pct_Abnormal": f"{(bad>2.6).sum()/len(bad)*100:.1f}%",
        "Confidence": f"{results[3]['conf'][mk].mean():.3f}"
    })
pd.DataFrame(rows).to_csv(TBL_DIR/"k3_phenotype_summary.csv", index=False)
print("\nK=3 Summary:")
print(pd.DataFrame(rows).to_string(index=False))

print("\n\nDONE.")
