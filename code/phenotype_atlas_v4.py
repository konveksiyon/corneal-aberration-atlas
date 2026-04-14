#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-surface Zernike phenotype atlas — v4
CF (anterior) + CB (posterior) combined analysis
=================================================
Pipeline stages:
  A) Data loading + cohort (Error==0, OD, first visit, age>=18)
  B) CF+CB Zernike extraction + block normalisation
  C) PCA (discovery/replication split)
  D) K=4/5/6 comparison (silhouette, stability, severity, redundancy)
  E) Final models (optimal K)
  F) Severity independence
  G) Atlas generation (CF + CB wavefront maps)
  H) Clinical characterisation
  I) Output tables
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from datetime import datetime
from math import factorial
from scipy.stats import kruskal
from scipy.special import comb
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, pairwise_distances
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import hashlib

# ==============================================================
# CONFIG
# ==============================================================
BASE   = Path(__file__).resolve().parents[1]
CSV    = BASE / "csv"
OUT    = BASE / "output"; OUT.mkdir(exist_ok=True)
FIG_DIR = OUT / "figures"; FIG_DIR.mkdir(exist_ok=True)
TBL_DIR = OUT / "tables"; TBL_DIR.mkdir(exist_ok=True)

DISC_FRAC = 0.70
K_COMPARE = [4, 5, 6]
N_INIT    = 20
N_BOOT    = 100
EPS       = 1e-10
MAX_PCA   = 20   # PCA cap (was 15, raised for CF+CB)
ORDERS    = [2, 3, 4, 5, 6]
HOA_ORDERS= [3, 4, 5, 6]
SURFACES  = ["CF", "CB"]

R = {}  # results dict

print("="*70)
print("DUAL-SURFACE ZERNIKE PHENOTYPE ATLAS - v4 (CF+CB)")
print("="*70)

# ==============================================================
# A) DATA LOADING + COHORT
# ==============================================================
print("\n[A] Data loading and cohort formation...")

df_z   = pd.read_csv(CSV/"ZERNIKE-WFA.CSV", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
df_bad = pd.read_csv(CSV/"BADisplay-LOAD.CSV", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
df_idx = pd.read_csv(CSV/"INDEX-LOAD.CSV", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)

for d in [df_z, df_bad, df_idx]:
    d.columns = d.columns.str.strip().str.rstrip(":")

R["raw_wfa"] = len(df_z); R["raw_bad"] = len(df_bad); R["raw_idx"] = len(df_idx)
print(f"  Raw: WFA={R['raw_wfa']}, BAD={R['raw_bad']}, IDX={R['raw_idx']}")

# Column names for the demographic identifier fields in the Pentacam export.
_SUBJ_FIELDS = ["Last Name", "First Name", "D.o.Birth"]
_EXAM_FIELDS = ["Exam Eye", "Exam Date", "Exam Time"]

def make_key(df):
    """Build hashed patient and exam identifiers from device-export fields."""
    for c in _SUBJ_FIELDS + _EXAM_FIELDS:
        df[c] = df[c].astype(str).str.strip()
    df["patient_id"] = df[_SUBJ_FIELDS].apply(
        lambda r: hashlib.sha256("_".join(r).encode()).hexdigest()[:16], axis=1)
    df["exam_key"] = df["patient_id"]+"_"+df["Exam Eye"]+"_"+df["Exam Date"]+"_"+df["Exam Time"]

for d in [df_z, df_bad, df_idx]:
    make_key(d)
    d.drop_duplicates(subset="exam_key", keep="first", inplace=True)

keys_all = set(df_z["exam_key"]) & set(df_bad["exam_key"]) & set(df_idx["exam_key"])
R["n_merged"] = len(keys_all)
print(f"  3-CSV intersection: {R['n_merged']}")

df = df_z[df_z["exam_key"].isin(keys_all)].copy()

# Error == 0 only
df["Error"] = pd.to_numeric(df["Error"], errors="coerce").fillna(99)
n_before = len(df)
df = df[df["Error"] == 0].copy()
R["n_error_excluded"] = n_before - len(df)
R["n_after_error"] = len(df)
print(f"  Error==0: {n_before} -> {len(df)} ({R['n_error_excluded']} excluded)")

# Parse Zernike columns for BOTH surfaces
def parse_nm(col):
    parts = col.split("(")[0].strip().split()
    return int(parts[1]), int(parts[2])

cf_cols_raw = [c for c in df.columns if "(CF)" in c and c.startswith("Z ")]
cb_cols_raw = [c for c in df.columns if "(CB)" in c and c.startswith("Z ")]

cf_info_all = [(c, *parse_nm(c)) for c in cf_cols_raw]
cb_info_all = [(c, *parse_nm(c)) for c in cb_cols_raw]

# Convert to numeric
for c in cf_cols_raw + cb_cols_raw:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Check completeness
cf_full_info = [(c,n,m) for c,n,m in cf_info_all if 2<=n<=6]
cb_full_info = [(c,n,m) for c,n,m in cb_info_all if 2<=n<=6]
cf_hoa_info  = [(c,n,m) for c,n,m in cf_info_all if 3<=n<=6]
cb_hoa_info  = [(c,n,m) for c,n,m in cb_info_all if 3<=n<=6]

z_cf_ok = df[[c for c,_,_ in cf_full_info]].notna().all(axis=1)
z_cb_ok = df[[c for c,_,_ in cb_full_info]].notna().all(axis=1)
z_all_ok = z_cf_ok & z_cb_ok
n_z_miss = (~z_all_ok).sum()

# BAD-D
df_bad_m = df_bad[df_bad["exam_key"].isin(df["exam_key"])][["exam_key","BAD D"]].copy()
df_bad_m["BAD D"] = pd.to_numeric(df_bad_m["BAD D"], errors="coerce")
df = df.merge(df_bad_m, on="exam_key", how="left")
bad_ok = df["BAD D"].notna()
n_bad_miss = (~bad_ok).sum()

# CCT
idx_cols = ["exam_key"]
for pc in ["Thinnest Pachy","Pachy Min","D0mm Pachy"]:
    if pc in df_idx.columns: idx_cols.append(pc); break
# Also grab clinical index columns from INDEX
for cc in ["ISV","IVA","KI","CKI","IHA","IHD","K Max (Front)"]:
    if cc in df_idx.columns and cc not in idx_cols: idx_cols.append(cc)
df_idx_m = df_idx[df_idx["exam_key"].isin(df["exam_key"])][idx_cols].copy()
pachy_col = [c for c in idx_cols if "Pachy" in c][0]
df_idx_m[pachy_col] = pd.to_numeric(df_idx_m[pachy_col], errors="coerce")
df = df.merge(df_idx_m, on="exam_key", how="left", suffixes=("","_idx"))
pcol = pachy_col+"_idx" if pachy_col+"_idx" in df.columns else pachy_col
cct_ok = df[pcol].notna() & (df[pcol] > 100)
n_cct_miss = (~cct_ok).sum()

# Recalculate z completeness on merged df
z_cf_ok2 = df[[c for c,_,_ in cf_full_info]].notna().all(axis=1)
z_cb_ok2 = df[[c for c,_,_ in cb_full_info]].notna().all(axis=1)
z_ok2 = z_cf_ok2 & z_cb_ok2

mask_all = z_ok2 & bad_ok & cct_ok
n_before2 = len(df)
df = df[mask_all].copy()
R["n_missing_excluded"] = n_before2 - len(df)
R["n_after_missing"] = len(df)
print(f"  Missing data: {n_before2} -> {len(df)}")
print(f"    Zernike CF+CB missing: {(~(z_cf_ok2 & z_cb_ok2)).sum()}, BAD-D missing: {n_bad_miss}, CCT invalid: {(~cct_ok).sum()}")

# Only OD
n_b = len(df)
n_left = (~df["Exam Eye"].str.contains("Right", na=False)).sum()
df = df[df["Exam Eye"].str.contains("Right", na=False)].copy()
R["n_left_excluded"] = n_b - len(df)
print(f"  Right eye only: {n_b} -> {len(df)} ({R['n_left_excluded']} left eyes excluded)")

# First exam
df["Exam Date Parsed"] = pd.to_datetime(df["Exam Date"], format="mixed", dayfirst=False, errors="coerce")
df = df.sort_values(["patient_id","Exam Date Parsed"])
n_b = len(df)
df = df.drop_duplicates(subset="patient_id", keep="first")
print(f"  First visit: {n_b} -> {len(df)} ({n_b-len(df)} repeat visits excluded)")

# Age >= 18
df["dob"] = pd.to_datetime(df[_SUBJ_FIELDS[2]], format="mixed", dayfirst=False, errors="coerce")
df["age"] = (df["Exam Date Parsed"] - df["dob"]).dt.days / 365.25
n_b = len(df)
df = df[df["age"] >= 18].copy()
R["n_age_excluded"] = n_b - len(df)
R["N"] = len(df)
print(f"  Age >= 18: {n_b} -> {len(df)} ({R['n_age_excluded']} excluded)")
print(f"\n  >>> FINAL COHORT: {R['N']} right eyes <<<")

# RMS columns
df["RMS (CF)"]     = pd.to_numeric(df.get("RMS (CF)",""), errors="coerce")
df["RMS HOA (CF)"] = pd.to_numeric(df.get("RMS HOA (CF)",""), errors="coerce")
rms_total = df["RMS (CF)"].values.astype(float)
rms_hoa   = df["RMS HOA (CF)"].values.astype(float)

# Demographics
R["age_mean"] = df["age"].mean(); R["age_sd"] = df["age"].std()
R["age_med"]  = df["age"].median(); R["age_q25"] = df["age"].quantile(0.25)
R["age_q75"]  = df["age"].quantile(0.75)
R["age_min"]  = df["age"].min(); R["age_max"] = df["age"].max()
print(f"  Yas: {R['age_med']:.1f} [{R['age_q25']:.1f}-{R['age_q75']:.1f}]")

# Clinical columns for later
clin_map = {}
for cname in ["ISV","IVA","KI","CKI","IHA","IHD"]:
    if cname in df.columns:
        df[cname] = pd.to_numeric(df[cname], errors="coerce"); clin_map[cname] = cname
clin_map["BAD D"] = "BAD D"
for kc in ["K Max (Front)","K Max"]:
    if kc in df.columns:
        df[kc] = pd.to_numeric(df[kc], errors="coerce"); clin_map["Kmax"] = kc; break
clin_map["CCT"] = pcol

# ==============================================================
# B) CF+CB BLOCK NORMALIZATION
# ==============================================================
print("\n[B] CF+CB Block normalization...")

# Combined info: (col_name, n, m, surface)
combined_full_info = [(c,n,m,"CF") for c,n,m in cf_full_info] + [(c,n,m,"CB") for c,n,m in cb_full_info]
combined_hoa_info  = [(c,n,m,"CF") for c,n,m in cf_hoa_info]  + [(c,n,m,"CB") for c,n,m in cb_hoa_info]

R["n_full_cf"] = len(cf_full_info); R["n_full_cb"] = len(cb_full_info)
R["n_full"] = len(combined_full_info)
R["n_hoa_cf"] = len(cf_hoa_info); R["n_hoa_cb"] = len(cb_hoa_info)
R["n_hoa"] = len(combined_hoa_info)

print(f"  Full: CF={R['n_full_cf']} + CB={R['n_full_cb']} = {R['n_full']} coefficients")
print(f"  HOA:  CF={R['n_hoa_cf']} + CB={R['n_hoa_cb']} = {R['n_hoa']} coefficients")

# Raw arrays
raw_cf_full = df[[c for c,_,_ in cf_full_info]].values.astype(float)
raw_cb_full = df[[c for c,_,_ in cb_full_info]].values.astype(float)
raw_cf_hoa  = df[[c for c,_,_ in cf_hoa_info]].values.astype(float)
raw_cb_hoa  = df[[c for c,_,_ in cb_hoa_info]].values.astype(float)

def get_block_idx(info_list, surface, order):
    """Get indices for a specific (surface, order) block."""
    return [i for i,(c,n,m,s) in enumerate(info_list) if s==surface and n==order]

def block_normalize(raw_combined, info_list, surfaces, orders):
    """Block L2 normalization: blocks = (surface, order) pairs."""
    N = raw_combined.shape[0]
    blocks = []
    block_spec = []  # (surface, order, start_idx, end_idx)
    for surf in surfaces:
        for o in orders:
            idx = get_block_idx(info_list, surf, o)
            if len(idx) == 0: continue
            b = raw_combined[:, idx].copy()
            norms = np.linalg.norm(b, axis=1, keepdims=True)
            b = b / (norms + EPS)
            block_spec.append((surf, o, len(idx)))
            blocks.append(b)
    cat = np.hstack(blocks)
    gnorm = np.linalg.norm(cat, axis=1, keepdims=True)
    cat = cat / (gnorm + EPS)
    return cat, block_spec

# Combined raw arrays
raw_full_combined = np.hstack([raw_cf_full, raw_cb_full])
raw_hoa_combined  = np.hstack([raw_cf_hoa, raw_cb_hoa])

u_full, blocks_full = block_normalize(raw_full_combined, combined_full_info, SURFACES, ORDERS)
u_hoa,  blocks_hoa  = block_normalize(raw_hoa_combined, combined_hoa_info, SURFACES, HOA_ORDERS)

print(f"  Shape vectors: Full={u_full.shape}, HOA={u_hoa.shape}")
print(f"  Full blocks: {[(s,o,n) for s,o,n in blocks_full]}")
print(f"  HOA blocks:  {[(s,o,n) for s,o,n in blocks_hoa]}")

R["z_standard"] = "OSA/ANSI"
R["z_dia"] = "6 mm"

# ==============================================================
# C) PCA
# ==============================================================
print("\n[C] PCA...")

np.random.seed(42)
patients = df["patient_id"].unique()
np.random.shuffle(patients)
n_disc = int(len(patients) * DISC_FRAC)
disc_set = set(patients[:n_disc])
disc_mask = df["patient_id"].isin(disc_set).values
repl_mask = ~disc_mask
R["n_disc"] = int(disc_mask.sum()); R["n_repl"] = int(repl_mask.sum())
print(f"  Discovery: {R['n_disc']}, Replication: {R['n_repl']}")

def do_pca(u, dm, label):
    ud = u[dm]
    mc = min(ud.shape)
    pca = PCA(n_components=mc, random_state=42)
    pca.fit(ud)
    cv = np.cumsum(pca.explained_variance_ratio_)
    k90 = int(np.searchsorted(cv, 0.90)) + 1
    npc = min(k90, MAX_PCA); npc = max(npc, 3)
    sc = pca.transform(u)[:, :npc]
    print(f"  [{label}] %90->{k90}PC, secilen={npc}PC (var={cv[npc-1]:.4f})")
    return pca, npc, sc, cv

pca_f, npc_f, sc_f, cv_f = do_pca(u_full, disc_mask, "FULL CF+CB")
pca_h, npc_h, sc_h, cv_h = do_pca(u_hoa,  disc_mask, "HOA CF+CB")
R["pca_f_npc"] = npc_f; R["pca_f_var"] = float(cv_f[npc_f-1])
R["pca_h_npc"] = npc_h; R["pca_h_var"] = float(cv_h[npc_h-1])

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, cv, npc, t in [(axes[0],cv_f,npc_f,"Full CF+CB"), (axes[1],cv_h,npc_h,"HOA CF+CB")]:
    xr = range(1, min(40, len(cv))+1)
    ax.plot(xr, cv[:len(xr)]*100, "bo-", ms=4)
    ax.axhline(90, color="red", ls="--", lw=0.8, label="90%")
    ax.axhline(95, color="orange", ls="--", lw=0.8, label="95%")
    ax.axvline(npc, color="green", ls="--", lw=1, label=f"Selected={npc}")
    ax.set_xlabel("PC"); ax.set_ylabel("Cum. Var (%)"); ax.set_title(f"PCA Scree - {t}")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR/"fig02_pca_scree.png", dpi=200); plt.close()

# ==============================================================
# D) CLUSTERING UTILITIES
# ==============================================================
def spherical_kmeans(X, K, n_init=10, max_iter=300, seed=42):
    rng = np.random.RandomState(seed)
    best_inertia, best_labels, best_centers = np.inf, None, None
    N, D = X.shape
    for run in range(n_init):
        idx = rng.choice(N, K, replace=False)
        centers = X[idx].copy()
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + EPS)
        for _ in range(max_iter):
            sims = X @ centers.T
            labels = sims.argmax(axis=1)
            new_c = np.zeros_like(centers)
            for k in range(K):
                mk = labels == k
                if mk.sum() == 0:
                    new_c[k] = X[rng.randint(N)]
                else:
                    new_c[k] = X[mk].mean(axis=0)
            new_c = new_c / (np.linalg.norm(new_c, axis=1, keepdims=True) + EPS)
            if np.max(np.abs(new_c - centers)) < 1e-6: break
            centers = new_c
        inertia = np.sum(1.0 - np.einsum("ij,ij->i", X, centers[labels]))
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels.copy(), centers.copy()
    return best_labels, best_centers, best_inertia

def vmf_em(X, K, n_init=10, max_iter=200, seed=42):
    rng = np.random.RandomState(seed)
    N, D = X.shape
    best_ll = -np.inf; best_lab = None; best_cen = None
    for run in range(n_init):
        idx = rng.choice(N, K, replace=False)
        mu = X[idx].copy()
        mu = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + EPS)
        kappa = np.full(K, 10.0)
        pi = np.full(K, 1.0/K)
        for it in range(max_iter):
            log_r = np.zeros((N, K))
            for k in range(K):
                log_r[:, k] = np.log(pi[k] + EPS) + kappa[k] * (X @ mu[k])
            log_r -= log_r.max(axis=1, keepdims=True)
            r = np.exp(log_r)
            r /= r.sum(axis=1, keepdims=True) + EPS
            Nk = r.sum(axis=0)
            pi = Nk / N
            for k in range(K):
                Rk = (r[:, k:k+1] * X).sum(axis=0)
                rbar = np.linalg.norm(Rk) / (Nk[k] + EPS)
                mu[k] = Rk / (np.linalg.norm(Rk) + EPS)
                rbar = min(rbar, 0.999)
                kappa[k] = rbar * (D - rbar**2) / (1 - rbar**2 + EPS)
                kappa[k] = np.clip(kappa[k], 0.1, 500)
        ll = 0
        for k in range(K):
            ll += np.sum(r[:, k] * (np.log(pi[k]+EPS) + kappa[k]*(X @ mu[k])))
        if ll > best_ll:
            best_ll = ll; best_lab = r.argmax(axis=1).copy(); best_cen = mu.copy()
    return best_lab, best_cen

def cosine_silhouette(X, labels):
    D = 1 - X @ X.T
    np.fill_diagonal(D, 0)
    return silhouette_score(D, labels, metric="precomputed")

def bootstrap_stability(X, K, n_boot=100, seed=42):
    rng = np.random.RandomState(seed)
    ref_lab, _, _ = spherical_kmeans(X, K, n_init=5, seed=seed)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        blab, _, _ = spherical_kmeans(X[idx], K, n_init=5, seed=seed+b+1)
        aris.append(adjusted_rand_score(ref_lab[idx], blab))
    return np.mean(aris), np.std(aris)

def severity_pred(labels, rms, K):
    X_s = rms.reshape(-1,1)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    acc_lr = cross_val_score(lr, X_s, labels, cv=cv, scoring="accuracy").mean()
    return acc_lr

def centroid_similarity(centers):
    K = len(centers)
    sims = []
    for i in range(K):
        for j in range(i+1, K):
            s = np.dot(centers[i], centers[j]) / (np.linalg.norm(centers[i])*np.linalg.norm(centers[j]) + EPS)
            sims.append(s)
    return max(sims), np.mean(sims)

# ==============================================================
# E) K=4/5/6 COMPARISON
# ==============================================================
print("\n[E] K=4/5/6 comparison...")

def k_compare(u, dm, rms_t, rms_h, label):
    Xd = u[dm]
    rows = []
    all_results = {}
    for K in K_COMPARE:
        print(f"    K={K}...", end=" ", flush=True)
        lab_skm, cen_skm, _ = spherical_kmeans(Xd, K, n_init=N_INIT, seed=42)
        sil_skm = cosine_silhouette(Xd, lab_skm)
        stab_m, stab_s = bootstrap_stability(Xd, K, N_BOOT, seed=42)
        sev_tl = severity_pred(lab_skm, rms_t[dm], K)
        sev_hl = severity_pred(lab_skm, rms_h[dm], K)
        mx_sim, mn_sim = centroid_similarity(cen_skm)
        sizes = np.bincount(lab_skm, minlength=K)
        print(f"sil={sil_skm:.3f} stab={stab_m:.3f} sev={sev_tl:.3f} maxSim={mx_sim:.3f}")
        rows.append({
            "K": K, "cluster_sizes": str(list(sizes)),
            "min_pct": f"{sizes.min()/len(Xd)*100:.1f}%",
            "max_pct": f"{sizes.max()/len(Xd)*100:.1f}%",
            "sil_skm": round(sil_skm,4),
            "stab_mean": round(stab_m,4), "stab_std": round(stab_s,4),
            "sev_total_lr": round(sev_tl,4),
            "sev_hoa_lr": round(sev_hl,4),
            "max_cent_sim": round(mx_sim,4), "mean_cent_sim": round(mn_sim,4),
            "chance": round(1.0/K,4)
        })
        all_results[K] = {"lab": lab_skm, "cen": cen_skm, "sil": sil_skm,
                          "stab": stab_m, "sev": sev_tl, "sim": mx_sim}
    # Composite score
    vals = {m: [all_results[K][m] for K in K_COMPARE] for m in ["sil","stab","sev","sim"]}
    def norm01(v): r = max(v)-min(v); return [(x-min(v))/(r+EPS) for x in v] if r>0 else [0.5]*len(v)
    sn = norm01(vals["sil"]); tn = norm01(vals["stab"])
    en = [1-x for x in norm01(vals["sev"])]; rn = [1-x for x in norm01(vals["sim"])]
    comp = [0.25*s+0.30*t+0.25*e+0.20*r for s,t,e,r in zip(sn,tn,en,rn)]
    best_K = K_COMPARE[int(np.argmax(comp))]
    for i, K in enumerate(K_COMPARE):
        rows[i]["composite"] = round(comp[i],4)
    return pd.DataFrame(rows), best_K, all_results

print(f"  Full CF+CB (n_disc={R['n_disc']}):")
df_kf, Kf, res_f = k_compare(u_full, disc_mask, rms_total, rms_hoa, "Full")
print(f"  HOA CF+CB:")
df_kh, Kh, res_h = k_compare(u_hoa, disc_mask, rms_total, rms_hoa, "HOA")
print(f"\n  >>> Optimal K: Full={Kf}, HOA={Kh}")

df_kf.to_csv(TBL_DIR/"k_comparison_full.csv", index=False)
df_kh.to_csv(TBL_DIR/"k_comparison_hoa.csv", index=False)

# ==============================================================
# F) FINAL MODELS
# ==============================================================
print(f"\n[F] Final models (Full K={Kf}, HOA K={Kh})...")

labels_f, centers_f, _ = spherical_kmeans(u_full, Kf, n_init=N_INIT, seed=42)
labels_h, centers_h, _ = spherical_kmeans(u_hoa,  Kh, n_init=N_INIT, seed=42)

conf_f = np.max(u_full @ centers_f.T, axis=1)
conf_h = np.max(u_hoa  @ centers_h.T, axis=1)

print(f"  Full: sizes={list(np.bincount(labels_f))}, conf={conf_f.mean():.3f}")
print(f"  HOA:  sizes={list(np.bincount(labels_h))}, conf={conf_h.mean():.3f}")

# ==============================================================
# G) SEVERITY INDEPENDENCE
# ==============================================================
print("\n[G] Severity independence...")

def full_sev_report(labels, u, rms_t, rms_h, K, raw_combined, info_list, surfaces, orders, label):
    s_tl = severity_pred(labels, rms_t, K)
    s_hl = severity_pred(labels, rms_h, K)
    # Raw clustering comparison
    raw_lab, _, _ = spherical_kmeans(raw_combined / (np.linalg.norm(raw_combined, axis=1, keepdims=True)+EPS),
                                      K, n_init=5, seed=42)
    raw_sev = severity_pred(raw_lab, rms_t, K)
    norm_sev = s_tl
    # Cosine separation
    sep = []
    for i in range(len(u)):
        same = labels == labels[i]; diff = ~same
        if same.sum() > 1 and diff.sum() > 0:
            s_same = np.mean(u[i] @ u[same].T)
            s_diff = np.mean(u[i] @ u[diff].T)
            sep.append(s_same - s_diff)
    cos_sep = np.mean(sep) if sep else 0
    res = {"Total_LR": s_tl, "HOA_LR": s_hl,
           "raw_sev": raw_sev, "norm_sev": norm_sev, "cos_sep": cos_sep,
           "chance": 1.0/K}
    print(f"  [{label}] Total_LR={s_tl:.3f}, HOA_LR={s_hl:.3f}")
    print(f"    Raw vs Norm: {raw_sev:.3f} vs {norm_sev:.3f}")
    return res

R["sev_f"] = full_sev_report(labels_f, u_full, rms_total, rms_hoa, Kf,
                              raw_full_combined, combined_full_info, SURFACES, ORDERS, "Full")
R["sev_h"] = full_sev_report(labels_h, u_hoa, rms_total, rms_hoa, Kh,
                              raw_hoa_combined, combined_hoa_info, SURFACES, HOA_ORDERS, "HOA")

# ==============================================================
# H) ATLAS URETIMI (CF + CB)
# ==============================================================
print("\n[H] Atlas uretimi (CF + CB)...")

def zernike_R(n, m, rho):
    m_abs = abs(m)
    result = np.zeros_like(rho)
    for s in range(int((n - m_abs) / 2) + 1):
        num = (-1)**s * factorial(n - s)
        den = factorial(s) * factorial(int((n + m_abs)/2) - s) * factorial(int((n - m_abs)/2) - s)
        result += (num / den) * rho**(n - 2*s)
    return result

def zernike_value(n, m, rho, theta):
    Rnm = zernike_R(n, m, rho)
    if m >= 0: return Rnm * np.cos(m * theta)
    else:      return Rnm * np.sin(abs(m) * theta)

def reconstruct_wf(coeffs, col_info, grid=100):
    """Reconstruct wavefront from Zernike coefficients for a single surface."""
    x = np.linspace(-1, 1, grid)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X**2 + Y**2); theta = np.arctan2(Y, X)
    W = np.zeros_like(rho)
    for j, (c_name, n, m) in enumerate(col_info):
        if j < len(coeffs):
            W += coeffs[j] * zernike_value(n, m, rho, theta)
    W[rho > 1] = np.nan
    return W

def split_center(center, info_list, surface, col_info_surface):
    """Extract surface-specific coefficients from combined centroid."""
    idx = [i for i, (c,n,m,s) in enumerate(info_list) if s == surface]
    return center[idx]

def make_atlas(labels, centers, u, raw_cf, raw_cb, cf_info, cb_info,
               combined_info, rms, K, tag, fig_name):
    """Generate atlas with 5 rows: CF Mean, CB Mean, CF Medoid, CB Medoid, CF Low/High RMS."""
    n_rows = 4  # CF Mean, CB Mean, CF Medoid, CB Medoid
    fig, axes = plt.subplots(n_rows, K, figsize=(3.2*K, 3.2*n_rows))
    if K == 1: axes = axes.reshape(-1, 1)
    cmap = plt.cm.RdBu_r
    row_labels = ["CF Mean Shape", "CB Mean Shape", "CF Medoid", "CB Medoid"]

    for k in range(K):
        mk = labels == k
        pct = mk.sum() / len(labels) * 100

        # Split centroid into CF and CB parts
        cf_center = split_center(centers[k], combined_info, "CF", cf_info)
        cb_center = split_center(centers[k], combined_info, "CB", cb_info)

        # Medoid (highest cosine similarity to centroid)
        sims_k = u[mk] @ centers[k]
        medoid_local = np.argmax(sims_k)
        medoid_global = np.where(mk)[0][medoid_local]

        maps = [
            reconstruct_wf(cf_center, cf_info),
            reconstruct_wf(cb_center, cb_info),
            reconstruct_wf(raw_cf[medoid_global], cf_info),
            reconstruct_wf(raw_cb[medoid_global], cb_info),
        ]

        for r, W in enumerate(maps):
            ax = axes[r, k]
            vm = max(np.nanmax(np.abs(W)), 0.01)
            ax.imshow(W, cmap=cmap, vmin=-vm, vmax=vm, extent=[-1.1,1.1,-1.1,1.1],
                      interpolation="bilinear")
            circle = Circle((0,0), 1.0, fill=False, ec="black", lw=1)
            ax.add_patch(circle)
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"P{k} (n={mk.sum()}, {pct:.1f}%)", fontsize=9, fontweight="bold")
            if k == 0:
                ax.set_ylabel(row_labels[r], fontsize=8, fontweight="bold")

    fig.suptitle(f"{tag} Phenotype Atlas (K={K}): CF + CB", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR/fig_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK {fig_name}")

# Generate K=4,5,6 atlases + BEST
for K_val in K_COMPARE:
    lab_t, cen_t, _ = spherical_kmeans(u_full, K_val, n_init=N_INIT, seed=42)
    make_atlas(lab_t, cen_t, u_full, raw_cf_full, raw_cb_full,
               cf_full_info, cb_full_info, combined_full_info,
               rms_total, K_val, "Full CF+CB", f"fig04_full_K{K_val}.png")

    lab_t2, cen_t2, _ = spherical_kmeans(u_hoa, K_val, n_init=N_INIT, seed=42)
    make_atlas(lab_t2, cen_t2, u_hoa, raw_cf_hoa, raw_cb_hoa,
               cf_hoa_info, cb_hoa_info, combined_hoa_info,
               rms_total, K_val, "HOA CF+CB", f"fig05_hoa_K{K_val}.png")

# BEST
make_atlas(labels_f, centers_f, u_full, raw_cf_full, raw_cb_full,
           cf_full_info, cb_full_info, combined_full_info,
           rms_total, Kf, "Full CF+CB", "fig04_full_BEST.png")
make_atlas(labels_h, centers_h, u_hoa, raw_cf_hoa, raw_cb_hoa,
           cf_hoa_info, cb_hoa_info, combined_hoa_info,
           rms_total, Kh, "HOA CF+CB", "fig05_hoa_BEST.png")

# ==============================================================
# I) KLINIK KARAKTERIZASYON
# ==============================================================
print("\n[I] Klinik karakterizasyon...")

def clinical_stats(labels, K, prefix):
    pvals = {}
    rows = []
    for vname, col in clin_map.items():
        vals = pd.to_numeric(df[col], errors="coerce").values
        groups = [vals[labels==k] for k in range(K)]
        groups_clean = [g[~np.isnan(g)] for g in groups]
        if all(len(g) > 1 for g in groups_clean):
            _, pv = kruskal(*groups_clean)
            pvals[vname] = pv
        for k in range(K):
            g = groups_clean[k]
            if len(g) > 0:
                rows.append({"Variable": vname, "Phenotype": k, "N": len(g),
                             "Median": round(np.median(g),2),
                             "IQR": f"[{np.percentile(g,25):.2f}-{np.percentile(g,75):.2f}]",
                             "Mean": round(np.mean(g),2), "SD": round(np.std(g),2)})
    return pvals, pd.DataFrame(rows) if rows else None

pv_f, cs_f = clinical_stats(labels_f, Kf, "Full")
pv_h, cs_h = clinical_stats(labels_h, Kh, "HOA")

# Save
if pv_f:
    pd.DataFrame([{"Variable":k,"p":v} for k,v in pv_f.items()]).to_csv(TBL_DIR/"clinical_pvalues_full.csv", index=False)
if pv_h:
    pd.DataFrame([{"Variable":k,"p":v} for k,v in pv_h.items()]).to_csv(TBL_DIR/"clinical_pvalues_hoa.csv", index=False)
if cs_f is not None: cs_f.to_csv(TBL_DIR/"clinical_summary_full.csv", index=False)
if cs_h is not None: cs_h.to_csv(TBL_DIR/"clinical_summary_hoa.csv", index=False)

# ==============================================================
# J) DEMOGRAFI + FENOTIP TABLOLARI
# ==============================================================
print("\n[J] Tablolar...")

# Demographics
demo_rows = []
def add_demo(name, vals):
    v = vals[~np.isnan(vals)]
    demo_rows.append({"Variable": name, "N": len(v),
        "Mean +/- SD": f"{v.mean():.2f} +/- {v.std():.2f}",
        "Median [IQR]": f"{np.median(v):.2f} [{np.percentile(v,25):.2f}-{np.percentile(v,75):.2f}]",
        "Range": f"{v.min():.2f} - {v.max():.2f}"})

add_demo("Age (years)", df["age"].values)
add_demo("Total RMS (um)", rms_total)
add_demo("HOA RMS (um)", rms_hoa)
for vn, col in clin_map.items():
    v = pd.to_numeric(df[col], errors="coerce").values
    add_demo(vn, v)

df_demo = pd.DataFrame(demo_rows)
df_demo.to_csv(TBL_DIR/"demographics.csv", index=False)

# Phenotype tables
def pheno_table(labels, K, rms_t, rms_h, conf):
    rows = []
    for k in range(K):
        mk = labels == k
        rt = rms_t[mk]; rh = rms_h[mk]; ck = conf[mk]
        rows.append({"Phenotype": k, "N": int(mk.sum()),
            "Pct": f"{mk.sum()/len(labels)*100:.1f}%",
            "Total_RMS": f"{np.nanmedian(rt):.2f} [{np.nanpercentile(rt,25):.2f}-{np.nanpercentile(rt,75):.2f}]",
            "HOA_RMS": f"{np.nanmedian(rh):.2f} [{np.nanpercentile(rh,25):.2f}-{np.nanpercentile(rh,75):.2f}]",
            "Confidence": f"{ck.mean():.3f}"})
    return pd.DataFrame(rows)

tf = pheno_table(labels_f, Kf, rms_total, rms_hoa, conf_f)
th = pheno_table(labels_h, Kh, rms_total, rms_hoa, conf_h)
tf.to_csv(TBL_DIR/"phenotype_full.csv", index=False)
th.to_csv(TBL_DIR/"phenotype_hoa.csv", index=False)

print(f"  Full (K={Kf}):")
for _, r in tf.iterrows():
    print(f"    P{r['Phenotype']}: N={r['N']}, {r['Pct']}, RMS={r['Total_RMS']}")
print(f"  HOA (K={Kh}):")
for _, r in th.iterrows():
    print(f"    P{r['Phenotype']}: N={r['N']}, {r['Pct']}, RMS={r['Total_RMS']}")

# Assignments
df_out = df[["exam_key","Exam Eye","RMS (CF)","RMS HOA (CF)","age"]].copy()
df_out["pheno_full"] = labels_f; df_out["pheno_hoa"] = labels_h
df_out["conf_full"] = conf_f; df_out["conf_hoa"] = conf_h
df_out.to_csv(TBL_DIR/"assignments.csv", index=False)

# Replication
print(f"\n[Replication]")
for lbl_name, lab, K in [("Full", labels_f, Kf), ("HOA", labels_h, Kh)]:
    dd = np.bincount(lab[disc_mask], minlength=K)
    rd = np.bincount(lab[repl_mask], minlength=K)
    parts = []
    for k in range(K):
        dp = dd[k]/disc_mask.sum()*100; rp = rd[k]/repl_mask.sum()*100
        parts.append(f"P{k}: D={dp:.1f}% R={rp:.1f}%")
    print(f"  {lbl_name}: {' | '.join(parts)}")

# ==============================================================
# K) ADDITIONAL FIGURES
# ==============================================================
print("\n[K] Additional figures...")

# K Comparison figure
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for row, (dfk, name) in enumerate([(df_kf, "Full CF+CB"), (df_kh, "HOA CF+CB")]):
    metrics = [("sil_skm","Silhouette","steelblue"), ("stab_mean","Stability (ARI)","seagreen"),
               ("sev_total_lr","Sev Pred (LR)","coral"), ("max_cent_sim","Max Centroid Sim","purple")]
    for col, (m, title, color) in enumerate(metrics):
        ax = axes[row, col]
        vals = dfk[m].values
        ax.bar(range(len(K_COMPARE)), vals, color=color, alpha=0.8)
        ax.set_xticks(range(len(K_COMPARE))); ax.set_xticklabels(K_COMPARE)
        ax.set_title(f"{name}\n{title}", fontsize=9)
        if "sev" in m.lower():
            for K_idx, K_val in enumerate(K_COMPARE):
                ax.axhline(1.0/K_val, color="gray", ls="--", lw=0.8)
        for i, v in enumerate(vals):
            ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(FIG_DIR/"fig03_k_comparison.png", dpi=200, bbox_inches="tight"); plt.close()
print("  OK fig03_k_comparison.png")

# Fingerprint plots
def fingerprint_plot(centers, col_info, K, tag, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [2, 1]})
    colors = plt.cm.Set2(np.linspace(0, 1, max(K, 3)))
    # Bar chart
    labels_z = [f"Z({n},{m})" for _, n, m, _ in col_info]
    x = np.arange(len(labels_z))
    w = 0.8 / K
    for k in range(K):
        ax1.bar(x + k*w, centers[k], w, color=colors[k], label=f"P{k}", alpha=0.85)
    ax1.set_xticks(x + 0.4); ax1.set_xticklabels(labels_z, rotation=90, fontsize=6)
    ax1.set_ylabel("Normalized Coefficient"); ax1.legend(fontsize=8)
    ax1.set_title(f"{tag} Centroid Fingerprint (K={K})")
    ax1.axhline(0, color="gray", lw=0.5)
    # Degree contribution heatmap
    surfaces_orders = []
    for surf in SURFACES:
        ords = ORDERS if "Full" in tag else HOA_ORDERS
        for o in ords:
            idx = [i for i,(c,n,m,s) in enumerate(col_info) if s==surf and n==o]
            if idx: surfaces_orders.append((f"{surf} O{o}", idx))
    heat = np.zeros((K, len(surfaces_orders)))
    for j, (so_name, idx) in enumerate(surfaces_orders):
        for k in range(K):
            heat[k, j] = np.linalg.norm(centers[k][idx])
    heat = heat / (heat.sum(axis=1, keepdims=True) + EPS)
    im = ax2.imshow(heat, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(surfaces_orders)))
    ax2.set_xticklabels([s for s,_ in surfaces_orders], rotation=45, fontsize=7)
    ax2.set_yticks(range(K)); ax2.set_yticklabels([f"P{k}" for k in range(K)])
    for i in range(K):
        for j in range(len(surfaces_orders)):
            ax2.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax2.set_title("Block Energy Distribution (Surface x Order)")
    fig.tight_layout(); fig.savefig(FIG_DIR/fname, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  OK {fname}")

fingerprint_plot(centers_f, combined_full_info, Kf, "Full CF+CB", "fig06a_fp_full.png")
fingerprint_plot(centers_h, combined_hoa_info,  Kh, "HOA CF+CB",  "fig06b_fp_hoa.png")

# Severity independence figure
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
colors_set2 = plt.cm.Set2(np.linspace(0, 1, max(Kf, Kh)))
for row, (lab, K, sev, name) in enumerate([
    (labels_f, Kf, R["sev_f"], "Full CF+CB"),
    (labels_h, Kh, R["sev_h"], "HOA CF+CB")]):
    # Total RMS box
    ax = axes[row, 0]
    data_t = [rms_total[lab==k] for k in range(K)]
    bp = ax.boxplot(data_t, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]): patch.set_facecolor(colors_set2[i])
    ax.set_title(f"{name}: Total RMS by Phenotype", fontsize=9)
    ax.set_xticklabels([f"P{k}" for k in range(K)])
    # HOA RMS box
    ax = axes[row, 1]
    data_h = [rms_hoa[lab==k] for k in range(K)]
    bp = ax.boxplot(data_h, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]): patch.set_facecolor(colors_set2[i])
    ax.set_title(f"{name}: HOA RMS by Phenotype", fontsize=9)
    ax.set_xticklabels([f"P{k}" for k in range(K)])
    # Severity pred bars
    ax = axes[row, 2]
    vals = [sev["Total_LR"], sev["HOA_LR"]]
    cols_bar = ["steelblue","coral"]
    lbls = ["Total RMS LR","HOA RMS LR"]
    bars = ax.bar(range(2), vals, color=cols_bar, alpha=0.8)
    ax.axhline(sev["chance"], color="red", ls="--", lw=1, label=f"Chance={sev['chance']:.3f}")
    for i, v in enumerate(vals): ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(2)); ax.set_xticklabels(lbls, fontsize=8)
    ax.set_title(f"{name}: Severity Predictability", fontsize=9)
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG_DIR/"fig07_severity_independence.png", dpi=200, bbox_inches="tight"); plt.close()
print("  OK fig07_severity_independence.png")

# Clinical characterization figure
def clinical_fig(labels, K, pvals, tag, fname):
    clin_vars = [v for v in clin_map.keys() if v in (pvals or {})]
    if not clin_vars: return
    nc = min(4, len(clin_vars)); nr = (len(clin_vars)+nc-1)//nc
    fig, axes = plt.subplots(nr, nc, figsize=(4.5*nc, 3.5*nr))
    axes_flat = np.array(axes).flatten() if nr*nc > 1 else [axes]
    colors_s = plt.cm.Set2(np.linspace(0, 1, K))
    for i, vn in enumerate(clin_vars):
        ax = axes_flat[i]
        col = clin_map[vn]
        vals = pd.to_numeric(df[col], errors="coerce").values
        data = [vals[labels==k][~np.isnan(vals[labels==k])] for k in range(K)]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for j, patch in enumerate(bp["boxes"]): patch.set_facecolor(colors_s[j])
        pv = pvals.get(vn, np.nan)
        pstr = f"p<0.001" if pv < 0.001 else f"p={pv:.3f}"
        ax.set_title(f"{vn} ({pstr})", fontsize=9)
        ax.set_xticklabels([f"P{k}" for k in range(K)], fontsize=7)
    for j in range(len(clin_vars), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"{tag} Post-hoc Clinical Characterization", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR/fname, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  OK {fname}")

clinical_fig(labels_f, Kf, pv_f, f"Full CF+CB (K={Kf})", "fig08a_clin_full.png")
clinical_fig(labels_h, Kh, pv_h, f"HOA CF+CB (K={Kh})", "fig08b_clin_hoa.png")

# Cohort flow diagram
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
steps = [
    (f"Pentacam HR Scheimpflug\n3 CSV (WFA + BAD + INDEX)\nN = {R['n_merged']}", 11),
    (f"Error == 0 filtresi\nN = {R['n_after_error']}", 9.5),
    (f"Missing data excluded\n(BAD-D, Zernike CF+CB, CCT)\nN = {R['n_after_missing']}", 8),
    (f"Right eye only (OD)\nFirst visit selected", 6.5),
    (f"Yas >= 18 filtresi", 5),
    (f"FINAL COHORT\nN = {R['N']} right eyes", 3.5),
]
for i, (txt, y) in enumerate(steps):
    color = "lightyellow" if i == len(steps)-1 else "lightcyan"
    ax.annotate(txt, xy=(5, y), fontsize=10, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="gray"))
    if i < len(steps)-1:
        ax.annotate("", xy=(5, steps[i+1][1]+0.6), xytext=(5, y-0.6),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
fig.savefig(FIG_DIR/"fig01_cohort_flow.png", dpi=200, bbox_inches="tight"); plt.close()
print("  OK fig01_cohort_flow.png")

# ==============================================================
# L) OUTPUT SUMMARY
# ==============================================================
print("\n[L] Output summary...")

# Collect extra data
def full_sim_matrix(centers):
    K = len(centers)
    m = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            m[i,j] = np.dot(centers[i], centers[j]) / (np.linalg.norm(centers[i])*np.linalg.norm(centers[j])+EPS)
    return m

sim_mat_f = full_sim_matrix(centers_f)
sim_mat_h = full_sim_matrix(centers_h)

def cluster_rms_detail(labels, rms_t, rms_h, K):
    rows = []
    for k in range(K):
        mk = labels == k
        rt = rms_t[mk]; rh = rms_h[mk]
        rows.append({"Phenotype": k, "N": int(mk.sum()),
            "Total_RMS_mean": round(np.nanmean(rt),3), "Total_RMS_sd": round(np.nanstd(rt),3),
            "Total_RMS_median": round(np.nanmedian(rt),3),
            "HOA_RMS_mean": round(np.nanmean(rh),3), "HOA_RMS_sd": round(np.nanstd(rh),3),
            "HOA_RMS_median": round(np.nanmedian(rh),3)})
    return pd.DataFrame(rows)

rms_det_f = cluster_rms_detail(labels_f, rms_total, rms_hoa, Kf)
rms_det_h = cluster_rms_detail(labels_h, rms_total, rms_hoa, Kh)

# Replication detail
def repl_detail(labels, dm, rm, K):
    dd = np.bincount(labels[dm], minlength=K)
    rd = np.bincount(labels[rm], minlength=K)
    rows = []
    for k in range(K):
        rows.append({"Phenotype":k, "Disc_N":int(dd[k]),
            "Disc_pct":f"{dd[k]/dm.sum()*100:.2f}%",
            "Repl_N":int(rd[k]), "Repl_pct":f"{rd[k]/rm.sum()*100:.2f}%",
            "Diff_pct":f"{abs(dd[k]/dm.sum()-rd[k]/rm.sum())*100:.2f}%"})
    return pd.DataFrame(rows)

repl_f = repl_detail(labels_f, disc_mask, repl_mask, Kf)
repl_h = repl_detail(labels_h, disc_mask, repl_mask, Kh)

# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "="*70)
print("SUMMARY (v4 — CF+CB)")
print("="*70)
print(f"  Cohort: {R['N']} right eyes")
print(f"  Features: CF({R['n_full_cf']}) + CB({R['n_full_cb']}) = {R['n_full']} (Full)")
print(f"  PCA Full: {R['pca_f_npc']} PCs ({R['pca_f_var']*100:.1f}%)")
print(f"  PCA HOA:  {R['pca_h_npc']} PCs ({R['pca_h_var']*100:.1f}%)")
print(f"  Full Atlas: K={Kf}, sizes={list(np.bincount(labels_f))}")
print(f"  HOA Atlas:  K={Kh}, sizes={list(np.bincount(labels_h))}")
print(f"  Sev pred Full: {R['sev_f']['Total_LR']:.3f} (chance {R['sev_f']['chance']:.3f})")
print(f"  Sev pred HOA:  {R['sev_h']['Total_LR']:.3f} (chance {R['sev_h']['chance']:.3f})")
print(f"\n  Outputs: {OUT}")
print("  DONE.")
