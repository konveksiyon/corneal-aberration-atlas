#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUPPLEMENTARY ANALYSES for reviewer response
=============================================
Runs on the same cohort/pipeline as phenotype_atlas_v4.py
Produces:
  1) K=2,3,4,5,6 full comparison table
  2) Composite weight sensitivity analysis (5 weight sets)
  3) Kruskal-Wallis effect sizes (eta-squared) for all clinical variables
  4) BAD-D diagnostic category distribution per phenotype
  5) Discovery-replication centroid concordance
  6) Cosine vs Euclidean silhouette comparison
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import json, sys, hashlib

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
CSV = BASE / "csv"
OUT = BASE / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-10
DISC_FRAC = 0.70
N_INIT = 20
N_BOOT = 100
MAX_PCA = 20
ORDERS = [2, 3, 4, 5, 6]
HOA_ORDERS = [3, 4, 5, 6]
SURFACES = ["CF", "CB"]

print("=" * 70)
print("SUPPLEMENTARY ANALYSES")
print("=" * 70)

# ─── A) DATA LOADING (same as v4) ─────────────────────────────────
print("\n[A] Loading data...")

df_z = pd.read_csv(CSV / "ZERNIKE-WFA.CSV", sep=";", encoding="latin-1",
                    on_bad_lines="skip", low_memory=False)
df_bad = pd.read_csv(CSV / "BADisplay-LOAD.CSV", sep=";", encoding="latin-1",
                      on_bad_lines="skip", low_memory=False)
df_idx = pd.read_csv(CSV / "INDEX-LOAD.CSV", sep=";", encoding="latin-1",
                      on_bad_lines="skip", low_memory=False)

for d in [df_z, df_bad, df_idx]:
    d.columns = d.columns.str.strip().str.rstrip(":")

_SUBJ_FIELDS = ["Last Name", "First Name", "D.o.Birth"]
_EXAM_FIELDS = ["Exam Eye", "Exam Date", "Exam Time"]

def make_key(df):
    """Build hashed patient and exam identifiers from device-export fields."""
    for c in _SUBJ_FIELDS + _EXAM_FIELDS:
        df[c] = df[c].astype(str).str.strip()
    df["patient_id"] = df[_SUBJ_FIELDS].apply(
        lambda r: hashlib.sha256("_".join(r).encode()).hexdigest()[:16], axis=1)
    df["exam_key"] = df["patient_id"] + "_" + df["Exam Eye"] + "_" + df["Exam Date"] + "_" + df["Exam Time"]

for d in [df_z, df_bad, df_idx]:
    make_key(d)
    d.drop_duplicates(subset="exam_key", keep="first", inplace=True)

keys_all = set(df_z["exam_key"]) & set(df_bad["exam_key"]) & set(df_idx["exam_key"])
df = df_z[df_z["exam_key"].isin(keys_all)].copy()
df["Error"] = pd.to_numeric(df["Error"], errors="coerce").fillna(99)
df = df[df["Error"] == 0].copy()

def parse_nm(col):
    parts = col.split("(")[0].strip().split()
    return int(parts[1]), int(parts[2])

cf_cols_raw = [c for c in df.columns if "(CF)" in c and c.startswith("Z ")]
cb_cols_raw = [c for c in df.columns if "(CB)" in c and c.startswith("Z ")]
cf_info_all = [(c, *parse_nm(c)) for c in cf_cols_raw]
cb_info_all = [(c, *parse_nm(c)) for c in cb_cols_raw]

for c in cf_cols_raw + cb_cols_raw:
    df[c] = pd.to_numeric(df[c], errors="coerce")

cf_full_info = [(c, n, m) for c, n, m in cf_info_all if 2 <= n <= 6]
cb_full_info = [(c, n, m) for c, n, m in cb_info_all if 2 <= n <= 6]

z_ok = (df[[c for c, _, _ in cf_full_info]].notna().all(axis=1) &
        df[[c for c, _, _ in cb_full_info]].notna().all(axis=1))

df_bad_m = df_bad[df_bad["exam_key"].isin(df["exam_key"])][["exam_key", "BAD D"]].copy()
df_bad_m["BAD D"] = pd.to_numeric(df_bad_m["BAD D"], errors="coerce")
df = df.merge(df_bad_m, on="exam_key", how="left")
bad_ok = df["BAD D"].notna()

idx_cols = ["exam_key"]
for pc in ["Thinnest Pachy", "Pachy Min", "D0mm Pachy"]:
    if pc in df_idx.columns:
        idx_cols.append(pc)
        break
for cc in ["ISV", "IVA", "KI", "CKI", "IHA", "IHD", "K Max (Front)"]:
    if cc in df_idx.columns and cc not in idx_cols:
        idx_cols.append(cc)
df_idx_m = df_idx[df_idx["exam_key"].isin(df["exam_key"])][idx_cols].copy()
pachy_col = [c for c in idx_cols if "Pachy" in c][0]
df_idx_m[pachy_col] = pd.to_numeric(df_idx_m[pachy_col], errors="coerce")
df = df.merge(df_idx_m, on="exam_key", how="left", suffixes=("", "_idx"))
pcol = pachy_col + "_idx" if pachy_col + "_idx" in df.columns else pachy_col
cct_ok = df[pcol].notna() & (df[pcol] > 100)

z_ok2 = (df[[c for c, _, _ in cf_full_info]].notna().all(axis=1) &
         df[[c for c, _, _ in cb_full_info]].notna().all(axis=1))
mask_all = z_ok2 & bad_ok & cct_ok
df = df[mask_all].copy()

df = df[df["Exam Eye"].str.contains("Right", na=False)].copy()
df["Exam Date Parsed"] = pd.to_datetime(df["Exam Date"], format="mixed", dayfirst=False, errors="coerce")
df = df.sort_values(["patient_id", "Exam Date Parsed"])
df = df.drop_duplicates(subset="patient_id", keep="first")
df["dob"] = pd.to_datetime(df[_SUBJ_FIELDS[2]], format="mixed", dayfirst=False, errors="coerce")
df["age"] = (df["Exam Date Parsed"] - df["dob"]).dt.days / 365.25
df = df[df["age"] >= 18].copy()
N = len(df)
print(f"  Final cohort: N={N}")

# Clinical columns
clin_map = {}
for cname in ["ISV", "IVA", "KI", "CKI", "IHA", "IHD"]:
    if cname in df.columns:
        df[cname] = pd.to_numeric(df[cname], errors="coerce")
        clin_map[cname] = cname
clin_map["BAD D"] = "BAD D"
for kc in ["K Max (Front)", "K Max"]:
    if kc in df.columns:
        df[kc] = pd.to_numeric(df[kc], errors="coerce")
        clin_map["Kmax"] = kc
        break
clin_map["CCT"] = pcol
df["RMS (CF)"] = pd.to_numeric(df.get("RMS (CF)", ""), errors="coerce")
df["RMS HOA (CF)"] = pd.to_numeric(df.get("RMS HOA (CF)", ""), errors="coerce")
rms_total = df["RMS (CF)"].values.astype(float)
rms_hoa = df["RMS HOA (CF)"].values.astype(float)

# ─── B) BLOCK NORMALIZATION ───────────────────────────────────────
print("\n[B] Block normalization...")

combined_full_info = [(c, n, m, "CF") for c, n, m in cf_full_info] + \
                     [(c, n, m, "CB") for c, n, m in cb_full_info]

def get_block_idx(info_list, surface, order):
    return [i for i, (c, n, m, s) in enumerate(info_list) if s == surface and n == order]

def block_normalize(raw_combined, info_list, surfaces, orders):
    blocks = []
    for surf in surfaces:
        for o in orders:
            idx = get_block_idx(info_list, surf, o)
            if len(idx) == 0: continue
            b = raw_combined[:, idx].copy()
            norms = np.linalg.norm(b, axis=1, keepdims=True)
            b = b / (norms + EPS)
            blocks.append(b)
    cat = np.hstack(blocks)
    gnorm = np.linalg.norm(cat, axis=1, keepdims=True)
    cat = cat / (gnorm + EPS)
    return cat

raw_cf = df[[c for c, _, _ in cf_full_info]].values.astype(float)
raw_cb = df[[c for c, _, _ in cb_full_info]].values.astype(float)
raw_combined = np.hstack([raw_cf, raw_cb])
u_full = block_normalize(raw_combined, combined_full_info, SURFACES, ORDERS)
print(f"  Full shape: {u_full.shape}")

# ─── C) PCA + SPLIT ──────────────────────────────────────────────
print("\n[C] PCA...")
np.random.seed(42)
patients = df["patient_id"].unique()
np.random.shuffle(patients)
n_disc = int(len(patients) * DISC_FRAC)
disc_set = set(patients[:n_disc])
disc_mask = df["patient_id"].isin(disc_set).values
repl_mask = ~disc_mask
print(f"  Discovery: {disc_mask.sum()}, Replication: {repl_mask.sum()}")

ud = u_full[disc_mask]
pca = PCA(n_components=min(ud.shape), random_state=42)
pca.fit(ud)
cv = np.cumsum(pca.explained_variance_ratio_)
k90 = int(np.searchsorted(cv, 0.90)) + 1
npc = min(k90, MAX_PCA)
npc = max(npc, 3)
sc_full = pca.transform(u_full)[:, :npc]
print(f"  Full: {npc} PCs, var={cv[npc-1]:.4f}")

# ─── D) CLUSTERING FUNCTIONS ─────────────────────────────────────
def spherical_kmeans(X, K, n_init=10, max_iter=300, seed=42):
    rng = np.random.RandomState(seed)
    best_inertia, best_labels, best_centers = np.inf, None, None
    Nn, D = X.shape
    for run in range(n_init):
        idx = rng.choice(Nn, K, replace=False)
        centers = X[idx].copy()
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + EPS)
        for _ in range(max_iter):
            sims = X @ centers.T
            labels = sims.argmax(axis=1)
            new_c = np.zeros_like(centers)
            for k in range(K):
                mk = labels == k
                if mk.sum() == 0:
                    new_c[k] = X[rng.randint(Nn)]
                else:
                    new_c[k] = X[mk].mean(axis=0)
            new_c = new_c / (np.linalg.norm(new_c, axis=1, keepdims=True) + EPS)
            if np.max(np.abs(new_c - centers)) < 1e-6:
                break
            centers = new_c
        inertia = np.sum(1.0 - np.einsum("ij,ij->i", X, centers[labels]))
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels.copy(), centers.copy()
    return best_labels, best_centers, best_inertia

def cosine_silhouette(X, labels):
    D = 1 - X @ X.T
    np.fill_diagonal(D, 0)
    D = np.maximum(D, 0)  # clip tiny negatives from float precision
    return silhouette_score(D, labels, metric="precomputed")

def euclidean_silhouette(X, labels):
    return silhouette_score(X, labels, metric="euclidean")

def bootstrap_stability(X, K, n_boot=100, seed=42):
    rng = np.random.RandomState(seed)
    ref_lab, _, _ = spherical_kmeans(X, K, n_init=5, seed=seed)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        blab, _, _ = spherical_kmeans(X[idx], K, n_init=5, seed=seed + b + 1)
        aris.append(adjusted_rand_score(ref_lab[idx], blab))
    return np.mean(aris), np.std(aris)

def severity_pred(labels, rms, K):
    X_s = rms.reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv_obj = StratifiedKFold(5, shuffle=True, random_state=42)
    acc_lr = cross_val_score(lr, X_s, labels, cv=cv_obj, scoring="accuracy").mean()
    return acc_lr

def centroid_similarity(centers):
    K = len(centers)
    sims = []
    for i in range(K):
        for j in range(i + 1, K):
            s = np.dot(centers[i], centers[j]) / (np.linalg.norm(centers[i]) * np.linalg.norm(centers[j]) + EPS)
            sims.append(s)
    return max(sims), np.mean(sims)

# ─── 1) K=2,3,4,5,6 FULL COMPARISON ─────────────────────────────
print("\n[1] K=2,3,4,5,6 comparison...")
K_RANGE = [2, 3, 4, 5, 6]
Xd = sc_full[disc_mask]
rms_d = rms_total[disc_mask]

k_results = {}
rows = []
for K in K_RANGE:
    print(f"  K={K}...", end=" ", flush=True)
    lab, cen, inertia = spherical_kmeans(Xd, K, n_init=N_INIT, seed=42)
    sil = cosine_silhouette(Xd, lab)
    sil_euc = euclidean_silhouette(Xd, lab)
    stab_m, stab_s = bootstrap_stability(Xd, K, N_BOOT, seed=42)
    sev_lr = severity_pred(lab, rms_d, K)
    mx_sim, mn_sim = centroid_similarity(cen)
    sizes = np.bincount(lab, minlength=K)
    min_pct = sizes.min() / len(Xd) * 100
    print(f"sil_cos={sil:.4f} sil_euc={sil_euc:.4f} stab={stab_m:.4f} sev={sev_lr:.4f} maxSim={mx_sim:.4f}")

    k_results[K] = {"lab": lab, "cen": cen, "sil": sil, "sil_euc": sil_euc,
                     "stab": stab_m, "stab_s": stab_s, "sev": sev_lr, "sim": mx_sim}
    rows.append({
        "K": K,
        "sizes": str(list(sizes)),
        "min_cluster_pct": round(min_pct, 1),
        "cosine_silhouette": round(sil, 4),
        "euclidean_silhouette": round(sil_euc, 4),
        "bootstrap_ARI_mean": round(stab_m, 4),
        "bootstrap_ARI_sd": round(stab_s, 4),
        "severity_LR_acc": round(sev_lr, 4),
        "max_centroid_sim": round(mx_sim, 4),
        "mean_centroid_sim": round(mn_sim, 4),
        "chance_level": round(1.0 / K, 4),
    })

df_k = pd.DataFrame(rows)
df_k.to_csv(OUT / "k_comparison_extended.csv", index=False)
print(f"  Saved: k_comparison_extended.csv")

# ─── 2) COMPOSITE WEIGHT SENSITIVITY ─────────────────────────────
print("\n[2] Composite weight sensitivity...")

weight_sets = {
    "Original (0.25/0.30/0.25/0.20)": (0.25, 0.30, 0.25, 0.20),
    "Equal (0.25/0.25/0.25/0.25)": (0.25, 0.25, 0.25, 0.25),
    "Stability-heavy (0.15/0.50/0.15/0.20)": (0.15, 0.50, 0.15, 0.20),
    "Silhouette-heavy (0.40/0.20/0.20/0.20)": (0.40, 0.20, 0.20, 0.20),
    "Severity-heavy (0.20/0.20/0.40/0.20)": (0.20, 0.20, 0.40, 0.20),
}

def norm01(v):
    r = max(v) - min(v)
    return [(x - min(v)) / (r + EPS) for x in v] if r > 0 else [0.5] * len(v)

vals = {m: [k_results[K][m] for K in K_RANGE] for m in ["sil", "stab", "sev", "sim"]}
sn = norm01(vals["sil"])
tn = norm01(vals["stab"])
en = [1 - x for x in norm01(vals["sev"])]
rn = [1 - x for x in norm01(vals["sim"])]

ws_rows = []
for wname, (w_sil, w_stab, w_sev, w_sim) in weight_sets.items():
    comp = [w_sil * s + w_stab * t + w_sev * e + w_sim * r
            for s, t, e, r in zip(sn, tn, en, rn)]
    best_idx = int(np.argmax(comp))
    best_K = K_RANGE[best_idx]
    ws_rows.append({
        "weight_scheme": wname,
        "w_silhouette": w_sil, "w_stability": w_stab,
        "w_severity": w_sev, "w_redundancy": w_sim,
        **{f"composite_K{K}": round(c, 4) for K, c in zip(K_RANGE, comp)},
        "optimal_K": best_K
    })
    print(f"  {wname}: optimal K={best_K} (scores: {[round(c,3) for c in comp]})")

df_ws = pd.DataFrame(ws_rows)
df_ws.to_csv(OUT / "weight_sensitivity.csv", index=False)
print(f"  Saved: weight_sensitivity.csv")

# ─── 3) EFFECT SIZES (eta-squared) ──────────────────────────────
print("\n[3] Effect sizes (Kruskal-Wallis eta-squared)...")

# Full-cohort labels (run on ALL data, not just discovery)
labels_all, centers_all, _ = spherical_kmeans(sc_full, 4, n_init=N_INIT, seed=42)

es_rows = []
for var_name, col_name in clin_map.items():
    vals_arr = pd.to_numeric(df[col_name], errors="coerce").values
    valid = ~np.isnan(vals_arr)
    if valid.sum() < 100:
        continue
    groups = [vals_arr[valid & (labels_all[valid.nonzero()[0]] == k) if False else
              (labels_all == k) & valid] for k in range(4)]
    # Filter empty groups
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        continue
    H_stat, p_val = kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    k_groups = len(groups)
    # eta-squared: (H - k + 1) / (n - k)
    eta_sq = (H_stat - k_groups + 1) / (n_total - k_groups)
    eta_sq = max(0, eta_sq)  # floor at 0
    es_rows.append({
        "variable": var_name,
        "H_statistic": round(H_stat, 2),
        "p_value": f"{p_val:.2e}",
        "n": n_total,
        "eta_squared": round(eta_sq, 4),
        "effect_interpretation": "large" if eta_sq >= 0.14 else ("medium" if eta_sq >= 0.06 else "small")
    })
    print(f"  {var_name}: H={H_stat:.1f}, p={p_val:.2e}, eta2={eta_sq:.4f} ({es_rows[-1]['effect_interpretation']})")

df_es = pd.DataFrame(es_rows)
df_es.to_csv(OUT / "effect_sizes.csv", index=False)
print(f"  Saved: effect_sizes.csv")

# ─── 4) DIAGNOSTIC DISTRIBUTION (BAD-D categories) ───────────────
print("\n[4] BAD-D diagnostic category distribution...")

bad_d_vals = pd.to_numeric(df["BAD D"], errors="coerce").values
categories = np.where(bad_d_vals < 1.6, "Normal (<1.6)",
             np.where(bad_d_vals <= 2.6, "Suspect (1.6-2.6)", "Abnormal (>2.6)"))

dd_rows = []
for k in range(4):
    mask_k = labels_all == k
    cats_k = categories[mask_k]
    n_k = mask_k.sum()
    for cat in ["Normal (<1.6)", "Suspect (1.6-2.6)", "Abnormal (>2.6)"]:
        n_cat = (cats_k == cat).sum()
        dd_rows.append({
            "phenotype": f"P{k}",
            "BAD_D_category": cat,
            "n": int(n_cat),
            "pct_within_phenotype": round(n_cat / n_k * 100, 1) if n_k > 0 else 0,
            "pct_within_category": round(n_cat / (categories == cat).sum() * 100, 1) if (categories == cat).sum() > 0 else 0
        })

# Overall distribution
for cat in ["Normal (<1.6)", "Suspect (1.6-2.6)", "Abnormal (>2.6)"]:
    n_cat = (categories == cat).sum()
    dd_rows.append({
        "phenotype": "Total",
        "BAD_D_category": cat,
        "n": int(n_cat),
        "pct_within_phenotype": round(n_cat / N * 100, 1),
        "pct_within_category": 100.0
    })

df_dd = pd.DataFrame(dd_rows)
df_dd.to_csv(OUT / "diagnostic_distribution.csv", index=False)
print(f"  Saved: diagnostic_distribution.csv")

# Print crosstab
print("\n  Crosstab (% within phenotype):")
for k in range(4):
    mask_k = labels_all == k
    cats_k = categories[mask_k]
    n_norm = (cats_k == "Normal (<1.6)").sum()
    n_susp = (cats_k == "Suspect (1.6-2.6)").sum()
    n_abn = (cats_k == "Abnormal (>2.6)").sum()
    n_k = mask_k.sum()
    print(f"  P{k} (n={n_k}): Normal={n_norm/n_k*100:.1f}% Suspect={n_susp/n_k*100:.1f}% Abnormal={n_abn/n_k*100:.1f}%")

# ─── 5) DISCOVERY-REPLICATION CENTROID CONCORDANCE ────────────────
print("\n[5] Discovery-replication centroid concordance...")

Xd_disc = sc_full[disc_mask]
Xd_repl = sc_full[repl_mask]

lab_disc, cen_disc, _ = spherical_kmeans(Xd_disc, 4, n_init=N_INIT, seed=42)
lab_repl_pred = (Xd_repl @ cen_disc.T).argmax(axis=1)  # assign replication to discovery centroids

# Also cluster replication independently
lab_repl_ind, cen_repl, _ = spherical_kmeans(Xd_repl, 4, n_init=N_INIT, seed=42)

# Centroid cosine similarity matrix
cos_sim_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        cos_sim_matrix[i, j] = np.dot(cen_disc[i], cen_repl[j]) / \
                                (np.linalg.norm(cen_disc[i]) * np.linalg.norm(cen_repl[j]) + EPS)

# Hungarian matching (greedy for 4x4)
from scipy.optimize import linear_sum_assignment
cost = 1 - cos_sim_matrix
row_ind, col_ind = linear_sum_assignment(cost)
matched_sims = [cos_sim_matrix[r, c] for r, c in zip(row_ind, col_ind)]
mean_concordance = np.mean(matched_sims)

print(f"  Centroid similarity matrix (disc x repl):")
for i in range(4):
    print(f"    Disc P{i}: [{', '.join(f'{cos_sim_matrix[i,j]:.4f}' for j in range(4))}]")
print(f"  Matched pairs: {list(zip(row_ind, col_ind))}")
print(f"  Matched similarities: {[round(s, 4) for s in matched_sims]}")
print(f"  Mean concordance: {mean_concordance:.4f}")

# ARI between projected and independent labels
ari_proj = adjusted_rand_score(lab_repl_pred, lab_repl_ind)
print(f"  ARI (projected vs independent): {ari_proj:.4f}")

cc_results = {
    "centroid_similarity_matrix": cos_sim_matrix.tolist(),
    "matched_pairs_disc_to_repl": list(zip(row_ind.tolist(), col_ind.tolist())),
    "matched_similarities": [round(s, 4) for s in matched_sims],
    "mean_concordance": round(mean_concordance, 4),
    "ARI_projected_vs_independent": round(ari_proj, 4)
}

with open(OUT / "centroid_concordance.json", "w") as f:
    json.dump(cc_results, f, indent=2)
print(f"  Saved: centroid_concordance.json")

# ─── 6) SILHOUETTE COMPARISON ────────────────────────────────────
print("\n[6] Cosine vs Euclidean silhouette comparison...")

sil_rows = []
for K in K_RANGE:
    sil_rows.append({
        "K": K,
        "cosine_silhouette": k_results[K]["sil"],
        "euclidean_silhouette": k_results[K]["sil_euc"],
        "difference": round(k_results[K]["sil"] - k_results[K]["sil_euc"], 4)
    })
    print(f"  K={K}: cosine={k_results[K]['sil']:.4f}, euclidean={k_results[K]['sil_euc']:.4f}")

df_sil = pd.DataFrame(sil_rows)
df_sil.to_csv(OUT / "silhouette_comparison.csv", index=False)
print(f"  Saved: silhouette_comparison.csv")

# ─── SUMMARY ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUPPLEMENTARY ANALYSES COMPLETE")
print("=" * 70)
print(f"Output files in: {OUT}")
print("  1. k_comparison_extended.csv")
print("  2. weight_sensitivity.csv")
print("  3. effect_sizes.csv")
print("  4. diagnostic_distribution.csv")
print("  5. centroid_concordance.json")
print("  6. silhouette_comparison.csv")
