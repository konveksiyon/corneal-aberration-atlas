import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_BASE = ROOT
CSV_DIR = ANALYSIS_BASE / "csv"
OUTPUT_DIR = ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PROVENANCE_DIR = OUTPUT_DIR / "provenance"
PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PATH = FIGURE_DIR / "Figure4_severity_independence.png"
METRICS_PATH = PROVENANCE_DIR / "reviewer_round2_metrics.json"
MANIFEST_PATH = PROVENANCE_DIR / "reviewer_round2_manifest.md"

DISC_FRAC = 0.70
MAX_PCA = 20
EPS = 1e-10
ORDERS = [2, 3, 4, 5, 6]
SURFACES = ["CF", "CB"]


_SUBJ_FIELDS = ["Last Name", "First Name", "D.o.Birth"]
_EXAM_FIELDS = ["Exam Eye", "Exam Date", "Exam Time"]

def make_key(df: pd.DataFrame) -> None:
    """Build hashed patient and exam identifiers from device-export fields."""
    df.columns = df.columns.str.strip().str.rstrip(":")
    for c in _SUBJ_FIELDS + _EXAM_FIELDS:
        df[c] = df[c].astype(str).str.strip()
    df["patient_id"] = df[_SUBJ_FIELDS].apply(
        lambda r: hashlib.sha256("_".join(r).encode()).hexdigest()[:16], axis=1)
    df["exam_key"] = (
        df["patient_id"]
        + "_"
        + df["Exam Eye"]
        + "_"
        + df["Exam Date"]
        + "_"
        + df["Exam Time"]
    )


def parse_nm(col: str) -> tuple[int, int]:
    parts = col.split("(")[0].strip().split()
    return int(parts[1]), int(parts[2])


def spherical_kmeans(
    x: np.ndarray,
    k: int,
    n_init: int = 20,
    max_iter: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    best_inertia = np.inf
    best_labels = None
    best_centers = None
    for run in range(n_init):
        rng = np.random.RandomState(seed + run)
        centers = x[rng.choice(len(x), k, replace=False)].copy()
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + EPS
        for _ in range(max_iter):
            sims = x @ centers.T
            labels = sims.argmax(axis=1)
            new_centers = np.zeros_like(centers)
            for idx in range(k):
                members = x[labels == idx]
                new_centers[idx] = members.mean(0) if len(members) else centers[idx]
            new_centers /= np.linalg.norm(new_centers, axis=1, keepdims=True) + EPS
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers
        sims = np.array([x[i] @ centers[labels[i]] for i in range(len(x))])
        inertia = np.sum(1 - sims)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers


def order_clusters(labels: np.ndarray, centers: np.ndarray, bad_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort([np.nanmedian(bad_d[labels == idx]) for idx in range(len(centers))])
    relabeled = np.zeros_like(labels)
    reordered = np.zeros_like(centers)
    for new_idx, old_idx in enumerate(order):
        relabeled[labels == old_idx] = new_idx
        reordered[new_idx] = centers[old_idx]
    return relabeled, reordered


def logistic_metrics(labels: np.ndarray, rms: np.ndarray) -> dict[str, float]:
    x = rms.reshape(-1, 1)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    y_pred = cross_val_predict(clf, x, labels, cv=cv, method="predict")
    y_proba = cross_val_predict(clf, x, labels, cv=cv, method="predict_proba")
    metrics = {
        "accuracy": accuracy_score(labels, y_pred),
        "balanced_accuracy": balanced_accuracy_score(labels, y_pred),
        "macro_f1": f1_score(labels, y_pred, average="macro"),
    }
    if len(np.unique(labels)) == 2:
        metrics["auc"] = roc_auc_score(labels, y_proba[:, 1])
    else:
        metrics["auc"] = roc_auc_score(y_true=labels, y_score=y_proba, multi_class="ovr", average="macro")
    return {key: float(value) for key, value in metrics.items()}


def block_normalize(
    raw_combined: np.ndarray,
    info_list: list[tuple[str, int, int, str]],
    surfaces: list[str],
    orders: list[int],
) -> np.ndarray:
    blocks = []
    for surface in surfaces:
        for order in orders:
            idx = [i for i, (_, n, _, s) in enumerate(info_list) if s == surface and n == order]
            if not idx:
                continue
            block = raw_combined[:, idx].copy()
            block /= np.linalg.norm(block, axis=1, keepdims=True) + EPS
            blocks.append(block)
    cat = np.hstack(blocks)
    return cat / (np.linalg.norm(cat, axis=1, keepdims=True) + EPS)


def load_analysis_tables() -> tuple[pd.DataFrame, dict[str, int]]:
    df_z = pd.read_csv(CSV_DIR / "ZERNIKE-WFA.CSV", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    df_bad = pd.read_csv(CSV_DIR / "BADisplay-LOAD.CSV", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    df_idx = pd.read_csv(CSV_DIR / "INDEX-LOAD.CSV", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)

    for table in [df_z, df_bad, df_idx]:
        make_key(table)
        table.drop_duplicates(subset="exam_key", keep="first", inplace=True)

    keys_all = set(df_z["exam_key"]) & set(df_bad["exam_key"]) & set(df_idx["exam_key"])
    df = df_z[df_z["exam_key"].isin(keys_all)].copy()
    counts: dict[str, int] = {"matched": len(df)}

    df["Error"] = pd.to_numeric(df["Error"], errors="coerce").fillna(99)
    before = len(df)
    df = df[df["Error"] == 0].copy()
    counts["error_excluded"] = before - len(df)

    cf_cols_raw = [c for c in df.columns if "(CF)" in c and c.startswith("Z ")]
    cb_cols_raw = [c for c in df.columns if "(CB)" in c and c.startswith("Z ")]
    cf_info_all = [(c, *parse_nm(c)) for c in cf_cols_raw]
    cb_info_all = [(c, *parse_nm(c)) for c in cb_cols_raw]

    for col in cf_cols_raw + cb_cols_raw:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cf_full_info = [(c, n, m) for c, n, m in cf_info_all if 2 <= n <= 6]
    cb_full_info = [(c, n, m) for c, n, m in cb_info_all if 2 <= n <= 6]
    z_cf_ok = df[[c for c, _, _ in cf_full_info]].notna().all(axis=1)
    z_cb_ok = df[[c for c, _, _ in cb_full_info]].notna().all(axis=1)
    z_ok = z_cf_ok & z_cb_ok
    counts["missing_zernike_excluded"] = int((~z_ok).sum())

    df_bad_m = df_bad[df_bad["exam_key"].isin(df["exam_key"])][["exam_key", "BAD D"]].copy()
    df_bad_m["BAD D"] = pd.to_numeric(df_bad_m["BAD D"], errors="coerce")
    df = df.merge(df_bad_m, on="exam_key", how="left")
    bad_ok = df["BAD D"].notna()

    idx_cols = ["exam_key"]
    for pachy in ["Thinnest Pachy", "Pachy Min", "D0mm Pachy"]:
        if pachy in df_idx.columns:
            idx_cols.append(pachy)
            break
    for col in ["ISV", "IVA", "KI", "CKI", "IHA", "IHD", "K Max (Front)"]:
        if col in df_idx.columns and col not in idx_cols:
            idx_cols.append(col)
    df_idx_m = df_idx[df_idx["exam_key"].isin(df["exam_key"])][idx_cols].copy()
    pachy_col = [c for c in idx_cols if "Pachy" in c][0]
    df_idx_m[pachy_col] = pd.to_numeric(df_idx_m[pachy_col], errors="coerce")
    df = df.merge(df_idx_m, on="exam_key", how="left", suffixes=("", "_idx"))
    pcol = pachy_col + "_idx" if pachy_col + "_idx" in df.columns else pachy_col
    df["CCT"] = pd.to_numeric(df[pcol], errors="coerce")
    cct_ok = df["CCT"].notna() & (df["CCT"] > 100)

    z_cf_ok = df[[c for c, _, _ in cf_full_info]].notna().all(axis=1)
    z_cb_ok = df[[c for c, _, _ in cb_full_info]].notna().all(axis=1)
    z_ok = z_cf_ok & z_cb_ok
    counts["missing_bad_d_excluded"] = int((z_ok & ~bad_ok).sum())
    counts["invalid_cct_excluded"] = int((z_ok & bad_ok & ~cct_ok).sum())

    df = df[z_ok & bad_ok & cct_ok].copy()

    df["Exam Date Parsed"] = pd.to_datetime(df["Exam Date"], format="mixed", errors="coerce")
    before = len(df)
    df = df[df["Exam Eye"].str.contains("Right", na=False)].copy()
    df = df.sort_values(["patient_id", "Exam Date Parsed"]).drop_duplicates(subset="patient_id", keep="first").copy()
    counts["eye_or_repeat_excluded"] = before - len(df)

    df["dob"] = pd.to_datetime(df[_SUBJ_FIELDS[2]], format="mixed", errors="coerce")
    df["age"] = (df["Exam Date Parsed"] - df["dob"]).dt.days / 365.25
    before = len(df)
    df = df[df["age"] >= 18].copy()
    counts["underage_excluded"] = before - len(df)
    counts["final_n"] = len(df)

    for col in ["RMS (CF)", "RMS HOA (CF)", "K Max (Front)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, counts


def prepare_feature_matrices(df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, int, int, str]], np.ndarray, np.ndarray]:
    cf_cols_raw = [c for c in df.columns if "(CF)" in c and c.startswith("Z ")]
    cb_cols_raw = [c for c in df.columns if "(CB)" in c and c.startswith("Z ")]
    cf_info = [(c, *parse_nm(c)) for c in cf_cols_raw if 2 <= parse_nm(c)[0] <= 6]
    cb_info = [(c, *parse_nm(c)) for c in cb_cols_raw if 2 <= parse_nm(c)[0] <= 6]
    combined_info = [(c, n, m, "CF") for c, n, m in cf_info] + [(c, n, m, "CB") for c, n, m in cb_info]
    raw_full = np.hstack(
        [
            df[[c for c, _, _ in cf_info]].values.astype(float),
            df[[c for c, _, _ in cb_info]].values.astype(float),
        ]
    )
    u_full = block_normalize(raw_full, combined_info, SURFACES, ORDERS)
    rms_total = df["RMS (CF)"].values.astype(float)
    rms_hoa = df["RMS HOA (CF)"].values.astype(float)
    return u_full, combined_info, rms_total, rms_hoa


def build_discovery_pca(
    df: pd.DataFrame,
    u_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    patients = df["patient_id"].unique()
    np.random.seed(42)
    np.random.shuffle(patients)
    n_disc = int(len(patients) * DISC_FRAC)
    disc_set = set(patients[:n_disc])
    disc_mask = df["patient_id"].isin(disc_set).values
    repl_mask = ~disc_mask

    pca = PCA(n_components=min(u_full[disc_mask].shape), random_state=42)
    pca.fit(u_full[disc_mask])
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    npc = min(int(np.searchsorted(cumulative_var, 0.90)) + 1, MAX_PCA)
    npc = max(npc, 3)
    scores = pca.transform(u_full)[:, :npc]

    meta = {
        "discovery_n": int(disc_mask.sum()),
        "replication_n": int(repl_mask.sum()),
        "npc": int(npc),
        "variance_retained": float(cumulative_var[npc - 1]),
    }
    return scores, disc_mask, repl_mask, meta


def compute_reproducibility(
    disc_scores: np.ndarray,
    repl_scores: np.ndarray,
    disc_bad_d: np.ndarray,
    k: int,
) -> dict[str, float]:
    labels_disc, centers_disc = spherical_kmeans(disc_scores, k)
    labels_disc, centers_disc = order_clusters(labels_disc, centers_disc, disc_bad_d)

    projected_labels = (repl_scores @ centers_disc.T).argmax(axis=1)
    labels_repl, centers_repl = spherical_kmeans(repl_scores, k)

    sim = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            sim[i, j] = np.dot(centers_disc[i], centers_repl[j]) / (
                np.linalg.norm(centers_disc[i]) * np.linalg.norm(centers_repl[j]) + EPS
            )

    row_ind, col_ind = linear_sum_assignment(1 - sim)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    labels_repl_matched = np.vectorize(mapping.get)(labels_repl)
    matched = [sim[row, col] for row, col in zip(row_ind, col_ind)]
    return {
        "mean_centroid_cosine_similarity": float(np.mean(matched)),
        "ari_projected_vs_independent": float(adjusted_rand_score(projected_labels, labels_repl_matched)),
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_figure(
    scores: np.ndarray,
    disc_mask: np.ndarray,
    df: pd.DataFrame,
    rms_total: np.ndarray,
    rms_hoa: np.ndarray,
    metrics_by_k: dict[int, dict[str, dict[str, float]]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    palette = ["#4E79A7", "#F28E2B", "#E15759"]

    for row, k in enumerate([2, 3]):
        x = scores[disc_mask]
        bad_d = df.loc[disc_mask, "BAD D"].values
        labels, centers = spherical_kmeans(x, k)
        labels, centers = order_clusters(labels, centers, bad_d)
        row_rms_total = rms_total[disc_mask]
        row_rms_hoa = rms_hoa[disc_mask]

        total_groups = [row_rms_total[labels == idx] for idx in range(k)]
        hoa_groups = [row_rms_hoa[labels == idx] for idx in range(k)]

        ax = axes[row, 0]
        bp = ax.boxplot(total_groups, patch_artist=True, showfliers=False)
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(palette[idx])
            patch.set_alpha(0.80)
        ax.set_title(f"K = {k}: Total RMS", fontsize=11)
        ax.set_ylabel("RMS (um)")
        ax.set_xticklabels([f"C{idx + 1}" for idx in range(k)])

        ax = axes[row, 1]
        bp = ax.boxplot(hoa_groups, patch_artist=True, showfliers=False)
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(palette[idx])
            patch.set_alpha(0.80)
        ax.set_title(f"K = {k}: HOA RMS", fontsize=11)
        ax.set_ylabel("RMS (um)")
        ax.set_xticklabels([f"C{idx + 1}" for idx in range(k)])

        ax = axes[row, 2]
        bar_labels = ["Total RMS", "HOA RMS"]
        acc_values = [
            metrics_by_k[k]["total_rms"]["accuracy"],
            metrics_by_k[k]["hoa_rms"]["accuracy"],
        ]
        bars = ax.bar(bar_labels, acc_values, color=["#4E79A7", "#59A14F"], alpha=0.85)
        ax.axhline(1.0 / k, color="#D62728", linestyle="--", linewidth=1.0)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"K = {k}: LR Accuracy", fontsize=11)
        for bar, value in zip(bars, acc_values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    fig.suptitle("Figure 4. Discovery-set severity dependence using logistic regression only", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURE_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_manifest(payload: dict) -> None:
    k2 = payload["severity_metrics"]["K2"]["total_rms"]
    k3 = payload["severity_metrics"]["K3"]["total_rms"]
    r2 = payload["reproducibility"]["K2"]
    r3 = payload["reproducibility"]["K3"]
    counts = payload["cohort_counts"]
    lines = [
        "# Reviewer Round 2 Assets",
        "",
        f"- Final cohort: {counts['final_n']} eyes after excluding {counts['missing_bad_d_excluded']} eyes with missing BAD-D.",
        f"- Discovery/replication split: {payload['pca']['discovery_n']} / {payload['pca']['replication_n']}.",
        f"- PCA retained {payload['pca']['npc']} components ({payload['pca']['variance_retained'] * 100:.2f}% variance).",
        f"- K=2 discovery LR metrics from total RMS: accuracy {format_pct(k2['accuracy'])}, balanced accuracy {format_pct(k2['balanced_accuracy'])}, macro-F1 {k2['macro_f1']:.3f}, AUC {k2['auc']:.3f}.",
        f"- K=3 discovery LR metrics from total RMS: accuracy {format_pct(k3['accuracy'])}, balanced accuracy {format_pct(k3['balanced_accuracy'])}, macro-F1 {k3['macro_f1']:.3f}, AUC {k3['auc']:.3f}.",
        f"- K=2 split-sample concordance: cosine similarity {r2['mean_centroid_cosine_similarity']:.3f}, ARI {r2['ari_projected_vs_independent']:.3f}.",
        f"- K=3 split-sample concordance: cosine similarity {r3['mean_centroid_cosine_similarity']:.3f}, ARI {r3['ari_projected_vs_independent']:.3f}.",
        f"- Updated LR-only Figure 4 written to: {FIGURE_PATH}",
    ]
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df, counts = load_analysis_tables()
    u_full, _, rms_total, rms_hoa = prepare_feature_matrices(df)
    scores, disc_mask, repl_mask, pca_meta = build_discovery_pca(df, u_full)

    severity_metrics = {
        "K2": {
            "total_rms": logistic_metrics(order_clusters(*spherical_kmeans(scores[disc_mask], 2), df.loc[disc_mask, "BAD D"].values)[0], rms_total[disc_mask]),
            "hoa_rms": logistic_metrics(order_clusters(*spherical_kmeans(scores[disc_mask], 2), df.loc[disc_mask, "BAD D"].values)[0], rms_hoa[disc_mask]),
        },
        "K3": {
            "total_rms": logistic_metrics(order_clusters(*spherical_kmeans(scores[disc_mask], 3), df.loc[disc_mask, "BAD D"].values)[0], rms_total[disc_mask]),
            "hoa_rms": logistic_metrics(order_clusters(*spherical_kmeans(scores[disc_mask], 3), df.loc[disc_mask, "BAD D"].values)[0], rms_hoa[disc_mask]),
        },
    }

    reproducibility = {
        "K2": compute_reproducibility(scores[disc_mask], scores[repl_mask], df.loc[disc_mask, "BAD D"].values, 2),
        "K3": compute_reproducibility(scores[disc_mask], scores[repl_mask], df.loc[disc_mask, "BAD D"].values, 3),
    }

    figure_metrics = {
        2: {
            "total_rms": severity_metrics["K2"]["total_rms"],
            "hoa_rms": severity_metrics["K2"]["hoa_rms"],
        },
        3: {
            "total_rms": severity_metrics["K3"]["total_rms"],
            "hoa_rms": severity_metrics["K3"]["hoa_rms"],
        },
    }
    generate_figure(scores, disc_mask, df, rms_total, rms_hoa, figure_metrics)

    payload = {
        "analysis_base": str(ANALYSIS_BASE),
        "cohort_counts": counts,
        "pca": pca_meta,
        "severity_metrics": severity_metrics,
        "reproducibility": reproducibility,
        "figure_path": str(FIGURE_PATH),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_manifest(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
