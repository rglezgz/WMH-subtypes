# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:18:56 2026

@author: rglez
"""

# -*- coding: utf-8 -*-
"""
WMH PIPELINE COMPLETO (ARREGLADO)

Includes EVERYTHING in a single integrated pipeline:
1.	Excel loading and integrity checks
2.	GLM/OLS residualization applied ONLY to clustering features (excluding Coord-x/y/z)
3.	Z-score normalization
4.	Automatic GMM clustering (optimal k selected by BIC on subset) + FULL dataset labels + maximum posterior probability (probmax)
5.	Main Excel output including raw data, cluster summaries, and statistical tests
6.	Visualization outputs: (BIC curve, PCA 3D, PCA 2D)
7.	Stacked percentage frequency plots by diagnostic group (Gpo)
8.	Interactive 3D HTML visualization using Plotly
9.	Advanced group statistics: Global chi-square test, Cramér’s V effect size, Pairwise post-hoc comparisons, Standardized residuals per cell
10.	Bubble matrix visualization displaying significant pairwise comparisons after FDR correction
11.	Correlation analyses:
•	Spearman lower-triangle correlation matrices
•	Computed for ALL subjects and stratified by cluster
•	Optional partial correlations after residualization by covariates
12.	Machine learning classification framework:
•	Repeated cross-validation
•	Out-of-fold ROC curves
•	Feature importance analysis
•	Robust SHAP explainability analysis
13.	Interactive visualization support for exploratory and presentation purposes
14.	Multiple-comparison correction procedures including FDR and Bonferroni adjustments
15.	Cluster confidence estimation using maximum posterior probability (probmax)
16.	Group-balanced statistical evaluation across diagnostic categories
17.	Support for repeated measures and subject-level dependency correction
18.	Robust multiclass LightGBM classification with explainability analyses (feature importance and SHAP)
19.	Fully reproducible end-to-end framework for multimodal WMH clustering, statistical characterization, visualization, correlation analysis, and predictive modeling

"""

import os
import re
import warnings
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mpl_colors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.stats import kruskal, chi2_contingency, spearmanr, norm as normal_dist
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
import plotly.io as pio
from scipy.optimize import linear_sum_assignment

# ---------------- LightGBM + SHAP ----------------
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise ImportError("No puedo importar lightgbm. Instala con: pip install lightgbm") from e

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False
    warnings.warn("No se pudo importar shap. SHAP se omitirá. Instala con: pip install shap")


# ======================================================================================
# ====================================== CONFIG ========================================
# ======================================================================================
excel_path = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\Ampliacion\WMH_clusters_metrics-all_with_clinic_final.xlsx"
out_dir = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\Ampliacion\clustering"
os.makedirs(out_dir, exist_ok=True)

base_name = os.path.splitext(os.path.basename(excel_path))[0]
excel_out_main = os.path.join(out_dir, f"{base_name}_CLUSTERING_GLM.xlsx")

# Features para clustering (GLM SOLO a estas, excepto NO_GLM_COLS)
requested_cols = [
    "T1", "GFA", "QA", "ISO", "ha", "ad", "fa", "rd", "rd1", "rd2",
    "fALFF", "Hurst", "Entropy",
    "Power slope", "Autocor",
    "WMH",
    "Coord-x", "Coord-y", "Coord-z",
]

glm_covars = ["Age", "Sex", "WMH_t0", "Education", "Size", "TIV", "Site"]

# Comparaciones post-cluster (sin corrección)
compare_vars = ["Sex", "Age", "Coord-x", "Coord-y", "Coord-z", "∆WMH", "Gpo", "Size"]

# NO residualizar coords
NO_GLM_COLS = ["Coord-x", "Coord-y", "Coord-z"]

# Clustering speed
FIT_SUBSET_N = 5000
K_RANGE = list(range(2, 11))
SIL_SAMPLE = 2500
DROP_NA_ROWS = False  # si True: drop NaNs en features/covars; si False: error.

# Correlaciones
DO_CORR_BLOCKS = True
vars_corr = [
    "WMH", "T1", "GFA", "QA", "ISO", "ha", "ad", "fa", "rd", "rd1", "rd2",
    "fALFF", "Hurst", "Entropy", "Power slope", "Autocor",
]
DO_PARTIAL = True  # residualiza vars_corr por covars
CORR_COVARS = glm_covars[:]  # usa las mismas covars
CORR_ALPHA = 0.05
CORR_USE_FDR = True
CORR_FDR_METHOD = "fdr_bh"
CORR_VMIN, CORR_VMAX = -0.5, 0.5
CORR_CMAP = "bwr"

CLUSTER_COL = "cluster_gmm_auto"

# ML clasificación
DO_ML = True
TARGET = "cluster_gmm_auto"
GPO_COL = "Gpo"           # solo reportes
SUBJ_COL = "subject_id"   # si no existe, se crea desde el índice

FEATURES = [
    "T1", "GFA", "QA", "ISO", "ha", "ad", "fa", "rd", "rd1", "rd2",
    "fALFF", "Hurst", "Entropy", "Power slope", "Autocor",
    "WMH",
    "Coord-x", "Coord-y", "Coord-z",
    "Age", "Sex", "WMH_t0", "Size", "Education", "TIV",
]

N_SPLITS = 5
N_REPEATS = 20
BASE_SEED = 42
RANDOM_STATE = 42
N_BOOT = 1000
GRID_N = 200
SEED = 42
DO_ROC_BY_GPO = True
DO_FEATURE_IMPORTANCE_AND_SHAP = True
MAX_SHAP_SAMPLES = 1500
SAVE_SHAP_PER_CLASS = True


# ======================================================================================
# ===================================== HELPERS =======================================
# ======================================================================================
def pick_subset(n, subset_n, seed=42):
    rng = np.random.default_rng(seed)
    if n <= subset_n:
        return np.arange(n)
    return rng.choice(n, size=subset_n, replace=False)

def safe_silhouette(X, labels, sample_size=2500, seed=42):
    labels = np.asarray(labels)
    if len(set(labels)) < 2:
        return np.nan
    ss = min(sample_size, len(labels))
    return float(silhouette_score(X, labels, sample_size=ss, random_state=seed))

def plot_curve(x, y, title, xlabel, ylabel, out_png, out_pdf=None):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    # PDF (opcional)
    if out_pdf is not None:
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def to_subscript(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

def lab_text(lab: int) -> str:
    return f"L{to_subscript(int(lab) + 1)}"

def make_cluster_color_map(labels_or_uniq, cmap_name="viridis",
                           pos_violet=0.10, pos_green=0.55, pos_yellow=0.95):
    """
    Fuerza:
      cluster 0 (L1) -> violeta
      cluster 1 (L2) -> verde
      cluster 2 (L3) -> amarillo
    """
    uniq = np.array(sorted(np.unique(np.asarray(labels_or_uniq).astype(int))))
    cmap = cm.get_cmap(cmap_name)
    pos_map = {0: pos_violet, 1: pos_green, 2: pos_yellow}

    # fallback si hubiera más clusters:
    default_positions = np.linspace(pos_violet, pos_yellow, len(uniq))
    out = {}
    for i, lab in enumerate(uniq):
        pos = pos_map.get(int(lab), float(default_positions[i]))
        out[int(lab)] = cmap(pos)
    return out

def rgba_to_plotly_rgba(rgba_tuple):
    r, g, b, a = rgba_tuple
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})"

def chi2_test(df, a, b):
    tab = pd.crosstab(df[a], df[b], dropna=False)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2, p, dof, _ = chi2_contingency(tab, correction=False)
    return float(chi2), float(p), int(dof)

def cluster_summary_table(df_in, label_col, compare_vars):
    df = df_in.copy()
    out = []

    numeric_cols = [c for c in compare_vars
                    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    for cl in sorted(df[label_col].dropna().unique()):
        sub = df[df[label_col] == cl]
        n = len(sub)

        row = {"cluster": int(cl), "cluster_L": lab_text(int(cl)), "N": int(n)}

        if "Sex" in compare_vars and "Sex" in sub.columns:
            nF = int((sub["Sex"] == "F").sum())
            nM = int((sub["Sex"] == "M").sum())
            row.update({
                "Sex_F_n": nF,
                "Sex_M_n": nM,
                "Sex_F_%": (100.0 * nF / n) if n else np.nan,
                "Sex_M_%": (100.0 * nM / n) if n else np.nan,
            })

        for c in numeric_cols:
            row[f"{c}_mean"] = float(np.mean(sub[c]))
            row[f"{c}_sd"] = float(np.std(sub[c], ddof=1)) if n > 1 else np.nan

        out.append(row)

    return pd.DataFrame(out)

def run_tests_original(df_in, label_col, compare_vars):
    df = df_in.copy()
    results = []

    for cat in ["Sex", "Gpo"]:
        if cat in compare_vars and cat in df.columns:
            out = chi2_test(df, label_col, cat)
            if out is not None:
                chi2, p, dof = out
                results.append({
                    "label_col": label_col,
                    "variable": cat,
                    "test": "Chi-square",
                    "stat": chi2,
                    "df": dof,
                    "p_value": p
                })

    numeric_cols = [c for c in compare_vars
                    if c in df.columns and c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    clusters = sorted(df[label_col].dropna().unique())

    if len(clusters) >= 2:
        for v in numeric_cols:
            groups = [df.loc[df[label_col] == k, v].to_numpy(dtype=float) for k in clusters]
            nonempty = [g for g in groups if len(g) > 0]
            if len(nonempty) >= 2:
                H, p = kruskal(*nonempty)
                results.append({
                    "label_col": label_col,
                    "variable": v,
                    "test": "Kruskal-Wallis",
                    "stat": float(H),
                    "df": int(len(nonempty) - 1),
                    "p_value": float(p)
                })

    res = pd.DataFrame(results)
    if not res.empty:
        res["p_fdr"] = multipletests(res["p_value"].values, method="fdr_bh")[1]
        res["p_bonf"] = multipletests(res["p_value"].values, method="bonferroni")[1]
        res = res.sort_values(["p_fdr", "p_value"])
    return res

def freq_by_gpo(df_in, label_col, gpo_col="Gpo"):
    if (label_col not in df_in.columns) or (gpo_col not in df_in.columns):
        return pd.DataFrame()
    return pd.crosstab(df_in[label_col], df_in[gpo_col], dropna=False)

def plot_freq_by_gpo_stacked(df_out, label_col, gpo_col, cluster_color,
                             out_png, out_svg=None, out_pdf=None,
                             bar_width=0.40, fig_w_per_group=0.90, fig_h=5.0):
    tab = pd.crosstab(df_out[label_col], df_out[gpo_col], dropna=False)
    if tab.shape[0] == 0 or tab.shape[1] == 0:
        raise ValueError("Tabla de contingencia vacía para frecuencias (revisa Gpo / clusters).")

    tab = tab.reindex([0, 1, 2], fill_value=0)

    desired_order = ["CN", "MCI", "AD", "PD"]
    present = [g for g in desired_order if g in tab.columns]
    others = [g for g in tab.columns if g not in present]
    tab = tab[present + others]

    col_sums = tab.sum(axis=0)
    tab_pct = tab.div(col_sums.replace(0, np.nan), axis=1) * 100

    if np.isfinite(tab_pct.to_numpy()).sum() == 0:
        raise ValueError("tab_pct quedó todo NaN (posible grupo con sum=0 o datos inválidos).")

    fig_w = max(3.0, fig_w_per_group * len(tab_pct.columns) + 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bottom = np.zeros(len(tab_pct.columns), dtype=float)
    x = np.arange(len(tab_pct.columns))  # posiciones

    # ✅ apilar en orden ASC para que L1 quede abajo
    for lab in [2, 1, 0]:
        if lab not in tab_pct.index:
            continue
        vals = tab_pct.loc[lab].to_numpy(dtype=float)
        ax.bar(
            x, vals, bottom=bottom, width=bar_width,
            color=cluster_color[int(lab)],
            label=lab_text(int(lab))
        )
        bottom += np.nan_to_num(vals, nan=0.0)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Frequency (% within group)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tab_pct.columns, rotation=0, fontsize=16)
    ax.tick_params(axis="y", labelsize=12)

    handles, labels_legend = ax.get_legend_handles_labels()
    ax.legend(handles, labels_legend, title="Clusters",
              bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_svg is not None:
        fig.savefig(out_svg, format="svg", bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print("[DONE] Frequency plot saved:", out_png)
    return tab, tab_pct

def encode_sex(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    sex_map = {"F": 0, "M": 1, "f": 0, "m": 1}
    out = series.map(sex_map)
    if out.isna().any():
        bad = series[out.isna()].unique()
        raise ValueError(f"Unexpected Sex values: {bad}. Expected F/M.")
    return out.astype(float)

def residualize_matrix(Y, cov_df):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    C = cov_df.to_numpy(dtype=float)
    Xcov = np.column_stack([np.ones(len(cov_df)), C])
    B = np.linalg.lstsq(Xcov, Y, rcond=None)[0]
    return Y - (Xcov @ B)

def stars_from_p(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ======================================================================================
# ====================================== 1) LOAD ======================================
# ======================================================================================
print(f"[INFO] Reading Excel: {excel_path}")
df = pd.read_excel(excel_path)
df["WMH"] = df["∆WMH"] / df["Follow-up time"]
df["T1"] = df["∆T1"] / df["Follow-up time"]
df["QA"] = df["∆QA"] / df["Follow-up time"]
df["GFA"] = df["∆GFA"] / df["Follow-up time"]
df["ISO"] = df["∆ISO"] / df["Follow-up time"]
df["ha"] = df["∆ha"] / df["Follow-up time"]
df["ad"] = df["∆ad"] / df["Follow-up time"]
df["fa"] = df["∆fa"] / df["Follow-up time"]
df["rd"] = df["∆rd"] / df["Follow-up time"]
df["rd1"] = df["∆rd1"] / df["Follow-up time"]
df["rd2"] = df["∆rd2"] / df["Follow-up time"]
df["fALFF"] = df["∆fALFF"] / df["Follow-up time"]
df["Hurst"] = df["∆Hurst"] / df["Follow-up time"]
df["Entropy"] = df["∆Entropy"] / df["Follow-up time"]
df["Power slope"] = df["∆Power slope"] / df["Follow-up time"]
df["Autocor"] = df["∆Autocor"] / df["Follow-up time"]





needed = list(set(requested_cols + glm_covars + compare_vars + FEATURES + [GPO_COL]))
missing_needed = [c for c in needed if c not in df.columns and c != SUBJ_COL]
if missing_needed:
    raise ValueError(f"Missing required columns in Excel: {missing_needed}")

# Si no existe subject_id, créalo
if SUBJ_COL not in df.columns:
    df[SUBJ_COL] = np.arange(len(df)).astype(int)

use_cols = [c for c in requested_cols if c in df.columns]
if len(use_cols) < 2:
    raise ValueError("Not enough clustering features found in Excel.")

check_cols = use_cols + glm_covars
na_rows = df[check_cols].isna().any(axis=1)
n_na = int(na_rows.sum())

if n_na > 0:
    msg = f"[ERROR] Found {n_na} rows with NaNs in clustering features/covars. "
    if DROP_NA_ROWS:
        print(msg + "Dropping those rows (DROP_NA_ROWS=True).")
        df = df.loc[~na_rows].reset_index(drop=True)
    else:
        raise ValueError(msg + "Set DROP_NA_ROWS=True to drop them, or fix the Excel.")

print(f"[INFO] Rows after NaN handling: {len(df)}")




# ======================================================================================
# ===================== 2) GLM / OLS correction (solo features) ========================
# ======================================================================================
glm_features = [c for c in use_cols if c not in NO_GLM_COLS]
raw_features = [c for c in use_cols if c in NO_GLM_COLS]

if len(glm_features) + len(raw_features) < 2:
    raise ValueError("Not enough features after GLM/RAW split.")

print("[INFO] GLM features (corrected):", glm_features)
print("[INFO] RAW features (NOT corrected):", raw_features)

cov_df = df[glm_covars].copy()
if "Sex" in glm_covars:
    cov_df["Sex"] = encode_sex(cov_df["Sex"])

C = cov_df[glm_covars].to_numpy(dtype=float)
Xcov = np.column_stack([np.ones(len(df)), C])

# GLM residualize ONLY glm_features
if len(glm_features) > 0:
    X_glm_raw = df[glm_features].to_numpy(dtype=float)
    B = np.linalg.lstsq(Xcov, X_glm_raw, rcond=None)[0]
    X_glm_resid = X_glm_raw - (Xcov @ B)
    X_glm_z = StandardScaler().fit_transform(X_glm_resid)
else:
    X_glm_z = None

# RAW part: z-score only
if len(raw_features) > 0:
    X_rawpart = df[raw_features].to_numpy(dtype=float)
    X_raw_z = StandardScaler().fit_transform(X_rawpart)
else:
    X_raw_z = None

# Combine for clustering
if X_glm_z is not None and X_raw_z is not None:
    X_z = np.column_stack([X_glm_z, X_raw_z])
elif X_glm_z is not None:
    X_z = X_glm_z
elif X_raw_z is not None:
    X_z = X_raw_z
else:
    raise RuntimeError("No features for clustering after processing.")

print("[INFO] X_z shape:", X_z.shape)


# ======================================================================================
# ===================== 3) All lesion for k selection/plots + GMM ======================
# ======================================================================================
n = len(df)
fit_idx = pick_subset(n, FIT_SUBSET_N, seed=42)
X_fit = X_z[fit_idx]

print(f"[INFO] Total rows: {n} | subset for selection/plots: {len(fit_idx)}")

print("\n=== [GMM] Auto-k by min BIC ===")
gmm_bics = []
gmm_models = {}

for k in K_RANGE:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
        n_init=2,
        max_iter=300
    )
    gmm.fit(X_fit)
    bic = gmm.bic(X_fit)
    gmm_bics.append(bic)
    gmm_models[k] = gmm
    print(f" k={k:2d} | BIC={bic:.0f}")

best_k_gmm = K_RANGE[int(np.argmin(gmm_bics))]
gmm_best = gmm_models[best_k_gmm]
print(f"[GMM] Best k = {best_k_gmm}")

bic_png = os.path.join(out_dir, f"{base_name}_GMM_bic.png")
bic_pdf = os.path.join(out_dir, f"{base_name}_GMM_bic.pdf")
plot_curve(
    K_RANGE, gmm_bics,
    title=f"GMM: BIC vs k (best k={best_k_gmm})",
    xlabel="k",
    ylabel="BIC",
    out_png=bic_png,
    out_pdf=bic_pdf
)
print("[DONE] BIC plot:", bic_png)

labels_gmm_full = gmm_best.predict(X_z).astype(int)
labels_gmm_fit = gmm_best.predict(X_fit).astype(int)
gmm_probmax_full = gmm_best.predict_proba(X_z).max(axis=1)

# ===================== RE-LABEL: swap 1 <-> 2 (keep 0) =====================
map_swap = {0: 2, 1: 0, 2: 1}
labels_gmm_full = np.vectorize(map_swap.get)(labels_gmm_full)
labels_gmm_fit = np.vectorize(map_swap.get)(labels_gmm_fit)

# ======================================================================================
# ===================== SAVE TRAINED MODEL =============================================
# ======================================================================================

import joblib

print("\n[INFO] Saving trained clustering model...")

# ----------------------------------------------------------
# SAVE SCALERS
# ----------------------------------------------------------
scaler_glm = None
scaler_raw = None

if len(glm_features) > 0:
    scaler_glm = StandardScaler()
    scaler_glm.fit(X_glm_resid)

if len(raw_features) > 0:
    scaler_raw = StandardScaler()
    scaler_raw.fit(X_rawpart)

# ----------------------------------------------------------
# SAVE EVERYTHING NEEDED FOR EXTERNAL APPLICATION
# ----------------------------------------------------------
model_bundle = {

    # ================= GMM =================
    "gmm_model": gmm_best,

    # ================= FEATURES =================
    "feature_order": glm_features + raw_features,
    "glm_features": glm_features,
    "raw_features": raw_features,

    # ================= COVARIATES =================
    "glm_covars": glm_covars,
    "NO_GLM_COLS": NO_GLM_COLS,

    # ================= GLM BETAS =================
    "glm_betas": B if len(glm_features) > 0 else None,

    # ================= SCALERS =================
    "scaler_glm": scaler_glm,
    "scaler_raw": scaler_raw,

    # ================= LABEL MAPPING =================
    "map_swap": map_swap,

    # ================= CONFIG =================
    "requested_cols": requested_cols,
    "vars_corr": vars_corr,

    # ================= INFO =================
    "best_k": best_k_gmm,
    "n_features": X_z.shape[1],
}

# ----------------------------------------------------------
# OUTPUT FILE
# ----------------------------------------------------------
model_out = os.path.join(
    out_dir,
    f"{base_name}_trained_GMM_model.pkl"
)

# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------
joblib.dump(model_bundle, model_out)

print("[DONE] Trained model saved:")
print(model_out)

# ======================================================================================
# ===================== 3b) GMM PARAMETERS: COMPONENT MEANS + TOP FEATURES + HEATMAP ===
# ======================================================================================
feature_names = glm_features + raw_features  # mismo orden que X_z

means = pd.DataFrame(gmm_best.means_, columns=feature_names)
means_swapped = means.copy()
means_swapped.index = [map_swap[i] for i in range(means.shape[0])]
means_swapped = means_swapped.sort_index()  # 0,1,2 => L1,L2,L3

# ---- (C) Reordenar columnas: ∆WMH primero y Follow-up time al final ----
cols = list(means_swapped.columns)

# 1) sacar Follow-up time (si está) para agregarlo al final después
follow = ["Follow-up time"] if "Follow-up time" in cols else []
cols = [c for c in cols if c != "Follow-up time"]

# 2) poner ∆WMH primero (si está)
if "WMH" in cols:
    cols = ["WMH"] + [c for c in cols if c != "WMH"]

# 3) agregar Follow-up time al final
cols = cols + follow

means_swapped = means_swapped[cols]

# ---- (D) Top features por cluster según |mean| (en z) ----
TOPN = 8
top_list = []

for cl in means_swapped.index:
    s = means_swapped.loc[cl]  # serie (features) para ese cluster
    top = s.abs().sort_values(ascending=False).head(TOPN).index
    tmp = pd.DataFrame({
        "cluster": int(cl),
        "cluster_L": lab_text(int(cl)),  # L₁, L₂, L₃...
        "feature": top,
        "mean_z": s.loc[top].values
    })
    top_list.append(tmp)

top_df = pd.concat(top_list, ignore_index=True)

# ---- (E) Exportar a Excel ----
out_xlsx = os.path.join(out_dir, f"{base_name}_GMM_component_means.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    means_swapped.to_excel(writer, sheet_name="component_means_z")
    top_df.to_excel(writer, index=False, sheet_name="top_features_by_cluster")

print("[DONE] GMM component means saved:", out_xlsx)

# ---- (F) Heatmap de medias por cluster (z) - VECTOR (PDF) ----



# Para que el texto sea editable en PDF (Illustrator/Inkscape)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42

M = means_swapped.copy()
M.index = [lab_text(int(i)) for i in M.index]   # L₁, L₂, L₃...

# Tamaño base original:
# height_old = 0.8 * M.shape[0] + 2
# Nuevo: 0.9 del tamaño actual
width  = 0.55 * M.shape[1] + 5
height = 0.9 * (0.8 * M.shape[0] + 2)

fig, ax = plt.subplots(figsize=(width, height))

# pcolormesh = celdas vectoriales en PDF
x = np.arange(M.shape[1] + 1)
y = np.arange(M.shape[0] + 1)

# ===== Symmetric z-score scale =====
absmax = np.nanmax(np.abs(M.values))

mesh = ax.pcolormesh(
    x,
    y,
    M.values,
    cmap="bwr",
    vmin=-0.4,
    vmax= 0.4,
    shading="flat",
    edgecolors="none")

# Ticks centrados en cada celda
ax.set_xticks(np.arange(M.shape[1]) + 0.5)
ax.set_yticks(np.arange(M.shape[0]) + 0.5)

ax.set_xticklabels(M.columns, rotation=60, ha="right", fontsize=10)
ax.set_yticklabels(M.index, fontsize=12)

# Para que la primera fila quede arriba (como imshow)
ax.invert_yaxis()

# Limpio (opcional)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.06)
cbar.set_label("Component mean (z)", fontsize=12)
cbar.ax.tick_params(labelsize=10)

ax.set_title("")  # sin título si quieres
fig.tight_layout()

out_png = os.path.join(out_dir, f"{base_name}_GMM_componentMeans_heatmap.png")
out_pdf = os.path.join(out_dir, f"{base_name}_GMM_componentMeans_heatmap.pdf")

# PNG (raster) para vista rápida
fig.savefig(out_png, dpi=300, bbox_inches="tight")

# PDF vector (las celdas quedan vectorizadas)
fig.savefig(out_pdf, bbox_inches="tight")

plt.show()
plt.close(fig)

print("[DONE] Heatmap:", out_png, out_pdf)



# ======================================================================================
# ===================== 4) DF OUTPUT + Summary + Tests =================================
# ======================================================================================
df_out = df.copy()
df_out[CLUSTER_COL] = labels_gmm_full
df_out["cluster_gmm_probmax"] = gmm_probmax_full

gmm_summary = cluster_summary_table(df_out, CLUSTER_COL, compare_vars)
gmm_tests = run_tests_original(df_out, CLUSTER_COL, compare_vars)
gmm_gpo_freq = freq_by_gpo(df_out, CLUSTER_COL, GPO_COL) if GPO_COL in df_out.columns else pd.DataFrame()

with pd.ExcelWriter(excel_out_main, engine="openpyxl") as writer:
    df_out.to_excel(writer, index=False, sheet_name="data_with_clusters")
    gmm_summary.to_excel(writer, index=False, sheet_name="gmm_cluster_summary")
    if not gmm_tests.empty:
        gmm_tests.to_excel(writer, index=False, sheet_name="cluster_tests")
    if not gmm_gpo_freq.empty:
        gmm_gpo_freq.to_excel(writer, sheet_name="gmm_freq_by_Gpo")

print("[DONE] Excel main:", excel_out_main)


# ======================================================================================
# ===================== 5) COLORS CONSISTENTES (matplotlib+plotly) ======================
# ======================================================================================
uniq = np.array(sorted(np.unique(labels_gmm_fit).astype(int)))
cluster_color = make_cluster_color_map(
    uniq,
    cmap_name="viridis",
    pos_violet=0.10,
    pos_green=0.55,
    pos_yellow=0.95
)


# ======================================================================================
# ===================== 6) PCA plots ===================================================
# ======================================================================================
pc3 = PCA(n_components=3, random_state=42).fit_transform(X_fit)
labels = labels_gmm_fit
views = [(20, 35), (20, 120), (20, 210), (60, 35)]

for elev, azim in views:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for lab in uniq:
        m = labels == lab
        ax.scatter(
            pc3[m, 0], pc3[m, 1], pc3[m, 2],
            s=10, alpha=0.65,
            color=cluster_color[int(lab)],
            label=lab_text(int(lab))
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"GMM (subset) k={len(uniq)} | elev={elev} azim={azim}")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right", title="Clusters", fontsize=9)

    out_png = os.path.join(out_dir, f"{base_name}_GMM_PCA3D_e{elev}_a{azim}.png")
    out_pdf = os.path.join(out_dir, f"{base_name}_GMM_PCA3D_e{elev}_a{azim}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("[DONE] PCA3D:", out_png)

pairs = [(0, 1, "PC1", "PC2"), (0, 2, "PC1", "PC3"), (1, 2, "PC2", "PC3")]

# ✅ orden de pintado: L1 -> L2 -> L3 (L3 al final para que quede arriba)
plot_order = [2, 1, 0]
plot_order = [lab for lab in plot_order if lab in set(uniq)]  # por si falta alguno

for a, b, xa, xb in pairs:
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab in plot_order:
        m = (labels == lab)
        ax.scatter(
            pc3[m, a], pc3[m, b],
            s=12, alpha=0.65,
            color=cluster_color[int(lab)],
            label=lab_text(int(lab))
        )
    ax.set_xlabel(xa)
    ax.set_ylabel(xb)
    ax.set_title(f"GMM (subset) {xa} vs {xb} | k={len(uniq)}")
    ax.legend(title="Clusters", fontsize=9)

    out_png = os.path.join(out_dir, f"{base_name}_GMM_{xa}_vs_{xb}.png")
    out_pdf = os.path.join(out_dir, f"{base_name}_GMM_{xa}_vs_{xb}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("[DONE] PCA2D:", out_png)


# ======================================================================================
# ===================== 7) FRECUENCIAS por Gpo (STACKED %) ==============================
# ======================================================================================
if GPO_COL in df_out.columns:
    freq_png = os.path.join(out_dir, f"{base_name}_freq.png")
    freq_svg = os.path.join(out_dir, f"{base_name}_freq.svg")
    freq_pdf = os.path.join(out_dir, f"{base_name}_freq.pdf")

    tab_counts, tab_pct = plot_freq_by_gpo_stacked(
        df_out, CLUSTER_COL, GPO_COL, cluster_color,
        out_png=freq_png, out_svg=freq_svg, out_pdf=freq_pdf,
        bar_width=0.6, fig_w_per_group=0.85, fig_h=5
    )

    # ✅ Asegurar orden L1, L2, L3... en tablas/export (filas = clusters)
    tab_counts = tab_counts.reindex([2, 1, 0])
    tab_pct = tab_pct.reindex([2, 1, 0])

    # ---- Stats avanzadas + Excel ----
    obs = tab_counts.values
    chi2, p, dof, expected = chi2_contingency(obs, correction=False)
    N = obs.sum()
    k_eff = min(tab_counts.shape[0] - 1, tab_counts.shape[1] - 1)
    cramers_v_global = np.sqrt(chi2 / (N * k_eff)) if (N > 0 and k_eff > 0) else np.nan

    global_summary = pd.DataFrame({
        "chi2": [float(chi2)],
        "df": [int(dof)],
        "p": [float(p)],
        "N": [int(N)],
        "Cramers_V": [float(cramers_v_global)]
    })

    cols = tab_counts.columns.tolist()
    pairs_list, stats_list = [], []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            g1, g2 = cols[i], cols[j]
            sub = tab_counts[[g1, g2]].values
            x2, pp, dfp, _ = chi2_contingency(sub, correction=False)
            N2 = sub.sum()
            k2 = min(sub.shape[0] - 1, sub.shape[1] - 1)
            v_pair = np.sqrt(x2 / (N2 * k2)) if (N2 > 0 and k2 > 0) else np.nan
            pairs_list.append((g1, g2))
            stats_list.append((float(x2), int(dfp), float(pp), float(v_pair), int(N2)))

    pair_df = pd.DataFrame(
        stats_list,
        columns=["chi2", "df", "p", "Cramers_V", "N"],
        index=[f"{a} vs {b}" for a, b in pairs_list],
    )

    if not pair_df.empty:
        pair_df["p_fdr"] = multipletests(pair_df["p"].values, method="fdr_bh")[1]
        pair_df["p_bonf"] = multipletests(pair_df["p"].values, method="bonferroni")[1]
        pair_df = pair_df.sort_values("p_fdr")

    # Residuales estandarizados por celda (usa expected de chi2_contingency)
    exp = expected
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    row_prop = row_sum / N if N else np.nan
    col_prop = col_sum / N if N else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        std_res = (obs - exp) / np.sqrt(exp * (1 - row_prop) * (1 - col_prop))

    std_res_df = pd.DataFrame(std_res, index=tab_counts.index, columns=tab_counts.columns)

    p_cell = 2 * normal_dist.sf(np.abs(std_res))
    p_cell_adj = multipletests(np.ravel(p_cell), method="fdr_bh")[1].reshape(p_cell.shape)

    p_cell_df = pd.DataFrame(p_cell, index=tab_counts.index, columns=tab_counts.columns)
    p_cell_fdr_df = pd.DataFrame(p_cell_adj, index=tab_counts.index, columns=tab_counts.columns)

    out_xlsx = os.path.join(out_dir, f"{base_name}_GMM_cluster_stats.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        global_summary.to_excel(writer, sheet_name="global_stats", index=False)
        if not pair_df.empty:
            pair_df.to_excel(writer, sheet_name="pairwise_stats")
        tab_counts.to_excel(writer, sheet_name="counts_numeric")
        tab_pct.to_excel(writer, sheet_name="percent_by_group_numeric")
        std_res_df.to_excel(writer, sheet_name="std_residuals")
        p_cell_df.to_excel(writer, sheet_name="p_cell")
        p_cell_fdr_df.to_excel(writer, sheet_name="p_cell_fdr")

    print("[DONE] Gpo stats Excel:", out_xlsx)
else:
    pair_df = pd.DataFrame()  # para bubble matrix si no hay Gpo


# ======================================================================================
# ===================== 8) PROBMAX: mean + SEM por cluster =============================
# ======================================================================================
probmax_xlsx = os.path.join(out_dir, f"{base_name}_GMM_probmax_by_cluster.xlsx")
probmax_png = os.path.join(out_dir, f"{base_name}_GMM_probmax_by_cluster.png")

tmp = df_out[[CLUSTER_COL, "cluster_gmm_probmax"]].copy()
tmp[CLUSTER_COL] = tmp[CLUSTER_COL].astype(int)

summary_prob = (
    tmp.groupby(CLUSTER_COL)["cluster_gmm_probmax"]
    .agg(["count", "mean", "std"])
    .rename(columns={"count": "N", "mean": "probmax_mean", "std": "probmax_sd"})
    .reset_index()
    .sort_values(CLUSTER_COL)
    .reset_index(drop=True)
)
summary_prob["probmax_sem"] = summary_prob["probmax_sd"] / np.sqrt(summary_prob["N"].clip(lower=1))

clusters_prob = summary_prob[CLUSTER_COL].to_numpy()
bar_colors = [cluster_color[int(c)] for c in clusters_prob]

plt.figure(figsize=(3.4 + 0.35 * len(clusters_prob), 3.2))
x = np.arange(len(clusters_prob))
y = summary_prob["probmax_mean"].to_numpy(float)
yerr = summary_prob["probmax_sem"].to_numpy(float)

plt.bar(x, y, yerr=yerr, capsize=3, color=bar_colors)
plt.xticks(x, [lab_text(int(c)) for c in clusters_prob], rotation=0)
plt.xlabel("Cluster")
plt.ylabel("Mean max posterior prob")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(probmax_png, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("[DONE] Probmax plot:", probmax_png)

with pd.ExcelWriter(probmax_xlsx, engine="openpyxl") as writer:
    summary_prob.to_excel(writer, index=False, sheet_name="probmax_summary")
    tmp.to_excel(writer, index=False, sheet_name="probmax_all_rows")

print("[DONE] Probmax Excel:", probmax_xlsx)


# ======================================================================================
# ===================== 9) INTERACTIVE 3D (HTML) =======================================
# ======================================================================================
fig = go.Figure()
for lab in uniq:
    m = labels == lab
    fig.add_trace(go.Scatter3d(
        x=pc3[m, 0], y=pc3[m, 1], z=pc3[m, 2],
        mode="markers",
        name=lab_text(int(lab)),
        marker=dict(
            size=3,
            opacity=0.65,
            color=rgba_to_plotly_rgba(cluster_color[int(lab)])
        )
    ))

fig.update_layout(
    title=f"GMM clusters (subset) - k={len(uniq)} (interactive)",
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    legend=dict(title="Clusters")
)

html_out = os.path.join(out_dir, f"{base_name}_GMM_PCA3D_interactive.html")
pio.write_html(fig, file=html_out, auto_open=False, include_plotlyjs="cdn")
print("[DONE] Interactive 3D saved:", html_out)


# ======================================================================================
# ===================== 10) BUBBLE MATRIX (SIG ONLY) ===================================
# ======================================================================================
def get_pair_row(a, b, pair_df):
    k1 = f"{a} vs {b}"
    k2 = f"{b} vs {a}"
    if k1 in pair_df.index:
        return pair_df.loc[k1]
    if k2 in pair_df.index:
        return pair_df.loc[k2]
    return None

def area_to_diameter_pt(s_area):
    return 2.0 * np.sqrt(np.array(s_area, dtype=float) / np.pi)

def map_to_sizes(raw, smin, smax):
    raw = np.asarray(raw, float)
    m = np.isfinite(raw)
    if m.sum() == 0:
        return np.array([], float), (np.nan, np.nan, np.nan)
    x = raw[m]
    lo, hi = (np.percentile(x, [5, 95]) if len(x) > 1 else (x.min(), x.max() + 1e-9))
    denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    z = np.clip((raw - lo) / denom, 0, 1)
    return (smin + (smax - smin) * z), (lo, hi, denom)

def ensure_group_order_from_df(df_out, group_col="Gpo"):
    return list(pd.unique(df_out[group_col].dropna()))

if not pair_df.empty:
    # ---------------- CONFIG ----------------
    P_COL = "p_fdr"
    ALPHA = 0.05
    SIZE_MODE = "cramers_v"  # "chi2" o "cramers_v"

    # ✅ lo que pediste:
    # - eliminar la FILA CN pero NO su columna
    # - eliminar la COLUMNA PD pero NO su fila
    DROP_ROWS = {"CN"}
    DROP_COLS = {"PD"}

    S_MIN, S_MAX = 650, 4200
    V_MAX = 6.0

    LEG_BASE_X = 0.70
    LEG_BASE_Y = -0.20
    LEG_GAP_PT = 30.0
    LEG_SCALE = 0.85
    BOTTOM_MARGIN = 0.30
    LEG_EDGE = "#666666"

    CHI2_LEG = np.array([15, 25, 75], dtype=float)
    CHI2_TEXT_SHIFT = 5.0

    V_LEG = np.array([0.10, 0.20, 0.40], dtype=float)
    V_TEXT_FMT = "{:.2f}"

    # ---------------- GROUPS ----------------
    groups_all = ensure_group_order_from_df(df_out, group_col=GPO_COL)
    rows = [g for g in groups_all if g not in DROP_ROWS]  # CN se va solo de filas
    cols = [g for g in groups_all if g not in DROP_COLS]  # PD se va solo de columnas

    R, C = len(rows), len(cols)
    if R == 0 or C == 0:
        raise ValueError("Después de DROP_ROWS/DROP_COLS, no quedan filas/columnas.")

    # ---------------- COLLECT SIGNIFICANT CELLS ----------------
    xs, ys, pvals = [], [], []
    chi2vals, vvals = [], []

    for i in range(R):
        for j in range(C):
            rr, cc = rows[i], cols[j]
            if rr == cc:
                continue

            # ✅ robusto: mantener solo triángulo inferior respecto al orden global
            if groups_all.index(rr) < groups_all.index(cc):
                continue

            r = get_pair_row(rr, cc, pair_df)
            if r is None:
                continue

            pv = float(r[P_COL]) if P_COL in r.index else np.nan
            if np.isfinite(pv) and pv < ALPHA:
                xs.append(j)
                ys.append(i)
                pvals.append(pv)
                chi2vals.append(float(r["chi2"]) if "chi2" in r.index else np.nan)
                if "Cramers_V" in r.index:
                    vvals.append(float(r["Cramers_V"]))
                else:
                    vvals.append(np.nan)

    pvals = np.array(pvals, float)
    chi2vals = np.array(chi2vals, float)
    vvals = np.array(vvals, float)

    if len(pvals) == 0:
        print("[WARN] No hay pares significativos (p < ALPHA). Bubble matrix sin burbujas.")

    # ---------------- SIZE DRIVER ----------------
    if SIZE_MODE == "cramers_v" and np.any(np.isfinite(vvals)):
        size_label = "V"
        driver_raw = np.clip(vvals, 0, None)
    else:
        size_label = "χ²"
        driver_raw = np.sqrt(np.clip(chi2vals, 0, None)) if len(chi2vals) else np.array([])

    sizes, _ = map_to_sizes(driver_raw, S_MIN, S_MAX)

    # ---------------- COLORS (by p) ----------------
    val_color = -np.log10(np.clip(pvals, 1e-300, 1.0)) if len(pvals) else np.array([])
    cmap_b = mpl_colors.LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"])

    vmin = -np.log10(10)  # 1
    vmax = V_MAX
    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    colors_rgba = cmap_b(norm(np.clip(val_color, vmin, vmax))) if len(val_color) else np.array([])

    # ---------------- PLOT ----------------
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(R - 0.5, -0.5)

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(R))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticklabels(rows)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-.5, R, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if len(xs) > 0:
        ax.scatter(xs, ys, s=sizes, c=colors_rgba, edgecolors="none", linewidths=0)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    p_start = ALPHA
    p_end = 10 ** (-vmax)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{p_start:g}", f"{p_end:g}"])

    # ---------------- SIZE LEGEND ----------------
    def build_size_legend(ax, fig, driver_raw, size_label):
        if size_label == "χ²":
            leg_driver = np.sqrt(np.clip(CHI2_LEG, 0, None))
            leg_sizes, _ = map_to_sizes(leg_driver, S_MIN, S_MAX)
            leg_sizes = leg_sizes * LEG_SCALE
            leg_text = [f"{int(v - CHI2_TEXT_SHIFT)}" for v in CHI2_LEG]
            title_txt = "χ²"
            title_index = 1
        else:
            leg_driver = np.clip(V_LEG, 0, None)
            m = np.isfinite(driver_raw)
            if m.sum() > 0:
                lo = np.percentile(driver_raw[m], 5)
                hi = np.percentile(driver_raw[m], 95) if m.sum() > 1 else (driver_raw[m].max() + 1e-9)
                denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
                z = np.clip((leg_driver - lo) / denom, 0, 1)
                leg_sizes = (S_MIN + (S_MAX - S_MIN) * z) * LEG_SCALE
            else:
                leg_sizes = (S_MIN + (S_MAX - S_MIN) * np.linspace(0.2, 0.8, len(V_LEG))) * LEG_SCALE
            leg_text = [V_TEXT_FMT.format(v) for v in V_LEG]
            title_txt = "V"
            title_index = 1

        diam = area_to_diameter_pt(leg_sizes)
        x_pt = np.zeros(len(leg_sizes))
        x_pt[0] = 0.0
        for k in range(1, len(leg_sizes)):
            x_pt[k] = x_pt[k-1] + (diam[k-1]/2 + diam[k]/2) + LEG_GAP_PT

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_w_pt = bbox.width * 72.0
        x_ax = x_pt / ax_w_pt

        xs_leg = LEG_BASE_X + x_ax
        y_leg = LEG_BASE_Y

        for x, s_area, t in zip(xs_leg, leg_sizes, leg_text):
            ax.scatter([x], [y_leg], s=[s_area], facecolors="none",
                       edgecolors=LEG_EDGE, linewidths=1.2,
                       transform=ax.transAxes, clip_on=False)
            ax.text(x, y_leg, t, ha="center", va="center",
                    fontsize=9, color="#444444",
                    transform=ax.transAxes, clip_on=False)

        ax.text(xs_leg[title_index], y_leg + 0.10, title_txt,
                ha="center", va="center",
                fontsize=11, color="#444444",
                transform=ax.transAxes, clip_on=False)

    build_size_legend(ax, fig, driver_raw, size_label)

    plt.tight_layout()
    plt.subplots_adjust(bottom=BOTTOM_MARGIN)

    out_png = os.path.join(out_dir, f"{base_name}_GMM_pairwise_frequency_bubble.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    out_pdf = os.path.join(out_dir, f"{base_name}_GMM_pairwise_frequency_bubble.pdf")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")

    plt.close()
    print("[DONE] Bubble-matrix saved:", out_png)


# ======================================================================================
# ===================== 11) CORRELACIONES (lower-triangle) ==============================
# =====================  AJUSTADAS POR SUJETO (GEE) ====================================
# ======================================================================================
# ✅ Lo que pediste (aplica a TODOS los plots de correlación):
# - quitar la FILA de "∆WMH" pero NO su columna
# - quitar la COLUMNA de "∆Autocor" pero NO su fila
DROP_ROW_ONLY = "WMH"
DROP_COL_ONLY = "Autocor"
cbar_label_size=16
# ---- imports (solo para esta sección) ----
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence  # o Exchangeable

def rank_z(x):
    """Spearman = ranks; luego z-score para que beta ~ r."""
    r = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    sd = np.nanstd(r, ddof=1)
    if sd < 1e-12:
        return (r - np.nanmean(r))  # todo constante -> sd~0
    return (r - np.nanmean(r)) / sd

def format_lancet(x, nd=2):
    """Formato tipo Lancet: 1.23 -> 1·23 (punto)."""
    return f"{x:.{nd}f}".replace(".", ".")

def compute_spearman_block_clustered(df_block, label="ALL"):
    """
    Spearman ajustada por dependencia intra-sujeto usando:
    - rank+z de cada variable (Spearman)
    - GEE con groups = subject_id (SUBJ_COL)
    - partial opcional: covars incluidas en el modelo
    Devuelve df_r, df_p, df_padj (lower triangle rellenado).
    """
    # OJO: usa SUBJ_COL ya definido arriba ("subject_id")
    use_cols = vars_corr + ([SUBJ_COL] if SUBJ_COL not in vars_corr else []) + (CORR_COVARS if DO_PARTIAL else [])

    miss = [c for c in use_cols if c not in df_block.columns]
    if miss:
        raise ValueError(f"[{label}] Faltan columnas: {miss}")

    # NaNs: mismo comportamiento que antes
    if df_block[use_cols].isna().any().any():
        bad_n = int(df_block[use_cols].isna().any(axis=1).sum())
        raise ValueError(f"[{label}] Hay {bad_n} filas con NaN en vars/covars/subject. Arregla o filtra.")

    df_use = df_block[use_cols].copy()

    # ranks+z para vars_corr
    Z = {v: rank_z(df_use[v].values) for v in vars_corr}

    # covars para partial (dentro del modelo)
    cov_df = None
    if DO_PARTIAL:
        cov_df = df_use[CORR_COVARS].copy()

        if "Sex" in CORR_COVARS:
            cov_df["Sex"] = encode_sex(cov_df["Sex"])

        # z-score covariables numéricas (recomendado)
        for c in cov_df.columns:
            if pd.api.types.is_numeric_dtype(cov_df[c]):
                sd = cov_df[c].std(ddof=1)
                if sd is None or sd < 1e-12:
                    cov_df[c] = cov_df[c] - cov_df[c].mean()
                else:
                    cov_df[c] = (cov_df[c] - cov_df[c].mean()) / sd

    groups = df_use[SUBJ_COL].astype("category")

    P = len(vars_corr)
    Rmat = np.full((P, P), np.nan, float)
    PVAL = np.full((P, P), np.nan, float)

    for i in range(P):
        Rmat[i, i] = 1.0
        PVAL[i, i] = 0.0

        y = Z[vars_corr[i]]

        for j in range(i):  # lower triangle
            x = Z[vars_corr[j]]

            Xcols = {"x": x}
            if DO_PARTIAL:
                for c in cov_df.columns:
                    Xcols[c] = cov_df[c].to_numpy(dtype=float)

            X = pd.DataFrame(Xcols)
            X = sm.add_constant(X, has_constant="add")

            # GEE cluster-robust por sujeto
            model = GEE(
                endog=y,
                exog=X,
                groups=groups,
                cov_struct=Independence()
            )
            res = model.fit()

            r_adj = float(res.params["x"])     # ~ "Spearman r" ajustada
            p_adj = float(res.pvalues["x"])    # p con dependencia corregida

            Rmat[i, j] = r_adj
            PVAL[i, j] = p_adj

    # FDR en triángulo inferior
    PADJ = PVAL.copy()
    if CORR_USE_FDR:
        tri = np.tril_indices(P, k=-1)
        pvec = PVAL[tri]
        padj = multipletests(pvec, method=CORR_FDR_METHOD)[1]
        PADJ[tri] = padj

    df_r = pd.DataFrame(Rmat, index=vars_corr, columns=vars_corr)
    df_p = pd.DataFrame(PVAL, index=vars_corr, columns=vars_corr)
    df_padj = pd.DataFrame(PADJ, index=vars_corr, columns=vars_corr)
    return df_r, df_p, df_padj


def plot_lower_triangle(df_r, df_p_use, out_png, out_pdf=None,
                        drop_row_only=DROP_ROW_ONLY, drop_col_only=DROP_COL_ONLY,
                        thr_text_white=0.35, tick_label_size=16, cell_text_size=14,
                        cbar_label_size=14, cbar_tick_size=12):
    """
    Heatmap lower-triangle con drops asimétricos:
    - quita SOLO la fila drop_row_only (pero deja su columna)
    - quita SOLO la columna drop_col_only (pero deja su fila)
    """
    rows = [v for v in df_r.index.tolist() if v != drop_row_only]
    cols = [v for v in df_r.columns.tolist() if v != drop_col_only]

    df_rp = df_r.loc[rows, cols]
    df_pp = df_p_use.loc[rows, cols]
    mat = df_rp.to_numpy(dtype=float)

    Pr, Pc = mat.shape
    fig, ax = plt.subplots(figsize=(0.6 * Pc + 6, 0.6 * Pr + 5))

    rr = np.arange(Pr)[:, None]
    cc = np.arange(Pc)[None, :]
    mask = cc > rr

    mat_plot = mat.copy()
    mat_plot[mask] = np.nan

    im = ax.imshow(mat_plot, vmin=CORR_VMIN, vmax=CORR_VMAX, cmap=CORR_CMAP, aspect="equal")

    ax.set_xticks(np.arange(Pc))
    ax.set_yticks(np.arange(Pr))
    ax.set_xticklabels(df_rp.columns.tolist(), rotation=45, ha="right", fontsize=tick_label_size)
    ax.set_yticklabels(df_rp.index.tolist(), fontsize=tick_label_size)

    ax.set_xticks(np.arange(-.5, Pc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, Pr, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # anotaciones (solo triángulo inferior) -> formato "Lancet" con punto medio (·)
    for i in range(Pr):
        for j in range(Pc):
            if j <= i:
                r = df_rp.iat[i, j]
                p = df_pp.iat[i, j]
                if np.isfinite(r):
                    txt_color = "white" if abs(r) > thr_text_white else "black"
                    r_txt = format_lancet(r, nd=2)
                    ax.text(j, i, f"{r_txt}{stars_from_p(p)}",
                            ha="center", va="center",
                            fontsize=cell_text_size, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman r", fontsize=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_tick_size)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print("[DONE] Corr heatmap PNG:", out_png)
    if out_pdf is not None:
        print("[DONE] Corr heatmap PDF:", out_pdf)


if DO_CORR_BLOCKS:

    # ---------------- ALL ----------------
    df_r_all, df_p_all, df_padj_all = compute_spearman_block_clustered(df_out, label="ALL")
    p_used_all = df_padj_all if CORR_USE_FDR else df_p_all

    png_all = os.path.join(out_dir, f"{base_name}_lowertri_spearman_ALL.png")
    pdf_all = os.path.join(out_dir, f"{base_name}_lowertri_spearman_ALL.pdf")
    plot_lower_triangle(df_r_all, p_used_all, png_all, out_pdf=pdf_all)

    # ---------------- BY CLUSTER ----------------
    if CLUSTER_COL not in df_out.columns:
        raise ValueError(f"No existe {CLUSTER_COL} en df_out.")

    clusters = sorted(pd.unique(df_out[CLUSTER_COL].dropna()))
    xlsx_out = os.path.join(out_dir, f"{base_name}_lowertri_spearman_ALL_and_clusters.xlsx")

    sheets = {
        "ALL_r": df_r_all,
        "ALL_p": df_p_all,
        ("ALL_pFDR" if CORR_USE_FDR else "ALL_p_used"): p_used_all
    }

    for cl in clusters:
        block = df_out[df_out[CLUSTER_COL] == cl].copy()
        if len(block) < 5:
            print(f"[WARN] Cluster {cl}: N={len(block)} muy pequeño, salto.")
            continue

        df_r, df_p, df_padj = compute_spearman_block_clustered(block, label=f"cluster_{cl}")
        p_used = df_padj if CORR_USE_FDR else df_p

        png_cl = os.path.join(out_dir, f"{base_name}_lowertri_spearman_{CLUSTER_COL}_{cl}.png")
        pdf_cl = os.path.join(out_dir, f"{base_name}_lowertri_spearman_{CLUSTER_COL}_{cl}.pdf")
        plot_lower_triangle(df_r, p_used, png_cl, out_pdf=pdf_cl)

        sheets[f"cl{cl}_r"[:31]] = df_r
        sheets[f"cl{cl}_p"[:31]] = df_p
        sheets[(f"cl{cl}_pFDR" if CORR_USE_FDR else f"cl{cl}_pused")[:31]] = p_used

    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        for name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, sheet_name=str(name)[:31])

    print("[DONE] Corr Excel:", xlsx_out)


# ======================================================================================
# ===================== 12) ML CLASIFICACION + ROC (OOF) ===============================
# ======================================================================================
# Cambios aplicados:
# - En ROC: clases mostradas como L₁, L₂, ... (NO 0,1,2)
# - ROC SIN TÍTULO (plt.title(""))
# - (Opcional) macro AUC como texto dentro del gráfico (no título)

def auc_ovr_macro(y_true, proba, labels):
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)

def align_proba_to_global(fold_classes, proba_fold, global_classes):
    out = np.full((proba_fold.shape[0], len(global_classes)), np.nan, float)
    idx_map = {c: i for i, c in enumerate(global_classes)}
    for j, c in enumerate(fold_classes):
        out[:, idx_map[c]] = proba_fold[:, j]
    return out

def strip_prefixes(feat_names):
    cleaned = []
    for f in feat_names:
        s = str(f)
        s = re.sub(r"^(num|cat)__", "", s)
        s = re.sub(r"^(num|cat)__[^_]+__", "", s)
        s = s.replace("onehot__", "").replace("imputer__", "")
        cleaned.append(s)
    return np.array(cleaned, dtype=object)

def get_feature_names_from_preprocess(prep: ColumnTransformer):
    try:
        names = prep.get_feature_names_out()
        return strip_prefixes(names)
    except Exception:
        return None

def build_splits(X, y, groups, n_splits, seed):
    has_repeats = groups.duplicated().any()
    if has_repeats:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, y, groups=groups))
            return splits, "StratifiedGroupKFold", True
        except Exception:
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(X, y, groups=groups))
            print("[WARN] No hay StratifiedGroupKFold. Uso GroupKFold (NO estratifica).")
            return splits, "GroupKFold", True
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
        return splits, "StratifiedKFold", False

def plot_multiclass_roc_with_se(y_true, proba, classes, out_png, cluster_color,
                                auc_macro_value=None, n_boot=1000, grid_n=200,
                                seed=42, show_auc_text=True):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, float)
    fpr_grid = np.linspace(0, 1, grid_n)

    plt.figure(figsize=(8, 6))

    for i, cl in enumerate(classes):
        y_i = (y_true == cl).astype(int)
        s_i = proba[:, i]
        if len(np.unique(y_i)) < 2:
            continue

        fpr_obs, tpr_obs, _ = roc_curve(y_i, s_i)
        auc_obs = auc(fpr_obs, tpr_obs)

        idx_all = np.arange(len(y_i))
        auc_boot = []
        tpr_boot = []

        for _ in range(n_boot):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            y_b = y_i[idx]
            s_b = s_i[idx]
            if len(np.unique(y_b)) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y_b, s_b)
            auc_boot.append(auc(fpr_b, tpr_b))
            tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
            tpr_interp[0] = 0.0
            tpr_boot.append(tpr_interp)

        color = cluster_color.get(int(cl), "black")
        name = lab_text(cl)

        if len(auc_boot) < 10:
            plt.plot(fpr_obs, tpr_obs, color=color, lw=2, label=f"{name} AUC={auc_obs:.3f}")
            continue

        auc_boot = np.asarray(auc_boot, float)
        tpr_boot = np.asarray(tpr_boot, float)
        auc_se = auc_boot.std(ddof=1)
        tpr_mean = tpr_boot.mean(axis=0)
        tpr_se = tpr_boot.std(axis=0, ddof=1) / np.sqrt(tpr_boot.shape[0])

        plt.plot(fpr_grid, tpr_mean, color=color, lw=2,
                 label=f"{name} AUC={auc_obs:.3f} ± {auc_se:.3f}")
        plt.fill_between(
            fpr_grid,
            np.clip(tpr_mean - tpr_se, 0, 1),
            np.clip(tpr_mean + tpr_se, 0, 1),
            color=color,
            alpha=0.18,
            linewidth=0
        )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("")

    if (auc_macro_value is not None) and show_auc_text:
        plt.text(
            0.98, 0.02,
            f"Macro AUC (OVR) = {auc_macro_value:.3f}",
            ha="right", va="bottom",
            transform=plt.gca().transAxes, fontsize=10
        )

    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print("[DONE] ROC guardada:", out_png)

def compute_shap_global_and_per_class(shap_values, K_expected=None):
    if isinstance(shap_values, list):
        per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
        shap_global = np.mean(np.vstack(per_class), axis=0)
        return shap_global, per_class

    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        shap_global = np.mean(np.abs(sv), axis=(0, 2))
        per_class = [np.mean(np.abs(sv[:, :, k]), axis=0) for k in range(sv.shape[2])]
        if K_expected is not None and sv.shape[2] != K_expected:
            print(f"[WARN] SHAP devolvió K={sv.shape[2]} clases, esperaba K={K_expected}.")
        return shap_global, per_class
    if sv.ndim == 2:
        shap_global = np.mean(np.abs(sv), axis=0)
        return shap_global, None
    raise ValueError(f"Formato shap_values inesperado: shape={sv.shape}")

def save_shap_summary_plot(shap_vals_2d, X_2d, feat_names, out_png,
                           max_display=30, plot_type=None):
    try:
        shap.summary_plot(
            shap_vals_2d, X_2d,
            feature_names=feat_names,
            max_display=max_display,
            plot_type=plot_type,
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[DONE] SHAP plot guardado:", out_png)
    except Exception as e:
        print("[WARN] No pude guardar SHAP plot:", repr(e))


# ======================================================================================
# ===================================== RUN ============================================
# ======================================================================================
if DO_ML:
    needed_ml = [TARGET, GPO_COL, SUBJ_COL] + FEATURES
    miss_ml = [c for c in needed_ml if c not in df_out.columns]
    if miss_ml:
        raise ValueError(f"Faltan columnas para ML: {miss_ml}")

    df_ml = df_out.copy()
    df_ml = df_ml[df_ml[TARGET].notna()].reset_index(drop=True)

    X = df_ml[FEATURES].copy()
    y = df_ml[TARGET].copy()
    groups = df_ml[SUBJ_COL].copy()
    gpo_vec = df_ml[GPO_COL].copy()

    classes_global = np.array(sorted(pd.unique(y)))
    K = len(classes_global)
    if K < 2:
        raise ValueError("El target tiene <2 clases. No se puede clasificar.")

    print(f"[INFO][ML] N={len(df_ml)} | K={K} clases: {classes_global}")

    cat_cols = [c for c in ["Sex"] if c in FEATURES]
    num_cols = [c for c in FEATURES if c not in cat_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe)
            ]), cat_cols),
        ],
        remainder="drop"
    )

    clf = LGBMClassifier(
        objective="multiclass",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])

    proba_oof_sum = np.zeros((len(df_ml), K), dtype=float)
    proba_oof_count = np.zeros((len(df_ml), K), dtype=int)
    pred_oof_last = np.full((len(df_ml),), None, object)

    auc_per_repeat = []
    cv_name_last = None

    for rep in range(N_REPEATS):
        seed_rep = BASE_SEED + rep
        splits_list, cv_name, has_repeats = build_splits(X, y, groups, N_SPLITS, seed_rep)
        cv_name_last = cv_name

        print(f"[INFO][ML] Repeat {rep+1}/{N_REPEATS} | CV={cv_name} | seed={seed_rep} | repeats={has_repeats}")

        proba_oof_rep = np.full((len(df_ml), K), np.nan, float)
        pred_oof_rep = np.full((len(df_ml),), None, object)

        for fold, (tr, te) in enumerate(splits_list, start=1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]

            pipe.set_params(clf__random_state=seed_rep)
            pipe.fit(Xtr, ytr)

            proba = pipe.predict_proba(Xte)
            fold_classes = pipe.named_steps["clf"].classes_
            proba_aligned = align_proba_to_global(fold_classes, proba, classes_global)

            proba_oof_rep[te, :] = proba_aligned
            pred_oof_rep[te] = pipe.predict(Xte)

        if not np.isfinite(proba_oof_rep).all():
            raise RuntimeError("OOF proba (rep) tiene NaNs. Suele pasar si algún fold no vio alguna clase.")

        auc_rep = auc_ovr_macro(y, proba_oof_rep, labels=classes_global)
        auc_per_repeat.append(float(auc_rep))
        print(f"[DONE][ML] Repeat {rep+1}: OOF AUC macro OVR = {auc_rep:.4f}")

        mask_finite = np.isfinite(proba_oof_rep)
        proba_oof_sum[mask_finite] += proba_oof_rep[mask_finite]
        proba_oof_count[mask_finite] += 1
        pred_oof_last = pred_oof_rep

    with np.errstate(divide="ignore", invalid="ignore"):
        proba_oof_mean = proba_oof_sum / np.maximum(proba_oof_count, 1)

    if not np.isfinite(proba_oof_mean).all():
        raise RuntimeError("proba_oof_mean tiene NaNs/inf. Revisa folds con clases faltantes.")

    auc_oof_mean = auc_ovr_macro(y, proba_oof_mean, labels=classes_global)
    print(f"[DONE][ML] OOF AUC global (macro OVR) promedio {N_REPEATS} rep: {auc_oof_mean:.4f}")

    # ---------------- ROC GLOBAL (L₁,L₂,...; sin título) ----------------
    roc_png_all = os.path.join(out_dir, f"{base_name}_ROC_OOF_ALLsubjects_rep{N_REPEATS}.png")
    plot_multiclass_roc_with_se(
        y_true=y,
        proba=proba_oof_mean,
        classes=classes_global,
        out_png=roc_png_all,
        cluster_color=cluster_color,
        auc_macro_value=auc_oof_mean,
        n_boot=N_BOOT,
        grid_n=GRID_N,
        seed=SEED,
        show_auc_text=True
    )

    # ---------------- ROC POR Gpo (L₁,L₂,...; sin título) ----------------
    if DO_ROC_BY_GPO:
        for g in pd.unique(gpo_vec.dropna()):
            m = (np.asarray(gpo_vec) == g)
            n_g = int(m.sum())
            if n_g < 30 or len(np.unique(np.asarray(y)[m])) < 2:
                print(f"[SKIP][ML] Gpo={g}: N={n_g} o 1 clase.")
                continue

            out_png = os.path.join(out_dir, f"{base_name}_ROC_OOFmean_Gpo_{g}_rep{N_REPEATS}.png")
            plot_multiclass_roc_with_se(
                y_true=np.asarray(y)[m],
                proba=proba_oof_mean[m, :],
                classes=classes_global,
                out_png=out_png,
                cluster_color=cluster_color,
                auc_macro_value=None,
                n_boot=N_BOOT,
                grid_n=GRID_N,
                seed=SEED,
                show_auc_text=False
            )

    # ---------------- Feature importance + SHAP (entrena final en todo) ----------------
    if DO_FEATURE_IMPORTANCE_AND_SHAP:
        pipe.set_params(clf__random_state=RANDOM_STATE)
        pipe.fit(X, y)

        prep_fitted = pipe.named_steps["prep"]
        clf_fitted = pipe.named_steps["clf"]

        feat_names = get_feature_names_from_preprocess(prep_fitted)
        if feat_names is None:
            feat_names = np.array([f"f{i}" for i in range(clf_fitted.booster_.num_feature())], dtype=object)

        booster = clf_fitted.booster_
        imp_gain = booster.feature_importance(importance_type="gain")
        fi = (
            pd.DataFrame({"feature": feat_names, "importance_gain": imp_gain})
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )

        featimp_xlsx = os.path.join(out_dir, f"{base_name}_FeatureImportance_LGBM_{TARGET}.xlsx")
        featimp_png = os.path.join(out_dir, f"{base_name}_FeatureImportance_LGBM_{TARGET}.png")

        with pd.ExcelWriter(featimp_xlsx, engine="openpyxl") as writer:
            fi.to_excel(writer, index=False, sheet_name="feature_importance_gain")

        topn = min(30, len(fi))
        plt.figure(figsize=(8, 10))
        plt.barh(fi.loc[:topn-1, "feature"][::-1], fi.loc[:topn-1, "importance_gain"][::-1])
        plt.xlabel("Importance (gain)")
        plt.title("")
        plt.tight_layout()
        plt.savefig(featimp_png, dpi=300, bbox_inches="tight")
        plt.close()

        print("[DONE][ML] Feature importance:", featimp_xlsx)
        print("[DONE][ML] Feature importance plot:", featimp_png)

        shap_dir = os.path.join(out_dir, f"{base_name}_SHAP_{TARGET}")
        os.makedirs(shap_dir, exist_ok=True)

        if _HAS_SHAP:
            X_tr = prep_fitted.transform(X)

            rng = np.random.default_rng(SEED)
            if X_tr.shape[0] > MAX_SHAP_SAMPLES:
                idx = rng.choice(np.arange(X_tr.shape[0]), size=MAX_SHAP_SAMPLES, replace=False)
                X_sh = X_tr[idx]
            else:
                X_sh = X_tr

            explainer = shap.TreeExplainer(clf_fitted)
            shap_values = explainer.shap_values(X_sh)

            shap_global, per_class = compute_shap_global_and_per_class(shap_values, K_expected=K)

            shap_imp = (
                pd.DataFrame({"feature": feat_names, "mean_abs_shap": shap_global})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )

            shap_xlsx = os.path.join(shap_dir, f"SHAP_importance_{base_name}_{TARGET}.xlsx")
            with pd.ExcelWriter(shap_xlsx, engine="openpyxl") as writer:
                shap_imp.to_excel(writer, index=False, sheet_name="shap_global")
                fi.to_excel(writer, index=False, sheet_name="lgbm_gain_importance")
                if SAVE_SHAP_PER_CLASS and per_class is not None:
                    for k_idx, cl in enumerate(classes_global):
                        dfk = (
                            pd.DataFrame({"feature": feat_names, "mean_abs_shap": per_class[k_idx]})
                            .sort_values("mean_abs_shap", ascending=False)
                        )
                        dfk.to_excel(writer, index=False, sheet_name=f"class_{lab_text(cl)}"[:31])

            print("[DONE][ML] SHAP Excel:", shap_xlsx)

            # Para plots: usa clase 0 por defecto
            if isinstance(shap_values, list):
                shap2d = shap_values[0]
            else:
                sv = np.asarray(shap_values)
                shap2d = sv[:, :, 0] if sv.ndim == 3 else sv

            shap_bee_png = os.path.join(shap_dir, f"SHAP_beeswarm_{base_name}_{TARGET}.png")
            save_shap_summary_plot(shap2d, X_sh, feat_names, shap_bee_png, max_display=30, plot_type=None)

            shap_bar_png = os.path.join(shap_dir, f"SHAP_summaryBar_{base_name}_{TARGET}.png")
            save_shap_summary_plot(shap2d, X_sh, feat_names, shap_bar_png, max_display=30, plot_type="bar")

            if SAVE_SHAP_PER_CLASS and per_class is not None and isinstance(shap_values, list):
                for k_idx, cl in enumerate(classes_global):
                    outp = os.path.join(
                        shap_dir,
                        f"SHAP_beeswarm_class_{lab_text(cl)}_{base_name}_{TARGET}.png"
                    )
                    save_shap_summary_plot(shap_values[k_idx], X_sh, feat_names, outp, max_display=30, plot_type=None)
        else:
            print("[INFO][ML] SHAP omitido (no está instalado).")

    # ---------------- Guardar OOF predictions ----------------
    oof_xlsx = os.path.join(out_dir, f"{base_name}_OOF_predictions_{TARGET}_rep{N_REPEATS}.xlsx")

    oof_df = df_ml[[TARGET, GPO_COL, SUBJ_COL]].copy()
    oof_df["pred_oof_last_repeat"] = pred_oof_last
    for i, c in enumerate(classes_global):
        oof_df[f"proba_oof_mean_{lab_text(c)}"] = proba_oof_mean[:, i]  # ✅ columnas con L₁,L₂,...

    summary_df = pd.DataFrame([{
        "target": TARGET,
        "cv": cv_name_last,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "AUC_OOF_macro_OVR_meanProba": float(auc_oof_mean),
        "AUC_per_repeat_mean": float(np.mean(auc_per_repeat)),
        "AUC_per_repeat_sd": float(np.std(auc_per_repeat, ddof=1)) if len(auc_per_repeat) > 1 else 0.0,
        "N": int(len(df_ml)),
        "classes": ", ".join(lab_text(c) for c in classes_global)  # ✅ L₁,L₂,...
    }])

    auc_rep_df = pd.DataFrame({
        "repeat": np.arange(1, N_REPEATS + 1),
        "auc_oof_macro_ovr": auc_per_repeat
    })

    with pd.ExcelWriter(oof_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        auc_rep_df.to_excel(writer, index=False, sheet_name="auc_by_repeat")
        oof_df.to_excel(writer, index=False, sheet_name="oof_mean")

    print("[DONE][ML] OOF Excel:", oof_xlsx)


# ======================================================================================
# ===================== HEATMAP por INDIVIDUO: subjects (Y) x lesion types (X) ==========
# ===================== PDF 100% VECTOR (pcolormesh) ====================================
# ======================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_subject_lesiontype_heatmap_counts_vector(df_in, subj_col, cluster_col,
                                                  out_png,
                                                  out_pdf=None,
                                                  max_subjects=200,
                                                  sort_subjects=True,
                                                  fig_w=3.5, fig_h=6.0,
                                                  cmap_name="OrRd",
                                                  cluster_order=(0, 1, 2)):
    """
    Heatmap de CONTEOS:
      filas = sujetos
      columnas = clusters (en el orden FINAL ya swappeado arriba)
    PDF totalmente vector: usa pcolormesh (no imshow) y guarda directo con savefig.
    """

    # Texto editable en Illustrator/Inkscape
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42

    if subj_col not in df_in.columns:
        raise ValueError(f"Missing subject column: {subj_col}")
    if cluster_col not in df_in.columns:
        raise ValueError(f"Missing cluster column: {cluster_col}")

    cl_use = df_in[cluster_col].astype(int)

    tab = pd.crosstab(df_in[subj_col], cl_use).sort_index()
    tab = tab.reindex(columns=list(cluster_order), fill_value=0)

    if sort_subjects:
        tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]

    if (max_subjects is not None) and (len(tab) > max_subjects):
        tab_plot = tab.iloc[:max_subjects].copy()
    else:
        tab_plot = tab.copy()

    x_labels = [lab_text(int(i)) for i in tab_plot.columns]

    Z = tab_plot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- VECTOR HEATMAP (each cell is a vector rectangle) ---
    x = np.arange(Z.shape[1] + 1)
    y = np.arange(Z.shape[0] + 1)

    mesh = ax.pcolormesh(
        x, y, Z,
        cmap=cmap_name,
        shading="flat",
        edgecolors="none"   # pon "k" + linewidth si quieres bordes
    )

    # Ticks centrados
    ax.set_xticks(np.arange(len(x_labels)) + 0.5)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=11)

    # Muchos sujetos => no mostrar ticks
    ax.set_yticks([])

    ax.set_xlabel("Cluster", fontsize=13)
    ax.set_ylabel("Subjects", fontsize=13)
    ax.set_title("")

    # Primera fila arriba (estilo imshow)
    ax.invert_yaxis()

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.08, pad=0.04)
    cbar.set_label("Number of lesions", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()

    # PNG (raster) para vista rápida
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print("[DONE] Heatmap PNG:", out_png)

    # PDF vector (sin PIL)
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
        print("[DONE] Heatmap PDF (VECTOR):", out_pdf)

    plt.close(fig)
    return tab, tab_plot


# --- llamada (PNG + PDF) ---
heat_png = os.path.join(out_dir, f"{base_name}_HEATMAP_subjects_x_clusters_COUNTS.png")
heat_pdf = os.path.join(out_dir, f"{base_name}_HEATMAP_subjects_x_clusters_COUNTS.pdf")

tab_counts_subj, tab_counts_top = plot_subject_lesiontype_heatmap_counts_vector(
    df_out,
    subj_col=SUBJ_COL,
    cluster_col=CLUSTER_COL,
    out_png=heat_png,
    out_pdf=heat_pdf,
    max_subjects=400,
    sort_subjects=True,
    fig_w=3.5, fig_h=6.0,
    cmap_name="OrRd",
    cluster_order=(0, 1, 2)
)

# (opcional) exportar la tabla
heat_xlsx = os.path.join(out_dir, f"{base_name}_HEATMAP_subjects_x_clusters_COUNTS.xlsx")
with pd.ExcelWriter(heat_xlsx, engine="openpyxl") as writer:
    tab_counts_subj.to_excel(writer, sheet_name="counts_all_subjects")
    tab_counts_top.to_excel(writer, sheet_name=f"counts_top{len(tab_counts_top)}")
print("[DONE] Heatmap tables Excel:", heat_xlsx)






# ======================================================================================
# =================== REPRODUCIBILITY (SUBSAMPLING 80/20 BY SUBJECT) ===================
# ======================================================================================
# Objetivo:
# - Tomar un SUBSAMPLE de sujetos (80% IN) sin reemplazo en cada iteración
# - Ajustar el GMM (k fijo) usando SOLO sujetos IN
# - Predecir labels para TODO el dataset
# - Alinear labels al modelo de referencia (full data) con Hungarian
# - Medir estabilidad: ARI_all, ARI_in, ARI_oob
# - Estabilidad por lesión: proporción de veces que cae en su cluster modal (solo predicciones OOB)
#
# Mantiene: covariance_type="full" y subsampling 80%
# Mejora estabilidad: n_init alto + reg_covar + max_iter mayor

import os
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# ---------------- CONFIG ----------------
N_SUBSAMPLE = 200         # iteraciones (200–1000)
SUBSEED = 42
IN_FRAC = 0.80            # 80% IN, 20% OOB
K_FIXED = best_k_gmm      # k elegido (por BIC u otro)
COV_TYPE = "full"         # pedido por ti
N_INIT = 30               # 🔥 mejora: muchos reinicios
MAX_ITER = 500            # 🔥 mejora: más iteraciones
REG_COVAR = 1e-4          # 🔥 mejora: regularización (prueba 1e-4 o 1e-3)

# Columnas (deben existir en tu script)
# SUBJ_COL = "subject_id"
# CLUSTER_COL = "cluster_gmm_auto"
# df_out = ...
# X_z = ...
# gmm_best = ...
# out_dir = ...
# base_name = ...

# ---------------- Reference labels ----------------
# Modelo entrenado en TODO el dataset (referencia)
labels_ref = gmm_best.predict(X_z).astype(int)
K_REF = int(labels_ref.max()) + 1

# ---------------- Helper: Hungarian alignment ----------------
def align_labels_hungarian(y_pred, y_ref):
    """
    Alinea etiquetas y_pred a y_ref maximizando acuerdos (Hungarian sobre matriz de conteos).
    Devuelve y_pred_aligned y el mapping.
    """
    y_pred = np.asarray(y_pred).astype(int)
    y_ref  = np.asarray(y_ref).astype(int)

    Kp = int(y_pred.max()) + 1
    Kr = int(y_ref.max()) + 1
    K = max(Kp, Kr)

    M = np.zeros((K, K), dtype=int)
    for a, b in zip(y_pred, y_ref):
        if a >= 0 and b >= 0:
            M[a, b] += 1

    row_ind, col_ind = linear_sum_assignment(-M)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    y_aligned = np.array([mapping.get(int(t), int(t)) for t in y_pred], dtype=int)
    return y_aligned, mapping

# ---------------- SUBJECT INDEXING ----------------
# ✅ Importante: subject_id como string
subj_ids = df_out[SUBJ_COL].astype(str).to_numpy()
uniq_subj = np.unique(subj_ids)

# Mapa sujeto -> indices de lesiones
subj_to_idx = {}
for i, s in enumerate(subj_ids):
    subj_to_idx.setdefault(s, []).append(i)

# ---------------- STORAGE ----------------
rng = np.random.default_rng(SUBSEED)

ari_all_list, ari_in_list, ari_oob_list = [], [], []
subs_in_counts, subs_oob_counts = [], []
les_in_counts, les_oob_counts = [], []

# Para estabilidad por lesión usando SOLO predicciones OOB:
# - Contamos votos solo cuando esa lesión estuvo OOB en esa iteración
label_votes_oob = np.zeros((len(df_out), K_REF), dtype=int)
oob_seen_count  = np.zeros((len(df_out),), dtype=int)

# ---------------- MAIN LOOP ----------------
n_in_subj = int(np.round(IN_FRAC * len(uniq_subj)))

for b in range(N_SUBSAMPLE):
    # ---- Elegir sujetos IN (80%) sin reemplazo ----
    in_subj = rng.choice(uniq_subj, size=n_in_subj, replace=False)
    in_set = set(in_subj)

    oob_subj = np.array([s for s in uniq_subj if s not in in_set], dtype=object)

    # ---- Indices de lesiones IN y OOB ----
    idx_in = []
    for s in in_subj:
        idx_in.extend(subj_to_idx[s])
    idx_in = np.array(idx_in, dtype=int)

    idx_oob = []
    for s in oob_subj:
        idx_oob.extend(subj_to_idx[s])
    idx_oob = np.array(idx_oob, dtype=int)

    # ---- Guardar counts ----
    subs_in_counts.append(len(in_subj))
    subs_oob_counts.append(len(oob_subj))
    les_in_counts.append(len(idx_in))
    les_oob_counts.append(len(idx_oob))

    # ---- Fit GMM en IN ----
    X_in = X_z[idx_in, :]

    gmm_b = GaussianMixture(
        n_components=K_FIXED,
        covariance_type=COV_TYPE,   # full (como pediste)
        reg_covar=REG_COVAR,        # 🔥 estabilidad numérica
        random_state=SUBSEED + b,
        n_init=N_INIT,              # 🔥 muchos reinicios
        max_iter=MAX_ITER
    )
    gmm_b.fit(X_in)

    # ---- Predict en TODO ----
    yb_full = gmm_b.predict(X_z).astype(int)

    # ---- Align a labels_ref ----
    yb_aligned, _ = align_labels_hungarian(yb_full, labels_ref)

    # ---- ARI global ----
    ari_all = adjusted_rand_score(labels_ref, yb_aligned)
    ari_all_list.append(float(ari_all))

    # ---- ARI IN y OOB (comparando solo esas lesiones) ----
    if len(idx_in) > 1:
        ari_in = adjusted_rand_score(labels_ref[idx_in], yb_aligned[idx_in])
    else:
        ari_in = np.nan

    if len(idx_oob) > 1:
        ari_oob = adjusted_rand_score(labels_ref[idx_oob], yb_aligned[idx_oob])
    else:
        ari_oob = np.nan

    ari_in_list.append(float(ari_in) if np.isfinite(ari_in) else np.nan)
    ari_oob_list.append(float(ari_oob) if np.isfinite(ari_oob) else np.nan)

    # ---- Votos OOB por lesión (solo cuando estuvo OOB) ----
    for i in idx_oob:
        lab = int(yb_aligned[i])
        if 0 <= lab < K_REF:
            label_votes_oob[i, lab] += 1
        oob_seen_count[i] += 1

    if (b + 1) % 25 == 0:
        print(f"[SUBSAMPLE] {b+1}/{N_SUBSAMPLE} | ARI_all={ari_all:.3f} | ARI_in={ari_in:.3f} | ARI_oob={ari_oob:.3f}")

# ---------------- SUMMARY ----------------
ari_all_arr = np.array(ari_all_list, dtype=float)
ari_in_arr  = np.array(ari_in_list, dtype=float)
ari_oob_arr = np.array(ari_oob_list, dtype=float)

def summarize(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    sd   = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    med  = float(np.median(x))
    iqr1 = float(np.percentile(x, 25))
    iqr3 = float(np.percentile(x, 75))
    return mean, sd, med, (iqr1, iqr3)

m_all, sd_all, med_all, (p25_all, p75_all) = summarize(ari_all_arr)
m_in,  sd_in,  med_in,  (p25_in,  p75_in)  = summarize(ari_in_arr)
m_oob, sd_oob, med_oob, (p25_oob, p75_oob) = summarize(ari_oob_arr)

print("\n[DONE] Subsampling reproducibility (80/20 by subject)")
print(f"Subjects IN (mean):  {np.mean(subs_in_counts):.1f} / {len(uniq_subj)}  | OOB (mean): {np.mean(subs_oob_counts):.1f}")
print(f"Lesions  IN (mean):  {np.mean(les_in_counts):.1f} / {len(df_out)}      | OOB (mean): {np.mean(les_oob_counts):.1f}")
print(f"ARI_all: {m_all:.3f} ± {sd_all:.3f} | med={med_all:.3f} [IQR {p25_all:.3f}-{p75_all:.3f}]")
print(f"ARI_in:  {m_in:.3f} ± {sd_in:.3f} | med={med_in:.3f} [IQR {p25_in:.3f}-{p75_in:.3f}]")
print(f"ARI_oob: {m_oob:.3f} ± {sd_oob:.3f} | med={med_oob:.3f} [IQR {p25_oob:.3f}-{p75_oob:.3f}]")

# ---------------- LESION-LEVEL STABILITY (OOB-ONLY) ----------------
# Proporción modal SOLO en las veces que la lesión estuvo OOB
vote_prop_oob = np.zeros_like(label_votes_oob, dtype=float)
valid = oob_seen_count > 0
vote_prop_oob[valid] = label_votes_oob[valid] / oob_seen_count[valid, None]

lesion_stability_oob = np.full((len(df_out),), np.nan, dtype=float)
lesion_label_mode_oob = np.full((len(df_out),), -1, dtype=int)

lesion_stability_oob[valid] = vote_prop_oob[valid].max(axis=1)
lesion_label_mode_oob[valid] = vote_prop_oob[valid].argmax(axis=1).astype(int)

df_out["oob_seen_count"] = oob_seen_count
df_out["bootstrap_stability_oob"] = lesion_stability_oob
df_out["bootstrap_label_mode_oob"] = lesion_label_mode_oob

stab_summary = pd.DataFrame({
    "IN_FRAC": [IN_FRAC],
    "N_SUBSAMPLE": [N_SUBSAMPLE],
    "K_FIXED": [K_FIXED],
    "covariance_type": [COV_TYPE],
    "n_init": [N_INIT],
    "max_iter": [MAX_ITER],
    "reg_covar": [REG_COVAR],
    "Subjects_IN_mean": [float(np.mean(subs_in_counts))],
    "Subjects_OOB_mean": [float(np.mean(subs_oob_counts))],
    "Lesions_IN_mean": [float(np.mean(les_in_counts))],
    "Lesions_OOB_mean": [float(np.mean(les_oob_counts))],
    "ARI_all_mean": [m_all],
    "ARI_all_sd": [sd_all],
    "ARI_all_median": [med_all],
    "ARI_all_p25": [p25_all],
    "ARI_all_p75": [p75_all],
    "ARI_in_mean": [m_in],
    "ARI_in_sd": [sd_in],
    "ARI_oob_mean": [m_oob],
    "ARI_oob_sd": [sd_oob],
    "lesion_stability_oob_mean": [float(np.nanmean(lesion_stability_oob))],
    "lesion_stability_oob_median": [float(np.nanmedian(lesion_stability_oob))],
})

# ---------------- SAVE EXCEL ----------------
sub_xlsx = os.path.join(out_dir, f"{base_name}_subsampling80_fullcov_reproducibility.xlsx")
with pd.ExcelWriter(sub_xlsx, engine="openpyxl") as writer:
    pd.DataFrame({
        "ARI_all": ari_all_arr,
        "ARI_in": ari_in_arr,
        "ARI_oob": ari_oob_arr
    }).to_excel(writer, index=False, sheet_name="ARI_per_iter")

    stab_summary.to_excel(writer, index=False, sheet_name="summary")

    df_out[[SUBJ_COL, CLUSTER_COL, "oob_seen_count", "bootstrap_label_mode_oob", "bootstrap_stability_oob"]].to_excel(
        writer, index=False, sheet_name="lesion_stability_OOBonly"
    )

print("[DONE] Subsampling Excel:", sub_xlsx)


# ======================================================================================
# ================= REPRODUCIBILITY (80/20 SUBJECT SUBSAMPLING) ========================
# ===================== (CONSENSUS STABILITY) ==========================
# ======================================================================================
# (label-switching proof, practical):
#   - 80% subjects IN (sin reemplazo) / 20% OOB por repetición
#   - Entrena GMM (covariance_type="full") SOLO con lesiones IN
#   - Predice labels para TODAS las lesiones en cada run
#   - Alinea cada run a un "anchor" usando Hungarian (para hacer comparables los IDs)
#   - Calcula:
#       * label_mode por lesión (consenso)
#       * estabilidad por lesión = proporción de runs donde cae en su label_mode
#       * (opcional) entropía por lesión
#   - Reporta estabilidad global + por cluster
#   - Genera plots y guarda Excel con summaries



# ------------------------------ CONFIG ------------------------------------
N_RUNS = 1000          # sube a 1000 como pediste
SEED = 42
IN_FRAC = 0.80

K_FIXED = best_k_gmm   # tu k elegido por BIC
COV_TYPE = "full"      # mantener full covariance

# Columns in df_out
SUBJ_COL = SUBJ_COL        # e.g., "subject_id"
CLUSTER_COL = CLUSTER_COL  # e.g., "cluster_gmm_auto" (del fit original, si existe)

# Output
out_dir = out_dir
base_name = base_name
os.makedirs(out_dir, exist_ok=True)

# ------------------------------ INPUTS ------------------------------------
# X_z: (n_lesions, n_features)
# df_out: dataframe con una fila por lesión

n_lesions = X_z.shape[0]
subj_ids = df_out[SUBJ_COL].astype(str).to_numpy()
uniq_subj = np.unique(subj_ids)
n_subj = len(uniq_subj)

print(f"[INFO] Lesions: {n_lesions} | Subjects: {n_subj} | IN_FRAC={IN_FRAC} | K={K_FIXED} | RUNS={N_RUNS}")

rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# STORAGE
# --------------------------------------------------------------------------
labels_runs = np.empty((N_RUNS, n_lesions), dtype=np.int32)

in_mask_runs = np.zeros((N_RUNS, n_lesions), dtype=bool)
oob_mask_runs = np.zeros((N_RUNS, n_lesions), dtype=bool)

n_in_subj_list, n_oob_subj_list = [], []
n_in_les_list, n_oob_les_list = [], []

# --------------------------------------------------------------------------
# MAIN LOOP: 80/20 subject subsampling
# --------------------------------------------------------------------------
for b in range(N_RUNS):
    n_in = int(np.round(IN_FRAC * n_subj))
    in_subj = rng.choice(uniq_subj, size=n_in, replace=False)

    in_mask = np.isin(subj_ids, in_subj)
    oob_mask = ~in_mask

    in_mask_runs[b] = in_mask
    oob_mask_runs[b] = oob_mask

    idx_in = np.where(in_mask)[0]
    X_in = X_z[idx_in, :]

    gmm_b = GaussianMixture(
        n_components=K_FIXED,
        covariance_type=COV_TYPE,
        random_state=SEED + b,
        n_init=5,
        max_iter=500,
        reg_covar=1e-6
    )
    gmm_b.fit(X_in)

    # labels para TODO el dataset
    labels_runs[b] = gmm_b.predict(X_z).astype(np.int32)

    # bookkeeping
    n_in_subj_list.append(len(in_subj))
    n_oob_subj_list.append(n_subj - len(in_subj))
    n_in_les_list.append(int(in_mask.sum()))
    n_oob_les_list.append(int(oob_mask.sum()))

    if (b + 1) % 50 == 0:
        print(f"[RUN] {b+1}/{N_RUNS} | IN_subj={len(in_subj)} | IN_les={in_mask.sum()}")

# --------------------------------------------------------------------------
# LABEL ALIGNMENT (Hungarian) TO ANCHOR
# --------------------------------------------------------------------------
def align_labels_hungarian(y_pred, y_ref):
    """
    Alinea etiquetas de y_pred a y_ref maximizando diagonal (conteos).
    Devuelve y_pred_aligned.
    """
    y_pred = np.asarray(y_pred).astype(int)
    y_ref  = np.asarray(y_ref).astype(int)

    K = max(int(y_pred.max()) + 1, int(y_ref.max()) + 1)

    M = np.zeros((K, K), dtype=int)
    for a, b in zip(y_pred, y_ref):
        if a >= 0 and b >= 0:
            M[a, b] += 1

    r, c = linear_sum_assignment(-M)
    mapping = {int(rr): int(cc) for rr, cc in zip(r, c)}
    y_aligned = np.vectorize(lambda t: mapping.get(int(t), int(t)))(y_pred)
    return y_aligned.astype(int)

anchor_idx = 0
labels_anchor = labels_runs[anchor_idx].copy()

labels_aligned_runs = np.empty_like(labels_runs)
labels_aligned_runs[anchor_idx] = labels_anchor

for b in range(N_RUNS):
    if b == anchor_idx:
        continue
    labels_aligned_runs[b] = align_labels_hungarian(labels_runs[b], labels_anchor)

# --------------------------------------------------------------------------
# CONSENSUS: MODE + STABILITY + (OPTIONAL) ENTROPY
# --------------------------------------------------------------------------
label_votes = np.zeros((n_lesions, K_FIXED), dtype=np.int32)

# más rápido que loop i por i: acumular con np.add.at
for b in range(N_RUNS):
    yb = labels_aligned_runs[b]
    # proteger por si aparece alguna etiqueta rara fuera de rango
    yb = np.clip(yb, 0, K_FIXED - 1)
    np.add.at(label_votes, (np.arange(n_lesions), yb), 1)

vote_prop = label_votes / np.maximum(label_votes.sum(axis=1, keepdims=True), 1)
lesion_mode = vote_prop.argmax(axis=1).astype(int)
lesion_stability = vote_prop.max(axis=1).astype(float)

# Entropy (opcional)
eps = 1e-12
lesion_entropy = (-np.sum(vote_prop * np.log(vote_prop + eps), axis=1)).astype(float)

print("\n[APPROACH 2 DONE] Anchor-aligned consensus stability")
print(f"Stability mean ± SD: {lesion_stability.mean():.3f} ± {lesion_stability.std(ddof=1):.3f}")
print(f"Stability median [IQR]: {np.median(lesion_stability):.3f} "
      f"[{np.percentile(lesion_stability,25):.3f}-{np.percentile(lesion_stability,75):.3f}]")

# --------------------------------------------------------------------------
# IN/OOB SUMMARY
# --------------------------------------------------------------------------
print("\n[IN/OOB SUMMARY] (across runs)")
print(f"Subjects IN (mean):  {np.mean(n_in_subj_list):.1f} / {n_subj}  | OOB (mean): {np.mean(n_oob_subj_list):.1f}")
print(f"Lesions  IN (mean):  {np.mean(n_in_les_list):.1f} / {n_lesions} | OOB (mean): {np.mean(n_oob_les_list):.1f}")

# --------------------------------------------------------------------------
# SAVE BACK TO df_out
# --------------------------------------------------------------------------
df_out["subsample80_label_mode"] = lesion_mode
df_out["subsample80_stability"] = lesion_stability
df_out["subsample80_entropy"] = lesion_entropy

# --------------------------------------------------------------------------
# CLUSTER-SPECIFIC STABILITY (por cluster consenso o por cluster original si existe)
# --------------------------------------------------------------------------
# Si quieres por cluster original (del fit final), usa CLUSTER_COL si existe.
# Si no, usamos el consenso (lesion_mode).
use_original_cluster = (CLUSTER_COL in df_out.columns)

cluster_group_col = CLUSTER_COL if use_original_cluster else "subsample80_label_mode"
df_out[cluster_group_col] = df_out[cluster_group_col].astype(int)

stab_by_cluster = (
    df_out.groupby(cluster_group_col)["subsample80_stability"]
    .agg(["count", "mean", "std", "median"])
    .reset_index()
    .rename(columns={cluster_group_col: "cluster"})
)
stab_by_cluster["p25"] = df_out.groupby(cluster_group_col)["subsample80_stability"].quantile(0.25).values
stab_by_cluster["p75"] = df_out.groupby(cluster_group_col)["subsample80_stability"].quantile(0.75).values

ent_by_cluster = (
    df_out.groupby(cluster_group_col)["subsample80_entropy"]
    .agg(["count", "mean", "std", "median"])
    .reset_index()
    .rename(columns={cluster_group_col: "cluster"})
)
ent_by_cluster["p25"] = df_out.groupby(cluster_group_col)["subsample80_entropy"].quantile(0.25).values
ent_by_cluster["p75"] = df_out.groupby(cluster_group_col)["subsample80_entropy"].quantile(0.75).values

print("\n[STABILITY BY CLUSTER]")
print(stab_by_cluster)

# --------------------------------------------------------------------------
# REPLICABILITY
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import os

plot_prefix = f"{base_name}_subsample80_runs{N_RUNS}"
os.makedirs(out_dir, exist_ok=True)

# ------------------ choose cluster grouping for plots ------------------
# Recommended: consensus clusters (lesion_mode) because it matches approach 2
cluster_group_col = "subsample80_label_mode"

# If you REALLY want the original clusters from full-data fit, uncomment:
# if (CLUSTER_COL in df_out.columns):
#     cluster_group_col = CLUSTER_COL

df_out[cluster_group_col] = df_out[cluster_group_col].astype(int)

# Ensure we have exactly the clusters present (usually 0..K-1)
clusters_sorted = sorted(df_out[cluster_group_col].unique())


# --------------------------------------------------------------------------
# SAVE EXCEL
# --------------------------------------------------------------------------
summary_df = pd.DataFrame({
    "N_RUNS": [N_RUNS],
    "IN_FRAC": [IN_FRAC],
    "K_FIXED": [K_FIXED],
    "COV_TYPE": [COV_TYPE],
    "subjects_total": [n_subj],
    "lesions_total": [n_lesions],
    "subjects_in_mean": [float(np.mean(n_in_subj_list))],
    "subjects_oob_mean": [float(np.mean(n_oob_subj_list))],
    "lesions_in_mean": [float(np.mean(n_in_les_list))],
    "lesions_oob_mean": [float(np.mean(n_oob_les_list))],
    "stability_mean": [float(lesion_stability.mean())],
    "stability_sd": [float(lesion_stability.std(ddof=1))],
    "stability_median": [float(np.median(lesion_stability))],
    "stability_p25": [float(np.percentile(lesion_stability, 25))],
    "stability_p75": [float(np.percentile(lesion_stability, 75))],
    "entropy_mean": [float(lesion_entropy.mean())],
    "entropy_sd": [float(lesion_entropy.std(ddof=1))],
    "entropy_median": [float(np.median(lesion_entropy))],
    "entropy_p25": [float(np.percentile(lesion_entropy, 25))],
    "entropy_p75": [float(np.percentile(lesion_entropy, 75))],
    "cluster_grouping": ["original" if use_original_cluster else "consensus"],
})

xlsx_path = os.path.join(out_dir, f"{plot_prefix}_approach2_only.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, index=False, sheet_name="summary")
    stab_by_cluster.to_excel(writer, index=False, sheet_name="stability_by_cluster")
    ent_by_cluster.to_excel(writer, index=False, sheet_name="entropy_by_cluster")

    cols_to_save = [SUBJ_COL, CLUSTER_COL, "subsample80_label_mode", "subsample80_stability", "subsample80_entropy"]
    cols_to_save = [c for c in cols_to_save if c in df_out.columns]
    df_out[cols_to_save].to_excel(writer, index=False, sheet_name="lesion_level")

print("\n[DONE] Saved Excel:", xlsx_path)


# ------------------------------------------------------------------
# RUN-LEVEL REPLICABILITY (ALL vs OOB)
# ------------------------------------------------------------------

run_stability_all = []
run_stability_oob = []

for b in range(N_RUNS):

    # ALL lesions
    match_all = labels_aligned_runs[b] == lesion_mode
    run_stability_all.append(np.mean(match_all))

    # OOB lesions only
    mask_oob = oob_mask_runs[b]
    if mask_oob.sum() > 0:
        match_oob = labels_aligned_runs[b][mask_oob] == lesion_mode[mask_oob]
        run_stability_oob.append(np.mean(match_oob))
    else:
        run_stability_oob.append(np.nan)

run_stability_all = np.array(run_stability_all)
run_stability_oob = np.array(run_stability_oob)

print("\n[RUN-LEVEL STABILITY]")
print(f"ALL  mean ± SD: {run_stability_all.mean():.3f} ± {run_stability_all.std(ddof=1):.3f}")
print(f"OOB  mean ± SD: {np.nanmean(run_stability_oob):.3f} ± {np.nanstd(run_stability_oob, ddof=1):.3f}")

# --------------------------------------------------------------------------
# RUN-LEVEL + LESION-LEVEL STABILITY (ALL vs IN vs OOB) + PLOTS (NO CLUSTERS)
# --------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_prefix = f"{base_name}_subsample80_runs{N_RUNS}"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# CONSENSUS LABELS
# -------------------------
consensus = lesion_mode.astype(int)  # shape: (n_lesions,)
n_lesions = consensus.shape[0]

# -------------------------
# RUN-LEVEL STABILITY (agreement with consensus)
#   - ALL: promedio sobre todas las lesiones
#   - IN:  promedio solo sobre lesiones IN de ese run
#   - OOB: promedio solo sobre lesiones OOB de ese run
# -------------------------
run_stability_all = np.full(N_RUNS, np.nan, dtype=float)
run_stability_in  = np.full(N_RUNS, np.nan, dtype=float)
run_stability_oob = np.full(N_RUNS, np.nan, dtype=float)

n_in_lesions_per_run  = np.zeros(N_RUNS, dtype=int)
n_oob_lesions_per_run = np.zeros(N_RUNS, dtype=int)

for b in range(N_RUNS):
    yb = labels_aligned_runs[b]

    # ALL
    run_stability_all[b] = np.mean(yb == consensus)

    # IN
    mask_in = in_mask_runs[b]
    n_in = int(mask_in.sum())
    n_in_lesions_per_run[b] = n_in
    if n_in > 0:
        run_stability_in[b] = np.mean(yb[mask_in] == consensus[mask_in])

    # OOB
    mask_oob = oob_mask_runs[b]
    n_oob = int(mask_oob.sum())
    n_oob_lesions_per_run[b] = n_oob
    if n_oob > 0:
        run_stability_oob[b] = np.mean(yb[mask_oob] == consensus[mask_oob])

print("\n[RUN-LEVEL STABILITY vs CONSENSUS]")
print(f"ALL mean ± SD: {np.nanmean(run_stability_all):.3f} ± {np.nanstd(run_stability_all, ddof=1):.3f}")
print(f"IN  mean ± SD: {np.nanmean(run_stability_in):.3f} ± {np.nanstd(run_stability_in, ddof=1):.3f}")
print(f"OOB mean ± SD: {np.nanmean(run_stability_oob):.3f} ± {np.nanstd(run_stability_oob, ddof=1):.3f}")

# -------------------------
# LESION-LEVEL STABILITY
#   A) ALL-runs (como antes): max proportion en vote_prop (ya lo puedes tener)
#      si NO lo tienes, lo recalculamos aquí desde labels_aligned_runs
#   B) OOB-only: votos solo cuando esa lesión estuvo OOB en ese run
# -------------------------

# A) ALL-runs lesion stability
label_votes_all = np.zeros((n_lesions, K_FIXED), dtype=np.int32)
for b in range(N_RUNS):
    yb = np.clip(labels_aligned_runs[b], 0, K_FIXED - 1)
    np.add.at(label_votes_all, (np.arange(n_lesions), yb), 1)

vote_prop_all = label_votes_all / np.maximum(label_votes_all.sum(axis=1, keepdims=True), 1)
lesion_label_mode_all = vote_prop_all.argmax(axis=1).astype(int)
lesion_stability_all = vote_prop_all.max(axis=1).astype(float)

# B) OOB-only lesion stability
label_votes_oob = np.zeros((n_lesions, K_FIXED), dtype=np.int32)
oob_seen_count  = np.zeros((n_lesions,), dtype=np.int32)

for b in range(N_RUNS):
    mask_oob = oob_mask_runs[b]
    if mask_oob.sum() == 0:
        continue

    yb = np.clip(labels_aligned_runs[b], 0, K_FIXED - 1)

    idx = np.where(mask_oob)[0]
    np.add.at(label_votes_oob, (idx, yb[idx]), 1)
    oob_seen_count[idx] += 1

valid_oob = oob_seen_count > 0
vote_prop_oob = np.zeros_like(label_votes_oob, dtype=float)
vote_prop_oob[valid_oob] = label_votes_oob[valid_oob] / oob_seen_count[valid_oob, None]

lesion_label_mode_oob = np.full(n_lesions, -1, dtype=int)
lesion_stability_oob  = np.full(n_lesions, np.nan, dtype=float)
lesion_label_mode_oob[valid_oob] = vote_prop_oob[valid_oob].argmax(axis=1).astype(int)
lesion_stability_oob[valid_oob]  = vote_prop_oob[valid_oob].max(axis=1).astype(float)

print("\n[LESION-LEVEL STABILITY]")
print(f"ALL-runs mean ± SD: {lesion_stability_all.mean():.3f} ± {lesion_stability_all.std(ddof=1):.3f}")
print(f"OOB-only mean ± SD: {np.nanmean(lesion_stability_oob):.3f} ± {np.nanstd(lesion_stability_oob, ddof=1):.3f}")
print(f"OOB-only coverage (mean oob_seen_count): {float(np.mean(oob_seen_count)):.1f} runs per lesion")

# Guardar en df_out
df_out["consensus_label_mode_all"] = lesion_label_mode_all
df_out["stability_allruns"] = lesion_stability_all

df_out["oob_seen_count"] = oob_seen_count
df_out["consensus_label_mode_oob"] = lesion_label_mode_oob
df_out["stability_oobonly"] = lesion_stability_oob

# -------------------------
# PLOTS (OOB)
# -------------------------

x = np.arange(1, N_RUNS + 1)


# Plot: solo OOB (línea)
plt.figure(figsize=(8, 4))
plt.plot(x, run_stability_oob, label="OOB", linewidth=2)
plt.ylim(0, 1)  # <- pedido
plt.xlim(0, 1000) 
plt.xlabel("Run")
plt.ylabel("Proportion matching consensus")
plt.legend(frameon=False)
plt.tight_layout()
run_oob_path = os.path.join(out_dir, f"{plot_prefix}_run_stability_OOB_only.png")
plt.savefig(run_oob_path, dpi=300)
plt.show()

print("\n[PLOTS SAVED]")

print(run_oob_path)

# -------------------------
# 95% CI helper (sobre promedio)
# -------------------------
def mean_sd_ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else np.nan
    ci_lo = m - 1.96 * se
    ci_hi = m + 1.96 * se
    return m, sd, ci_lo, ci_hi, n

m_all, sd_all, ci_all_lo, ci_all_hi, n_all = mean_sd_ci95(run_stability_all)
m_in,  sd_in,  ci_in_lo,  ci_in_hi,  n_in  = mean_sd_ci95(run_stability_in)
m_oob, sd_oob, ci_oob_lo, ci_oob_hi, n_oob = mean_sd_ci95(run_stability_oob)

m_lall, sd_lall, ci_lall_lo, ci_lall_hi, n_lall = mean_sd_ci95(lesion_stability_all)
m_loob, sd_loob, ci_loob_lo, ci_loob_hi, n_loob = mean_sd_ci95(lesion_stability_oob)




# -------------------------
# SAVE EXCEL (separado ALL / IN / OOB)
# -------------------------
run_level_df = pd.DataFrame({
    "run": np.arange(N_RUNS),
    "n_in_lesions": n_in_lesions_per_run,
    "n_oob_lesions": n_oob_lesions_per_run,
    "stability_ALL": run_stability_all,
    "stability_IN": run_stability_in,
    "stability_OOB": run_stability_oob
})

summary_df = pd.DataFrame({
    "N_RUNS": [N_RUNS],
    "IN_FRAC": [IN_FRAC],
    "K_FIXED": [K_FIXED],
    "COV_TYPE": [COV_TYPE],

    # Run-level stability (vs consensus)
    "run_stability_ALL_mean": [m_all],
    "run_stability_ALL_sd": [sd_all],
    "run_stability_ALL_ci95_lo": [ci_all_lo],
    "run_stability_ALL_ci95_hi": [ci_all_hi],
    "run_stability_ALL_n": [n_all],

    "run_stability_IN_mean": [m_in],
    "run_stability_IN_sd": [sd_in],
    "run_stability_IN_ci95_lo": [ci_in_lo],
    "run_stability_IN_ci95_hi": [ci_in_hi],
    "run_stability_IN_n": [n_in],

    "run_stability_OOB_mean": [m_oob],
    "run_stability_OOB_sd": [sd_oob],
    "run_stability_OOB_ci95_lo": [ci_oob_lo],
    "run_stability_OOB_ci95_hi": [ci_oob_hi],
    "run_stability_OOB_n": [n_oob],

    # OOB coverage
    "oob_seen_count_mean": [float(np.mean(oob_seen_count))],
    "oob_seen_count_min": [int(np.min(oob_seen_count))],
    "oob_seen_count_max": [int(np.max(oob_seen_count))],
})

lesion_all_df = df_out[[SUBJ_COL]].copy()
if CLUSTER_COL in df_out.columns:
    lesion_all_df[CLUSTER_COL] = df_out[CLUSTER_COL]
lesion_all_df["consensus_label_mode_all"] = df_out["consensus_label_mode_all"]
lesion_all_df["stability_allruns"] = df_out["stability_allruns"]

lesion_oob_df = df_out[[SUBJ_COL]].copy()
if CLUSTER_COL in df_out.columns:
    lesion_oob_df[CLUSTER_COL] = df_out[CLUSTER_COL]
lesion_oob_df["oob_seen_count"] = df_out["oob_seen_count"]
lesion_oob_df["consensus_label_mode_oob"] = df_out["consensus_label_mode_oob"]
lesion_oob_df["stability_oobonly"] = df_out["stability_oobonly"]

xlsx_path = os.path.join(out_dir, f"{plot_prefix}_stability_ALL_IN_OOB.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, index=False, sheet_name="summary")
    run_level_df.to_excel(writer, index=False, sheet_name="run_level_ALL_IN_OOB")
    lesion_all_df.to_excel(writer, index=False, sheet_name="lesion_level_ALL")
    

print("\n[DONE] Saved Excel:", xlsx_path)

