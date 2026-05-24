# -*- coding: utf-8 -*-
"""


Steps performed by this script:

1. Builds subject-level lesion counts for the three WMH clusters.
2. Computes total WMH lesion volume and log-transformed lesion volume.
3. Fits binomial GLM models to test which variables predict the frequency of each WMH lesion cluster.
4. Applies FDR correction to GLM p-values.
5. Generates forest plots for the GLM results.
6. Builds subject-level OLS datasets for GM and CSF analyses.
7. Fits robust OLS models predicting GM and CSFfrom lesion counts, total WMH volume, and covariates.
8. Runs extra GM models testing total WMH burden before and after correction by lesion-class volume.
9. Applies FDR correction to OLS and extra GM models.
10. Compares lesion size distributions across clusters using Mann–Whitney tests and Cohen’s d.

"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from statsmodels.stats.multitest import multipletests


# =========================
# 0) EXPORT SETTINGS
# =========================
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.02


# =========================
# 1) CONFIG
# =========================
INPUT_XLSX = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\Ampliacion\clustering\WMH_clusters_metrics-all_with_clinic_final_CLUSTERING_GLM.xlsx"
SHEET_NAME = 0

OUT_DIR_BASE = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\Ampliacion\clustering"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(OUT_DIR_BASE, f"subject_level_freq_{RUN_TAG}")
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_XLSX = os.path.join(
    OUT_DIR,
    "subject_level_frequency_GLM_OLS_MAIN_extra_GM_CORRECTED.xlsx"
)

PLOT_FILENAME_GLM = "forest_GLM_3clusters_logOR_FDR_LOG_LesionVolumeT2_WITH_GPO.png"
PLOT_FILENAME_OLS_MAIN = "forest_OLS_MAIN_GM_T2_CSF_T2_stdBeta_FDR.png"
PLOT_FILENAME_EXTRA_GM = "wmh_vertical_beta_plot_FDR_HC1.png"

FIG_W = 7.0
FIG_H = 2.9

EFFECTS_CONT = [
    "Age",
    "ΔPP", "ΔHR", "ΔWGTKG",
    "Education",
    "APOE_E4_count"
]

COVARS_CATEG_GLM = ["Sex", "Gpo"]
COVARS_CONT_GLM = ["log_Lesion_Volume_T2"]

TERM_ORDER_PLOT = [
    "APOE_E4_count", "Sex", "Age", "Education",
    "ΔPP", "ΔHR", "ΔWGTKG"
]

TERM_LABELS = {
    "APOE_E4_count": "APOE ε4",
    "Sex": "Sex (Male)",
    "Age": "Age",
    "Education": "Education",
    "ΔPP": "ΔPP",
    "ΔHR": "ΔHR",
    "ΔWGTKG": "ΔWeight"
}


# =========================
# 2) HELPERS
# =========================
def apoe_e4_count(geno):
    if pd.isna(geno):
        return np.nan
    s = str(geno).upper().replace(" ", "")
    return float(s.count("E4"))


def assert_unique_col(df, col):
    if list(df.columns).count(col) != 1:
        raise ValueError(f"Problema con columna: {col} falta o duplicada")


def safe_to_numeric(df, col):
    assert_unique_col(df, col)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _star_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def match_tick_style(ax):
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11, pad=4)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.title.set_size(12)


def _bucket_term(term):
    if term.startswith("Sex_"):
        return "Sex"
    if term.endswith("_z"):
        return term.replace("_z", "")
    return term


def add_fdr_by_group(df, p_col="p", group_cols=(), method="fdr_bh", out_col="p_fdr"):
    df = df.copy()
    df[out_col] = np.nan

    if group_cols is None or len(group_cols) == 0:
        pvals = df[p_col].to_numpy(dtype=float)
        ok = np.isfinite(pvals)
        if ok.sum() > 0:
            _, p_corr, _, _ = multipletests(pvals[ok], method=method)
            df.loc[df.index[ok], out_col] = p_corr
        return df

    for _, sub_idx in df.groupby(list(group_cols)).groups.items():
        sub_idx = np.array(list(sub_idx))
        pvals = df.loc[sub_idx, p_col].to_numpy(dtype=float)
        ok = np.isfinite(pvals)
        if ok.sum() > 0:
            _, p_corr, _, _ = multipletests(pvals[ok], method=method)
            df.loc[sub_idx[ok], out_col] = p_corr

    return df


def residualize_variable(df_in, y_col, adjust_cols):
    """
    Residualiza y_col por adjust_cols.
    Devuelve residuos alineados al índice original.
    """
    tmp = df_in[[y_col] + adjust_cols].copy()
    tmp = tmp.apply(pd.to_numeric, errors="coerce")

    valid_idx = tmp.dropna().index
    resid = pd.Series(index=df_in.index, dtype=float)

    if len(valid_idx) < 3:
        return resid

    X = sm.add_constant(
        tmp.loc[valid_idx, adjust_cols],
        has_constant="add"
    ).astype(float)

    y = tmp.loc[valid_idx, y_col].astype(float)

    res = sm.OLS(y, X).fit()
    resid.loc[valid_idx] = res.resid

    return resid


# =========================
# 3) GLM FUNCTIONS
# =========================
def fit_binomial_freq_glm(df_subj, success_col, effect_terms_z):
    y = (df_subj[success_col] / df_subj["n_total"]).astype(float).to_numpy()

    X = df_subj.copy()
    X = pd.get_dummies(X, columns=COVARS_CATEG_GLM, drop_first=True)

    model_cols = (
        effect_terms_z +
        [f"{c}_z" for c in COVARS_CONT_GLM] +
        [c for c in X.columns if c.startswith("Sex_") or c.startswith("Gpo_")]
    )

    X = sm.add_constant(X[model_cols], has_constant="add").astype(float)
    w = df_subj["n_total"].astype(float).to_numpy()

    res = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=w
    ).fit(cov_type="HC1")

    return res, X.columns


def make_glm_table(res, cols, outcome_label):
    rows = []

    for term in cols:
        if term == "const":
            continue

        beta = float(res.params[term])
        se = float(res.bse[term])
        p = float(res.pvalues[term])

        rows.append({
            "outcome_frequency": outcome_label,
            "term": term,
            "beta": beta,
            "SE": se,
            "OR": float(np.exp(beta)),
            "CI95_low": float(np.exp(beta - 1.96 * se)),
            "CI95_high": float(np.exp(beta + 1.96 * se)),
            "p": p
        })

    return pd.DataFrame(rows)


# =========================
# 4) OLS FUNCTIONS
# =========================
def fit_ols_standardizedY(df_subj, ycol, cont_cols, dummy_cols=("Sex", "Gpo")):
    """
    Estandariza Y y predictores continuos.
    Ajusta OLS con cov_type HC1.
    """
    X = df_subj.copy()

    X = pd.get_dummies(
        X,
        columns=list(dummy_cols),
        drop_first=True
    )

    dum_cols = [
        c for c in X.columns
        if c.startswith("Sex_") or c.startswith("Gpo_")
    ]

    sc = StandardScaler()

    X_cont_z = pd.DataFrame(
        sc.fit_transform(X[cont_cols].astype(float)),
        columns=[f"{c}_z" for c in cont_cols],
        index=X.index
    )

    X_dum = X[dum_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    Xmat = pd.concat([X_cont_z, X_dum], axis=1)
    Xmat = sm.add_constant(Xmat, has_constant="add")
    Xmat = Xmat.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    y_raw = pd.to_numeric(df_subj[ycol], errors="coerce").to_numpy(dtype=float)

    mu = np.nanmean(y_raw)
    sd = np.nanstd(y_raw, ddof=0)

    if sd == 0 or not np.isfinite(sd):
        raise ValueError(f"La variable dependiente {ycol} tiene SD=0 o no finita.")

    y_z = (y_raw - mu) / sd

    res = sm.OLS(y_z, Xmat).fit(cov_type="HC1")

    return res, Xmat.columns


def make_ols_table(res, cols, outcome, model_name):
    rows = []

    for term in cols:
        if term == "const":
            continue

        beta = float(res.params[term])
        se = float(res.bse[term])

        rows.append({
            "model": model_name,
            "outcome": outcome,
            "term": term,
            "beta_std": beta,
            "SE": se,
            "t": float(res.tvalues[term]),
            "p": float(res.pvalues[term]),
            "CI95_low": float(beta - 1.96 * se),
            "CI95_high": float(beta + 1.96 * se),
            "N": int(res.nobs),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj)
        })

    return pd.DataFrame(rows)


# =========================
# 5) PLOT FUNCTIONS
# =========================
def plot_ols_main_forest(
    ols_df_in,
    out_dir,
    filename,
    outcomes=("GM_T2", "CSF_T2"),
    show=True
):
    lesion_terms = [
        "n0_z",
        "n1_z",
        "n2_z",
        "log_Lesion_Volume_T2_z"
    ]

    labels = {
        "n0_z": "L$_{1}$ count",
        "n1_z": "L$_{2}$ count",
        "n2_z": "L$_{3}$ count",
        "log_Lesion_Volume_T2_z": r"Total WMH$_{vol}$"
    }

    dfp = ols_df_in[ols_df_in["term"].isin(lesion_terms)].copy()

    if dfp.empty:
        print("⚠ No se encontraron términos de lesión en OLS principal.")
        return None, None, None

    dfp_xlim = dfp[dfp["outcome"].isin(outcomes)].copy()

    lo_all = dfp_xlim["CI95_low"].to_numpy(dtype=float)
    hi_all = dfp_xlim["CI95_high"].to_numpy(dtype=float)
    ok_all = np.isfinite(lo_all) & np.isfinite(hi_all)

    if np.any(ok_all):
        x_min = float(np.nanmin(lo_all[ok_all]))
        x_max = float(np.nanmax(hi_all[ok_all]))
        pad_x = 0.20 * (x_max - x_min + 1e-12)
        ols_xlim = (x_min - pad_x, x_max + pad_x)
    else:
        ols_xlim = None

    fig, axes = plt.subplots(
        1,
        len(outcomes),
        figsize=(FIG_W, FIG_H * 0.6),
        sharey=True
    )

    fig.subplots_adjust(top=0.90, bottom=0.22, wspace=0.30)

    if len(outcomes) == 1:
        axes = [axes]

    y_ticks = np.arange(len(lesion_terms), dtype=float)
    y_pts = y_ticks.copy()
    pad_y = 0.35

    for ax, out in zip(axes, outcomes):
        sub = dfp[dfp["outcome"] == out].set_index("term").reindex(lesion_terms)

        beta = sub["beta_std"].to_numpy(dtype=float)
        lo = sub["CI95_low"].to_numpy(dtype=float)
        hi = sub["CI95_high"].to_numpy(dtype=float)
        pval = sub["p_fdr"].to_numpy(dtype=float)

        ok = np.isfinite(beta) & np.isfinite(lo) & np.isfinite(hi)

        if np.any(ok):
            xerr = np.vstack([beta[ok] - lo[ok], hi[ok] - beta[ok]])
            ax.errorbar(beta[ok], y_pts[ok], xerr=xerr, fmt="o", capsize=3)

        ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black")

        outcome_labels = {
            "GM_T2": "Grey matter",
            "CSF_T2": "Cerebrospinal fluid"
        }

        ax.set_title(outcome_labels.get(out, out))
        ax.set_xlabel("Standardized beta 95% CI")
        ax.set_yticks(y_ticks)
        ax.set_ylim(-pad_y, (len(lesion_terms) - 1) + pad_y)

        if ols_xlim is not None:
            ax.set_xlim(ols_xlim)

        if ax is axes[0]:
            ax.set_yticklabels([labels[t] for t in lesion_terms])
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)

        for i in range(len(lesion_terms)):
            if np.isfinite(beta[i]) and np.isfinite(pval[i]):
                s = _star_from_p(float(pval[i]))
                if s:
                    ax.text(
                        beta[i],
                        y_pts[i] - 0.05,
                        s,
                        ha="center",
                        va="bottom",
                        fontsize=12
                    )

        ax.invert_yaxis()
        match_tick_style(ax)

    png_path = os.path.join(out_dir, filename)
    svg_path = os.path.join(out_dir, filename.replace(".png", ".svg"))
    pdf_path = os.path.join(out_dir, filename.replace(".png", ".pdf"))

    fig.savefig(png_path, dpi=300, bbox_inches=None)
    fig.savefig(svg_path, bbox_inches=None)
    fig.savefig(pdf_path, format="pdf", bbox_inches=None)

    print("📊 OLS MAIN Plot PNG:", png_path)
    print("📊 OLS MAIN Plot SVG:", svg_path)
    print("📊 OLS MAIN Plot PDF:", pdf_path)

    if show:
        plt.show()

    plt.close(fig)

    return png_path, svg_path, pdf_path


# =========================
# 6) LOAD DATA
# =========================
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

dups = df.columns[df.columns.duplicated()].tolist()

if dups:
    print("⚠ Columnas duplicadas:", dups)
    df = df.loc[:, ~df.columns.duplicated()].copy()

assert_unique_col(df, "cluster_gmm_auto")

df["cluster_gmm_auto"] = (
    pd.to_numeric(df["cluster_gmm_auto"], errors="coerce")
    .round()
    .astype("Int64")
)

print("✅ Conteo clusters:", df["cluster_gmm_auto"].value_counts(dropna=False).to_dict())


# =========================
# 7) SUBJECT-LEVEL BUILD GLM
# =========================
ct = df.groupby(["subject_id", "cluster_gmm_auto"]).size().unstack(fill_value=0)

for k in [0, 1, 2]:
    if k not in ct.columns:
        ct[k] = 0

ct = ct[[0, 1, 2]]
ct.columns = ["n0", "n1", "n2"]
ct["n_total"] = ct.sum(axis=1)

need_cols = [
    "Age", "Sex", "Gpo", "Education", "APOE_GENOTYPE",
    "ΔPP", "ΔHR", "ΔWGTKG",
    "Lesion Volume (ml)_T2"
]

for c in need_cols + ["Size", "subject_id", "cluster_gmm_auto"]:
    assert_unique_col(df, c)

subj = df.groupby("subject_id")[need_cols].first()
subj["APOE_E4_count"] = subj["APOE_GENOTYPE"].apply(apoe_e4_count)

size_sum = df.groupby("subject_id")["Size"].sum().rename("size_total")

subj = subj.join(ct).join(size_sum).reset_index()

subj["Lesion Volume (ml)_T2"] = pd.to_numeric(
    subj["Lesion Volume (ml)_T2"],
    errors="coerce"
)

subj["log_Lesion_Volume_T2"] = np.log1p(subj["Lesion Volume (ml)_T2"])

numeric_cols = [
    "Age", "Education", "APOE_E4_count",
    "Lesion Volume (ml)_T2",
    "log_Lesion_Volume_T2",
    "ΔPP", "ΔHR", "ΔWGTKG",
    "n0", "n1", "n2", "n_total"
]

for c in numeric_cols:
    subj = safe_to_numeric(subj, c)

subj["Gpo"] = subj["Gpo"].astype(str).str.strip()
subj["Sex"] = subj["Sex"].astype(str).str.strip()

subj["Gpo"] = pd.Categorical(
    subj["Gpo"],
    categories=["CN", "MCI", "AD", "PD"],
    ordered=False
)

req_glm = (
    EFFECTS_CONT +
    COVARS_CONT_GLM +
    ["Sex", "Gpo", "n0", "n1", "n2", "n_total"]
)

d = subj.dropna(subset=req_glm).copy().reset_index(drop=True)

print("N sujetos GLM:", len(d))

if len(d) == 0:
    na_counts = subj[req_glm].isna().sum().sort_values(ascending=False)
    print("⚠ d vacío. NA counts:\n", na_counts)
    raise ValueError("d quedó vacío tras dropna subset=req_glm.")


# =========================
# 8) Z-SCORE GLM
# =========================
cont_to_z = list(dict.fromkeys(EFFECTS_CONT + COVARS_CONT_GLM))

bad_cols = [c for c in cont_to_z if not pd.api.types.is_numeric_dtype(d[c])]

if bad_cols:
    raise ValueError(f"No numéricas en cont_to_z: {bad_cols}")

Z = StandardScaler().fit_transform(d[cont_to_z].astype(float))

for i, c in enumerate(cont_to_z):
    d[f"{c}_z"] = Z[:, i]

effect_terms_z = [f"{c}_z" for c in EFFECTS_CONT]


# =========================
# 9) FIT GLM
# =========================
glm_tables = []

for label, succ in [("cluster0", "n0"), ("cluster1", "n1"), ("cluster2", "n2")]:
    res, cols = fit_binomial_freq_glm(d, succ, effect_terms_z)
    outlabel = f"{label}: {succ}/n_total"
    glm_tables.append(make_glm_table(res, cols, outlabel))

full_params_df = pd.concat(glm_tables, ignore_index=True)

full_params_df = add_fdr_by_group(
    full_params_df,
    p_col="p",
    group_cols=("outcome_frequency",),
    out_col="p_fdr"
)

sex_dummy_terms = [
    t for t in full_params_df["term"].unique()
    if t.startswith("Sex_")
]

plot_terms_set = set(effect_terms_z) | set(sex_dummy_terms)

main_df = full_params_df[full_params_df["term"].isin(plot_terms_set)].copy()
main_df["term_plot"] = main_df["term"].apply(_bucket_term)

main_df = (
    main_df
    .sort_values("p")
    .drop_duplicates(subset=["outcome_frequency", "term_plot"], keep="first")
)


# =========================
# 10) EXPORT GLM
# =========================
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    d.to_excel(writer, sheet_name="GLM_subject_data", index=False)

    main_df.sort_values(["outcome_frequency", "term_plot"]).to_excel(
        writer,
        sheet_name="GLM_main_for_plot_FDR",
        index=False
    )

    full_params_df.sort_values(["outcome_frequency", "p_fdr"]).to_excel(
        writer,
        sheet_name="GLM_all_params_FDR",
        index=False
    )


# =========================
# 11) PLOT GLM
# =========================
def _cluster_key(s):
    try:
        return int(s.split(":")[0].replace("cluster", ""))
    except Exception:
        return 999


outcomes_glm = sorted(main_df["outcome_frequency"].unique(), key=_cluster_key)[:3]
panel_titles = [r"L$_1$", r"L$_2$", r"L$_3$"]

glm_sub_all = main_df[main_df["outcome_frequency"].isin(outcomes_glm)].copy()
glm_sub_all = glm_sub_all[glm_sub_all["term_plot"].isin(TERM_ORDER_PLOT)].copy()

glm_lo_all = (glm_sub_all["beta"] - 1.96 * glm_sub_all["SE"]).to_numpy(dtype=float)
glm_hi_all = (glm_sub_all["beta"] + 1.96 * glm_sub_all["SE"]).to_numpy(dtype=float)

ok_all = np.isfinite(glm_lo_all) & np.isfinite(glm_hi_all)

if np.any(ok_all):
    x_min = float(np.nanmin(glm_lo_all[ok_all]))
    x_max = float(np.nanmax(glm_hi_all[ok_all]))
    pad = 0.08 * (x_max - x_min + 1e-12)
    glm_xlim = (x_min - pad, x_max + pad)
else:
    glm_xlim = None

fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H * 0.8), sharey=True)
fig.subplots_adjust(wspace=0.35)

y = np.arange(len(TERM_ORDER_PLOT)) * 0.8

for j, (ax, out) in enumerate(zip(axes, outcomes_glm)):
    sub = main_df[main_df["outcome_frequency"] == out].copy()
    sub = sub.set_index("term_plot").reindex(TERM_ORDER_PLOT)

    beta = sub["beta"].to_numpy(dtype=float)
    lo = (sub["beta"] - 1.96 * sub["SE"]).to_numpy(dtype=float)
    hi = (sub["beta"] + 1.96 * sub["SE"]).to_numpy(dtype=float)
    pval = sub["p_fdr"].to_numpy(dtype=float)

    ok = np.isfinite(beta) & np.isfinite(lo) & np.isfinite(hi)

    if np.any(ok):
        xerr = np.vstack([beta[ok] - lo[ok], hi[ok] - beta[ok]])
        ax.errorbar(beta[ok], y[ok], xerr=xerr, fmt="o", capsize=3)

    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_title(panel_titles[j])
    ax.set_xlabel("log odds ratio")
    ax.set_xticks([-0.25, 0, 0.25])
    ax.set_xticklabels(["-0.25", "0", "0.25"])
    ax.set_yticks(y)

    if glm_xlim is not None:
        ax.set_xlim(glm_xlim)

    if j == 0:
        ax.set_yticklabels([TERM_LABELS.get(t, t) for t in TERM_ORDER_PLOT])
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)

    for i in range(len(TERM_ORDER_PLOT)):
        if np.isfinite(beta[i]) and np.isfinite(pval[i]):
            s = _star_from_p(float(pval[i]))
            if s:
                ax.text(
                    beta[i],
                    y[i] - 0.03,
                    s,
                    ha="center",
                    va="bottom",
                    fontsize=12
                )

    ax.invert_yaxis()
    match_tick_style(ax)

fig.tight_layout()

png_glm = os.path.join(OUT_DIR, PLOT_FILENAME_GLM)
svg_glm = os.path.join(OUT_DIR, PLOT_FILENAME_GLM.replace(".png", ".svg"))
pdf_glm = os.path.join(OUT_DIR, PLOT_FILENAME_GLM.replace(".png", ".pdf"))

fig.savefig(png_glm, dpi=300)
fig.savefig(svg_glm)
fig.savefig(pdf_glm, format="pdf")

plt.show()
plt.close(fig)

print("📊 GLM Plot PNG:", png_glm)
print("📊 GLM Plot SVG:", svg_glm)
print("📊 GLM Plot PDF:", pdf_glm)


# =========================
# 12) SUBJECT-LEVEL BUILD OLS
# =========================
ct_ols = df.groupby(["subject_id", "cluster_gmm_auto"]).size().unstack(fill_value=0)

for k in [0, 1, 2]:
    if k not in ct_ols.columns:
        ct_ols[k] = 0

ct_ols = ct_ols[[0, 1, 2]]
ct_ols.columns = ["n0", "n1", "n2"]
ct_ols["n_total"] = ct_ols[["n0", "n1", "n2"]].sum(axis=1)

ss_ols = df.groupby(["subject_id", "cluster_gmm_auto"])["Size"].sum().unstack(fill_value=0)

for k in [0, 1, 2]:
    if k not in ss_ols.columns:
        ss_ols[k] = 0.0

ss_ols = ss_ols[[0, 1, 2]]
ss_ols.columns = ["size0", "size1", "size2"]
ss_ols["size_total"] = ss_ols[["size0", "size1", "size2"]].sum(axis=1)

need_base = [
    "Age", "Sex", "Gpo", "Education",
    "GM_T2", "CSF_T2", "TIV",
    "Lesion Volume (ml)_T2"
]

for c in need_base + ["subject_id"]:
    assert_unique_col(df, c)

subj_base = df.groupby("subject_id")[need_base].first()

subj_ols = subj_base.join(ct_ols).join(ss_ols).reset_index()

subj_ols["Lesion Volume (ml)_T2"] = pd.to_numeric(
    subj_ols["Lesion Volume (ml)_T2"],
    errors="coerce"
)

subj_ols["log_Lesion_Volume_T2"] = np.log1p(subj_ols["Lesion Volume (ml)_T2"])
subj_ols["log_size_total"] = np.log1p(subj_ols["size_total"])

num_cols = [
    "Age", "Education",
    "GM_T2", "CSF_T2", "TIV",
    "n0", "n1", "n2", "n_total",
    "size0", "size1", "size2", "size_total",
    "Lesion Volume (ml)_T2",
    "log_Lesion_Volume_T2",
    "log_size_total"
]

for c in num_cols:
    subj_ols = safe_to_numeric(subj_ols, c)

subj_ols["Gpo"] = subj_ols["Gpo"].astype(str).str.strip()
subj_ols["Sex"] = subj_ols["Sex"].astype(str).str.strip()

subj_ols["Gpo"] = pd.Categorical(
    subj_ols["Gpo"],
    categories=["CN", "MCI", "AD", "PD"],
    ordered=False
)

req_ols = [
    "Sex", "Gpo",
    "Age", "Education",
    "n0", "n1", "n2",
    "TIV",
    "GM_T2", "CSF_T2",
    "log_Lesion_Volume_T2"
]

d_ols = subj_ols.dropna(subset=req_ols).copy().reset_index(drop=True)

print("N sujetos OLS:", len(d_ols))

if len(d_ols) == 0:
    na_counts = subj_ols[req_ols].isna().sum().sort_values(ascending=False)
    print("⚠ d_ols vacío. NA counts:\n", na_counts)
    raise ValueError("d_ols quedó vacío tras dropna subset=req_ols.")


# =========================
# 13) OLS PRINCIPAL
# =========================
ols_outcomes = ["GM_T2", "CSF_T2"]

CONT_COLS_OLS_MAIN = [
    "Age",
    "Education",
    "TIV",
    "n0",
    "n1",
    "n2",
    "log_Lesion_Volume_T2"
]

ols_tables_main = []

for ycol in ols_outcomes:
    res, cols = fit_ols_standardizedY(
        d_ols,
        ycol,
        cont_cols=CONT_COLS_OLS_MAIN,
        dummy_cols=("Sex", "Gpo")
    )

    ols_tables_main.append(
        make_ols_table(
            res,
            cols,
            outcome=ycol,
            model_name="OLS_MAIN"
        )
    )

ols_df_main = pd.concat(ols_tables_main, ignore_index=True)

ols_df_main = add_fdr_by_group(
    ols_df_main,
    p_col="p",
    group_cols=("outcome",),
    out_col="p_fdr"
)


# =========================
# 14) EXTRA WMH → GM ANALYSIS SIN GPO
#     CORREGIDO: volumen por clase + HC1 + FDR por modelo
# =========================
d_extra = d_ols.copy()

# Log volúmenes por clase lesional
d_extra["log_size0"] = np.log1p(d_extra["size0"])
d_extra["log_size1"] = np.log1p(d_extra["size1"])
d_extra["log_size2"] = np.log1p(d_extra["size2"])

# Residualizar log WMH total por volumen L1/L2/L3
d_extra["log_Lesion_resid_L1vol"] = residualize_variable(
    d_extra,
    y_col="log_Lesion_Volume_T2",
    adjust_cols=["log_size0"]
)

d_extra["log_Lesion_resid_L2vol"] = residualize_variable(
    d_extra,
    y_col="log_Lesion_Volume_T2",
    adjust_cols=["log_size1"]
)

d_extra["log_Lesion_resid_L3vol"] = residualize_variable(
    d_extra,
    y_col="log_Lesion_Volume_T2",
    adjust_cols=["log_size2"]
)

extra_models = {
    "GM_logWM_uncorrected_noGpo": [
        "Age",
        "Education",
        "TIV",
        "log_Lesion_Volume_T2"
    ],
    "GM_logWM_corrected_by_L1_volume": [
        "Age",
        "Education",
        "TIV",
        "log_Lesion_resid_L1vol"
    ],
    "GM_logWM_corrected_by_L2_volume": [
        "Age",
        "Education",
        "TIV",
        "log_Lesion_resid_L2vol"
    ],
    "GM_logWM_corrected_by_L3_volume": [
        "Age",
        "Education",
        "TIV",
        "log_Lesion_resid_L3vol"
    ]
}

extra_tables = []

for model_name, cont_cols in extra_models.items():
    req_extra = ["GM_T2", "Sex"] + cont_cols

    dx = d_extra.dropna(subset=req_extra).copy().reset_index(drop=True)

    if len(dx) < 5:
        print(f"⚠ {model_name}: N insuficiente")
        continue

    res, cols = fit_ols_standardizedY(
        dx,
        "GM_T2",
        cont_cols=cont_cols,
        dummy_cols=("Sex",)  # SIN Gpo
    )

    extra_tables.append(
        make_ols_table(
            res,
            cols,
            outcome="GM_T2",
            model_name=model_name
        )
    )

if len(extra_tables) == 0:
    raise ValueError("No se pudo ajustar ningún modelo extra GM.")

extra_gm_df = pd.concat(extra_tables, ignore_index=True)

# FDR por modelo sobre todos los términos de ese modelo
extra_gm_df = add_fdr_by_group(
    extra_gm_df,
    p_col="p",
    group_cols=("model",),
    out_col="p_fdr"
)

lesion_terms = [
    "log_Lesion_Volume_T2_z",
    "log_Lesion_resid_L1vol_z",
    "log_Lesion_resid_L2vol_z",
    "log_Lesion_resid_L3vol_z"
]

extra_gm_plot_df = (
    extra_gm_df[
        extra_gm_df["term"].isin(lesion_terms)
    ]
    .copy()
    .reset_index(drop=True)
)

extra_gm_plot_df["sig_fdr"] = extra_gm_plot_df["p_fdr"].apply(_star_from_p)

# Orden correcto para plot
model_order = list(extra_models.keys())

extra_gm_plot_df["model"] = pd.Categorical(
    extra_gm_plot_df["model"],
    categories=model_order,
    ordered=True
)

extra_gm_plot_df = extra_gm_plot_df.sort_values("model").reset_index(drop=True)

print("=== EXTRA GM PLOT TERMS ===")
print(extra_gm_plot_df[
    ["model", "term", "beta_std", "CI95_low", "CI95_high", "p", "p_fdr", "sig_fdr"]
])


# =========================
# 15) EXPORT OLS TABLES
# =========================
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    d_ols.to_excel(
        writer,
        sheet_name="OLS_subject_data",
        index=False
    )

    d_extra.to_excel(
        writer,
        sheet_name="OLS_extra_GM_subject_data",
        index=False
    )

    ols_df_main.sort_values(["outcome", "p_fdr"]).to_excel(
        writer,
        sheet_name="OLS_MAIN_FDR",
        index=False
    )

    extra_gm_df.sort_values(["model", "p_fdr"]).to_excel(
        writer,
        sheet_name="EXTRA_GM_all_terms_FDR",
        index=False
    )

    extra_gm_plot_df.to_excel(
        writer,
        sheet_name="EXTRA_GM_WMH_beta_FDR",
        index=False
    )


# =========================
# 16) PLOT OLS PRINCIPAL
# =========================
png_ols_main, svg_ols_main, pdf_ols_main = plot_ols_main_forest(
    ols_df_main,
    OUT_DIR,
    filename=PLOT_FILENAME_OLS_MAIN,
    outcomes=("GM_T2", "CSF_T2"),
    show=True
)


# =========================
# 17) COMBINED PLOT:
#     LEFT  = Size by cluster, sin líneas/asteriscos/d values
#             Estadísticas + effect size exportadas a Excel
#     RIGHT = EXTRA GM beta plot (WMH → GM)
# =========================
import itertools
from scipy.stats import mannwhitneyu

PLOT_FILENAME_COMBINED = "combined_size_and_wmh_beta_plot.png"
SIZE_STATS_FILENAME = "size_cluster_pairwise_stats_effect_sizes.xlsx"


# -------------------------
# 17.1) LEFT PANEL DATA: SIZE BY CLUSTER
# -------------------------
df_size = df[["cluster_gmm_auto", "Size"]].dropna().copy()


def remove_outliers_size(group):
    q1 = group["Size"].quantile(0.25)
    q3 = group["Size"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return group[(group["Size"] >= lower) & (group["Size"] <= upper)]


def cohens_d_size(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan

    pooled_sd = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) +
         (ny - 1) * np.var(y, ddof=1)) /
        (nx + ny - 2)
    )

    if pooled_sd == 0 or not np.isfinite(pooled_sd):
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_sd


def cluster_label(c):
    c_int = int(c)
    label_map = {0: "L1", 1: "L2", 2: "L3"}
    return label_map.get(c_int, f"L{c_int + 1}")


def p_to_sig_label(p):
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


df_size_clean = (
    df_size
    .groupby("cluster_gmm_auto", group_keys=False)
    .apply(remove_outliers_size)
)

clusters = sorted(df_size_clean["cluster_gmm_auto"].unique())

# Resumen descriptivo por cluster
size_summary_df = (
    df_size_clean
    .groupby("cluster_gmm_auto")["Size"]
    .agg(
        N="count",
        mean="mean",
        std="std",
        median="median",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
        min="min",
        max="max"
    )
    .reset_index()
)
size_summary_df["cluster_label"] = size_summary_df["cluster_gmm_auto"].apply(cluster_label)

# Estadística Mann-Whitney + Cohen's d por pares
pairs = []
pvals = []
ds = []
u_stats = []
n1_list = []
n2_list = []
mean1_list = []
mean2_list = []
median1_list = []
median2_list = []

for c1, c2 in itertools.combinations(clusters, 2):
    g1 = df_size_clean.loc[df_size_clean["cluster_gmm_auto"] == c1, "Size"]
    g2 = df_size_clean.loc[df_size_clean["cluster_gmm_auto"] == c2, "Size"]

    if len(g1) > 0 and len(g2) > 0:
        u_stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
        d_val = cohens_d_size(g1, g2)
    else:
        u_stat = np.nan
        p = np.nan
        d_val = np.nan

    pairs.append((c1, c2))
    pvals.append(p)
    ds.append(d_val)
    u_stats.append(u_stat)
    n1_list.append(len(g1))
    n2_list.append(len(g2))
    mean1_list.append(float(np.nanmean(g1)) if len(g1) > 0 else np.nan)
    mean2_list.append(float(np.nanmean(g2)) if len(g2) > 0 else np.nan)
    median1_list.append(float(np.nanmedian(g1)) if len(g1) > 0 else np.nan)
    median2_list.append(float(np.nanmedian(g2)) if len(g2) > 0 else np.nan)

pvals_arr = np.asarray(pvals, dtype=float)
pvals_corr = np.full_like(pvals_arr, np.nan, dtype=float)
ok_p = np.isfinite(pvals_arr)

if ok_p.sum() > 0:
    _, p_corr_tmp, _, _ = multipletests(pvals_arr[ok_p], method="fdr_bh")
    pvals_corr[ok_p] = p_corr_tmp

size_pairwise_stats_df = pd.DataFrame({
    "cluster_1": [c1 for c1, _ in pairs],
    "cluster_2": [c2 for _, c2 in pairs],
    "cluster_1_label": [cluster_label(c1) for c1, _ in pairs],
    "cluster_2_label": [cluster_label(c2) for _, c2 in pairs],
    "N_cluster_1": n1_list,
    "N_cluster_2": n2_list,
    "mean_cluster_1": mean1_list,
    "mean_cluster_2": mean2_list,
    "median_cluster_1": median1_list,
    "median_cluster_2": median2_list,
    "mannwhitney_U": u_stats,
    "p_uncorrected": pvals_arr,
    "p_fdr_bh": pvals_corr,
    "sig_fdr": [p_to_sig_label(p) for p in pvals_corr],
    "cohens_d": ds
})

size_stats_xlsx = os.path.join(OUT_DIR, SIZE_STATS_FILENAME)

with pd.ExcelWriter(size_stats_xlsx, engine="openpyxl") as writer:
    size_summary_df.to_excel(writer, sheet_name="Size_summary_by_cluster", index=False)
    size_pairwise_stats_df.to_excel(writer, sheet_name="Pairwise_stats_effect_size", index=False)

# También agrega estas tablas al Excel principal
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    size_summary_df.to_excel(writer, sheet_name="Size_summary_by_cluster", index=False)
    size_pairwise_stats_df.to_excel(writer, sheet_name="Size_pairwise_stats", index=False)

print("📄 Size stats Excel:", size_stats_xlsx)


# -------------------------
# 17.2) RIGHT PANEL DATA: EXTRA GM BETA PLOT
# -------------------------
plot_df = extra_gm_plot_df.copy()

x_beta = np.arange(len(plot_df))

beta = plot_df["beta_std"].to_numpy(dtype=float)
ci_low = plot_df["CI95_low"].to_numpy(dtype=float)
ci_high = plot_df["CI95_high"].to_numpy(dtype=float)
p_fdr = plot_df["p_fdr"].to_numpy(dtype=float)

yerr = np.vstack([
    beta - ci_low,
    ci_high - beta
])

short_labels = [
    "Uncorrected",
    "Adj L1 vol",
    "Adj L2 vol",
    "Adj L3 vol"
]


# -------------------------
# 17.3) COMBINED FIGURE
# -------------------------
fig, axes = plt.subplots(
    1,
    2,
    figsize=(FIG_W, FIG_H * 0.9),
    gridspec_kw={"width_ratios": [1.1, 1.0]}
)

ax1, ax2 = axes


# -------------------------
# LEFT SUBPLOT: SIZE BOXPLOT SIN ANOTACIONES
# -------------------------
df_size_clean.boxplot(
    column="Size",
    by="cluster_gmm_auto",
    showfliers=False,
    ax=ax1,
    medianprops={"color": "orange", "linewidth": 1.8},
    boxprops={"color": "black"},
    whiskerprops={"color": "black"},
    capprops={"color": "black"}
)

ax1.grid(False)
ax1.set_title("")          # quita título automático "Size"
ax1.set_xlabel("")
ax1.set_ylabel("")         # quita título/label izquierdo que dice "Size"
ax1.set_ylim(0, 2500)
ax1.set_xticklabels([cluster_label(c) for c in clusters])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(axis="x", labelsize=10)
ax1.tick_params(axis="y", labelsize=9)


# -------------------------
# RIGHT SUBPLOT: WMH → GM BETA PLOT
# -------------------------
ax2.errorbar(
    x_beta,
    beta,
    yerr=yerr,
    fmt="o",
    capsize=3,
    markersize=5,
    linewidth=1.5
)

ax2.plot(x_beta, beta, linewidth=1.5)
ax2.axhline(0, linestyle="--", linewidth=1.2, color="black")

finite_ci = np.isfinite(ci_low) & np.isfinite(ci_high)

if np.any(finite_ci):
    y_range_beta = float(np.nanmax(ci_high[finite_ci]) - np.nanmin(ci_low[finite_ci]))
    star_offset = 0.05 * (y_range_beta + 1e-12)
else:
    star_offset = 0.03

for i in range(len(plot_df)):
    if (
        np.isfinite(beta[i]) and
        np.isfinite(ci_high[i]) and
        np.isfinite(p_fdr[i])
    ):
        star = _star_from_p(float(p_fdr[i]))

        if star:
            ax2.text(
                x_beta[i],
                ci_high[i] + star_offset,
                star,
                ha="center",
                va="bottom",
                fontsize=12
            )

ax2.set_xticks(x_beta)


ax2.set_ylabel("Standardized β", fontsize=9)
ax2.tick_params(axis="y", labelsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


# -------------------------
# SAVE COMBINED FIGURE
# -------------------------
plt.suptitle("")
plt.tight_layout()

png_combined = os.path.join(OUT_DIR, PLOT_FILENAME_COMBINED)
svg_combined = os.path.join(
    OUT_DIR,
    PLOT_FILENAME_COMBINED.replace(".png", ".svg")
)
pdf_combined = os.path.join(
    OUT_DIR,
    PLOT_FILENAME_COMBINED.replace(".png", ".pdf")
)

fig.savefig(png_combined, dpi=300)
fig.savefig(svg_combined)
fig.savefig(pdf_combined, format="pdf")

plt.show()
plt.close(fig)

print("📊 COMBINED Plot PNG:", png_combined)
print("📊 COMBINED Plot SVG:", svg_combined)
print("📊 COMBINED Plot PDF:", pdf_combined)

# Para mantener compatibilidad con prints anteriores, estos nombres apuntan ahora al plot combinado
png_extra_gm = png_combined
svg_extra_gm = svg_combined
pdf_extra_gm = pdf_combined

# =========================
# 18) FINAL PRINTS
# =========================
print("✅ OK ->", OUTPUT_XLSX)
print("📁 Outputs en:", OUT_DIR)

print("📊 GLM Plot PNG:", png_glm)
print("📊 GLM Plot SVG:", svg_glm)
print("📊 GLM Plot PDF:", pdf_glm)

print("📊 OLS MAIN Plot PNG:", png_ols_main)
print("📊 OLS MAIN Plot SVG:", svg_ols_main)
print("📊 OLS MAIN Plot PDF:", pdf_ols_main)

print("📊 COMBINED Plot PNG:", png_combined)
print("📊 COMBINED Plot SVG:", svg_combined)
print("📊 COMBINED Plot PDF:", pdf_combined)
print("📄 Size stats Excel:", size_stats_xlsx)

print("N sujetos usados GLM:", len(d))
print("N sujetos usados OLS:", len(d_ols))

print("N sujetos usados EXTRA GM por modelo:")
print(extra_gm_plot_df[[
    "model",
    "N",
    "beta_std",
    "CI95_low",
    "CI95_high",
    "p",
    "p_fdr",
    "sig_fdr"
]])

print("✅ GLM terminado")
print("✅ OLS principal terminado")
print("✅ Extra GM WMH → GM sin corregir terminado")
print("✅ Extra GM WMH → GM corregido por L1 volume terminado")
print("✅ Extra GM WMH → GM corregido por L2 volume terminado")
print("✅ Extra GM WMH → GM corregido por L3 volume terminado")
