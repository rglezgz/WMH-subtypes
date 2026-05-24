# -*- coding: utf-8 -*-
"""
WMH clusters -> fMRI metrics DELTA only (time2 - time0)
- meanTS_delta and medianTS_delta only
- Skip subjects with <180 usable volumes in ANY session
- Uses global TRIM_END computed from valid subjects
- Adds 'sub-' prefix to subject_id if missing
- Verbose per-subject debug: prints why rows are skipped.

KEY FIX:
- Resample BOLD time0 (b0) to time2 grid (b2) BEFORE extracting d0,
  so d0 and mask live in the same space.
- Resample WMH cluster mask to fMRI using a 3D reference volume (first vol of time2).
"""

import os, re, glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi
from scipy.signal import welch
from nilearn import image as nimg, datasets
from antropy import spectral_entropy
from collections import Counter

# ================= CONFIG =================
excel_path = r"F:\long\test\All\SALIDAS\whm_mask_only\WMH_clusters_summary_WMfilter_mean02.xlsx"
excel_out  = excel_path  # overwrite same Excel

fmriprep_root   = r"F:\long\test\All\SALIDAS\fMRI\fmriprep"
wmh_mask_folder = r"F:\long\test\All\SALIDAS\whm_mask_only\WMH_masks_WMfilter_mean02"

TRIM_START = 5
MIN_USABLE = 180

STRUCT26 = ndi.generate_binary_structure(3, 3)
DILATE_ITERS = 1

BOLD_TEMPLATE = "{sub}_{ses}_task-resting_space-MNI152NLin2009cAsym_desc-preproc_bold.nii*"

PRINT_FIRST_N_FAILURES_PER_SUBJECT = 25

# ================= HELPERS =================

def norm_sub_id(s):
    s = str(s).strip()
    return s if s.startswith("sub-") else f"sub-{s}"

def ses_num(s):
    m = re.search(r"ses-(\d+)", s)
    return int(m.group(1)) if m else 999999

def find_sessions(sub):
    p = glob.glob(os.path.join(fmriprep_root, sub, "ses-*"))
    return sorted([os.path.basename(x) for x in p if os.path.isdir(x)], key=ses_num)

def find_bold(sub, ses):
    patt = os.path.join(fmriprep_root, sub, ses, "func",
                        BOLD_TEMPLATE.format(sub=sub, ses=ses))
    f = sorted(glob.glob(patt))
    if not f:
        raise FileNotFoundError(patt)
    return f[0]

def find_wmh(sub):
    patt1 = os.path.join(wmh_mask_folder, f"*{sub}*WMHmask.nii*")
    patt2 = os.path.join(wmh_mask_folder, f"*{sub.replace('sub-','')}*WMHmask.nii*")
    f = sorted(glob.glob(patt1) + glob.glob(patt2))
    if not f:
        raise FileNotFoundError(f"WMH mask not found for {sub}")
    return f[0]

def label_mask(m_bool):
    lbl, _ = ndi.label(m_bool.astype(np.uint8), structure=STRUCT26)
    return lbl

def dilate(mask_bool):
    return ndi.binary_dilation(mask_bool.astype(bool), structure=STRUCT26, iterations=DILATE_ITERS)

def nvols(path):
    img = nib.load(path)
    return int(img.shape[3])

def wm_mask_fmri(ref3d):
    wm = datasets.load_mni152_wm_mask()
    wm = nib.as_closest_canonical(wm)
    wm_rs = nimg.resample_to_img(
        wm, ref3d,
        interpolation="nearest",
        force_resample=True,
        copy_header=True
    )
    return (wm_rs.get_fdata() > 0)

# ================= METRICS =================

def alff_falff(ts, tr):
    f, p = welch(ts, fs=1/float(tr), nperseg=len(ts))
    low = (f >= 0.01) & (f <= 0.1)
    alff = np.sqrt(np.sum(p[low]))
    tot = np.sqrt(np.sum(p))
    return float(alff), float(alff/tot) if tot > 0 else np.nan

def hurst_rs(ts):
    ts = np.asarray(ts, float)
    ts = ts - ts.mean()
    n = len(ts)
    if n < 20:
        return np.nan
    sizes = np.unique(np.logspace(np.log10(10), np.log10(max(11, n//2)), 10).astype(int))
    rs = []
    for s in sizes:
        segs = n // s
        if segs < 2:
            continue
        vals = []
        for i in range(segs):
            x = ts[i*s:(i+1)*s]
            y = np.cumsum(x - x.mean())
            R = y.max() - y.min()
            S = x.std()
            if S > 0:
                vals.append(R/S)
        if vals:
            rs.append((s, np.mean(vals)))
    if len(rs) < 3:
        return np.nan
    s, r = zip(*rs)
    return float(np.polyfit(np.log(s), np.log(r), 1)[0])

def metric_power_slope(ts, tr):
    f, pxx = welch(ts, fs=1/float(tr), nperseg=len(ts))
    m = (f > 0) & np.isfinite(pxx) & (pxx > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(f[m]), np.log(pxx[m]), 1)[0])

def metric_var_of_var(ts, win=20):
    ts = np.asarray(ts, float)
    if len(ts) <= win + 2:
        return np.nan
    v = np.array([np.var(ts[i:i+win]) for i in range(len(ts)-win)], float)
    return float(np.var(v))

def metric_autocorr(ts, lag=1):
    ts = np.asarray(ts, float)
    if len(ts) <= lag + 2:
        return np.nan
    a = ts[:-lag]
    b = ts[lag:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def metrics(ts, tr):
    a, f = alff_falff(ts, tr)
    return {
        "RSFA": float(np.std(ts)),
        "ALFF": a,
        "fALFF": f,
        "Hurst": hurst_rs(ts),
        "Entropy": float(spectral_entropy(ts, sf=1/float(tr), method="welch")),
        "PowerSlope": metric_power_slope(ts, tr),
        "VarOfVar": metric_var_of_var(ts, win=20),
        "Autocorr": metric_autocorr(ts, lag=1)
    }

# ================= QC STEP =================
print("[QC] Scanning subjects for >=180 usable in both sessions...")

subjects = sorted([s for s in os.listdir(fmriprep_root) if s.startswith("sub-")])
valid = set()
qc = []

for sub in subjects:
    try:
        ses = find_sessions(sub)
        if len(ses) < 2:
            continue
        f0, f2 = find_bold(sub, ses[0]), find_bold(sub, ses[-1])
        n0, n2 = nvols(f0), nvols(f2)
        u0, u2 = n0 - TRIM_START, n2 - TRIM_START
        if u0 >= MIN_USABLE and u2 >= MIN_USABLE:
            valid.add(sub)
            qc.append((sub, ses[0], ses[-1], n0, n2))
    except Exception:
        continue

if not valid:
    raise RuntimeError("No valid subjects (>=180 usable volumes in both sessions).")

min_global = min(min(x[3], x[4]) for x in qc)
TRIM_END = int(min_global)
T_USED = TRIM_END - TRIM_START
if T_USED < MIN_USABLE:
    raise RuntimeError(f"T_USED={T_USED} < {MIN_USABLE}. Check your dataset.")

print(f"[QC] Valid subjects: {len(valid)} / {len(subjects)}")
print(f"[QC] Using TRIM: [{TRIM_START}:{TRIM_END}] -> {T_USED} vols\n")

# ================= MAIN =================
print(f"[INFO] Reading Excel: {excel_path}")
df = pd.read_excel(excel_path)

if "region" in df.columns:
    df = df[df["region"].astype(str).str.lower() == "wmh"].copy()

base = ["RSFA","ALFF","fALFF","Hurst","Entropy","PowerSlope","VarOfVar","Autocorr"]
for m in base:
    for s in ["meanTS", "medianTS"]:
        col = f"{m}_{s}_delta"
        if col not in df.columns:
            df[col] = np.nan

if "warn_short_bold" not in df.columns:
    df["warn_short_bold"] = ""

# --- subject state ---
current = None
b0=b2=None
b0_rs=None
ref3=None
d0=d2=None
tr=None
wm=None
wmh=None
lab=None
cluster_cache = {}

rows_filled_total = 0
rows_skipped_total = 0
subject_summary = {}

def fail(i, reason, detail, sub, x, y, z, ssum):
    msg = reason if detail == "" else f"{reason}: {detail}"
    df.at[i, "warn_short_bold"] = msg
    ssum["skipped"] += 1
    ssum["fails"][reason] += 1

    global rows_skipped_total
    rows_skipped_total += 1

    if ssum["printed_fail"] < PRINT_FIRST_N_FAILURES_PER_SUBJECT:
        print(f"  [FAIL] row={i} peak=({x},{y},{z}) -> {msg}")
        ssum["printed_fail"] += 1

for i, row in df.iterrows():

    sub = norm_sub_id(row["subject_id"])

    if sub not in valid:
        df.at[i, "warn_short_bold"] = "SKIPPED (<180 usable)"
        rows_skipped_total += 1
        continue

    if sub != current:

        if current is not None:
            ssum_prev = subject_summary[current]
            print(f"[SUBJECT SUMMARY] {current}: filled={ssum_prev['filled']} skipped={ssum_prev['skipped']} top_fail={ssum_prev['fails'].most_common(3)}\n")

        subject_summary[sub] = {"filled": 0, "skipped": 0, "fails": Counter(), "printed_fail": 0}

        print("================================")
        print(f"[SUBJECT] {sub}")
        print("================================")

        current = None
        cluster_cache = {}

        try:
            ses = find_sessions(sub)
            ses0, ses2 = ses[0], ses[-1]
            f0, f2 = find_bold(sub, ses0), find_bold(sub, ses2)

            b0 = nib.load(f0)
            b2 = nib.load(f2)
            tr = float(b2.header.get_zooms()[3])

            # --- CRITICAL FIX: resample b0 to b2 grid if needed ---
            if (b0.shape[:3] != b2.shape[:3]) or (not np.allclose(b0.affine, b2.affine, atol=1e-5)):
                b0_rs = nimg.resample_to_img(
                    b0, b2,
                    interpolation="continuous",
                    force_resample=True,
                    copy_header=True
                )
                print(f"[INFO] b0 resampled to b2 grid: b0 {b0.shape[:3]} -> {b0_rs.shape[:3]}")
            else:
                b0_rs = b0

            # 3D ref = first volume of time2 (for mask/WM resampling)
            ref3 = nimg.index_img(b2, 0)
            ref3 = nib.as_closest_canonical(ref3)

            d0 = b0_rs.get_fdata()[..., TRIM_START:TRIM_END].astype(np.float32)
            d2 = b2.get_fdata()[..., TRIM_START:TRIM_END].astype(np.float32)

            wm = wm_mask_fmri(ref3)

            wmh = nib.as_closest_canonical(nib.load(find_wmh(sub)))
            lab = label_mask(wmh.get_fdata() > 0)

            current = sub
            print(f"[OK INIT] sessions {ses0}->{ses2} | TR={tr:.3f}s | vols used={d0.shape[3]}")
            print(f"         b0 shape={b0.shape} | b0_rs shape={b0_rs.shape} | b2 shape={b2.shape}")
            print(f"         ref3 shape={ref3.shape} | WMH shape={lab.shape}")

        except Exception as e:
            msg = f"INIT FAIL: {e}"
            print(f"[BUG INIT] {sub} | {msg}")
            df.at[i, "warn_short_bold"] = msg
            subject_summary[sub]["skipped"] += 1
            subject_summary[sub]["fails"]["INIT_FAIL"] += 1
            rows_skipped_total += 1
            current = None
            continue

    if current is None:
        continue

    ssum = subject_summary[sub]

    x = int(np.round(row["peak_x"]))
    y = int(np.round(row["peak_y"]))
    z = int(np.round(row["peak_z"]))

    if not (0 <= x < lab.shape[0] and 0 <= y < lab.shape[1] and 0 <= z < lab.shape[2]):
        fail(i, "PEAK_OOB", f"WMHshape={lab.shape}", sub, x, y, z, ssum)
        continue

    lid = int(lab[x, y, z])
    if lid == 0:
        fail(i, "LABEL0", "peak outside WMH", sub, x, y, z, ssum)
        continue

    # cluster -> fMRI (cached)
    if lid in cluster_cache:
        cl_final = cluster_cache[lid]
    else:
        try:
            cl = (lab == lid)
            cl_img = nib.Nifti1Image(cl.astype(np.uint8), wmh.affine, wmh.header)

            cl_rs_img = nimg.resample_to_img(
                cl_img, ref3,
                interpolation="nearest",
                force_resample=True,
                copy_header=True
            )
            cl_fmri = (cl_rs_img.get_fdata() > 0.5)

        except Exception as e:
            fail(i, "RESAMPLE_FAIL", str(e), sub, x, y, z, ssum)
            continue

        if not np.any(cl_fmri):
            fail(i, "EMPTY_AFTER_RESAMPLE", "", sub, x, y, z, ssum)
            continue

        cl_fmri = dilate(cl_fmri)

        if wm is not None and wm.shape == cl_fmri.shape:
            cl_final = cl_fmri & wm
        else:
            cl_final = cl_fmri

        if not np.any(cl_final):
            fail(i, "EMPTY_AFTER_WM", "", sub, x, y, z, ssum)
            continue

        cluster_cache[lid] = cl_final

    # IMPORTANT: check both d0 and d2 shapes explicitly (and print both)
    if d0.shape[:3] != cl_final.shape or d2.shape[:3] != cl_final.shape:
        fail(i, "SHAPE_MISMATCH",
             f"d0={d0.shape[:3]} d2={d2.shape[:3]} mask={cl_final.shape}",
             sub, x, y, z, ssum)
        continue

    v0 = d0[cl_final, :]
    v2 = d2[cl_final, :]

    if v0.size == 0 or v2.size == 0:
        fail(i, "EMPTY_VOX_TS", "", sub, x, y, z, ssum)
        continue

    ts0_mean = np.mean(v0, axis=0)
    ts2_mean = np.mean(v2, axis=0)
    ts0_median = np.median(v0, axis=0)
    ts2_median = np.median(v2, axis=0)

    try:
        M0m, M2m = metrics(ts0_mean, tr), metrics(ts2_mean, tr)
        M0d, M2d = metrics(ts0_median, tr), metrics(ts2_median, tr)
    except Exception as e:
        fail(i, "METRIC_FAIL", str(e), sub, x, y, z, ssum)
        continue

    for k in base:
        df.at[i, f"{k}_meanTS_delta"]   = M2m[k] - M0m[k]
        df.at[i, f"{k}_medianTS_delta"] = M2d[k] - M0d[k]

    df.at[i, "warn_short_bold"] = ""
    ssum["filled"] += 1
    rows_filled_total += 1

    if ssum["filled"] <= 5:
        print(f"  [OK] row={i} lid={lid} nvox={int(v0.shape[0])} vols={int(v0.shape[1])}")

# last subject summary
if current is not None:
    ssum_last = subject_summary[current]
    print(f"[SUBJECT SUMMARY] {current}: filled={ssum_last['filled']} skipped={ssum_last['skipped']} top_fail={ssum_last['fails'].most_common(3)}\n")

print(f"\n[GLOBAL SUMMARY] Rows filled: {rows_filled_total}")
print(f"[GLOBAL SUMMARY] Rows skipped: {rows_skipped_total}")

df.to_excel(excel_out, index=False)
print(f"[DONE] Saved (overwrite): {excel_out}")
