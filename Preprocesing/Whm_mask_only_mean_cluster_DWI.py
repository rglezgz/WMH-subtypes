
# -*- coding: utf-8 -*-
"""
WMH clusters (peaks en espacio WMH) -> cluster aislado -> resample nearest a DWI -> binario ->
dilatación 1 voxel en DWI -> recorte WM en DWI -> promedio de DWI diff (time2-time0) en DWI.

1- The WMH peak coordinates (given in WMH voxel space) are used to identify the corresponding WMH connected component.

2- That WMH cluster is isolated in WMH space and resampled (nearest-neighbor) to DWI space.

3- The cluster is dilated by 1 DWI voxel and restricted to white matter using an MNI WM mask resampled to DWI space.

4- For each DWI metric, the mean difference (time2 – time0) is computed inside the final DWI-space cluster mask.
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

# ---------------- CONFIG ----------------

excel_path = r"F:\long\test\All\SALIDAS\whm_mask_only\WMH_clusters_summary_WMfilter_mean02.xlsx"
excel_out  = excel_path  # o excel_path.replace(".xlsx", "_with_DWIdiff_clusterDilatedInDWI.xlsx")

qsiprep_root = r"F:\long\test\All\SALIDAS\DWI\qsirecon-DSIStudio"
wmh_mask_folder = r"F:\long\test\All\SALIDAS\whm_mask_only\WMH_masks_WMfilter_mean02"

metrics = [
    ("gqi",    "gfa"),
    ("gqi",    "iso"),
    ("gqi",    "qa"),
    ("tensor", "ad"),
    ("tensor", "fa"),
    ("tensor", "ha"),
    ("tensor", "rd"),
    ("tensor", "rd1"),
    ("tensor", "rd2"),
]

STRUCT26 = ndi.generate_binary_structure(3, 3)

FORCE_DILATION_ITERS = None  # None -> 1 voxel DWI
RESCUE_PEAK_IF_LABEL0 = True
RESCUE_RADIUS_VOX = 2

# --------------- HELPERS ----------------

def label_mask(mask_bool: np.ndarray) -> np.ndarray:
    lbl, _ = ndi.label(mask_bool.astype(np.uint8), structure=STRUCT26)
    return lbl

def dilate_bool(mask_bool: np.ndarray, n_iter: int) -> np.ndarray:
    if n_iter <= 0:
        return mask_bool.astype(bool)
    out = mask_bool.astype(bool)
    for _ in range(n_iter):
        out = ndi.binary_dilation(out, structure=STRUCT26)
    return out

def find_subject_folder(subject_id_raw: str) -> str:
    sid = subject_id_raw.strip()
    candidates = []
    candidates.append(os.path.join(qsiprep_root, sid))
    if not sid.startswith("sub-"):
        candidates.append(os.path.join(qsiprep_root, f"sub-{sid}"))
    if sid.startswith("sub-"):
        candidates.append(os.path.join(qsiprep_root, sid[4:]))

    for gp in glob.glob(os.path.join(qsiprep_root, f"*{sid}*")):
        if os.path.isdir(gp):
            candidates.append(gp)

    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Carpeta no encontrada para {sid}")

def find_sessions(subfolder: str):
    return sorted(glob.glob(os.path.join(subfolder, "ses-*")))

def find_metric_file(sub_name: str, ses: str, model: str, param: str) -> str:
    dwi_folder = os.path.join(qsiprep_root, sub_name, ses, "dwi")
    pattern_nii = os.path.join(
        dwi_folder,
        f"{sub_name}_{ses}_space-MNI152NLin2009cAsym_model-{model}_param-{param}_dwimap.nii"
    )
    pattern_niigz = pattern_nii + ".gz"
    files = sorted(glob.glob(pattern_nii) + glob.glob(pattern_niigz))
    if not files:
        raise FileNotFoundError(f"No archivo NIfTI para {sub_name}/{ses} {model}-{param} en {dwi_folder}")
    return files[0]

def find_wmh_mask_for_subject(folder: str, subject_id_raw: str) -> str:
    patt = os.path.join(folder, f"*{subject_id_raw}*WMHmask.nii*")
    files = sorted(glob.glob(patt))
    if not files:
        raise FileNotFoundError(f"No se encontró WMHmask para {subject_id_raw} en {folder}")
    return files[0]

def get_wm_mask_in_dwi_space(ref_dwi_img_ras: nib.Nifti1Image) -> np.ndarray:
    from nilearn import image as nimg
    from nilearn import datasets

    wm_mni = datasets.load_mni152_wm_mask()
    wm_mni = nib.as_closest_canonical(wm_mni)

    # 👇 FIX warnings: explicitar futuros defaults
    wm_res = nimg.resample_to_img(
        wm_mni,
        ref_dwi_img_ras,
        interpolation="nearest",
        force_resample=True,
        copy_header=True
    )
    return (wm_res.get_fdata() > 0).astype(bool)

def rescue_peak_to_nearest_label(labels: np.ndarray, peak_xyz: tuple[int,int,int], radius: int):
    px, py, pz = peak_xyz
    nx, ny, nz = labels.shape

    x0 = max(0, px - radius); x1 = min(nx - 1, px + radius)
    y0 = max(0, py - radius); y1 = min(ny - 1, py + radius)
    z0 = max(0, pz - radius); z1 = min(nz - 1, pz + radius)

    best = None
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                lab = int(labels[x, y, z])
                if lab == 0:
                    continue
                d2 = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
                cand = (d2, lab, x, y, z)
                if best is None or cand < best:
                    best = cand

    if best is None:
        return 0, px, py, pz
    _, lab, x, y, z = best
    return lab, x, y, z

# --------------- MAIN -------------------

print(f"[INFO] Leyendo Excel: {excel_path}")
df = pd.read_excel(excel_path)

if "region" in df.columns:
    before = len(df)
    df = df[df["region"].astype(str).str.lower() == "wmh"].copy()
    print(f"[INFO] Filtrando region='wmh': {before} -> {len(df)} filas")

for model, param in metrics:
    if param not in df.columns:
        df[param] = np.nan

current_subject = None
sub_name = None
subject_folder = None
ses1 = None
ses2 = None

wmh_img_ras = None
labels_wmh = None
wmh_shape = None

ref_dwi_img = None
ref_dwi_ras = None
wm_bool_dwi = None

diff_arrays_dwi = {}
diff_ok = {}
cluster_cache_dwi = {}

dilate_iters = 1 if FORCE_DILATION_ITERS is None else int(FORCE_DILATION_ITERS)

for idx, row in df.iterrows():

    subject_id_raw = str(row["subject_id"]).strip()

    peak_x = int(np.round(row["peak_x"]))
    peak_y = int(np.round(row["peak_y"]))
    peak_z = int(np.round(row["peak_z"]))

    if subject_id_raw != current_subject:

        print(f"\n[INFO] === Sujeto nuevo: {subject_id_raw} ===")

        current_subject = None
        sub_name = None
        subject_folder = None
        ses1 = None
        ses2 = None

        wmh_img_ras = None
        labels_wmh = None
        wmh_shape = None

        ref_dwi_img = None
        ref_dwi_ras = None
        wm_bool_dwi = None

        diff_arrays_dwi = {}
        diff_ok = {}
        cluster_cache_dwi = {}

        try:
            subject_folder = find_subject_folder(subject_id_raw)
        except FileNotFoundError as e:
            print("[WARN]", e)
            continue

        sub_name = os.path.basename(subject_folder)
        current_subject = subject_id_raw

        sessions = find_sessions(subject_folder)
        if len(sessions) < 2:
            print(f"[WARN] Menos de 2 sesiones para {subject_id_raw}")
            current_subject = None
            continue

        ses1 = os.path.basename(sessions[0])
        ses2 = os.path.basename(sessions[1])
        print(f"  [INFO] Sesiones usadas: {ses1} (time0), {ses2} (time2)")

        try:
            wmh_path = find_wmh_mask_for_subject(wmh_mask_folder, subject_id_raw)
        except FileNotFoundError as e:
            print("[WARN]", e)
            current_subject = None
            continue

        wmh_img = nib.load(wmh_path)
        wmh_img_ras = nib.as_closest_canonical(wmh_img)
        wmh_data = wmh_img_ras.get_fdata()
        wmh_shape = wmh_data.shape
        labels_wmh = label_mask(wmh_data > 0)

        # diffs DWI
        ref_dwi_img = None
        for model, param in metrics:
            print(f"   [INFO] Procesando {model}-{param} ...")
            try:
                f1 = find_metric_file(sub_name, ses1, model, param)
                f2 = find_metric_file(sub_name, ses2, model, param)
            except FileNotFoundError as e:
                print("     [WARN]", e)
                diff_ok[param] = False
                diff_arrays_dwi[param] = None
                continue

            img1 = nib.load(f1)
            img2 = nib.load(f2)

            d1 = img1.get_fdata().astype(np.float32)
            d2 = img2.get_fdata().astype(np.float32)
            diff = d2 - d1

            diff_arrays_dwi[param] = diff
            diff_ok[param] = True

            if ref_dwi_img is None:
                ref_dwi_img = img2

        if ref_dwi_img is None:
            print(f"[WARN] No se pudo obtener referencia DWI para {subject_id_raw}")
            current_subject = None
            continue

        ref_dwi_ras = nib.as_closest_canonical(ref_dwi_img)

        # WM mask en DWI
        try:
            wm_bool_dwi = get_wm_mask_in_dwi_space(ref_dwi_ras)
        except Exception as e:
            print(f"[WARN] No se pudo obtener WM mask en DWI space para {subject_id_raw}: {e}")
            wm_bool_dwi = None

        dilate_iters = 1 if FORCE_DILATION_ITERS is None else int(FORCE_DILATION_ITERS)
        print(f"  [INFO] Dilatación en DWI space: {dilate_iters} iteración(es) (~{dilate_iters} voxel DWI)")

        # sanity: shapes deben coincidir
        any_param = next((p for _, p in metrics if diff_ok.get(p, False)), None)
        if any_param is not None and wm_bool_dwi is not None:
            if wm_bool_dwi.shape != diff_arrays_dwi[any_param].shape:
                print(f"[WARN] Shape mismatch WM vs DWI diff: wm={wm_bool_dwi.shape} diff={diff_arrays_dwi[any_param].shape}")

    if current_subject is None:
        continue

    # cluster id por peak en WMH
    nx, ny, nz = wmh_shape
    if not (0 <= peak_x < nx and 0 <= peak_y < ny and 0 <= peak_z < nz):
        print(f"[WARN] {current_subject} fila {idx}: peak fuera WMH ({peak_x},{peak_y},{peak_z}).")
        continue

    label_id = int(labels_wmh[peak_x, peak_y, peak_z])

    if label_id == 0 and RESCUE_PEAK_IF_LABEL0:
        lab2, rx, ry, rz = rescue_peak_to_nearest_label(labels_wmh, (peak_x, peak_y, peak_z), RESCUE_RADIUS_VOX)
        if lab2 != 0:
            label_id = lab2
        else:
            print(f"[WARN] {current_subject} fila {idx}: peak fuera WMH (label=0) incluso tras rescate.")
            continue

    if label_id == 0:
        print(f"[WARN] {current_subject} fila {idx}: peak fuera WMH (label=0).")
        continue

    # cluster final en DWI (cache)
    if label_id in cluster_cache_dwi:
        cluster_final_dwi = cluster_cache_dwi[label_id]
    else:
        cluster0_wmh = (labels_wmh == label_id)

        cluster0_img = nib.Nifti1Image(
            cluster0_wmh.astype(np.uint8),
            wmh_img_ras.affine,
            wmh_img_ras.header
        )

        try:
            cluster0_in_dwi_img = resample_from_to(cluster0_img, ref_dwi_ras, order=0)  # nearest
        except Exception as e:
            print(f"[WARN] Resample cluster->DWI falló (label={label_id}) {current_subject}: {e}")
            continue

        cluster0_in_dwi = (cluster0_in_dwi_img.get_fdata() > 0.5)  # binario

        if not np.any(cluster0_in_dwi):
            print(f"[WARN] Cluster label={label_id} vacío tras resample->DWI ({current_subject}).")
            continue

        cluster_dil_dwi = dilate_bool(cluster0_in_dwi, dilate_iters)

        if wm_bool_dwi is not None:
            if wm_bool_dwi.shape != cluster_dil_dwi.shape:
                print(f"[WARN] Shape mismatch cluster vs WM: cluster={cluster_dil_dwi.shape} wm={wm_bool_dwi.shape}")
                continue
            cluster_final_dwi = cluster_dil_dwi & wm_bool_dwi
        else:
            cluster_final_dwi = cluster_dil_dwi

        if not np.any(cluster_final_dwi):
            print(f"[WARN] Cluster final vacío tras WM recorte en DWI ({current_subject}, label={label_id}).")
            continue

        cluster_cache_dwi[label_id] = cluster_final_dwi

    # promedios
    for model, param in metrics:
        if not diff_ok.get(param, False):
            continue
        diff_data = diff_arrays_dwi.get(param, None)
        if diff_data is None:
            continue

        if diff_data.shape != cluster_final_dwi.shape:
            print(f"[WARN] Shape mismatch diff vs cluster: diff={diff_data.shape} cluster={cluster_final_dwi.shape} ({current_subject}, {param})")
            continue

        vals = diff_data[cluster_final_dwi]
        vals = vals[np.isfinite(vals)]
        df.at[idx, param] = np.nan if vals.size == 0 else float(vals.mean())

print(f"\n[INFO] Guardando Excel: {excel_out}")
df.to_excel(excel_out, index=False)
print("[INFO] Listo.")
