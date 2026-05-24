# -*- coding: utf-8 -*-
"""
Script: agregar columna T1_diff al Excel de clusters WMH.

CAMBIO vs versión anterior:
- NO se hace min-max [0,1]. Se usa la intensidad original.
- T1_diff = T1_time2 - T1_time0 (en el espacio de T1_time2), y luego se resamplea al espacio WMH.

Pipeline por sujeto:
1) Carga T1 time0 y time2 (sin normalizar).
2) Calcula T1_diff = T2 - T0.
3) Carga máscara WMH del sujeto (*_WMHmask.nii / nii.gz).
4) Descarga/carga máscara WM del MNI152 (nilearn) y la resamplea al espacio del sujeto (WMH mask).
5) Resamplea T1_diff al espacio WMH.
6) Etiqueta clústeres WMH (26 vecinos).
7) Para cada clúster, crea una máscara DILATADA en N_VOX_DILATION vóxeles,
   pero recortada para que quede 100% dentro de WM (MNI152 WM resampleada).
8) Para cada fila del Excel: identifica el clúster por el pico (peak_x,y,z) y promedia T1_diff
   dentro del clúster dilatado.

NOTAS:
- peak_x/peak_y/peak_z deben estar en coordenadas VOXEL del espacio WMH.
- Usar WM MNI152 solo es anatómicamente válido si el espacio del sujeto es MNI.

Requisitos:
- nibabel, scipy, pandas, numpy
- nilearn
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi
from typing import Optional

# ---------------------------------------------------------------------
# 1) CONFIGURACIÓN: EDITA ESTAS RUTAS
# ---------------------------------------------------------------------

excel_path = r"F:\long\test\All\SALIDAS\whm_mask_only\WMH_clusters_summary_WMfilter_mean02.xlsx"

t1_time0_folder = r"F:\long\test\All\SALIDAS\T1w\T0"
t1_time2_folder = r"F:\long\test\All\SALIDAS\T1w\T2"

wmh_mask_folder = r"F:\long\test\All\SALIDAS\whm_mask_only\WMH_masks_WMfilter_mean02"

excel_out = excel_path  # o excel_path.replace(".xlsx", "_with_T1diff.xlsx")

# Dilatación en número de vóxeles (iteraciones)
N_VOX_DILATION = 1

# Si existe columna 'region' y quieres quedarte solo con wmh
FILTER_REGION_WMH = True

# ---------------------------------------------------------------------
# 2) HELPERS
# ---------------------------------------------------------------------

def find_nifti_for_subject(folder: str, subject_id: str) -> str:
    """Busca un .nii o .nii.gz en 'folder' cuyo nombre contenga subject_id."""
    pattern1 = os.path.join(folder, f"*{subject_id}*.nii")
    pattern2 = os.path.join(folder, f"*{subject_id}*.nii.gz")
    candidates = sorted(glob.glob(pattern1) + glob.glob(pattern2))
    if not candidates:
        raise FileNotFoundError(f"No se encontró NIfTI para {subject_id} en {folder}")
    return candidates[0]

def load_nifti(path: str, reorient_to_ras: bool = True) -> nib.Nifti1Image:
    img = nib.load(path)
    if reorient_to_ras:
        img = nib.as_closest_canonical(img)
    return img

def find_wmh_mask_for_subject(folder: str, subject_id: str) -> str:
    """Busca máscara WMH del sujeto en folder (nombre contiene subject_id y 'WMHmask')."""
    p1 = os.path.join(folder, f"*{subject_id}*WMHmask.nii")
    p2 = os.path.join(folder, f"*{subject_id}*WMHmask.nii.gz")
    candidates = sorted(glob.glob(p1) + glob.glob(p2))
    if not candidates:
        raise FileNotFoundError(f"No se encontró WMH mask para {subject_id} en {folder}")
    return candidates[0]

def label_clusters(mask_bool: np.ndarray) -> tuple[np.ndarray, int]:
    """Etiqueta clústeres 3D (26 vecinos)."""
    structure = ndi.generate_binary_structure(rank=3, connectivity=3)  # 26 vecinos
    labeled, nlab = ndi.label(mask_bool.astype(np.uint8), structure=structure)
    return labeled, nlab

def dilate_cluster_inside_mask(cluster_mask: np.ndarray,
                               restrict_mask: np.ndarray,
                               n_vox: int) -> np.ndarray:
    """
    Dilata cluster_mask n_vox iteraciones (en voxeles) y recorta dentro de restrict_mask.
    """
    structure = ndi.generate_binary_structure(rank=3, connectivity=3)  # 26 vecinos
    out = cluster_mask.astype(bool)
    for _ in range(max(0, n_vox)):
        out = ndi.binary_dilation(out, structure=structure)
    out = out & restrict_mask.astype(bool)
    return out

def get_wm_mask_in_subject_space(
    subject_img: nib.Nifti1Image,
    reorient_to_ras: bool = True,
    wm_mask_mode: str = "mni152_wm",           # "mni152_wm" o "custom"
    wm_mask_path: Optional[str] = None,
) -> nib.Nifti1Image:
    """
    Devuelve máscara de sustancia blanca (WM) en el espacio (grilla) de subject_img.

    - "mni152_wm": usa nilearn.datasets.load_mni152_wm_mask() (se descarga y cachea automáticamente).
    - "custom": resamplea una máscara WM propia al espacio del sujeto.
    """
    try:
        from nilearn import image as nimg
        from nilearn import datasets
    except Exception as e:
        raise RuntimeError(
            "Necesitas nilearn instalado para usar el filtro de WM.\n"
            "Instala: pip install nilearn"
        ) from e

    subj = subject_img
    if reorient_to_ras:
        subj = nib.as_closest_canonical(subj)

    if wm_mask_mode.lower() == "custom":
        if not wm_mask_path:
            raise ValueError("wm_mask_mode='custom' requiere wm_mask_path.")
        wm_img = load_nifti(wm_mask_path, reorient_to_ras=reorient_to_ras)
        wm_res = nimg.resample_to_img(wm_img, subj, interpolation="nearest")
        wm_bin = (wm_res.get_fdata() > 0).astype(np.uint8)
        return nimg.new_img_like(subj, wm_bin)

    wm_mni = datasets.load_mni152_wm_mask()
    wm_mni = nib.as_closest_canonical(wm_mni)
    wm_res = nimg.resample_to_img(wm_mni, subj, interpolation="nearest")
    wm_bin = (wm_res.get_fdata() > 0).astype(np.uint8)
    return nimg.new_img_like(subj, wm_bin)

# ---------------------------------------------------------------------
# 3) CARGAR EXCEL
# ---------------------------------------------------------------------

print(f"[INFO] Leyendo Excel: {excel_path}")
df = pd.read_excel(excel_path)

if FILTER_REGION_WMH and "region" in df.columns:
    before = len(df)
    df = df[df["region"].astype(str).str.lower() == "wmh"].copy()
    print(f"[INFO] Filtrando region='wmh': {before} -> {len(df)} filas")

if "T1_diff" not in df.columns:
    df["T1_diff"] = np.nan

# ---------------------------------------------------------------------
# 4) CACHE POR SUJETO
# ---------------------------------------------------------------------

current_subject = None
wmh_shape = None
labels_wmh = None
wm_bool = None
t1_diff_data = None
dilated_by_label = None

# ---------------------------------------------------------------------
# 5) LOOP FILA A FILA
# ---------------------------------------------------------------------

for idx, row in df.iterrows():
    subject_id = str(row["subject_id"])

    peak_x = int(row["peak_x"])
    peak_y = int(row["peak_y"])
    peak_z = int(row["peak_z"])

    # --- recargar todo si cambia el sujeto
    if subject_id != current_subject:
        print(f"\n[INFO] Cambiando a sujeto: {subject_id}")

        # ---- cargar T1s (sin normalizar)
        try:
            t1_time0_path = find_nifti_for_subject(t1_time0_folder, subject_id)
            t1_time2_path = find_nifti_for_subject(t1_time2_folder, subject_id)
        except FileNotFoundError as e:
            print(f"[WARN] {e}. Se salta sujeto.")
            current_subject = None
            continue

        img_t0 = load_nifti(t1_time0_path, reorient_to_ras=True)
        img_t2 = load_nifti(t1_time2_path, reorient_to_ras=True)

        data_t0 = img_t0.get_fdata()
        data_t2 = img_t2.get_fdata()

        # ---- T1_diff = T2 - T0 (SIN min-max)
        diff_data = data_t2 - data_t0
        diff_img  = nib.Nifti1Image(diff_data, img_t2.affine, img_t2.header)

        # ---- cargar WMH mask del sujeto
        try:
            wmh_mask_path = find_wmh_mask_for_subject(wmh_mask_folder, subject_id)
        except FileNotFoundError as e:
            print(f"[WARN] {e}. Se salta sujeto.")
            current_subject = None
            continue

        wmh_img = load_nifti(wmh_mask_path, reorient_to_ras=True)
        wmh_data = wmh_img.get_fdata()
        wmh_bool = wmh_data > 0
        wmh_shape = wmh_img.shape

        # ---- obtener WM mask (MNI152) en el MISMO espacio que wmh_img
        try:
            wm_mask_img = get_wm_mask_in_subject_space(
                subject_img=wmh_img,          # <- WM en espacio de la WMH
                reorient_to_ras=True,
                wm_mask_mode="mni152_wm",
                wm_mask_path=None
            )
        except Exception as e:
            print(f"[WARN] No se pudo cargar/resamplear WM MNI152 para {subject_id}: {e}")
            current_subject = None
            continue

        wm_bool = wm_mask_img.get_fdata() > 0

        # ---- resamplear T1_diff al espacio WMH
        t1_diff_img_rs = resample_from_to(diff_img, wmh_img, order=3)  # trilinear
        t1_diff_data = t1_diff_img_rs.get_fdata()

        # ---- label clusters WMH
        labels_wmh, nlab = label_clusters(wmh_bool)

        # ---- precomputar máscaras dilatadas por label, recortadas dentro de WM
        dilated_by_label = {}
        for lab in range(1, nlab + 1):
            cl = (labels_wmh == lab)
            dilated_by_label[lab] = dilate_cluster_inside_mask(
                cluster_mask=cl,
                restrict_mask=wm_bool,         # <- todo dentro de WM (MNI152 WM resampleada)
                n_vox=N_VOX_DILATION
            )

        current_subject = subject_id

    if current_subject is None:
        continue

    # ---- chequear pico dentro del volumen
    nx, ny, nz = wmh_shape
    if not (0 <= peak_x < nx and 0 <= peak_y < ny and 0 <= peak_z < nz):
        print(f"[WARN] {subject_id} fila {idx}: pico fuera de límites ({peak_x},{peak_y},{peak_z}).")
        continue

    # ---- encontrar cluster por peak
    label_id = int(labels_wmh[peak_x, peak_y, peak_z])
    if label_id == 0:
        print(f"[WARN] {subject_id} fila {idx}: pico cae fuera de WMH (label=0).")
        continue

    cluster_mask_dil = dilated_by_label.get(label_id, None)
    if cluster_mask_dil is None or not np.any(cluster_mask_dil):
        print(f"[WARN] {subject_id} fila {idx}: máscara dilatada vacía para label {label_id}.")
        continue

    # ---- promedio de T1_diff dentro del cluster dilatado (limitado a WM)
    vals = t1_diff_data[cluster_mask_dil]
    vals = vals[np.isfinite(vals)]
    mean_val = float(vals.mean()) if vals.size else np.nan

    df.at[idx, "T1_diff"] = mean_val

# ---------------------------------------------------------------------
# 6) GUARDAR
# ---------------------------------------------------------------------

print(f"\n[INFO] Guardando Excel actualizado en: {excel_out}")
df.to_excel(excel_out, index=False)
print("[INFO] Listo. 'T1_diff' = promedio dentro de clúster WMH dilatado (voxeles) y recortado a WM (MNI152).")
