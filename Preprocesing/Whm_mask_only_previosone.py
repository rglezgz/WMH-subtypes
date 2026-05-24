# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 12:09:26 2025

@author: White
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 23:25:03 2025

@author: White

WMH t0/t1 – Salida combinada SOLO WMH (sin mirror) + filtro por máscara de Sustancia Blanca (WM)
+ Plot axial de la suma grupal WMH_sum.nii.gz

- Lee TODOS los .nii/.nii.gz en carpetas t0/t1 (sin patrones).
- Etiqueta clústeres 3D (26 vecinos), filtra por tamaño >= min_cluster_size.
- Por clúster toma hasta 'central_voxels' voxeles centrales alrededor del PICO (máximo dentro del clúster).
- Acepta clúster si:
    (A) mean(central_voxels) > min_mean_wmh   (p.ej. > 0.2)  [se evalúa en t1 para clústeres t1 y en t0 para t0_promoted]
    (B) fracción de centrales dentro de máscara WM >= min_wm_fraction
- Combinación:
    • t1: agrega clústeres aceptados en t1
    • t0_promoted: agrega clústeres aceptados en t0 SOLO si en t1 esos centrales están “cero”
      (fracción >= min_zero_fraction usando t1_zero_epsilon)
- Guarda:
    • Una máscara combinada por sujeto: *_WMHmask.nii.gz
    • Un Excel con clusters + summary (incluye mean_wmh_t0, mean_wmh_t1 y delta)
    • Una suma grupal WMH_sum.nii.gz
    • Un PNG axial WMH_sum_axial.png

NOTA:
- Si tus WMH no están en MNI, el filtro wm_mask_mode="mni152_wm" puede no ser válido.
  En ese caso usa wm_mask_mode="custom" con una máscara WM en el mismo espacio.
"""

import os
import re
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi

# --------------------------- Parámetros de tolerancia ---------------------------
t1_zero_epsilon   = 0.05
min_zero_fraction = 0.60

# Filtros principales (umbral de mean y fracción WM)
min_mean_wmh = 0.2
min_wm_fraction = 0.60

# --------------------------- Utilidades ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def subjects_from_folder(folder: str, include_gz: bool = True) -> List[Tuple[str, str]]:
    nii = glob.glob(os.path.join(folder, "*.nii"))
    niigz = glob.glob(os.path.join(folder, "*.nii.gz")) if include_gz else []
    files = sorted(nii + niigz)
    out: List[Tuple[str, str]] = []
    for f in files:
        name = os.path.basename(f)
        subj = re.sub(r"\.nii(\.gz)?$", "", name, flags=re.IGNORECASE)
        out.append((subj, f))
    print(f"[INFO] {len(out)} NIfTI encontrados en {folder} (include_gz={include_gz})")
    if len(out) == 0:
        print("[WARN] No se encontraron NIfTI. Revisa la carpeta.")
    return out

def base_id_from_filename(path_or_name: str) -> str:
    name = os.path.basename(path_or_name)
    name = re.sub(r'\.nii(\.gz)?$', '', name, flags=re.IGNORECASE)
    base = re.split(r'_ses-', name, maxsplit=1)[0]
    return base

def load_nifti(path: str, reorient_to_ras: bool = True) -> nib.Nifti1Image:
    img = nib.load(path)
    if reorient_to_ras:
        img = nib.as_closest_canonical(img)
    return img

def binarize_for_labels(data: np.ndarray, prob_threshold: float = 0.0) -> np.ndarray:
    data = np.nan_to_num(data, nan=0.0)
    return (data > prob_threshold).astype(np.uint8)

def label_clusters(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    structure = ndi.generate_binary_structure(rank=3, connectivity=3)  # 26 vecinos
    labeled, nlab = ndi.label(binary, structure=structure)
    return labeled, nlab

def peak_and_central(coords: np.ndarray, data_src: np.ndarray, n_pick: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    - coords: Nx3 (vox) del clúster
    - data_src: volumen de donde se obtiene el pico (máximo dentro del clúster)
    - n_pick: cantidad de centrales (cerca del pico)
    """
    vals = data_src[coords[:, 0], coords[:, 1], coords[:, 2]]
    max_val = np.max(vals)
    cand_idx = np.flatnonzero(vals == max_val)

    if cand_idx.size == 1:
        peak = coords[cand_idx[0]].astype(int)
    else:
        centroid_cont = coords.mean(axis=0)
        d2c = ((coords[cand_idx] - centroid_cont) ** 2).sum(axis=1)
        peak = coords[cand_idx[np.argmin(d2c)]].astype(int)

    d2p = ((coords - peak) ** 2).sum(axis=1)
    order = np.argsort(d2p)
    central_coords = coords[order[: min(n_pick, coords.shape[0])]].astype(int)
    return peak, central_coords

def fraction_zero(vals: np.ndarray, epsilon: float) -> float:
    return float(np.mean(vals <= epsilon))

# --------------------------- Máscara WM (MNI152 o custom) ---------------------------

def get_wm_mask_in_subject_space(
    subject_img: nib.Nifti1Image,
    reorient_to_ras: bool = True,
    wm_mask_mode: str = "mni152_wm",           # "mni152_wm" o "custom"
    wm_mask_path: Optional[str] = None,
) -> nib.Nifti1Image:
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

def wm_fraction_for_coords(wm_mask_data: np.ndarray, coords: np.ndarray) -> float:
    vals = wm_mask_data[coords[:, 0], coords[:, 1], coords[:, 2]]
    return float(np.mean(vals > 0.5))

# --------------------------- Estructura para Excel ---------------------------

@dataclass
class ClusterInfo:
    label_id: int
    size_vox_full: int
    peak_xyz: Tuple[float, float, float]
    central_count: int
    accepted: bool
    time: str
    mean_wmh_t0: float
    mean_wmh_t1: float
    delta_mean_wmh: float
    wm_fraction: float
    reason: str

# --------------------------- Procesamiento por tiempo ---------------------------

def process_time_wmh_only(
    data_t1: np.ndarray,
    data_t0: np.ndarray,
    labeled_src: np.ndarray,
    wm_mask_data: np.ndarray,
    min_cluster_size: int,
    central_voxels: int,
    time_tag: str,
    wmh_accum: np.ndarray,
    min_mean_wmh_local: float,
    min_wm_fraction_local: float,
) -> List[ClusterInfo]:
    """
    Procesa clústeres provenientes de labeled_src (típicamente t1).
    - Para cada clúster aceptado:
        - mean_wmh_t1 y mean_wmh_t0 se calculan usando las MISMAS central_coords
        - delta = t1 - t0
    - El criterio de mean se evalúa en t1 (porque estos clústeres son t1).
    """
    details: List[ClusterInfo] = []
    labels = [lab for lab in np.unique(labeled_src) if lab != 0]

    for lab in labels:
        coords = np.argwhere(labeled_src == lab)
        size = int(coords.shape[0])

        if size < min_cluster_size:
            details.append(ClusterInfo(int(lab), size, tuple(coords.mean(axis=0).astype(float)), 0,
                                       False, time_tag, float("nan"), float("nan"), float("nan"), float("nan"),
                                       "< min size"))
            continue

        peak, central_coords = peak_and_central(coords, data_t1, n_pick=central_voxels)

        vals_t1 = data_t1[central_coords[:, 0], central_coords[:, 1], central_coords[:, 2]]
        vals_t0 = data_t0[central_coords[:, 0], central_coords[:, 1], central_coords[:, 2]]

        if np.isnan(vals_t1).any() or np.isnan(vals_t0).any():
            details.append(ClusterInfo(int(lab), size, tuple(peak.astype(float)), int(central_coords.shape[0]),
                                       False, time_tag, float("nan"), float("nan"), float("nan"), float("nan"),
                                       "rejected: NaN in central voxels (t0/t1)"))
            continue

        mean_t1 = float(np.mean(vals_t1))
        mean_t0 = float(np.mean(vals_t0))
        delta = mean_t1 - mean_t0

        if mean_t1 <= min_mean_wmh_local:
            details.append(ClusterInfo(int(lab), size, tuple(peak.astype(float)), int(central_coords.shape[0]),
                                       False, time_tag, mean_t0, mean_t1, delta, float("nan"),
                                       f"rejected: mean_wmh(t1)={mean_t1:.4f} <= {min_mean_wmh_local:.2f}"))
            continue

        frac_wm = wm_fraction_for_coords(wm_mask_data, central_coords)
        if frac_wm < min_wm_fraction_local:
            details.append(ClusterInfo(int(lab), size, tuple(peak.astype(float)), int(central_coords.shape[0]),
                                       False, time_tag, mean_t0, mean_t1, delta, frac_wm,
                                       f"rejected: wm_fraction={frac_wm:.2f} < {min_wm_fraction_local:.2f}"))
            continue

        wmh_accum[central_coords[:, 0], central_coords[:, 1], central_coords[:, 2]] = 1
        details.append(ClusterInfo(int(lab), size, tuple(peak.astype(float)), int(central_coords.shape[0]),
                                   True, time_tag, mean_t0, mean_t1, delta, frac_wm,
                                   f"ok (mean_t1={mean_t1:.4f}, mean_t0={mean_t0:.4f}, wm_fraction={frac_wm:.2f})"))
    return details

# --------------------------- Por sujeto: combina t1 + t0_promoted ---------------------------

def process_subject_pair_wmh_only(
    path_t0: str,
    path_t1: str,
    min_cluster_size: int,
    central_voxels: int,
    prob_threshold: float,
    reorient_to_ras: bool,
    out_mask_dir: str,
    wm_mask_mode: str,
    wm_mask_path: Optional[str],
    min_mean_wmh_local: float,
    min_wm_fraction_local: float,
) -> Tuple[str, List[ClusterInfo]]:

    img_t1 = load_nifti(path_t1, reorient_to_ras=reorient_to_ras)
    data_t1 = img_t1.get_fdata()

    img_t0 = load_nifti(path_t0, reorient_to_ras=reorient_to_ras)
    data_t0 = img_t0.get_fdata()

    if data_t1.shape != data_t0.shape:
        raise ValueError(f"Shapes distintos t1 {data_t1.shape} vs t0 {data_t0.shape} para {os.path.basename(path_t1)}")

    wm_mask_img = get_wm_mask_in_subject_space(
        subject_img=img_t1,
        reorient_to_ras=reorient_to_ras,
        wm_mask_mode=wm_mask_mode,
        wm_mask_path=wm_mask_path
    )
    wm_mask_data = wm_mask_img.get_fdata()

    wmh_out = np.zeros_like(data_t1, dtype=np.uint8)
    details_all: List[ClusterInfo] = []

    # ---------- t1 ----------
    bin_t1 = binarize_for_labels(data_t1, prob_threshold=prob_threshold)
    labeled_t1, _ = label_clusters(bin_t1)

    details_all += process_time_wmh_only(
        data_t1=data_t1,
        data_t0=data_t0,
        labeled_src=labeled_t1,
        wm_mask_data=wm_mask_data,
        min_cluster_size=min_cluster_size,
        central_voxels=central_voxels,
        time_tag="t1",
        wmh_accum=wmh_out,
        min_mean_wmh_local=min_mean_wmh_local,
        min_wm_fraction_local=min_wm_fraction_local
    )

    # ---------- t0_promoted ----------
    bin_t0 = binarize_for_labels(data_t0, prob_threshold=prob_threshold)
    labeled_t0, _ = label_clusters(bin_t0)
    labels_t0 = [lab for lab in np.unique(labeled_t0) if lab != 0]

    for lab in labels_t0:
        coords = np.argwhere(labeled_t0 == lab)
        size = int(coords.shape[0])

        if size < min_cluster_size:
            details_all.append(ClusterInfo(int(lab), size, tuple(coords.mean(axis=0).astype(float)), 0,
                                           False, "t0_promoted", float("nan"), float("nan"), float("nan"), float("nan"),
                                           "< min size"))
            continue

        # pico/centrales según t0 (porque el clúster viene de t0)
        peak, central_coords = peak_and_central(coords, data_t0, n_pick=central_voxels)

        vals_t0 = data_t0[central_coords[:, 0], central_coords[:, 1], central_coords[:, 2]]
        vals_t1 = data_t1[central_coords[:, 0], central_coords[:, 1], central_coords[:, 2]]

        if np.isnan(vals_t0).any() or np.isnan(vals_t1).any():
            details_all.append(ClusterInfo(int(lab), size, tuple(peak.astype(float)), int(central_coords.shape[0]),
                                           False, "t0_promoted", float("nan"), float("nan"), float("nan"), float("nan"),
                                           "rejected: NaN in central voxels (t0/t1)"))
            continue

        mean_t0 = float(np.mean(vals_t0))
        mean_t1 = float(np.mean(vals_t1))
        delta = mean_t1 - mean_t0

        # criterio de mean se evalúa en t0 para t0_promoted
        if mean_t0 <= min_mean_wmh_local:
            details_all.append(ClusterInfo(int(lab), size, tuple(peak.astype(float)), int(central_coords.shape[0]),
                                           False, "t0_promoted", mean_t0, mean_t1, delta, float("nan"),
                                           f"rejected: mean_wmh(t0)={mean_t0:.4f} <= {min_mean_wmh_local:.2f}"))
            continue

        frac_wm = wm_fraction_for_coords(wm_mask_data, central_coords)
        if frac_wm < min_wm_fraction_local:
            details_all.append(ClusterInfo(int(lab), size, tuple(peak.astype(float)), int(central_coords.shape[0]),
                                           False, "t0_promoted", mean_t0, mean_t1, delta, frac_wm,
                                           f"rejected: wm_fraction={frac_wm:.2f} < {min_wm_fraction_local:.2f}"))
            continue

        # condición de promoción: que en t1 sea "cero" suficiente
        zfrac = fraction_zero(vals_t1, t1_zero_epsilon)
        if zfrac < min_zero_fraction:
            details_all.append(ClusterInfo(int(lab), size, tuple(peak.astype(float)), int(central_coords.shape[0]),
                                           False, "t0_promoted", mean_t0, mean_t1, delta, frac_wm,
                                           f"rejected: t0->t1 not zero enough (zero_frac={zfrac:.2f} < {min_zero_fraction:.2f})"))
            continue

        wmh_out[central_coords[:, 0], central_coords[:, 1], central_coords[:, 2]] = 1
        details_all.append(ClusterInfo(int(lab), size, tuple(peak.astype(float)), int(central_coords.shape[0]),
                                       True, "t0_promoted", mean_t0, mean_t1, delta, frac_wm,
                                       f"ok (t0_promoted; mean_t0={mean_t0:.4f}, mean_t1={mean_t1:.4f}, delta={delta:.4f}, wm_fraction={frac_wm:.2f}, t1_zero_frac={zfrac:.2f})"))

    ensure_dir(out_mask_dir)
    base = os.path.basename(path_t1).replace(".nii.gz", "").replace(".nii", "")
    out_mask_path = os.path.join(out_mask_dir, f"{base}_WMHmask.nii.gz")
    nib.save(nib.Nifti1Image(wmh_out.astype(np.uint8), img_t1.affine, img_t1.header.copy()), out_mask_path)

    return out_mask_path, details_all

# --------------------------- Excel y orquestación ---------------------------

def run_combined_wmh_only(
    folder_t0: str,
    folder_t1: str,
    out_excel: str,
    out_mask_dir: str,
    min_cluster_size: int = 100,
    central_voxels: int = 100,
    prob_threshold: float = 0.01,
    reorient_to_ras: bool = True,
    wm_mask_mode: str = "mni152_wm",
    wm_mask_path: Optional[str] = None,
    min_mean_wmh_local: float = 0.2,
    min_wm_fraction_local: float = 0.60,
) -> None:
    ensure_dir(out_mask_dir)
    ensure_dir(os.path.dirname(out_excel))

    subs_t0_list = subjects_from_folder(folder_t0, include_gz=True)
    subs_t1_list = subjects_from_folder(folder_t1, include_gz=True)

    subs_t0 = {base_id_from_filename(sid): p for sid, p in subs_t0_list}
    subs_t1 = {base_id_from_filename(sid): p for sid, p in subs_t1_list}

    all_ids = sorted(set(subs_t1.keys()))
    if not all_ids:
        print("[WARN] No hay sujetos en t1. Nada que hacer.")
        return

    rows = []
    summary = []

    for sid in all_ids:
        p1 = subs_t1[sid]
        p0 = subs_t0.get(sid, None)
        if p0 is None:
            print(f"[INFO] {sid}: sin t0; se procesa solo t1 (t0 se iguala a t1 para compatibilidad)")
            p0 = p1

        try:
            mask_path, details = process_subject_pair_wmh_only(
                path_t0=p0,
                path_t1=p1,
                min_cluster_size=min_cluster_size,
                central_voxels=central_voxels,
                prob_threshold=prob_threshold,
                reorient_to_ras=reorient_to_ras,
                out_mask_dir=out_mask_dir,
                wm_mask_mode=wm_mask_mode,
                wm_mask_path=wm_mask_path,
                min_mean_wmh_local=min_mean_wmh_local,
                min_wm_fraction_local=min_wm_fraction_local
            )
        except Exception as e:
            print(f"[ERROR] {sid}: {e}")
            continue

        total_candidates = sum(1 for d in details if d.size_vox_full >= min_cluster_size)
        accepted = sum(1 for d in details if d.accepted and d.size_vox_full >= min_cluster_size)

        # (Opcional) resumen de deltas por sujeto (solo aceptados)
        deltas = [d.delta_mean_wmh for d in details if d.accepted and np.isfinite(d.delta_mean_wmh)]
        mean_delta = float(np.mean(deltas)) if len(deltas) > 0 else float("nan")

        for d in details:
            if not d.accepted:
                continue
            px, py, pz = d.peak_xyz
            rows.append({
                "subject_id": sid,
                "time": d.time,
                "cluster_label": int(d.label_id),
                "size_vox_full": int(d.size_vox_full),
                "size_vox": int(d.central_count),
                "peak_x": float(px),
                "peak_y": float(py),
                "peak_z": float(pz),
                "mean_wmh_t0": float(d.mean_wmh_t0),
                "mean_wmh_t1": float(d.mean_wmh_t1),
                "delta_mean_wmh": float(d.delta_mean_wmh),
                "wm_fraction": float(d.wm_fraction),
                "accepted": 1,
                "reason": d.reason
            })

        summary.append({
            "subject_id": sid,
            "clusters_total": total_candidates,
            "clusters_valid": accepted,
            "percent_valid": (accepted / total_candidates * 100.0) if total_candidates > 0 else 0.0,
            "mean_delta_accepted": mean_delta,
            "wmh_mask_path": mask_path
        })

    df_clusters = pd.DataFrame(rows, columns=[
        "subject_id", "time", "cluster_label",
        "size_vox_full", "size_vox",
        "peak_x", "peak_y", "peak_z",
        "mean_wmh_t0", "mean_wmh_t1", "delta_mean_wmh",
        "wm_fraction",
        "accepted", "reason"
    ])
    df_summary = pd.DataFrame(summary, columns=[
        "subject_id", "clusters_total", "clusters_valid", "percent_valid",
        "mean_delta_accepted",
        "wmh_mask_path"
    ])

    with pd.ExcelWriter(out_excel, engine="openpyxl") as w:
        df_clusters.to_excel(w, index=False, sheet_name="clusters")
        df_summary.to_excel(w, index=False, sheet_name="summary")

    print(f"[OK] Excel generado: {out_excel}")

# --------------------------- Suma grupal ---------------------------

def sum_niftis_in_folder(folder: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    files = sorted(glob.glob(os.path.join(folder, "*.nii")) + glob.glob(os.path.join(folder, "*.nii.gz")))
    if not files:
        raise RuntimeError(f"No se encontraron NIfTI en {folder}")
    ref = nib.load(files[0]); ref = nib.as_closest_canonical(ref)
    shape, affine, header = ref.shape, ref.affine, ref.header.copy()
    acc = np.zeros(shape, dtype=np.float32)
    for f in files:
        img = nib.load(f); img = nib.as_closest_canonical(img)
        if img.shape != shape:
            raise ValueError(f"Shape inconsistente: {os.path.basename(f)} {img.shape} != {shape}")
        acc += np.asanyarray(img.get_fdata(), dtype=np.float32)
    return nib.Nifti1Image(acc, affine, header), acc

# --------------------------- Plot axial WMH_sum ---------------------------

def plot_wmh_sum_axial(wmh_sum_path: str, out_png: str, n_slices: int = 12):
    """
    Genera un PNG con cortes axiales (z) de la suma grupal.
    - Usa MNI152 template como fondo si está disponible.
    - Colormap Reds truncado (empieza en tonos claros).
    """
    try:
        from nilearn import plotting, datasets
        from matplotlib import cm, colors
        import numpy as np
        import nibabel as nib
    except Exception as e:
        raise RuntimeError("Necesitas nilearn + matplotlib para el plot. pip install nilearn matplotlib") from e

    wmh_sum_img = nib.load(wmh_sum_path)
    wmh_sum_img = nib.as_closest_canonical(wmh_sum_img)
    wmh_data = wmh_sum_img.get_fdata()

    # Fondo anatómico (template MNI)
    try:
        bg_img = datasets.load_mni152_template(resolution=2)
    except Exception:
        bg_img = wmh_sum_img

    # Colormap truncado (Reds)
    def truncate_cmap(base_cmap, minval=0.30, maxval=1.0, n=256):
        new_colors = base_cmap(np.linspace(minval, maxval, n))
        return colors.LinearSegmentedColormap.from_list('trunc_' + base_cmap.name, new_colors)

    cmap_red = truncate_cmap(cm.get_cmap('Reds'), 0.30, 1.0)

    # Cortes z: donde haya señal, si no rango típico
    any_mask = (wmh_data > 0)
    if np.any(any_mask):
        affine = wmh_sum_img.affine
        iz = np.where(any_mask)[2]
        z_min, z_max = int(iz.min()), int(iz.max())
        z_idxs = np.linspace(z_min, z_max, n_slices).astype(int)
        z_mm = [float((affine @ np.array([0, 0, k, 1]))[2]) for k in z_idxs]
    else:
        z_mm = list(np.linspace(-40, 70, n_slices))

    # Escala
    vmin = 0.8
    vmax = float(np.percentile(wmh_data[wmh_data >= vmin], 99)) if np.any(wmh_data >= vmin) else 1.0
    vmax = max(vmax, 10.0)

    d = plotting.plot_stat_map(
        wmh_sum_img, bg_img=bg_img,
        display_mode='z', cut_coords=z_mm,
        threshold=0,
        colorbar=True,
        cmap=cmap_red,
        vmin=vmin, vmax=vmax,
        symmetric_cbar=False
    )
    d.savefig(out_png, dpi=300)
    d.close()
    print(f"[PLOT] WMH axial guardado en: {out_png}")

# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    folder_t0 = r"F:\long\test\All\WMH_0"
    folder_t1 = r"F:\long\test\All\WMH_1"

    out_dir      = r"F:\long\test\All\SALIDAS\whm_mask_only"
    out_excel    = os.path.join(out_dir, "WMH_clusters_summary_WMfilter_mean02.xlsx")
    out_mask_dir = os.path.join(out_dir, "WMH_masks_WMfilter_mean02")

    ensure_dir(out_dir)
    ensure_dir(out_mask_dir)

    # Hiperparámetros
    min_cluster_size = 100
    central_voxels   = 100
    prob_threshold   = 0.01
    reorient_to_ras  = True

    # Filtros
    min_mean_wmh_local = 0.2
    min_wm_fraction_local = 0.60

    # WM mask:
    # - "mni152_wm": usa datasets.load_mni152_wm_mask()
    # - "custom": usa tu máscara WM
    wm_mask_mode = "mni152_wm"
    wm_mask_path = None  # si custom: r"F:\...\wm_mask.nii.gz"

    # 1) Ejecuta pipeline y Excel
    run_combined_wmh_only(
        folder_t0=folder_t0,
        folder_t1=folder_t1,
        out_excel=out_excel,
        out_mask_dir=out_mask_dir,
        min_cluster_size=min_cluster_size,
        central_voxels=central_voxels,
        prob_threshold=prob_threshold,
        reorient_to_ras=reorient_to_ras,
        wm_mask_mode=wm_mask_mode,
        wm_mask_path=wm_mask_path,
        min_mean_wmh_local=min_mean_wmh_local,
        min_wm_fraction_local=min_wm_fraction_local
    )

    # 2) Suma grupal
    wmh_sum_img, _ = sum_niftis_in_folder(out_mask_dir)
    wmh_sum_path = os.path.join(out_dir, "WMH_sum.nii.gz")
    nib.save(wmh_sum_img, wmh_sum_path)
    print(f"[OK] Suma WMH guardada en: {wmh_sum_path}")

    # 3) Plot axial WMH_sum
    wmh_ax_png = os.path.join(out_dir, "WMH_sum_axial.png")
    plot_wmh_sum_axial(wmh_sum_path, wmh_ax_png, n_slices=12)
