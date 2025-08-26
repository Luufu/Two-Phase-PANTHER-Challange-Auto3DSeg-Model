#!/usr/bin/env python3
# Crops MRI images AND labels using Stage-1 masks, but the bbox is computed from PANCREAS ONLY (label==2),
# then expanded by +30 mm margin. If pancreas is missing, falls back to union of all labels (>0).
# Handles image stems like 10000_0001_0000.mha and label/mask stems like 10000_0001.mha

import json, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

# ---------- CONFIG ----------
BASE = Path("/home/keshav/PANTHER_Task2_Auto3DSeg")
IMAGES_DIR = BASE / "data_cropped" / "imagesTr"     # e.g., 10000_0001_0000.mha
LABELS_DIR = BASE / "data_cropped" / "labelsTr"     # e.g., 10000_0001.mha
MASKS_DIR  = BASE / "output"                        # Stage-1 masks (0=bg, 1=tumor, 2=pancreas)
OUT_IMG_DIR = BASE / "data_stage2_predcrop" / "imagesTr"
OUT_LBL_DIR = BASE / "data_stage2_predcrop" / "labelsTr"
META_PATH = BASE / "data_stage2_predcrop" / "crop_metadata.json"
MARGIN_MM = 30.0  # 3 cm
# ----------------------------

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)
META_PATH.parent.mkdir(parents=True, exist_ok=True)

def has_vol(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith((".mha", ".nii.gz", ".nii", ".nrrd"))

def strip_ext(name: str) -> str:
    for ext in (".nii.gz", ".nii", ".mha", ".nrrd"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name

def img_stem_to_label_stem(img_stem: str) -> str:
    # Convert "10000_0001_0000" -> "10000_0001"
    return img_stem[:-5] if img_stem.endswith("_0000") else img_stem

def read_img(p: Path) -> sitk.Image:
    return sitk.ReadImage(str(p))

def write_img(img: sitk.Image, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(p))

def bbox_from_mask(arr_zyx: np.ndarray):
    coords = np.where(arr_zyx > 0)
    if coords[0].size == 0:
        return None
    z0, z1 = int(coords[0].min()), int(coords[0].max()) + 1
    y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
    x0, x1 = int(coords[2].min()), int(coords[2].max()) + 1
    return (z0, z1, y0, y1, x0, x1)

def expand_bbox_zyx(bbox, spacing_xyz, shape_zyx, margin_mm: float):
    # ITK spacing (sx, sy, sz); array is (Z,Y,X) -> convert mm to voxels per axis
    sx, sy, sz = spacing_xyz
    mz = int(round(margin_mm / sz))  # along Z
    my = int(round(margin_mm / sy))  # along Y
    mx = int(round(margin_mm / sx))  # along X
    z0, z1, y0, y1, x0, x1 = bbox
    z0 = max(0, z0 - mz); y0 = max(0, y0 - my); x0 = max(0, x0 - mx)
    z1 = min(shape_zyx[0], z1 + mz); y1 = min(shape_zyx[1], y1 + my); x1 = min(shape_zyx[2], x1 + mx)
    return (z0, z1, y0, y1, x0, x1)

def crop_like(img: sitk.Image, bbox):
    z0, z1, y0, y1, x0, x1 = bbox
    arr = sitk.GetArrayFromImage(img)[z0:z1, y0:y1, x0:x1]
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing())
    out.SetDirection(img.GetDirection())
    new_origin = img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0)))
    out.SetOrigin(new_origin)
    return out

def find_label_for_image(img_path: Path) -> Path | None:
    img_stem = strip_ext(img_path.name)            # e.g., 10000_0001_0000
    base = img_stem_to_label_stem(img_stem)        # -> 10000_0001
    for cand in (LABELS_DIR / f"{base}.mha",
                 LABELS_DIR / f"{base}.nii.gz",
                 LABELS_DIR / f"{base}.nii",
                 LABELS_DIR / f"{base}.nrrd"):
        if cand.exists():
            return cand
    return None

def find_mask_for_image(img_path: Path) -> Path | None:
    img_stem = strip_ext(img_path.name)            # 10000_0001_0000
    base = img_stem_to_label_stem(img_stem)        # 10000_0001
    cands = [p for p in MASKS_DIR.rglob("*") if p.is_file() and has_vol(p)]
    def starts(p, s): return strip_ext(p.name).startswith(s)
    # preference: exact full stem with 'staple', then full stem, then base with 'staple', then base
    tiers = [
        [p for p in cands if starts(p, img_stem) and "staple" in p.name.lower()],
        [p for p in cands if starts(p, img_stem)],
        [p for p in cands if starts(p, base) and "staple" in p.name.lower()],
        [p for p in cands if starts(p, base)],
    ]
    for tier in tiers:
        if tier:
            return sorted(tier)[0]
    return None

# load or init metadata
meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

images = [p for p in IMAGES_DIR.iterdir() if has_vol(p)]
if not images:
    print(f"❌ No images found in {IMAGES_DIR}"); sys.exit(2)

for img_path in sorted(images):
    lbl_path = find_label_for_image(img_path)
    if lbl_path is None:
        print(f"[SKIP] No label for {img_path.name}")
        continue
    msk_path = find_mask_for_image(img_path)
    if msk_path is None:
        print(f"[SKIP] No Stage-1 mask for {img_path.name}")
        continue

    img = read_img(img_path)
    lbl = read_img(lbl_path)
    msk = read_img(msk_path)

    img_arr = sitk.GetArrayFromImage(img)
    lbl_arr = sitk.GetArrayFromImage(lbl)
    msk_arr = sitk.GetArrayFromImage(msk)

    # resample mask/label to image grid if needed (NN to preserve labels)
    if msk_arr.shape != img_arr.shape:
        print(f"[INFO] Resampling mask -> image grid for {img_path.name}")
        msk = sitk.Resample(msk, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        msk_arr = sitk.GetArrayFromImage(msk)
    if lbl_arr.shape != img_arr.shape:
        print(f"[INFO] Resampling label -> image grid for {img_path.name}")
        lbl = sitk.Resample(lbl, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        lbl_arr = sitk.GetArrayFromImage(lbl)

    # ---- PANCREAS-ONLY bbox (label==2), with safe fallback to union (>0) ----
    # Round to ints in case mask was saved as float probabilities
    msk_arr = np.rint(msk_arr).astype(np.int16)
    pancreas_only = (msk_arr == 2).astype(np.uint8)
    bbox = bbox_from_mask(pancreas_only)
    if bbox is None:
        union = (msk_arr > 0).astype(np.uint8)
        bbox = bbox_from_mask(union)
        if bbox is None:
            print(f"[WARN] Empty prediction for {img_path.name}; skipping.")
            continue
        else:
            print(f"[INFO] {img_path.name}: pancreas missing; using union bbox.")

    bbox_m = expand_bbox_zyx(bbox, img.GetSpacing(), img_arr.shape, MARGIN_MM)
    cimg = crop_like(img, bbox_m)
    clbl = crop_like(lbl, bbox_m)

    # preserve filenames
    out_img = OUT_IMG_DIR / img_path.name          # keep 10000_0001_0000.mha
    out_lbl = OUT_LBL_DIR / lbl_path.name          # keep 10000_0001.mha
    write_img(cimg, out_img)
    write_img(clbl, out_lbl)

    key = strip_ext(img_path.name)                 # use full image stem as key
    meta[key] = {
        "image_file": str(img_path),
        "label_file": str(lbl_path),
        "mask_file": str(msk_path),
        "cropped_image": str(out_img),
        "cropped_label": str(out_lbl),
        "bbox_zyx": [int(v) for v in bbox_m],
        "orig_size_zyx": list(img_arr.shape),
        "spacing_xyz": list(img.GetSpacing()),
        "direction": list(img.GetDirection()),
        "origin": list(img.GetOrigin()),
        "margin_mm": MARGIN_MM,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[OK] {img_path.name}  ->  {out_img.name}  (+{MARGIN_MM:.0f}mm, pancreas-only)")

print(f"\n✅ Cropped images: {OUT_IMG_DIR}")
print(f"✅ Cropped labels: {OUT_LBL_DIR}")
print(f"🗒️ Metadata: {META_PATH}")
