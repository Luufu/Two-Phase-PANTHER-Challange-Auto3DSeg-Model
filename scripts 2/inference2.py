#!/usr/bin/env python3
# Stage-2 inference for PANTHER Task 2
# Pipeline:
#   Input  : data_stage2_predcrop/imagesTr (cropped around pancreas)
#   Infer  : segresnet folds -> per-fold preds
#   Ensemble: STAPLE on cropped grid -> OUT_CROPPED
#   Uncrop1: bbox -> /data_cropped geometry (using crop_metadata.json) -> OUT_STEP1
#   Uncrop2: percent window -> /data geometry (using original crop percents) -> OUT_FINAL
#
# Label convention here assumes integers with 0=bg, 1=tumor, 2=pancreas.
# If your dataset uses 1=pancreas, 2=tumor, swap the loop order in the STAPLE block.

import os, sys, json
import numpy as np
import SimpleITK as sitk
import importlib.util

# ----------------------------
# Paths (set to your project)
# ----------------------------
BASE = "/home/keshav/PANTHER_Task2_Auto3DSeg"

# ORIGINALS (final target geometry for step-2 uncrop)
DATA_IMG_DIR       = os.path.join(BASE, "data", "imagesTr")             # <case>_0000.mha

# Stage-1 cropped dataset (used only for percent uncrop reference window)
DATA_CROPPED_DIR   = os.path.join(BASE, "data_cropped")

# Stage-2 (pred-cropped) dataset & metadata
STAGE2_DIR         = os.path.join(BASE, "data_stage2_predcrop")
STAGE2_IMG_DIR     = os.path.join(STAGE2_DIR, "imagesTr")
STAGE2_META_PATH   = os.path.join(STAGE2_DIR, "crop_metadata.json")     # bbox metadata
INFER_JSON_PATH    = os.path.join(STAGE2_DIR, "inference_cases.json")   # <-- your JSON lives here

# Trained Stage-2 results (5 folds)
RESULTS_STAGE2     = os.path.join(BASE, "results_stage2")               # segresnet_0..4 inside

# Temps & outputs
TMP_STAGE2         = os.path.join(BASE, "tmp_infer_stage2")
OUT_CROPPED        = os.path.join(BASE, "output_stage2_cropped")        # ensemble on cropped grid
OUT_STEP1          = os.path.join(BASE, "output_stage2_uncropped_to_data_cropped")  # bbox-based uncrop
OUT_FINAL          = os.path.join(BASE, "output_stage2_uncropped")      # percent-based uncrop (/data)

for d in (TMP_STAGE2, OUT_CROPPED, OUT_STEP1, OUT_FINAL):
    os.makedirs(d, exist_ok=True)

# ----------------------------
# Percent window used in your FIRST crop (Task 2)
# ----------------------------
Z_MIN_PCT, Z_MAX_PCT = 0.148, 1.000
Y_MIN_PCT, Y_MAX_PCT = 0.323, 0.705
X_MIN_PCT, X_MAX_PCT = 0.328, 0.790

# ----------------------------
# Helpers
# ----------------------------
def _strip_ext(name: str) -> str:
    for ext in (".nii.gz", ".nii", ".mha", ".nrrd"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name

def _case_id_from_image_path(p: str) -> str:
    s = _strip_ext(os.path.basename(p))
    # remove channel suffix if present
    if s.endswith("_0000"):
        s = s[:-5]
    return s

def _load_segmenter(segmenter_py):
    sys.path.insert(0, os.path.dirname(segmenter_py))
    spec = importlib.util.spec_from_file_location("segmenter", segmenter_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_segmenter

def _bounds_from_original(orig_img: sitk.Image):
    D, H, W = sitk.GetArrayFromImage(orig_img).shape
    z0 = int(D * Z_MIN_PCT); z1 = int(D * Z_MAX_PCT)
    y0 = int(H * Y_MIN_PCT); y1 = int(H * Y_MAX_PCT)
    x0 = int(W * X_MIN_PCT); x1 = int(W * X_MAX_PCT)
    return (z0, z1, y0, y1, x0, x1), (D, H, W)

def _paste_back_percent(cropped_img: sitk.Image, orig_ref: sitk.Image):
    """Step-2: paste cropped array into percent window of /data canvas."""
    bbox, (D, H, W) = _bounds_from_original(orig_ref)
    z0, z1, y0, y1, x0, x1 = bbox
    crop_arr = sitk.GetArrayFromImage(cropped_img)

    target = (z1 - z0, y1 - y0, x1 - x0)
    dz = min(target[0], crop_arr.shape[0])
    dy = min(target[1], crop_arr.shape[1])
    dx = min(target[2], crop_arr.shape[2])

    # handle rounding – center-crop to target if needed
    if (dz, dy, dx) != crop_arr.shape:
        cz0 = (crop_arr.shape[0] - dz) // 2
        cy0 = (crop_arr.shape[1] - dy) // 2
        cx0 = (crop_arr.shape[2] - dx) // 2
        crop_arr = crop_arr[cz0:cz0+dz, cy0:cy0+dy, cx0:cx0+dx]

    canvas = np.zeros((D, H, W), dtype=crop_arr.dtype)
    canvas[z0:z0+dz, y0:y0+dy, x0:x0+dx] = crop_arr

    out = sitk.GetImageFromArray(canvas)
    out.SetSpacing(orig_ref.GetSpacing())
    out.SetDirection(orig_ref.GetDirection())
    out.SetOrigin(orig_ref.GetOrigin())
    return out

def _paste_back_bbox(pred_path: str, meta_entry: dict, out_path: str):
    """Step-1: paste cropped pred back into /data_cropped canvas using bbox metadata."""
    pred = sitk.ReadImage(pred_path)
    p_arr = sitk.GetArrayFromImage(pred)

    Z, Y, X = meta_entry["orig_size_zyx"]
    z0, z1, y0, y1, x0, x1 = meta_entry["bbox_zyx"]

    canvas = np.zeros((Z, Y, X), dtype=p_arr.dtype)
    tz, ty, tx = z1 - z0, y1 - y0, x1 - x0
    dz = min(tz, p_arr.shape[0]); dy = min(ty, p_arr.shape[1]); dx = min(tx, p_arr.shape[2])
    if (dz, dy, dx) != p_arr.shape:
        cz0 = (p_arr.shape[0] - dz) // 2
        cy0 = (p_arr.shape[1] - dy) // 2
        cx0 = (p_arr.shape[2] - dx) // 2
        p_arr = p_arr[cz0:cz0+dz, cy0:cy0+dy, cx0:cx0+dx]

    canvas[z0:z0+dz, y0:y0+dy, x0:x0+dx] = p_arr

    out = sitk.GetImageFromArray(canvas)
    out.SetSpacing(tuple(meta_entry["spacing_xyz"]))
    out.SetDirection(tuple(meta_entry["direction"]))
    out.SetOrigin(tuple(meta_entry["origin"]))
    sitk.WriteImage(out, out_path)

# ----------------------------
# Build testing list from Stage-2 inference_cases.json
# (fallback: list STAGE2_IMG_DIR if JSON missing/empty)
# ----------------------------
testing_list = []

if os.path.exists(INFER_JSON_PATH):
    with open(INFER_JSON_PATH, "r") as f:
        infer = json.load(f)

    # accept a few common shapes: {"inference":[...]} or {"testing":[...]} or plain list
    candidate_lists = []
    if isinstance(infer, dict):
        for key in ("inference", "testing", "images", "cases"):
            if key in infer and isinstance(infer[key], list):
                candidate_lists = infer[key]
                break
        if not candidate_lists:
            # dataset-style? use "validation" or "training" as last resort
            for key in ("validation", "training"):
                if key in infer and isinstance(infer[key], list):
                    candidate_lists = infer[key]
                    break
    elif isinstance(infer, list):
        candidate_lists = infer

    for e in candidate_lists or []:
        img_rel = e.get("image") if isinstance(e, dict) else None
        if isinstance(img_rel, (list, tuple)): img_rel = img_rel[0]
        if not img_rel:
            continue
        # If path is relative, it should be relative to STAGE2_DIR
        stage2_img = img_rel if os.path.isabs(img_rel) else os.path.join(STAGE2_DIR, img_rel)
        # If entry contains "case_id", use it; else derive from image name
        cid = (e.get("case_id") if isinstance(e, dict) else None) or _case_id_from_image_path(stage2_img)
        if os.path.exists(stage2_img):
            testing_list.append({"image": stage2_img, "case_id": cid})

# Fallback to listing the stage-2 images dir
if not testing_list:
    if not os.path.isdir(STAGE2_IMG_DIR):
        print(f"❌ No inference list and missing {STAGE2_IMG_DIR}")
        sys.exit(2)
    for fn in sorted(os.listdir(STAGE2_IMG_DIR)):
        if fn.endswith((".mha", ".nii", ".nii.gz")):
            p = os.path.join(STAGE2_IMG_DIR, fn)
            testing_list.append({"image": p, "case_id": _case_id_from_image_path(p)})

if not testing_list:
    print("❌ No valid stage-2 inference images found.")
    sys.exit(2)

# unique by case_id
uniq = {}
for e in testing_list:
    uniq[e["case_id"]] = e
testing_list = list(uniq.values())
case_ids = [e["case_id"] for e in testing_list]
print(f"🧪 Stage-2: {len(case_ids)} cases to run.")

test_case_json = os.path.join(TMP_STAGE2, "test_case.json")
with open(test_case_json, "w") as f:
    json.dump({"testing": testing_list}, f, indent=2)

# ----------------------------
# Locate fold configs/segmenters
# ----------------------------
cfgs, seg_scripts = [], []
for fold in range(5):
    cfgs.append(os.path.join(RESULTS_STAGE2, f"segresnet_{fold}", "configs", "hyper_parameters.yaml"))
    seg_scripts.append(os.path.join(RESULTS_STAGE2, f"segresnet_{fold}", "scripts", "segmenter.py"))
    if not os.path.exists(cfgs[-1]) or not os.path.exists(seg_scripts[-1]):
        print(f"⚠️  Missing config or segmenter for fold {fold}: {cfgs[-1]} / {seg_scripts[-1]}")

def _collect_fold_preds(fold_idx):
    print(f"\n🔁 Stage-2 inference — Fold {fold_idx}")
    run_segmenter = _load_segmenter(seg_scripts[fold_idx])
    out_dir = os.path.join(TMP_STAGE2, f"fold_{fold_idx}")
    os.makedirs(out_dir, exist_ok=True)
    override = {
        "infer#enabled": True,
        "infer#output_path": os.path.join(out_dir, "prediction_testing"),
        "infer#save_mask": True,
        "infer#data_list_key": "testing",
        "ckpt_name": os.path.join(RESULTS_STAGE2, f"segresnet_{fold_idx}", "model", "model.pt"),
        "data_list_file_path": test_case_json
    }
    run_segmenter(config_file=cfgs[fold_idx], **override)
    return override["infer#output_path"]

# ----------------------------
# Run inference across folds
# ----------------------------
preds_per_case = {cid: [] for cid in case_ids}
for i in range(5):
    if not (os.path.exists(cfgs[i]) and os.path.exists(seg_scripts[i])):
        print(f"⏭️  Skipping fold {i} due to missing files.")
        continue
    pred_root = _collect_fold_preds(i)
    for cid in case_ids:
        hit = None
        for root, _, files in os.walk(pred_root):
            for fn in files:
                if cid in fn and fn.endswith((".mha", ".nii.gz", ".nii")):
                    hit = os.path.join(root, fn); break
            if hit: break
        if hit: preds_per_case[cid].append(hit)

# ----------------------------
# STAPLE ensemble on cropped grid
# ----------------------------
for cid, plist in preds_per_case.items():
    if not plist:
        print(f"❌ No predictions for {cid}; skipping ensemble.")
        continue
    print(f"🧠 STAPLE ensemble (cropped) for {cid} …")
    ref = sitk.ReadImage(plist[0])
    final_arr = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)

    # If your label mapping is 1=pancreas, 2=tumor, swap the order below.
    for label in [1, 2]:  # 1=tumor, 2=pancreas (adjust if needed)
        bin_masks = []
        for pf in plist:
            im = sitk.ReadImage(pf)
            bm = sitk.BinaryThreshold(im, label, label, 1, 0)
            bin_masks.append(bm)
        st = sitk.STAPLEImageFilter()
        prob = st.Execute(bin_masks)
        res = sitk.BinaryThreshold(prob, 0.5, 1.0, label, 0)
        arr = sitk.GetArrayFromImage(res)
        final_arr[arr == label] = label

    out_cropped = sitk.GetImageFromArray(final_arr)
    out_cropped.CopyInformation(ref)
    out_path = os.path.join(OUT_CROPPED, f"{cid}.mha")
    sitk.WriteImage(out_cropped, out_path)
    print(f"✅ Saved cropped ensemble: {out_path}")

# ----------------------------
# Load bbox metadata (for Step-1 uncrop to /data_cropped)
# ----------------------------
if not os.path.exists(STAGE2_META_PATH):
    print(f"❌ Missing metadata for step-1 uncrop: {STAGE2_META_PATH}")
    sys.exit(2)

with open(STAGE2_META_PATH, "r") as f:
    meta = json.load(f)

# meta may be dict-of-entries or list-of-entries; build a robust lookup
def _build_meta_maps(meta_obj):
    basename2meta, cid2meta = {}, {}
    if isinstance(meta_obj, dict):
        # could be {cid: entry} or {something: entry}; use both key and entry fields
        for k, v in meta_obj.items():
            if isinstance(v, dict):
                if "cropped_image" in v:
                    basename2meta[os.path.basename(v["cropped_image"])] = v
                if "case_id" in v:
                    cid2meta[v["case_id"]] = v
            # also map by key as a possible cid
            cid2meta[k] = v if isinstance(v, dict) else {"_raw": v}
    elif isinstance(meta_obj, list):
        for v in meta_obj:
            if not isinstance(v, dict): continue
            if "cropped_image" in v:
                basename2meta[os.path.basename(v["cropped_image"])] = v
            if "case_id" in v:
                cid2meta[v["case_id"]] = v
    return basename2meta, cid2meta

croppedbasename2meta, cid2meta = _build_meta_maps(meta)

def _find_meta_for_case(cid: str):
    # prefer lookup by the exact cropped image name used during testing
    cropped_img_basename = None
    for e in testing_list:
        if e["case_id"] == cid:
            cropped_img_basename = os.path.basename(e["image"])
            break
    if cropped_img_basename and cropped_img_basename in croppedbasename2meta:
        return croppedbasename2meta[cropped_img_basename]
    # fallback: by case_id itself or case_id+'_0000'
    if cid in cid2meta: return cid2meta[cid]
    if f"{cid}_0000" in cid2meta: return cid2meta[f"{cid}_0000"]
    return None

# ----------------------------
# Step 1 UN-CROP: bbox → /data_cropped geometry
# ----------------------------
for cid in case_ids:
    cropped_pred = os.path.join(OUT_CROPPED, f"{cid}.mha")
    if not os.path.exists(cropped_pred):
        print(f"[STEP1][SKIP] No cropped pred for {cid}")
        continue
    me = _find_meta_for_case(cid)
    if me is None:
        print(f"[STEP1][SKIP] No metadata match for {cid}")
        continue
    out_step1 = os.path.join(OUT_STEP1, f"{cid}.mha")
    _paste_back_bbox(cropped_pred, me, out_step1)
    print(f"[STEP1][OK] {cid} -> {out_step1}  (/data_cropped geometry)")

# ----------------------------
# Step 2 UN-CROP: percent window → /data geometry
# ----------------------------
for cid in case_ids:
    step1_pred = os.path.join(OUT_STEP1, f"{cid}.mha")
    if not os.path.exists(step1_pred):
        print(f"[STEP2][SKIP] {cid}: missing step-1 file {step1_pred}")
        continue

    orig_path = os.path.join(DATA_IMG_DIR, f"{cid}_0000.mha")
    if not os.path.exists(orig_path):
        print(f"[STEP2][SKIP] {cid}: original not found at {orig_path}")
        continue

    try:
        orig = sitk.ReadImage(orig_path)
        p1   = sitk.ReadImage(step1_pred)
    except Exception as e:
        print(f"[STEP2][SKIP] {cid}: read error ({e})")
        continue

    out_full = _paste_back_percent(p1, orig)
    out_path = os.path.join(OUT_FINAL, f"{cid}.mha")
    sitk.WriteImage(out_full, out_path)
    print(f"[STEP2][OK] {cid} -> {out_path}  (/data geometry)")

print("\n✅ Stage-2 inference + two-hop uncrop complete.")
print(f"   Cropped preds             : {OUT_CROPPED}")
print(f"   Uncropped to /data_cropped: {OUT_STEP1}")
print(f"   Uncropped to /data        : {OUT_FINAL}")
