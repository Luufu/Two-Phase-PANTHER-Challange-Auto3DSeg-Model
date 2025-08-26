# import os
# import sys
# import importlib.util
# import json
# import torch
# import SimpleITK as sitk
# import numpy as np

# # Setup
# BASE_PATH = "/home/keshav/PANTHER_Task1_Auto3DSeg"
# DATA_PATH = os.path.join(BASE_PATH, "data")
# RESULTS_PATH = os.path.join(BASE_PATH, "results")
# TMP_PATH = os.path.join(BASE_PATH, "tmp_infer")
# OUTPUT_PATH = os.path.join(BASE_PATH, "output")

# os.makedirs(TMP_PATH, exist_ok=True)
# os.makedirs(OUTPUT_PATH, exist_ok=True)

# #  Choose an image from training set to test
# test_image_filename = "10000_0001_0000.mha"
# input_image_path = os.path.join(DATA_PATH, "imagesTr", test_image_filename)
# output_json_file = os.path.join(TMP_PATH, "test_case.json")

# #  Create dataset.json format for testing
# test_image_rel_path = os.path.join("imagesTr", test_image_filename)

# with open(output_json_file, "w") as f:
#     json.dump({
#         "testing": [{
#             "image": input_image_path,
#             "case_id": "test_case_0"
#         }]
#     }, f, indent=4)


# #  Locate each fold’s config and segmenter
# config_files = []
# segmenter_paths = []
# for fold in range(5):
#     config_files.append(os.path.join(RESULTS_PATH, f"segresnet_{fold}", "configs", "hyper_parameters.yaml"))
#     segmenter_paths.append(os.path.join(RESULTS_PATH, f"segresnet_{fold}", "scripts", "segmenter.py"))

# #  Load segmenter.py dynamically
# def load_segmenter_function(segmenter_path):
#     import importlib.util
#     import sys

#     # Add the scripts folder to sys.path so 'utils' can be imported
#     sys.path.insert(0, os.path.dirname(segmenter_path))

#     spec = importlib.util.spec_from_file_location("segmenter", segmenter_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module.run_segmenter


# #  Run inference for each fold
# predictions = []
# for i in range(5):
#     print(f" Running inference for Fold {i}")
#     print(f"→ Running `run_segmenter` for Fold {i}")
#     print(f"  Config: {config_files[i]}")
#     print(f"  Segmenter: {segmenter_paths[i]}")
#     print(f"  Output: {os.path.join(TMP_PATH, f'predictions_{i}', 'prediction_testing')}")
#     print(f"  Test JSON: {output_json_file}")

#     run_segmenter = load_segmenter_function(segmenter_paths[i])
#     output_dir = os.path.join(TMP_PATH, f"predictions_{i}")
#     os.makedirs(output_dir, exist_ok=True)

#     override = {
#     "infer#enabled": True,
#     "infer#output_path": os.path.join(output_dir, "prediction_testing"),
#     "infer#save_mask": True,
#     "infer#data_list_key": "testing",
#     "ckpt_name": os.path.join(RESULTS_PATH, f"segresnet_{i}", "model", "model.pt"),

#     #  This is what datafold_read() needs:
#     "data_list_file_path": output_json_file
# }


#     run_segmenter(config_file=config_files[i], **override)

#     pred_folder = os.path.join(output_dir, "prediction_testing")
#     pred_file = None
#     for root, _, files in os.walk(pred_folder):
#         for f in files:
#             if f.endswith(".nii.gz") or f.endswith(".mha"):
#                 pred_file = os.path.join(root, f)
#                 break
#         if pred_file:
#             break

#     if pred_file:
#         predictions.append(pred_file)
#     else:
#         print(f" No prediction found in {pred_folder}")

# #  Perform label-wise majority voting for multi-class output
# if not predictions:
#     raise RuntimeError(" No predictions found from any fold!")

# print(" Performing label-wise ensemble...")
# pred_arrays = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in predictions]
# pred_stack = np.stack(pred_arrays, axis=0)  # shape: (num_folds, D, H, W)

# # Prepare empty output
# ensemble_array = np.zeros_like(pred_stack[0], dtype=np.uint8)

# # Perform majority voting label-wise
# for label in [1, 2]:  # tumor and pancreas
#     label_votes = (pred_stack == label).astype(np.uint8)
#     label_sum = np.sum(label_votes, axis=0)
#     ensemble_array[label_sum >= 3] = label  # 3/5 majority

# # Save ensembled result
# output_file = os.path.join(OUTPUT_PATH, "ensemble_output.mha")
# ensemble_image = sitk.GetImageFromArray(ensemble_array)
# ensemble_image.CopyInformation(sitk.ReadImage(predictions[0]))
# sitk.WriteImage(ensemble_image, output_file)

# print(f"Ensemble saved at {output_file}")

# import os
# import sys
# import importlib.util
# import json
# import torch
# import SimpleITK as sitk
# import numpy as np

# # === Setup paths ===
# BASE_PATH = "/home/keshav/PANTHER_Task1_Auto3DSeg"
# DATA_PATH = os.path.join(BASE_PATH, "data")
# RESULTS_PATH = os.path.join(BASE_PATH, "results")
# TMP_PATH = os.path.join(BASE_PATH, "tmp_infer")
# OUTPUT_PATH = os.path.join(BASE_PATH, "output")

# os.makedirs(TMP_PATH, exist_ok=True)
# os.makedirs(OUTPUT_PATH, exist_ok=True)

# # === Load inference cases from JSON ===
# inference_cases_path = os.path.join(DATA_PATH, "inference_cases.json")
# with open(inference_cases_path, "r") as f:
#     inference_data = json.load(f)["inference"]

# # === Create unified test_case.json ===
# test_case_json = os.path.join(TMP_PATH, "test_case.json")
# test_entries = []
# case_ids = []

# for case in inference_data:
#     image_path = os.path.join(DATA_PATH, case["image"][0])
#     case_id = case["case_id"]
#     case_ids.append(case_id)
#     test_entries.append({
#         "image": image_path,
#         "case_id": case_id
#     })

# with open(test_case_json, "w") as f:
#     json.dump({"testing": test_entries}, f, indent=4)

# # === Locate each fold’s config and segmenter ===
# config_files = []
# segmenter_paths = []
# for fold in range(5):
#     config_files.append(os.path.join(RESULTS_PATH, f"segresnet_{fold}", "configs", "hyper_parameters.yaml"))
#     segmenter_paths.append(os.path.join(RESULTS_PATH, f"segresnet_{fold}", "scripts", "segmenter.py"))

# def load_segmenter_function(segmenter_path):
#     sys.path.insert(0, os.path.dirname(segmenter_path))
#     spec = importlib.util.spec_from_file_location("segmenter", segmenter_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module.run_segmenter

# # === Run inference across all folds ===
# predictions_dict = {cid: [] for cid in case_ids}

# for i in range(5):
#     print(f"\n🔁 Running inference for Fold {i}")
#     run_segmenter = load_segmenter_function(segmenter_paths[i])
#     output_dir = os.path.join(TMP_PATH, f"predictions_{i}")
#     os.makedirs(output_dir, exist_ok=True)

#     override = {
#         "infer#enabled": True,
#         "infer#output_path": os.path.join(output_dir, "prediction_testing"),
#         "infer#save_mask": True,
#         "infer#data_list_key": "testing",
#         "ckpt_name": os.path.join(RESULTS_PATH, f"segresnet_{i}", "model", "model.pt"),
#         "data_list_file_path": test_case_json
#     }

#     run_segmenter(config_file=config_files[i], **override)

#     pred_folder = os.path.join(output_dir, "prediction_testing")
#     for cid in case_ids:
#         # Search for any file under pred_folder that contains the case_id in its name
#         found = False
#         for root, _, files in os.walk(pred_folder):
#             for f in files:
#                 if cid in f and (f.endswith(".mha") or f.endswith(".nii.gz")):
#                     full_path = os.path.join(root, f)
#                     predictions_dict[cid].append(full_path)
#                     found = True
#                     break
#             if found:
#                 break
#         if not found:
#             print(f"⚠️ No prediction for {cid} in Fold {i}")


# # === Perform ensemble for each case ===
# for cid, preds in predictions_dict.items():
#     if not preds:
#         print(f"❌ No predictions for {cid}, skipping ensemble.")
#         continue

#     print(f"\n🧠 Performing ensemble for {cid}...")
#     pred_arrays = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in preds]
#     pred_stack = np.stack(pred_arrays, axis=0)

#     ensemble_array = np.zeros_like(pred_stack[0], dtype=np.uint8)
#     for label in [1, 2]:  # label 1 = tumor, 2 = pancreas
#         label_votes = (pred_stack == label).astype(np.uint8)
#         label_sum = np.sum(label_votes, axis=0)
#         ensemble_array[label_sum >= 3] = label  # majority vote (3 of 5)

#     ensemble_image = sitk.GetImageFromArray(ensemble_array)
#     ensemble_image.CopyInformation(sitk.ReadImage(preds[0]))
#     output_file = os.path.join(OUTPUT_PATH, f"{cid}_ensemble_output.mha")
#     sitk.WriteImage(ensemble_image, output_file)
#     print(f"✅ Saved ensemble output for {cid} at: {output_file}")


# import os
# import sys
# import json
# import numpy as np
# import SimpleITK as sitk
# import importlib.util

# # === Setup paths ===
# BASE_PATH = "/home/keshav/PANTHER_Task1_Auto3DSeg"
# DATA_PATH = os.path.join(BASE_PATH, "data_cropped")
# RESULTS_PATH = os.path.join(BASE_PATH, "results")
# TMP_PATH = os.path.join(BASE_PATH, "tmp_infer")
# OUTPUT_PATH = os.path.join(BASE_PATH, "output")

# os.makedirs(TMP_PATH, exist_ok=True)
# os.makedirs(OUTPUT_PATH, exist_ok=True)

# # === Load inference cases from JSON ===
# inference_cases_path = os.path.join(DATA_PATH, "inference_cases.json")
# with open(inference_cases_path, "r") as f:
#     inference_data = json.load(f)["inference"]

# # === Create test_case.json ===
# test_case_json = os.path.join(TMP_PATH, "test_case.json")
# test_entries = []
# case_ids = []

# for case in inference_data:
#     image_path = os.path.join(DATA_PATH, case["image"][0])
#     case_id = case["case_id"]
#     case_ids.append(case_id)
#     test_entries.append({
#         "image": image_path,
#         "case_id": case_id
#     })

# with open(test_case_json, "w") as f:
#     json.dump({"testing": test_entries}, f, indent=4)

# # === Load segmenters ===
# config_files = []
# segmenter_paths = []
# for fold in range(5):
#     config_files.append(os.path.join(RESULTS_PATH, f"segresnet_{fold}", "configs", "hyper_parameters.yaml"))
#     segmenter_paths.append(os.path.join(RESULTS_PATH, f"segresnet_{fold}", "scripts", "segmenter.py"))

# def load_segmenter_function(segmenter_path):
#     sys.path.insert(0, os.path.dirname(segmenter_path))
#     spec = importlib.util.spec_from_file_location("segmenter", segmenter_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module.run_segmenter

# # === Run inference for each fold ===
# predictions_dict = {cid: [] for cid in case_ids}

# for i in range(5):
#     print(f"\n🔁 Running inference for Fold {i}")
#     run_segmenter = load_segmenter_function(segmenter_paths[i])
#     output_dir = os.path.join(TMP_PATH, f"predictions_{i}")
#     os.makedirs(output_dir, exist_ok=True)

#     override = {
#         "infer#enabled": True,
#         "infer#output_path": os.path.join(output_dir, "prediction_testing"),
#         "infer#save_mask": True,
#         "infer#data_list_key": "testing",
#         "ckpt_name": os.path.join(RESULTS_PATH, f"segresnet_{i}", "model", "model.pt"),
#         "data_list_file_path": test_case_json
#     }

#     run_segmenter(config_file=config_files[i], **override)

#     pred_folder = os.path.join(output_dir, "prediction_testing")
#     for cid in case_ids:
#         for root, _, files in os.walk(pred_folder):
#             for f in files:
#                 if cid in f and (f.endswith(".mha") or f.endswith(".nii.gz")):
#                     predictions_dict[cid].append(os.path.join(root, f))
#                     break

# # === Perform STAPLE Ensemble ===
# for cid, preds in predictions_dict.items():
#     if not preds:
#         print(f"❌ No predictions for {cid}, skipping ensemble.")
#         continue

#     print(f"\n🧠 Performing STAPLE ensemble for {cid}...")
#     ref_image = sitk.ReadImage(preds[0])
#     final_array = np.zeros(sitk.GetArrayFromImage(ref_image).shape, dtype=np.uint8)

#     for label in [1, 2]:  # Tumor and Pancreas
#         binary_masks = []
#         for pred_file in preds:
#             img = sitk.ReadImage(pred_file)
#             bin_img = sitk.BinaryThreshold(img, label, label, 1, 0)
#             binary_masks.append(bin_img)

#         staple = sitk.STAPLEImageFilter()
#         prob_img = staple.Execute(binary_masks)
#         result_img = sitk.BinaryThreshold(prob_img, 0.5, 1.0, label, 0)

#         result_array = sitk.GetArrayFromImage(result_img)
#         final_array[result_array == label] = label

#     out_img = sitk.GetImageFromArray(final_array)
#     out_img.CopyInformation(ref_image)
#     out_path = os.path.join(OUTPUT_PATH, f"{cid}_staple_output.mha")
#     sitk.WriteImage(out_img, out_path)
#     print(f"✅ Saved STAPLE output: {out_path}")


import os
import sys
import json
import numpy as np
import SimpleITK as sitk
import importlib.util
from typing import Optional, List

# =========================
# Paths / constants
# =========================
BASE_PATH = "/home/keshav/PANTHER_Task2_Auto3DSeg"
DATA_PATH = os.path.join(BASE_PATH, "data_cropped")               # contains dataset.json + inference_cases.json
RESULTS_PATH = os.path.join(BASE_PATH, "results")                  # segresnet_{fold}/...
TMP_PATH = os.path.join(BASE_PATH, "tmp_infer")                    # temp predictions by fold
OUTPUT_PATH = os.path.join(BASE_PATH, "output")                    # final STAPLE masks

# Stage-2 cropped dataset (output of this script)
STAGE2_IMG_DIR = os.path.join(BASE_PATH, "data_stage2_predcrop", "imagesTr")
STAGE2_LBL_DIR = os.path.join(BASE_PATH, "data_stage2_predcrop", "labelsTr")
STAGE2_META    = os.path.join(BASE_PATH, "data_stage2_predcrop", "crop_metadata.json")
MARGIN_MM = 30.0  # 3 cm

# Also look here for labels when attaching to inference cases
LABELS_DIR = os.path.join(DATA_PATH, "labelsTr")

os.makedirs(TMP_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(STAGE2_IMG_DIR, exist_ok=True)
os.makedirs(STAGE2_LBL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STAGE2_META), exist_ok=True)

# =========================
# Helpers
# =========================
def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def to_abs(path_or_rel):
    # dataset.json paths are relative to DATA_PATH
    return path_or_rel if os.path.isabs(path_or_rel) else os.path.join(DATA_PATH, path_or_rel)

def _strip_ext(name: str):
    for ext in (".nii.gz", ".nii", ".mha", ".nrrd"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name

def _poss_label_paths(base_stem: str) -> List[str]:
    """Build a list of candidate label paths in LABELS_DIR for a given stem (no extension)."""
    exts = (".mha", ".nii.gz", ".nii", ".nrrd")
    return [os.path.join(LABELS_DIR, base_stem + e) for e in exts]

def _img_stem_variants(img_stem: str) -> List[str]:
    """
    Given image stem like '10001_0001_0000', return likely label stems:
      - exact: 10001_0001_0000
      - drop '_0000': 10001_0001
      - if someone saved label with channel: still try base + '_0000'
    """
    variants = [img_stem]
    if img_stem.endswith("_0000"):
        variants.append(img_stem[:-5])  # drop _0000
    # Also consider dropping any trailing _000X
    parts = img_stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 4:
        variants.append("_".join(parts[:-1]))
    # Deduplicate
    seen = set()
    uniq = []
    for v in variants:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq

def _guess_label_for_image(image_abs_path: str) -> Optional[str]:
    """Try multiple stems & extensions to find a label file."""
    stem = _strip_ext(os.path.basename(image_abs_path))  # e.g., 10001_0001_0000
    for s in _img_stem_variants(stem):
        for cand in _poss_label_paths(s):
            if os.path.exists(cand):
                return cand
    return None

def derive_case_id(entry):
    # Prefer explicit case_id if present; otherwise derive from image/label stems
    if "case_id" in entry and entry["case_id"]:
        return entry["case_id"]

    def stem(p):
        n = os.path.basename(p)
        return _strip_ext(n)

    if "label" in entry and entry["label"]:
        return stem(entry["label"])
    if "image" in entry and entry["image"]:
        im = entry["image"][0] if isinstance(entry["image"], (list, tuple)) else entry["image"]
        s = stem(im)
        # prefer dropping channel suffix for case_id
        if s.endswith("_0000"): s = s[:-5]
        return s
    return None

def load_segmenter_function(segmenter_path):
    sys.path.insert(0, os.path.dirname(segmenter_path))
    spec = importlib.util.spec_from_file_location("segmenter", segmenter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_segmenter

def bbox_from_mask(arr_zyx):
    coords = np.where(arr_zyx > 0)
    if coords[0].size == 0:
        return None
    z0, z1 = int(coords[0].min()), int(coords[0].max()) + 1
    y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
    x0, x1 = int(coords[2].min()), int(coords[2].max()) + 1
    return (z0, z1, y0, y1, x0, x1)

def expand_bbox_zyx(bbox, spacing_xyz, shape_zyx, margin_mm):
    # ITK spacing: (sx, sy, sz); numpy array order is (Z, Y, X)
    sx, sy, sz = spacing_xyz
    mz = int(round(margin_mm / sz))  # along Z
    my = int(round(margin_mm / sy))  # along Y
    mx = int(round(margin_mm / sx))  # along X
    z0, z1, y0, y1, x0, x1 = bbox
    z0 = max(0, z0 - mz); y0 = max(0, y0 - my); x0 = max(0, x0 - mx)
    z1 = min(shape_zyx[0], z1 + mz); y1 = min(shape_zyx[1], y1 + my); x1 = min(shape_zyx[2], x1 + mx)
    return (z0, z1, y0, y1, x0, x1)

def crop_like(img, bbox):
    z0, z1, y0, y1, x0, x1 = bbox
    arr = sitk.GetArrayFromImage(img)[z0:z1, y0:y1, x0:x1]
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing())
    out.SetDirection(img.GetDirection())
    new_origin = img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0)))
    out.SetOrigin(new_origin)
    return out

def read_img(path): return sitk.ReadImage(path)
def write_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path)

# =========================
# Collect ALL cases (dataset.json + inference_cases.json)
# =========================
dataset_json_path    = os.path.join(DATA_PATH, "dataset.json")
inference_cases_path = os.path.join(DATA_PATH, "inference_cases.json")

dataset = read_json(dataset_json_path) or {}
infer   = read_json(inference_cases_path) or {}

all_entries = []

# dataset.json may have "training", "validation", "testing"
for key in ("training", "validation", "testing"):
    if key in dataset and isinstance(dataset[key], list):
        for e in dataset[key]:
            img_rel = e.get("image")
            if isinstance(img_rel, (list, tuple)):
                img_rel = img_rel[0]
            if img_rel is None:
                continue
            entry = {
                "image": to_abs(img_rel),
                "label": to_abs(e["label"]) if "label" in e and e["label"] else None,
                "case_id": derive_case_id(e),
            }
            all_entries.append(entry)

# inference cases (images; now attach label if present)
if "inference" in infer and isinstance(infer["inference"], list):
    for e in infer["inference"]:
        img_rel = e.get("image")
        if isinstance(img_rel, (list, tuple)):
            img_rel = img_rel[0]
        if img_rel is None:
            continue
        img_abs = to_abs(img_rel)
        label_abs = _guess_label_for_image(img_abs)
        if label_abs is None:
            # helpful debug
            stem = _strip_ext(os.path.basename(img_abs))
            print(f"[DEBUG] No label found for inference image '{img_abs}'. Tried stems: {_img_stem_variants(stem)} in {LABELS_DIR}")
        entry = {
            "image": img_abs,
            "label": label_abs,  # may be None if truly not found
            "case_id": e.get("case_id") or derive_case_id({"image": img_abs}),
        }
        all_entries.append(entry)

# Deduplicate by case_id (prefer entries that have labels)
cases = {}
for e in all_entries:
    cid = e["case_id"]
    if cid is None:
        continue
    if cid not in cases:
        cases[cid] = e
    else:
        # keep label if present
        if cases[cid].get("label") is None and e.get("label"):
            cases[cid]["label"] = e["label"]

print(f"Collected {len(cases)} unique cases. With labels: {sum(1 for v in cases.values() if v.get('label'))}.")

# =========================
# Build a single test_case.json for inference of ALL cases
# =========================
test_case_json = os.path.join(TMP_PATH, "test_case.json")
test_entries = [{"image": v["image"], "case_id": cid} for cid, v in sorted(cases.items())]
with open(test_case_json, "w") as f:
    json.dump({"testing": test_entries}, f, indent=4)
case_ids = list(cases.keys())

# =========================
# Load segmenters / configs
# =========================
config_files = []
segmenter_paths = []
for fold in range(5):
    config_files.append(os.path.join(RESULTS_PATH, f"segresnet_{fold}", "configs", "hyper_parameters.yaml"))
    segmenter_paths.append(os.path.join(RESULTS_PATH, f"segresnet_{fold}", "scripts", "segmenter.py"))

def run_inference_fold(fold_idx):
    print(f"\n🔁 Running inference for Fold {fold_idx}")
    run_segmenter = load_segmenter_function(segmenter_paths[fold_idx])
    output_dir = os.path.join(TMP_PATH, f"predictions_{fold_idx}")
    os.makedirs(output_dir, exist_ok=True)

    override = {
        "infer#enabled": True,
        "infer#output_path": os.path.join(output_dir, "prediction_testing"),
        "infer#save_mask": True,
        "infer#data_list_key": "testing",
        "ckpt_name": os.path.join(RESULTS_PATH, f"segresnet_{fold_idx}", "model", "model.pt"),
        "data_list_file_path": test_case_json
    }
    run_segmenter(config_file=config_files[fold_idx], **override)
    return os.path.join(output_dir, "prediction_testing")

# =========================
# Run inference across folds
# =========================
predictions_dict = {cid: [] for cid in case_ids}
pred_dirs = []
for i in range(5):
    pred_dir = run_inference_fold(i)
    pred_dirs.append(pred_dir)
    # accumulate predictions per case_id
    for cid in case_ids:
        found = False
        for root, _, files in os.walk(pred_dir):
            for f in files:
                if cid in f and (f.endswith(".mha") or f.endswith(".nii.gz")):
                    predictions_dict[cid].append(os.path.join(root, f))
                    found = True
                    break
            if found: break

# =========================
# STAPLE ensemble to OUTPUT_PATH
# =========================
for cid, preds in predictions_dict.items():
    if not preds:
        print(f"❌ No predictions for {cid}, skipping ensemble.")
        continue

    print(f"\n🧠 Performing STAPLE ensemble for {cid}...")
    ref_image = sitk.ReadImage(preds[0])
    final_arr = np.zeros(sitk.GetArrayFromImage(ref_image).shape, dtype=np.uint8)

    # Combine labels 1 and 2 (robust across training conventions)
    for label in [1, 2]:
        binary_masks = []
        for pred_file in preds:
            img = sitk.ReadImage(pred_file)
            bin_img = sitk.BinaryThreshold(img, label, label, 1, 0)
            binary_masks.append(bin_img)

        staple = sitk.STAPLEImageFilter()
        prob_img = staple.Execute(binary_masks)
        result_img = sitk.BinaryThreshold(prob_img, 0.5, 1.0, label, 0)

        result_array = sitk.GetArrayFromImage(result_img)
        final_arr[result_array == label] = label

    out_img = sitk.GetImageFromArray(final_arr)
    out_img.CopyInformation(ref_image)
    out_path = os.path.join(OUTPUT_PATH, f"{cid}_staple_output.mha")
    sitk.WriteImage(out_img, out_path)
    print(f"✅ Saved STAPLE output: {out_path}")

# =========================
# Crop ALL images (+labels if available) using STAPLE mask
# =========================
# load or init crop metadata
if os.path.exists(STAGE2_META):
    with open(STAGE2_META, "r") as f:
        crop_meta = json.load(f)
else:
    crop_meta = {}

def crop_case(cid, img_path, lbl_path, mask_path):
    try:
        img = sitk.ReadImage(img_path)
        msk = sitk.ReadImage(mask_path)
    except Exception as e:
        print(f"[SKIP] {cid}: cannot read image/mask ({e})")
        return

    img_arr = sitk.GetArrayFromImage(img)
    msk_arr = sitk.GetArrayFromImage(msk)

    # resample mask to image grid if needed
    if msk_arr.shape != img_arr.shape:
        print(f"[INFO] Resampling mask -> image grid for {cid}")
        msk = sitk.Resample(msk, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        msk_arr = sitk.GetArrayFromImage(msk)

    bbox = bbox_from_mask(msk_arr)
    if bbox is None:
        print(f"[WARN] {cid}: empty pancreas prediction; skipping crop.")
        return

    bbox_m = expand_bbox_zyx(bbox, img.GetSpacing(), img_arr.shape, MARGIN_MM)
    cimg = crop_like(img, bbox_m)

    # save cropped image with original filename
    out_img_name = os.path.basename(img_path)
    out_img_path = os.path.join(STAGE2_IMG_DIR, out_img_name)
    write_img(cimg, out_img_path)

    out_lbl_path = None
    if lbl_path is not None and os.path.exists(lbl_path):
        try:
            lbl = sitk.ReadImage(lbl_path)
            lbl_arr = sitk.GetArrayFromImage(lbl)
            if lbl_arr.shape != img_arr.shape:
                print(f"[INFO] Resampling label -> image grid for {cid}")
                lbl = sitk.Resample(lbl, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            clbl = crop_like(lbl, bbox_m)
            out_lbl_name = os.path.basename(lbl_path)
            out_lbl_path = os.path.join(STAGE2_LBL_DIR, out_lbl_name)
            write_img(clbl, out_lbl_path)
        except Exception as e:
            print(f"[WARN] {cid}: could not crop label ({e})")

    # record metadata
    crop_meta[cid] = {
        "image_file": img_path,
        "label_file": lbl_path,
        "mask_file": mask_path,
        "cropped_image": out_img_path,
        "cropped_label": out_lbl_path,
        "bbox_zyx": [int(v) for v in bbox_m],
        "orig_size_zyx": list(img_arr.shape),
        "spacing_xyz": list(img.GetSpacing()),
        "direction": list(img.GetDirection()),
        "origin": list(img.GetOrigin()),
        "margin_mm": MARGIN_MM,
    }
    with open(STAGE2_META, "w") as f:
        json.dump(crop_meta, f, indent=2)
    print(f"[OK] Cropped {cid} -> {out_img_name}{' + label' if out_lbl_path else ''}")

# Iterate all cases and crop
for cid, v in sorted(cases.items()):
    mask_path = os.path.join(OUTPUT_PATH, f"{cid}_staple_output.mha")
    if not os.path.exists(mask_path):
        print(f"[SKIP] {cid}: no STAPLE mask at {mask_path}")
        continue
    crop_case(cid, v["image"], v.get("label"), mask_path)

print("\n✅ All done.")
print(f"   Cropped images -> {STAGE2_IMG_DIR}")
print(f"   Cropped labels -> {STAGE2_LBL_DIR}")
print(f"   Metadata       -> {STAGE2_META}")
