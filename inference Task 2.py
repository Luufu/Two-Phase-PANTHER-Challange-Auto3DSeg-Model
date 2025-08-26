# /opt/algorithm/inference/gc_entrypoint.py
import os, sys, json, shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from typing import List, Tuple, Optional
import yaml  # for config rewrite

SUPPORTED_EXTS = (".mha", ".nii.gz", ".nii", ".nrrd", ".tif", ".tiff")

# ==== Task 2 training window + margin (keep in sync with training) ====
Z_MIN_PCT, Z_MAX_PCT = 0.148, 1.000
Y_MIN_PCT, Y_MAX_PCT = 0.323, 0.705
X_MIN_PCT, X_MAX_PCT = 0.328, 0.790
MARGIN_MM = float(os.environ.get("STAGE2_MARGIN_MM", "30.0"))

# Class IDs (override if needed)
TUMOR_LABEL    = int(os.environ.get("TUMOR_LABEL", "1"))
PANCREAS_LABEL = int(os.environ.get("PANCREAS_LABEL", "2"))

ROOT = Path("/opt/algorithm")
RES1 = ROOT / "results"          # segresnet_0..4 (Stage-1)
RES2 = ROOT / "results_stage2"   # segresnet_0..4 (Stage-2)
TMP  = ROOT / "work"
TMP.mkdir(parents=True, exist_ok=True)

# ---------------- CPU/GPU fallback + VRAM cap ----------------
if os.environ.get("FORCE_CPU", "0") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

def _cap_cuda_memory():
    """Optionally cap VRAM per process via env CUDA_MEM_FRAC, e.g. 0.6."""
    try:
        import torch
        if torch.cuda.is_available():
            frac = float(os.environ.get("CUDA_MEM_FRAC", "0"))
            if 0.0 < frac <= 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(frac, 0)
                    print(f"[INFO] Set CUDA per-process memory fraction to {frac:.3f}")
                except Exception as e:
                    print(f"[WARN] Could not set CUDA memory cap: {e}")
    except Exception:
        pass

def _log_device():
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            try:
                names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
            except Exception:
                names = "CUDA devices"
            print(f"[INFO] CUDA available: {n} device(s): {names}")
        else:
            print("[INFO] CUDA not available -> using CPU.")
    except Exception as e:
        print(f"[INFO] torch not available in entrypoint ({e}) -> using CPU.")
# --------------------------------------------------------------

def _uuid_from_path(p: Path) -> str:
    name = p.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return p.stem

def _find_input() -> Path:
    # Prefer Task-2 interface path, then Task-1 (just in case), then any supported
    for preferred in (Path("/input/images/abdominal-t2-mri"),
                      Path("/input/images/abdominal-t1-mri")):
        if preferred.exists():
            for p in preferred.rglob("*"):
                if p.is_file() and p.name.lower().endswith(SUPPORTED_EXTS):
                    return p
    ip = Path("/input")
    for p in ip.rglob("*"):
        if p.is_file() and p.name.lower().endswith(SUPPORTED_EXTS):
            return p
    raise FileNotFoundError("No 3D medical image found under /input")

def _sitk_read(p: Path) -> sitk.Image: return sitk.ReadImage(str(p))
def _sitk_write(img: sitk.Image, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(p))

# ---------- Non-empty window helpers ----------
def _nonempty_slice_bounds(lo_pct: float, hi_pct: float, size: int):
    lo = int(np.floor(size * lo_pct))
    hi = int(np.ceil(size * hi_pct))
    lo = max(0, min(lo, size - 1))
    hi = max(lo + 1, min(hi, size))
    return lo, hi

def _percent_bounds(ref: sitk.Image):
    D, H, W = sitk.GetArrayFromImage(ref).shape
    z0, z1 = _nonempty_slice_bounds(Z_MIN_PCT, Z_MAX_PCT, D)
    y0, y1 = _nonempty_slice_bounds(Y_MIN_PCT, Y_MAX_PCT, H)
    x0, x1 = _nonempty_slice_bounds(X_MIN_PCT, X_MAX_PCT, W)
    return (z0, z1, y0, y1, x0, x1), (D, H, W)
# ----------------------------------------------

def _crop_percent(img: sitk.Image) -> Tuple[sitk.Image,dict]:
    (z0,z1,y0,y1,x0,x1),(D,H,W)=_percent_bounds(img)
    arr = sitk.GetArrayFromImage(img)[z0:z1, y0:y1, x0:x1]
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing()); out.SetDirection(img.GetDirection()); out.SetOrigin(img.GetOrigin())
    meta = {"orig_shape_zyx":[D,H,W], "percent_bbox_zyx":[z0,z1,y0,y1,x0,x1]}
    return out, meta

def _paste_percent(cropped: sitk.Image, orig_ref: sitk.Image) -> sitk.Image:
    (z0,z1,y0,y1,x0,x1),(D,H,W)=_percent_bounds(orig_ref)
    c = sitk.GetArrayFromImage(cropped)
    target=(z1-z0, y1-y0, x1-x0)
    dz=min(target[0], c.shape[0]); dy=min(target[1], c.shape[1]); dx=min(target[2], c.shape[2])
    if (dz,dy,dx)!=c.shape:
        cz=(c.shape[0]-dz)//2; cy=(c.shape[1]-dy)//2; cx=(c.shape[2]-dx)//2
        c=c[cz:cz+dz, cy:cy+dy, cx:cx+dx]
    canvas=np.zeros((D,H,W), dtype=c.dtype)
    canvas[z0:z0+dz, y0:y0+dy, x0:x0+dx]=c
    out=sitk.GetImageFromArray(canvas)
    out.SetSpacing(orig_ref.GetSpacing()); out.SetDirection(orig_ref.GetDirection()); out.SetOrigin(orig_ref.GetOrigin())
    return out

def _bbox_from(arr: np.ndarray):
    zz,yy,xx = np.where(arr>0)
    if zz.size==0: return None
    return int(zz.min()), int(zz.max())+1, int(yy.min()), int(yy.max())+1, int(xx.min()), int(xx.max())+1

def _expand_bbox(bbox, spacing_xyz, shape_zyx, margin_mm):
    sx,sy,sz = spacing_xyz  # spacing order: (x,y,z)
    mz,my,mx = int(round(margin_mm/sz)), int(round(margin_mm/sy)), int(round(margin_mm/sx))
    z0,z1,y0,y1,x0,x1 = bbox
    z0=max(0,z0-mz); y0=max(0,y0-my); x0=max(0,x0-mx)
    z1=min(shape_zyx[0], z1+mz); y1=min(shape_zyx[1], y1+my); x1=min(shape_zyx[2], x1+mx)
    return z0,z1,y0,y1,x0,x1

def _ensure_nonempty_bbox(bbox, shape_zyx):
    z0,z1,y0,y1,x0,x1 = bbox
    Z,Y,X = shape_zyx
    if z1 <= z0: z1 = min(z0 + 1, Z)
    if y1 <= y0: y1 = min(y0 + 1, Y)
    if x1 <= x0: x1 = min(x0 + 1, X)
    return z0,z1,y0,y1,x0,x1

def _crop_bbox(img: sitk.Image, bbox):
    z0,z1,y0,y1,x0,x1=bbox
    arr=sitk.GetArrayFromImage(img)[z0:z1, y0:y1, x0:x1]
    out=sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing()); out.SetDirection(img.GetDirection())
    out.SetOrigin(img.TransformIndexToPhysicalPoint((int(x0),int(y0),int(z0))))
    return out

def _write_testing_json(img_path: Path, out_json: Path, case_id: str):
    # IMPORTANT: many segmenters expect a STRING for "image"
    data = {"testing": [{"image": str(img_path), "case_id": case_id}]}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2))

def _load_segmenter(run_py: Path):
    import importlib.util, sys
    sys.path.insert(0, str(run_py.parent))
    spec = importlib.util.spec_from_file_location("segmenter", str(run_py))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.run_segmenter

# ---------- Config rewrite + checkpoint alias ----------
_PREFIX_MAP = [
    ("/home/keshav/PANTHER_Task2_Auto3DSeg/results_stage2", str(RES2)),
    ("/home/keshav/PANTHER_Task2_Auto3DSeg/results",       str(RES1)),
    ("/home/keshav/PANTHER_Task2_Auto3DSeg",               str(ROOT)),
    ("/home/keshav",                                       str(TMP / "hosthome")),  # fallback
]

def _remap_path(s: str) -> str:
    for src, dst in _PREFIX_MAP:
        if isinstance(s, str) and s.startswith(src):
            return s.replace(src, dst, 1)
    return s

def _rewrite_cfg_paths_with_infer(orig_cfg: Path, out_dir: Path, pred_dir: Path,
                                  test_json: Path, ckpt: Path) -> Path:
    with open(orig_cfg, "r") as f:
        cfg = yaml.safe_load(f) or {}

    def _walk(x):
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, str):
            return _remap_path(x)
        return x

    cfg = _walk(cfg)

    # Ensure a sane infer block many segmenters honor
    infer = cfg.get("infer", {})
    infer.update({
        "enabled": True,
        "save_mask": True,
        "output_path": str(pred_dir),
        "data_list_key": "testing",
        "data_list_file_path": str(test_json),
        # Some scripts check these names:
        "ckpt_name": str(ckpt),
    })
    cfg["infer"] = infer

    # Common top-level paths some algos honor
    for k in ("output_path", "output_dir", "work_dir", "working_dir", "log_dir"):
        cfg[k] = str(out_dir)

    # Try to set a ckpt if the key exists / is observed by the algo
    for k in ("ckpt_name", "pretrained_ckpt", "resume_ckpt"):
        cfg[k] = str(ckpt)

    rewritten = out_dir / "hyper_parameters.runtime.yaml"
    rewritten.parent.mkdir(parents=True, exist_ok=True)
    with open(rewritten, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return rewritten

def _ensure_ckpt_alias(root_results: Path, fold_idx: int, ckpt_real: Path):
    """
    If any config still points at a host-like path, create an alias there.
    E.g., /opt/algorithm/work/hosthome/PANTHER_Task2_Auto3DSeg/results/segresnet_k/model/model.pt
    """
    sub = root_results.name  # 'results' or 'results_stage2'
    alias = TMP / "hosthome" / "PANTHER_Task2_Auto3DSeg" / sub / f"segresnet_{fold_idx}" / "model" / "model.pt"
    if not alias.exists():
        alias.parent.mkdir(parents=True, exist_ok=True)
        try:
            alias.symlink_to(ckpt_real)
            print(f"[INFO] Linked alias ckpt -> {alias} -> {ckpt_real}")
        except Exception as e:
            shutil.copy2(ckpt_real, alias)
            print(f"[INFO] Copied alias ckpt -> {alias} (symlink failed: {e})")
# ----------------------------------------------------------

def _infer_folds(root: Path, test_json: Path, tmp_root: Path,
                 case_id: str, forbid_exact: Path) -> List[Path]:
    """
    Run inference for 5 folds and collect a predicted SEG per fold.

    IMPORTANT:
    - Accept ANY image saved under the fold's out_dir/pred_dir except the exact input file.
    - This matches a segmenter that saves with the *same base filename* as input
      (e.g., work/data_cropped_img.nii.gz or work/stage2/img.nii.gz).
    """
    os.environ.setdefault("HOME", str(TMP / "home"))
    (TMP / "home").mkdir(parents=True, exist_ok=True)

    def _is_image_file(p: Path) -> bool:
        fn = p.name.lower()
        return p.is_file() and any(fn.endswith(ext) for ext in SUPPORTED_EXTS)

    def _search_for_pred(out_dir: Path, pred_dir: Path) -> Optional[Path]:
        # 1) non-recursive in pred_dir (exclude the true input)
        if pred_dir.exists():
            files = [f for f in pred_dir.iterdir() if _is_image_file(f)]
            files = [p for p in files if p.resolve() != forbid_exact.resolve()]
            if files:
                return max(files, key=lambda p: p.stat().st_size)

        # 2) recursive under out_dir (exclude the true input)
        all_imgs=[]
        for r,_,files in os.walk(out_dir):
            for f in files:
                p = Path(r)/f
                if _is_image_file(p) and p.resolve() != forbid_exact.resolve():
                    all_imgs.append(p)
        if all_imgs:
            # As a last resort choose the largest file (masks are often smaller, but
            # your segmenter writes labels using input basename, which is fine)
            print(f"[WARN] Last-resort fallback: selecting largest image under {out_dir}.")
            return max(all_imgs, key=lambda p: p.stat().st_size)
        return None

    preds: List[Path] = []
    for k in range(5):
        cfg = root/f"segresnet_{k}"/"configs"/"hyper_parameters.yaml"
        seg = root/f"segresnet_{k}"/"scripts"/"segmenter.py"
        ckpt= root/f"segresnet_{k}"/"model"/"model.pt"
        if not (cfg.exists() and seg.exists() and ckpt.exists()):
            print(f"[WARN] Fold {k}: missing files -> cfg:{cfg.exists()} seg:{seg.exists()} ckpt:{ckpt.exists()} (skipping)")
            continue

        out_dir  = tmp_root/f"fold_{k}"
        pred_dir = out_dir/"prediction_testing"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Write a runtime YAML that *explicitly* enables infer and points at our paths
        safe_cfg = _rewrite_cfg_paths_with_infer(cfg, out_dir, pred_dir, test_json, ckpt)
        _ensure_ckpt_alias(root, k, ckpt)

        run = _load_segmenter(seg)

        override = {
            # Namespaced keys some segmenters parse:
            "infer#enabled": True,
            "infer#output_path": str(pred_dir),
            "infer#save_mask": True,
            "infer#data_list_key": "testing",
            "infer#ckpt_name": str(ckpt),
            # Dataset + path helpers:
            "data_list_file_path": str(test_json),
            "data_file_base_dir": str(ROOT),
            # Extra common keys some algos honor
            "output_path": str(pred_dir),
            "work_dir": str(out_dir),
            "pretrained_ckpt": str(ckpt),
            "resume_ckpt": str(ckpt),
        }

        print(f"[INFO] Fold {k}: cfg={cfg}")
        print(f"[INFO] Fold {k}: using rewritten cfg={safe_cfg}")
        print(f"[INFO] Fold {k}: ckpt={ckpt}")
        print(f"[INFO] Fold {k}: out_path={pred_dir}")

        try:
            run(config_file=str(safe_cfg), **override)
        except TypeError:
            run(str(safe_cfg), **override)

        hit = _search_for_pred(out_dir, pred_dir)

        # If still nothing, retry with ORIGINAL YAML (no rewrite) just in case
        if hit is None:
            print(f"[WARN] Fold {k}: nothing found after rewritten cfg; retrying with ORIGINAL YAML …")
            try:
                run(config_file=str(cfg), **override)
            except TypeError:
                run(str(cfg), **override)
            hit = _search_for_pred(out_dir, pred_dir)

        if hit is not None:
            print(f"[INFO] Fold {k}: found prediction -> {hit}")
            preds.append(hit)
        else:
            print(f"[WARN] Fold {k}: no valid prediction artifact found under {out_dir}")
    return preds


def _staple(preds: List[Path]) -> sitk.Image:
    if not preds:
        raise RuntimeError("No predictions to ensemble.")
    ref=sitk.ReadImage(str(preds[0]))
    final=np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
    for label in (TUMOR_LABEL, PANCREAS_LABEL):
        bins=[]
        for pf in preds:
            img=sitk.ReadImage(str(pf))
            bins.append(sitk.BinaryThreshold(img, label, label, 1, 0))
        st=sitk.STAPLEImageFilter()
        prob=st.Execute(bins)
        res=sitk.BinaryThreshold(prob, 0.5, 1.0, label, 0)
        arr=sitk.GetArrayFromImage(res)
        final[arr==label]=label
    out=sitk.GetImageFromArray(final)
    out.CopyInformation(ref)
    return out

def _bounds_json(mask_final_np: np.ndarray):
    bxs={}
    for label,name in [(TUMOR_LABEL,"tumor"),(PANCREAS_LABEL,"pancreas")]:
        arr=(mask_final_np==label).astype(np.uint8)
        bbox=_bbox_from(arr)
        bxs[name]=bbox if bbox is not None else None
    return bxs

def main():
    in_img_path=_find_input()
    # case_id for logs / config; uuid for output file name
    base_stem = in_img_path.name
    if base_stem.lower().endswith(".nii.gz"):
        base_stem = base_stem[:-7]
    else:
        base_stem = Path(base_stem).stem
    case_id = base_stem.replace("_0000","")
    print(f"[INFO] Case: {case_id}")

    _log_device()
    _cap_cuda_memory()

    orig=_sitk_read(in_img_path)

    # 1) Percent crop (non-empty)
    pct_img, _ = _crop_percent(orig)

    # 2) Stage-1 inference on percent-cropped
    t1_dir=TMP/"stage1"; t1_dir.mkdir(exist_ok=True, parents=True)
    t1_json=t1_dir/"test.json"
    cropped_input = TMP/"data_cropped_img.mha"
    _sitk_write(pct_img, cropped_input)
    _write_testing_json(cropped_input, t1_json, case_id)
    preds1=_infer_folds(RES1, t1_json, t1_dir, case_id, forbid_exact=cropped_input)
    staple1=_staple(preds1)

    # 3) Pancreas-only bbox (+margin), fallback to union, enforce non-empty
    arr1=np.rint(sitk.GetArrayFromImage(staple1)).astype(np.int16)
    pancreas=(arr1==PANCREAS_LABEL).astype(np.uint8)
    bbox=_bbox_from(pancreas)
    if bbox is None:
        union=(arr1>0).astype(np.uint8)
        bbox=_bbox_from(union)
        if bbox is None:
            raise RuntimeError("Empty Stage-1 mask; cannot crop Stage-2.")
    bbox=_expand_bbox(bbox, pct_img.GetSpacing(), sitk.GetArrayFromImage(pct_img).shape, MARGIN_MM)
    bbox=_ensure_nonempty_bbox(bbox, sitk.GetArrayFromImage(pct_img).shape)
    stage2_img=_crop_bbox(pct_img, bbox)
    bbox_meta={"bbox_zyx":[int(v) for v in bbox],
               "orig_size_zyx": list(sitk.GetArrayFromImage(pct_img).shape),
               "spacing_xyz": list(pct_img.GetSpacing()),
               "direction": list(pct_img.GetDirection()),
               "origin": list(pct_img.GetOrigin())}

    # 4) Stage-2 inference on bbox-cropped
    t2_dir=TMP/"stage2"; t2_dir.mkdir(exist_ok=True, parents=True)
    stage2_input = t2_dir/"img.mha"
    _sitk_write(stage2_img, stage2_input)
    t2_json=t2_dir/"test.json"; _write_testing_json(stage2_input, t2_json, case_id)
    preds2=_infer_folds(RES2, t2_json, t2_dir, case_id, forbid_exact=stage2_input)
    staple2=_staple(preds2)

    # 5) Uncrop step-1: bbox -> percent-cropped canvas
    p_arr=np.zeros(bbox_meta["orig_size_zyx"], dtype=np.uint8)
    s2_arr=sitk.GetArrayFromImage(staple2)
    z0,z1,y0,y1,x0,x1=bbox_meta["bbox_zyx"]
    dz,dy,dx=s2_arr.shape
    p_arr[z0:z0+dz, y0:y0+dy, x0:x0+dx]=s2_arr
    step1=sitk.GetImageFromArray(p_arr)
    step1.SetSpacing(tuple(bbox_meta["spacing_xyz"]))
    step1.SetDirection(tuple(bbox_meta["direction"]))
    step1.SetOrigin(tuple(bbox_meta["origin"]))

    # 6) Uncrop step-2: percent-cropped canvas -> original canvas
    final=_paste_percent(step1, orig)

    # 7) Save outputs — GC expects a binary tumor mask (0/1)
    arr_final = sitk.GetArrayFromImage(final)
    tumor_bin = (arr_final == TUMOR_LABEL).astype(np.uint8)
    final_bin = sitk.GetImageFromArray(tumor_bin)
    final_bin.CopyInformation(final)

    OUTPUT_SLUG = os.environ.get("OUTPUT_IMAGE_SLUG", "pancreatic-tumor-segmentation")
    out_dir = Path("/output") / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    uuid = _uuid_from_path(in_img_path)  # must mirror input UUID
    out_mask = out_dir / f"{uuid}.mha"
    _sitk_write(final_bin, out_mask)

    bxs=_bounds_json(arr_final)
    (Path("/output")/"bounds.json").write_text(json.dumps(bxs, indent=2))

    print(f"[OK] Wrote {out_mask} and /output/bounds.json")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
