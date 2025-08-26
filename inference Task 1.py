# /opt/algorithm/inference/gc_entrypoint.py
import os, sys, json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from typing import List, Tuple

SUPPORTED_EXTS = (".mha", ".nii.gz", ".nii", ".nrrd", ".tif", ".tiff")

# Your training window + margin (keep in sync with training)
Z_MIN_PCT, Z_MAX_PCT = 0.10, 0.90
Y_MIN_PCT, Y_MAX_PCT = 0.10, 0.90
X_MIN_PCT, X_MAX_PCT = 0.10, 0.90
MARGIN_MM = 30.0

ROOT = Path("/opt/algorithm")
RES1 = ROOT / "results"          # segresnet_0..4
RES2 = ROOT / "results_stage2"   # segresnet_0..4
TMP  = ROOT / "work"
TMP.mkdir(parents=True, exist_ok=True)

# ---------------- CPU/GPU fallback ----------------
# If you want to force CPU even when a GPU is present, run with: -e FORCE_CPU=1
if os.environ.get("FORCE_CPU", "0") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
        # torch not importable here (some segmenters import it themselves) — assume CPU
        print(f"[INFO] torch not available in entrypoint ({e}) -> using CPU.")
# --------------------------------------------------

def _uuid_from_path(p: Path) -> str:
    name = p.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return p.stem

def _find_input() -> Path:
    # Prefer the GC interface path
    preferred = Path("/input/images/abdominal-t1-mri")
    if preferred.exists():
        for p in preferred.rglob("*"):
            if p.is_file() and p.name.lower().endswith((".mha", ".tif", ".tiff")):
                return p
    # Fallback: any supported file under /input
    ip = Path("/input")
    for p in ip.rglob("*"):
        if p.is_file() and p.name.lower().endswith(SUPPORTED_EXTS):
            return p
    raise FileNotFoundError("No 3D medical image found under /input")

def _sitk_read(p: Path) -> sitk.Image: return sitk.ReadImage(str(p))
def _sitk_write(img: sitk.Image, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(p))

def _percent_bounds(ref: sitk.Image):
    D,H,W = sitk.GetArrayFromImage(ref).shape
    z0,z1 = int(D*Z_MIN_PCT), int(D*Z_MAX_PCT)
    y0,y1 = int(H*Y_MIN_PCT), int(H*Y_MAX_PCT)
    x0,x1 = int(W*X_MIN_PCT), int(W*X_MAX_PCT)
    return (z0,z1,y0,y1,x0,x1),(D,H,W)

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
    sx,sy,sz = spacing_xyz
    mz,my,mx = int(round(margin_mm/sz)), int(round(margin_mm/sy)), int(round(margin_mm/sx))
    z0,z1,y0,y1,x0,x1 = bbox
    z0=max(0,z0-mz); y0=max(0,y0-my); x0=max(0,x0-mx)
    z1=min(shape_zyx[0], z1+mz); y1=min(shape_zyx[1], y1+my); x1=min(shape_zyx[2], x1+mx)
    return z0,z1,y0,y1,x0,x1

def _crop_bbox(img: sitk.Image, bbox):
    z0,z1,y0,y1,x0,x1=bbox
    arr=sitk.GetArrayFromImage(img)[z0:z1, y0:y1, x0:x1]
    out=sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing()); out.SetDirection(img.GetDirection())
    out.SetOrigin(img.TransformIndexToPhysicalPoint((int(x0),int(y0),int(z0))))
    return out

def _write_testing_json(img_path: Path, out_json: Path, case_id: str):
    data={"testing":[{"image": str(img_path), "case_id": case_id}]}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2))

def _load_segmenter(run_py: Path):
    import importlib.util, sys
    sys.path.insert(0, str(run_py.parent))
    spec = importlib.util.spec_from_file_location("segmenter", str(run_py))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.run_segmenter

def _infer_folds(root: Path, test_json: Path, tmp_root: Path, case_id: str) -> List[Path]:
    def _collect_any_pred(search_root: Path) -> Tuple[Path | None, list[Path]]:
        cand=[]
        for r,_,files in os.walk(search_root):
            for f in files:
                fl=f.lower()
                if any(fl.endswith(ext) for ext in SUPPORTED_EXTS):
                    cand.append(Path(r)/f)
        if not cand:
            return None, []
        # prefer case-id match; otherwise largest file
        for p in cand:
            if case_id in p.name:
                return p, cand
        return max(cand, key=lambda p: p.stat().st_size), cand

    preds=[]
    for k in range(5):
        cfg = root/f"segresnet_{k}"/"configs"/"hyper_parameters.yaml"
        seg = root/f"segresnet_{k}"/"scripts"/"segmenter.py"
        ckpt= root/f"segresnet_{k}"/"model"/"model.pt"
        if not (cfg.exists() and seg.exists() and ckpt.exists()):
            print(f"[WARN] Fold {k}: missing files -> cfg:{cfg.exists()} seg:{seg.exists()} ckpt:{ckpt.exists()} (skipping)")
            continue

        run=_load_segmenter(seg)
        out_dir = tmp_root/f"fold_{k}"; out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir/"prediction_testing"

        override={
            "infer#enabled": True,
            "infer#output_path": str(out_path),
            "infer#save_mask": True,
            "infer#data_list_key": "testing",
            "ckpt_name": str(ckpt),
            "data_list_file_path": str(test_json),
        }

        print(f"[INFO] Fold {k}: cfg={cfg}")
        print(f"[INFO] Fold {k}: ckpt={ckpt}")
        print(f"[INFO] Fold {k}: out_path={out_path}")

        try:
            run(config_file=str(cfg), **override)
        except TypeError:
            run(str(cfg), **override)

        hit, cand = _collect_any_pred(out_dir)
        if hit is not None:
            print(f"[INFO] Fold {k}: found prediction -> {hit}")
            preds.append(hit)
        else:
            print(f"[WARN] Fold {k}: no predictions under {out_dir}")
            # mini listing for debugging
            for r,_,files in os.walk(out_dir):
                keep=[f for f in files if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTS)]
                if keep:
                    print(f"[DBG] {r}: {keep}")
    return preds


def _staple(preds: List[Path]) -> sitk.Image:
    if not preds: raise RuntimeError("No predictions to ensemble.")
    ref=sitk.ReadImage(str(preds[0]))
    final=np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
    for label in (1,2):  # tumor, pancreas
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
    for label,name in [(1,"tumor"),(2,"pancreas")]:
        arr=(mask_final_np==label).astype(np.uint8)
        bbox=_bbox_from(arr)
        bxs[name]=bbox if bbox is not None else None
    return bxs

def main():
    # 0) Locate input
    in_img_path=_find_input()
    case_id = in_img_path.stem.replace(".nii","").replace(".mha","")
    print(f"[INFO] Case: {case_id}")

    _log_device()  # <-- logs GPU/CPU mode and respects FORCE_CPU

    orig=_sitk_read(in_img_path)

    # 1) Percent crop
    pct_img, pct_meta = _crop_percent(orig)

    # 2) Stage-1 inference on percent-cropped
    t1_dir=TMP/"stage1"; t1_dir.mkdir(exist_ok=True, parents=True)
    t1_json=t1_dir/"test.json"; _write_testing_json(TMP/"data_cropped_img.mha", t1_json, case_id)
    _sitk_write(pct_img, TMP/"data_cropped_img.mha")
    preds1=_infer_folds(RES1, t1_json, t1_dir, case_id)
    staple1=_staple(preds1)  # on percent-cropped grid

    # 3) Pancreas-only bbox (+30mm), fallback to union
    arr1=np.rint(sitk.GetArrayFromImage(staple1)).astype(np.int16)
    pancreas=(arr1==2).astype(np.uint8)
    bbox=_bbox_from(pancreas)
    if bbox is None:
        union=(arr1>0).astype(np.uint8)
        bbox=_bbox_from(union)
        if bbox is None:
            raise RuntimeError("Empty Stage-1 mask; cannot crop Stage-2.")
    bbox=_expand_bbox(bbox, pct_img.GetSpacing(), sitk.GetArrayFromImage(pct_img).shape, MARGIN_MM)
    stage2_img=_crop_bbox(pct_img, bbox)
    bbox_meta={"bbox_zyx":[int(v) for v in bbox],
               "orig_size_zyx": list(sitk.GetArrayFromImage(pct_img).shape),
               "spacing_xyz": list(pct_img.GetSpacing()),
               "direction": list(pct_img.GetDirection()),
               "origin": list(pct_img.GetOrigin())}

    # 4) Stage-2 inference on bbox-cropped
    t2_dir=TMP/"stage2"; t2_dir.mkdir(exist_ok=True, parents=True)
    _sitk_write(stage2_img, t2_dir/"img.mha")
    t2_json=t2_dir/"test.json"; _write_testing_json(t2_dir/"img.mha", t2_json, case_id)
    preds2=_infer_folds(RES2, t2_json, t2_dir, case_id)
    staple2=_staple(preds2)  # on bbox-cropped grid

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

    # 7) Save outputs — match GC interface path
    # GC expects a *binary tumor* mask (0 background, 1 tumor)
    arr_final = sitk.GetArrayFromImage(final)
    tumor_bin = (arr_final == 1).astype(np.uint8)  # keep only label 1 = tumor
    final_bin = sitk.GetImageFromArray(tumor_bin)
    final_bin.CopyInformation(final)  # spacing/direction/origin

    # Output folder: /output/images/pancreatic-tumor-segmentation/<uuid>.mha
    OUTPUT_SLUG = os.environ.get("OUTPUT_IMAGE_SLUG", "pancreatic-tumor-segmentation")
    out_dir = Path("/output") / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    uuid = _uuid_from_path(in_img_path)
    out_mask = out_dir / f"{uuid}.mha"
    _sitk_write(final_bin, out_mask)

    # Optional artifact (ignored by GC unless defined in the interface)
    (Path("/output")/"bounds.json").write_text(
        json.dumps(_bounds_json(arr_final), indent=2)
    )
    print(f"[OK] Wrote {out_mask} and /output/bounds.json")


    # BBox JSON in original grid
    fin_np = sitk.GetArrayFromImage(final)
    bxs=_bounds_json(fin_np)
    (Path("/output")/"bounds.json").write_text(json.dumps(bxs, indent=2))
    print(f"[OK] Wrote {out_mask} and /output/bounds.json")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
