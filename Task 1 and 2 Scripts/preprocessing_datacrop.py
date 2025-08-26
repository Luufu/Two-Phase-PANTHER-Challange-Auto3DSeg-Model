#!/usr/bin/env python3
import os
import re
import pandas as pd
import SimpleITK as sitk

# ===== Paths (Task 2) =====
csv_path       = "/home/keshav/PANTHER_Task2_Auto3DSeg/scripts/bounding_box_stats.csv"
base_data_path = "/home/keshav/PANTHER_Task2_Auto3DSeg/data"
output_path    = "/home/keshav/PANTHER_Task2_Auto3DSeg/data_cropped"

os.makedirs(os.path.join(output_path, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(output_path, "labelsTr"), exist_ok=True)

# ===== Helpers =====
def normalize_case_id(x: object) -> str:
    """
    10379.0 -> '10379'
    '10379_0000' -> '10379'
    '/path/10379_0000.mha' -> '10379'
    """
    s = str(x).strip().split("/")[-1]
    s = s.replace(".mha", "")
    s = s.split("_")[0]
    if re.fullmatch(r"\d+(?:\.0+)?", s):
        s = str(int(float(s)))
    return s

def clamp_idx(lo_pct: float, hi_pct: float, size: int) -> tuple[int, int]:
    """Convert % bounds into clamped, non-empty integer slice indices."""
    a = max(0, min(int(round(size * lo_pct)), size - 1))
    b = max(a + 1, min(int(round(size * hi_pct)), size))
    return a, b

def maybe_cast_for_itk(img: sitk.Image, is_label: bool) -> sitk.Image:
    """
    Preserve original pixel type unless it's Int8 (unsupported by ITK Python).
    If Int8: cast labels -> UInt8, images -> Float32.
    """
    if img.GetPixelID() == sitk.sitkInt8:  # aka signed char / SC
        return sitk.Cast(img, sitk.sitkUInt8 if is_label else sitk.sitkFloat32)
    return img

# ===== Read CSV (keep Case as string) =====
df = pd.read_csv(csv_path, dtype={"Case": str})
print("CSV columns:", df.columns.tolist())

# ===== Crop bounds (consistent symmetric margin derived from your CSV) =====
x_min_pct, x_max_pct = 0.328, 0.790
y_min_pct, y_max_pct = 0.323, 0.705
z_min_pct, z_max_pct = 0.148, 1.000

for _, row in df.iterrows():
    # Task 1 style: normalise then zero-pad to 5
    stem_core = normalize_case_id(row["Case"])
    case_id   = stem_core.zfill(5)

    # Primary Task-1-like paths (lowercase dirs, zero-padded stems)
    image_path = os.path.join(base_data_path, "imagesTr", f"{case_id}_0000.mha")
    label_path = os.path.join(base_data_path, "labelsTr", f"{case_id}.mha")

    # Fallback to unpadded stems if that's what exists on disk
    if not (os.path.exists(image_path) and os.path.exists(label_path)) and stem_core != case_id:
        alt_img = os.path.join(base_data_path, "imagesTr", f"{stem_core}_0000.mha")
        alt_lab = os.path.join(base_data_path, "labelsTr", f"{stem_core}.mha")
        if os.path.exists(alt_img) and os.path.exists(alt_lab):
            image_path, label_path, case_id = alt_img, alt_lab, stem_core

    print(f"\n🧪 Processing case {case_id}")
    print(f"📂 Looking for image: {image_path}")
    print(f"📂 Looking for label: {label_path}")

    if not (os.path.exists(image_path) and os.path.exists(label_path)):
        print(f"❌ File missing for case {case_id}")
        continue

    # --- Read (SITK returns arrays as [Z, Y, X]) ---
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)
    image_np = sitk.GetArrayFromImage(image)
    label_np = sitk.GetArrayFromImage(label)

    D, H, W = image_np.shape
    print(f"📐 Image shape: {image_np.shape}")

    # --- Crop indices (clamped) ---
    x_min, x_max = clamp_idx(x_min_pct, x_max_pct, W)
    y_min, y_max = clamp_idx(y_min_pct, y_max_pct, H)
    z_min, z_max = clamp_idx(z_min_pct, z_max_pct, D)

    print(f"🔪 Cropping ranges: X({x_min}:{x_max}), Y({y_min}:{y_max}), Z({z_min}:{z_max})")

    cropped_image = image_np[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_label = label_np[z_min:z_max, y_min:y_max, x_min:x_max]

    if cropped_image.size == 0 or cropped_label.size == 0:
        print(f"⚠️ Empty crop — skipping case {case_id}")
        continue

    print(f"✅ Cropped shape: {cropped_image.shape}")

    # --- Back to ITK & preserve geometry ---
    cropped_image_itk = sitk.GetImageFromArray(cropped_image)
    cropped_label_itk = sitk.GetImageFromArray(cropped_label)

    # Copy spacing/direction
    cropped_image_itk.SetSpacing(image.GetSpacing())
    cropped_image_itk.SetDirection(image.GetDirection())
    cropped_label_itk.SetSpacing(label.GetSpacing())
    cropped_label_itk.SetDirection(label.GetDirection())

    # Update origin (SITK origin order is x,y,z)
    origin  = image.GetOrigin()
    spacing = image.GetSpacing()
    new_origin = [
        origin[0] + x_min * spacing[0],
        origin[1] + y_min * spacing[1],
        origin[2] + z_min * spacing[2],
    ]
    cropped_image_itk.SetOrigin(new_origin)
    cropped_label_itk.SetOrigin(new_origin)

    # --- Keep original dtypes (only auto-fix Int8) ---
    cropped_image_itk = maybe_cast_for_itk(cropped_image_itk, is_label=False)
    cropped_label_itk = maybe_cast_for_itk(cropped_label_itk, is_label=True)

    # --- Save (Task 1 naming) ---
    image_out = os.path.join(output_path, "imagesTr", f"{case_id}_0000.mha")
    label_out = os.path.join(output_path, "labelsTr", f"{case_id}.mha")

    print(f"💾 Saving to:\n  {image_out}\n  {label_out}")
    sitk.WriteImage(cropped_image_itk, image_out)
    sitk.WriteImage(cropped_label_itk, label_out)
    print("✅ Saved.")
