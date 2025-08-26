import os
import csv
import SimpleITK as sitk
import numpy as np
from glob import glob

# Set paths
BASE_PATH = "/home/keshav/PANTHER_Task2_Auto3DSeg"
image_dir = os.path.join(BASE_PATH, "data", "imagesTr")
label_dir = os.path.join(BASE_PATH, "data", "labelsTr")
output_csv = os.path.join(BASE_PATH, "scripts", "bounding_box_stats.csv")

# Header for CSV
header = [
    "Case", "Image max X", "Image max Y", "Image max Z",
    "Tumor min X", "Tumor min Y", "Tumor min Z",
    "Tumor max X", "Tumor max Y", "Tumor max Z",
    "Pancreas min X", "Pancreas min Y", "Pancreas min Z",
    "Pancreas max X", "Pancreas max Y", "Pancreas max Z",
    "Tumor min X %", "Tumor min Y %", "Tumor min Z %",
    "Tumor max X %", "Tumor max Y %", "Tumor max Z %",
    "Pancreas min X %", "Pancreas min Y %", "Pancreas min Z %",
    "Pancreas max X %", "Pancreas max Y %", "Pancreas max Z %"
]

# Helper function to get bounding box of a label
def get_bounding_box(arr, label_val):
    coords = np.argwhere(arr == label_val)
    if coords.size == 0:
        return [None] * 6
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return list(mins) + list(maxs)

# Begin analysis
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for label_path in sorted(glob(os.path.join(label_dir, "*.mha"))):
        case = os.path.basename(label_path).replace(".mha", "")
        image_path = os.path.join(image_dir, f"{case}_0000.mha")

        # Load image and label
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        image_arr = sitk.GetArrayFromImage(image)  # z, y, x
        label_arr = sitk.GetArrayFromImage(label)

        img_shape = image_arr.shape[::-1]  # x, y, z

        # Tumor = 1, Pancreas = 2
        tumor_box = get_bounding_box(label_arr, 1)
        pancreas_box = get_bounding_box(label_arr, 2)

        if None in tumor_box or None in pancreas_box:
            print(f"Skipping {case} due to missing labels.")
            continue

        # Convert boxes to (x, y, z) ordering
        tumor_box = [tumor_box[2], tumor_box[1], tumor_box[0], tumor_box[5], tumor_box[4], tumor_box[3]]
        pancreas_box = [pancreas_box[2], pancreas_box[1], pancreas_box[0], pancreas_box[5], pancreas_box[4], pancreas_box[3]]

        tumor_min_pct = [tumor_box[i] / img_shape[i] for i in range(3)]
        tumor_max_pct = [tumor_box[i+3] / img_shape[i] for i in range(3)]
        pancreas_min_pct = [pancreas_box[i] / img_shape[i] for i in range(3)]
        pancreas_max_pct = [pancreas_box[i+3] / img_shape[i] for i in range(3)]

        writer.writerow([
            case, *img_shape,
            *tumor_box[:3], *tumor_box[3:],
            *pancreas_box[:3], *pancreas_box[3:],
            *tumor_min_pct, *tumor_max_pct,
            *pancreas_min_pct, *pancreas_max_pct
        ])
