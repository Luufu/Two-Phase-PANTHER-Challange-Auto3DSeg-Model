# import os
# import json
# import random
# from sklearn.model_selection import KFold

# # === Configuration ===
# data_root = "/home/keshav/PANTHER_Task2_Auto3DSeg/data_cropped"
# images_dir = os.path.join(data_root, "imagesTr")
# labels_dir = os.path.join(data_root, "labelsTr")
# output_json_path = os.path.join(data_root, "dataset.json")
# num_folds = 5
# seed = 42

# # === Get image-label pairs ===
# image_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".mha"))
# label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".mha"))

# image_label_pairs = []
# for img_file in image_files:
#     case_id = img_file.replace("_0000.mha", "")
#     label_file = f"{case_id}.mha"
#     if label_file in label_files:
#         image_label_pairs.append({
#             "image": [f"imagesTr/{img_file}"],  # MONAI expects list
#             "label": f"labelsTr/{label_file}"
#         })

# # === Shuffle and assign folds ===
# random.seed(seed)
# random.shuffle(image_label_pairs)

# kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
# final_entries = []

# for fold_idx, (_, val_idx) in enumerate(kf.split(image_label_pairs)):
#     for i in val_idx:
#         entry = image_label_pairs[i].copy()
#         entry["fold"] = fold_idx
#         final_entries.append(entry)

# # === Final dataset structure ===
# dataset = {"training": final_entries}

# # === Save dataset.json ===
# with open(output_json_path, "w") as f:
#     json.dump(dataset, f, indent=4)

# print(f"✅ Saved dataset.json to: {output_json_path}")



import os
import json
import random
from sklearn.model_selection import KFold

# === Configuration ===
data_root = "/home/keshav/PANTHER_Task2_Auto3DSeg/data_stage2_predcrop"
images_dir = os.path.join(data_root, "imagesTr")
labels_dir = os.path.join(data_root, "labelsTr")
output_json_path = os.path.join(data_root, "dataset.json")
inference_json_path = os.path.join(data_root, "inference_cases.json")
num_folds = 5
seed = 42
heldout_fraction = 0.1  # 10% held out for testing

# === Get image-label pairs ===
image_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".mha"))
label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".mha"))

image_label_pairs = []
for img_file in image_files:
    case_id = img_file.replace("_0000.mha", "")
    label_file = f"{case_id}.mha"
    if label_file in label_files:
        image_label_pairs.append({
            "image": [f"imagesTr/{img_file}"],
            "label": f"labelsTr/{label_file}",
            "case_id": case_id
        })

# === Shuffle and split
random.seed(seed)
random.shuffle(image_label_pairs)

num_total = len(image_label_pairs)
num_heldout = int(heldout_fraction * num_total)
heldout_samples = image_label_pairs[:num_heldout]
training_samples = image_label_pairs[num_heldout:]

# === K-Fold splitting on training samples
kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
final_entries = []
for fold_idx, (_, val_idx) in enumerate(kf.split(training_samples)):
    for i in val_idx:
        entry = training_samples[i].copy()
        entry["fold"] = fold_idx
        final_entries.append(entry)

# === Save dataset.json
with open(output_json_path, "w") as f:
    json.dump({"training": final_entries}, f, indent=4)

# === Save held-out inference list
with open(inference_json_path, "w") as f:
    json.dump({"inference": heldout_samples}, f, indent=4)

print(f"✅ Saved training dataset to: {output_json_path}")
print(f"✅ Saved held-out inference cases to: {inference_json_path}")

