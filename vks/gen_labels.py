import os
import shutil
from sklearn.model_selection import train_test_split

# input dataset path
DATASET_PATH = "archive"

# output dataset path
OUTPUT_PATH = "dataset"
IMG_TRAIN = os.path.join(OUTPUT_PATH, "images/train")
IMG_VAL = os.path.join(OUTPUT_PATH, "images/val")
LBL_TRAIN = os.path.join(OUTPUT_PATH, "labels/train")
LBL_VAL = os.path.join(OUTPUT_PATH, "labels/val")

for p in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(p, exist_ok=True)

# classes (sorted for consistency)
classes = sorted(os.listdir(DATASET_PATH))
class_map = {c: i for i, c in enumerate(classes)}

print("Classes:", class_map)

all_images, all_labels = [], []

for cls in classes:
    cls_id = class_map[cls]
    folder = os.path.join(DATASET_PATH, cls)
    
    for img_file in os.listdir(folder):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(folder, img_file)
        all_images.append(img_path)
        all_labels.append(cls_id)

# split train/val
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)

def save_yolo(img_list, label_list, img_out, lbl_out):
    for img_path, cls_id in zip(img_list, label_list):
        img_name = os.path.basename(img_path)
        out_img = os.path.join(img_out, img_name)
        shutil.copy(img_path, out_img)

        # label file
        label_file = os.path.splitext(img_name)[0] + ".txt"
        with open(os.path.join(lbl_out, label_file), "w") as f:
            # whole image bbox (x_center=0.5, y_center=0.5, w=1, h=1)
            f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

save_yolo(train_imgs, train_labels, IMG_TRAIN, LBL_TRAIN)
save_yolo(val_imgs, val_labels, IMG_VAL, LBL_VAL)

print("Dataset prepared successfully ✅")
