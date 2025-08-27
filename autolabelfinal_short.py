import os, glob, shutil, random, yaml, torch
import torchvision.transforms as T
from PIL import Image

# ------------------- Config -------------------
DATASET_DIR = "dataset_limited"
OUTPUT_DIR = "dataset_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model + transform
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').eval()
transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Classes
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
class_to_idx = {cls: i for i, cls in enumerate(classes)}
print("✅ Classes:", class_to_idx)

# ------------------- Collect + split -------------------
all_images = [(p, class_to_idx[cls]) 
              for cls in classes 
              for p in glob.glob(os.path.join(DATASET_DIR, cls, "*.*"))]
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
splits = {"train": all_images[:split_idx], "val": all_images[split_idx:]}

# ------------------- Processing -------------------
def process_and_save(images, split):
    for img_path, cls_id in images:
        try:
            img = Image.open(img_path).convert("RGB"); w, h = img.size
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                attn = model.get_last_selfattention(tensor)[0,:,0,1:].mean(0).reshape(14,14).cpu()
            mask = T.Resize((h,w))(attn[None,None])[0,0] > attn.mean()
            coords = torch.nonzero(mask)
            if not len(coords): continue

            y_min, x_min = coords.min(0).values; y_max, x_max = coords.max(0).values
            x_c, y_c = ((x_min+x_max)/2)/w, ((y_min+y_max)/2)/h
            bw, bh = (x_max-x_min)/w, (y_max-y_min)/h

            out_img = os.path.join(OUTPUT_DIR, split, "images", os.path.basename(img_path))
            out_label = os.path.join(OUTPUT_DIR, split, "labels", os.path.splitext(os.path.basename(img_path))[0]+".txt")
            os.makedirs(os.path.dirname(out_img), exist_ok=True)
            os.makedirs(os.path.dirname(out_label), exist_ok=True)
            shutil.copy(img_path, out_img)
            with open(out_label, "w") as f: f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

        except Exception as e: print(f"⚠️ {img_path}: {e}")

for split, imgs in splits.items(): process_and_save(imgs, split)

# ------------------- data.yaml -------------------
yaml.dump({
    "train": os.path.join(OUTPUT_DIR, "train/images"),
    "val": os.path.join(OUTPUT_DIR, "val/images"),
    "nc": len(classes), "names": classes
}, open(os.path.join(OUTPUT_DIR, "data.yaml"), "w"))

print("✅ Dataset prepared & data.yaml created!")
