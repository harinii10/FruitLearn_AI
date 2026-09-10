import os
import json
import random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.preprocessing import get_transforms, DINOv2Augmentation

CLASSES = [
    "Apple_Good", "Apple_Bad",
    "Banana_Good", "Banana_Bad",
    "Guava_Good", "Guava_Bad",
    "Lime_Good", "Lime_Bad",
    "Orange_Good", "Orange_Bad",
    "Pomegranate_Good", "Pomegranate_Bad"
]

FRUIT_COLORS = {
    "Apple": (220, 20, 60),
    "Banana": (255, 225, 50),
    "Guava": (144, 238, 144),
    "Lime": (50, 205, 50),
    "Orange": (255, 140, 0),
    "Pomegranate": (178, 34, 34)
}

def generate_synthetic_image(class_name, image_size=256):
    """Generates a synthetic fruit image with quality characteristics."""
    fruit_type, quality = class_name.split("_")
    base_color = FRUIT_COLORS.get(fruit_type, (200, 200, 200))
    
    img = Image.new("RGB", (image_size, image_size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw background pattern/texture
    for _ in range(20):
        x = random.randint(0, image_size)
        y = random.randint(0, image_size)
        r = random.randint(2, 6)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(220, 220, 220))

    # Draw main fruit shape
    cx, cy = image_size // 2, image_size // 2
    r = random.randint(70, 90)
    
    if fruit_type == "Banana":
        # Draw curved arc shape for banana
        draw.chord([cx-r, cy-r//2, cx+r, cy+r//2], start=0, end=180, fill=base_color, outline=(180, 150, 20))
    else:
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=base_color, outline=(50, 50, 50))
    
    # Add defect spots for Bad quality
    if quality == "Bad":
        for _ in range(random.randint(5, 12)):
            bx = random.randint(cx-r+15, cx+r-15)
            by = random.randint(cy-r+15, cy+r-15)
            br = random.randint(4, 12)
            draw.ellipse([bx-br, by-br, bx+br, by+br], fill=(80, 40, 20))
            
    return img

def create_synthetic_fruitsgb_dataset(output_dir="data/raw", samples_per_class=40):
    """Creates a local synthetic FruitsGB dataset structure for reproducible offline runs."""
    os.makedirs(output_dir, exist_ok=True)
    total_images = 0
    for cls_name in CLASSES:
        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        # Check existing count
        existing = len([f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png'))])
        if existing < samples_per_class:
            for i in range(existing, samples_per_class):
                img = generate_synthetic_image(cls_name)
                img.save(os.path.join(cls_dir, f"{cls_name}_{i+1:04d}.jpg"))
        total_images += len(os.listdir(cls_dir))
    print(f"[Dataset Setup] FruitsGB dataset verified at '{output_dir}' with {total_images} total images across 12 classes.")

def create_stratified_split(data_dir="data/raw", splits_dir="data/splits", seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Creates and saves stratified train/val/test splits to disk to prevent data leakage."""
    os.makedirs(splits_dir, exist_ok=True)
    split_file = os.path.join(splits_dir, "split_info.json")

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            return json.load(f)

    filepaths = []
    labels = []
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
        filepaths, labels, test_size=val_test_ratio, random_state=seed, stratify=labels
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_test_paths, val_test_labels, test_size=relative_test_ratio, random_state=seed, stratify=val_test_labels
    )

    split_info = {
        "classes": CLASSES,
        "class_to_idx": class_to_idx,
        "train": [{"path": p, "label": l} for p, l in zip(train_paths, train_labels)],
        "val": [{"path": p, "label": l} for p, l in zip(val_paths, val_labels)],
        "test": [{"path": p, "label": l} for p, l in zip(test_paths, test_labels)]
    }

    with open(split_file, "w") as f:
        json.dump(split_info, f, indent=4)

    print(f"[Dataset Split] Created stratified splits: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    return split_info

class FruitsGBDataset(Dataset):
    """PyTorch Dataset for FruitsGB fruit quality classification."""
    def __init__(self, samples, transform=None, ssl_mode=False, dinov2_mode=False):
        self.samples = samples
        self.transform = transform
        self.ssl_mode = ssl_mode
        self.dinov2_mode = dinov2_mode
        if dinov2_mode:
            self.dinov2_aug = DINOv2Augmentation()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = item["path"]
        label = item["label"]

        image = Image.open(image_path).convert("RGB")

        if self.dinov2_mode and self.ssl_mode:
            view1, view2 = self.dinov2_aug(image)
            return view1, view2

        if self.transform:
            image = self.transform(image)

        if self.ssl_mode:
            return image  # Stage 1: Returns image only, quality labels are NOT used
        else:
            return image, label  # Stage 2: Returns image and label
