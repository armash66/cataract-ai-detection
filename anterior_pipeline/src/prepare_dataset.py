import os
import shutil
import kagglehub

# Download dataset
dataset_root = kagglehub.dataset_download(
    "nandanp6/cataract-image-dataset"
)
print("Dataset downloaded to:", dataset_root)

# Actual source directory
SOURCE_BASE = os.path.join(dataset_root, "processed_images")

# Project root (one level above src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
TARGET_BASE = os.path.join(PROJECT_ROOT, "dataset")

CLASSES = ["cataract", "normal"]
SPLITS = ["train", "test"]

for cls in CLASSES:
    os.makedirs(os.path.join(TARGET_BASE, cls), exist_ok=True)

total_count = {cls: 0 for cls in CLASSES}

for split in SPLITS:
    for cls in CLASSES:
        src_dir = os.path.join(SOURCE_BASE, split, cls)
        dst_dir = os.path.join(TARGET_BASE, cls)

        if not os.path.exists(src_dir):
            print(f"⚠️ Missing: {src_dir}")
            continue

        for img in os.listdir(src_dir):
            src_file = os.path.join(src_dir, img)
            dst_file = os.path.join(dst_dir, f"{split}_{img}")

            if os.path.isfile(src_file):
                shutil.copy(src_file, dst_file)
                total_count[cls] += 1

for cls, count in total_count.items():
    print(f"✅ {cls}: {count} images")

print("\n🎉 Anterior eye dataset prepared successfully!")
