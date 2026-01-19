import os
import pandas as pd
import shutil
import kagglehub

# ==============================
# 1. Download dataset via Kaggle API
# ==============================
path = kagglehub.dataset_download(
    "andrewmvd/ocular-disease-recognition-odir5k"
)

print("Dataset downloaded at:", path)

# ==============================
# 2. Correct paths (based on version 2)
# ==============================
csv_path = os.path.join(path, "full_df.csv")
img_dir = os.path.join(path, "preprocessed_images")

# ==============================
# 3. Read labels CSV
# ==============================
df = pd.read_csv(csv_path)

# ==============================
# 4. Keep only Normal (N) or Cataract (C)
# ==============================
binary_df = df[(df["N"] == 1) | (df["C"] == 1)]

print("Total images selected:", len(binary_df))

# ==============================
# 5. Create output folders
# ==============================
os.makedirs("dataset/normal", exist_ok=True)
os.makedirs("dataset/cataract", exist_ok=True)

# ==============================
# 6. Copy images into folders
# ==============================
copied = 0

for _, row in binary_df.iterrows():
    label = "normal" if row["N"] == 1 else "cataract"

    image_name = row["filename"]

    if isinstance(image_name, str):
        src = os.path.join(img_dir, image_name)
        dst = os.path.join("dataset", label, image_name)

        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1

print(f"✅ Dataset prepared successfully! Images copied: {copied}")
