import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = "cataract_model.pth"
IMAGE_PATH = "test_image.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATASET JUST TO GET CLASS MAPPING
# =========================
dataset = datasets.ImageFolder("dataset")
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

print("Using class mapping:", idx_to_class)

# =========================
# TRANSFORM (MUST MATCH TRAINING)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD MODEL
# =========================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# LOAD IMAGE
# =========================
img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

# =========================
# INFERENCE
# =========================
with torch.no_grad():
    logits = model(img)
    probs = torch.softmax(logits, dim=1)[0]

pred_idx = torch.argmax(probs).item()
pred_class = idx_to_class[pred_idx]

# =========================
# PRINT RAW TRUTH
# =========================
print("\nRaw probabilities:")
for i, p in enumerate(probs):
    print(f"{idx_to_class[i]}: {p.item()*100:.2f}%")

# =========================
# EARLY DETECTION LOGIC
# =========================
cataract_prob = probs[[k for k,v in idx_to_class.items() if v=="cataract"][0]].item()*100

if cataract_prob >= 85:
    status = "Cataract Detected (High Confidence)"
elif cataract_prob >= 50:
    status = "⚠️ Possible Early Cataract (Screening Recommended)"
else:
    status = "Normal"

print("\n--- FINAL DECISION ---")
print("Predicted class:", pred_class)
print("Status:", status)
