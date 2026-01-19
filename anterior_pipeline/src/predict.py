import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Device configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Paths (ROBUST)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "anterior_cataract_model_finetuned.pth")
)

# -----------------------------
# Classes
# -----------------------------
CLASSES = ["Cataract", "Normal"]

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load model (once)
# -----------------------------
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

MODEL = load_model()

# -----------------------------
# Prediction function (USED BY WEB + CLI)
# -----------------------------
def predict_image(image_path: str):
    """
    Predict cataract from an anterior eye image.

    Returns:
        label (str): "Cataract" or "Normal"
        confidence (float): percentage confidence
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label = CLASSES[pred.item()]
    confidence = conf.item() * 100

    return label, round(confidence, 2)

# -----------------------------
# CLI usage
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    label, confidence = predict_image(img_path)

    print(f"Cataract Probability: {confidence if label=='Cataract' else 100-confidence:.2f}%")
    print(f"Normal Probability: {confidence if label=='Normal' else 100-confidence:.2f}%")
    print(f"Status: {label.upper()}")
