import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import sys
import os

# -----------------------
# Config
# -----------------------
MODEL_PATH = "anterior_cataract_model.pth"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# Load Model (NEW API)
# -----------------------
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

class_names = ["cataract", "normal"]

# -----------------------
# Predict Function
# -----------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    cataract_prob = probs[0].item() * 100
    normal_prob = probs[1].item() * 100
    pred_class = class_names[probs.argmax().item()]

    print(f"\nCataract Probability: {cataract_prob:.2f}%")
    print(f"Normal Probability: {normal_prob:.2f}%")
    print(f"Status: {pred_class.upper()}")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
