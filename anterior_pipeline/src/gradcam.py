import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# -----------------------------
# Device
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
# Transform
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
# Load model
# -----------------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# Hook storage
# -----------------------------
gradients = None
activations = None


def save_gradient(grad):
    global gradients
    gradients = grad


def forward_hook(module, input, output):
    global activations
    activations = output
    output.register_hook(save_gradient)


# Register hook on last conv layer
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)

# -----------------------------
# Generate Grad-CAM
# -----------------------------
def generate_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Forward
    output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()

    # Backward
    model.zero_grad()
    output[0, class_idx].backward()

    # Compute CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    cam = activations[0]

    for i in range(cam.shape[0]):
        cam[i] *= pooled_gradients[i]

    cam = torch.mean(cam, dim=0).detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (224, 224))

    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )

    # Ensure same dtype
    heatmap = np.float32(heatmap) / 255
    img = np.float32(img) / 255

    # Blend heatmap with image
    overlay = heatmap * 0.4 + img * 0.6
    overlay = np.uint8(255 * overlay)


    # Save result
    output_path = os.path.join(BASE_DIR, "gradcam_result.png")
    cv2.imwrite(output_path, overlay)

    return output_path


# -----------------------------
# CLI usage
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python gradcam.py <path_to_image>")
        sys.exit(1)

    path = sys.argv[1]
    out = generate_gradcam(path)
    print(f"Grad-CAM saved to {out}")
