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
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "anterior_cataract_model_finetuned.pth")
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)
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
    """
    Returns:
        dict:
            prediction
            confidence
            original_image_path
            gradcam_image_path
    """

    # -----------------------------
    # Load image
    # -----------------------------
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    # -----------------------------
    # Forward
    # -----------------------------
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    class_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, class_idx].item()

    # -----------------------------
    # Backward
    # -----------------------------
    model.zero_grad()
    output[0, class_idx].backward()

    # -----------------------------
    # Compute Grad-CAM (7×7)
    # -----------------------------
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    cam = activations[0]

    for i in range(cam.shape[0]):
        cam[i] *= pooled_gradients[i]

    cam = torch.mean(cam, dim=0).detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)

    # -----------------------------
    # Load original image (224×224)
    # -----------------------------
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (224, 224))

    original_out_path = os.path.join(OUTPUT_DIR, "original_display.png")
    cv2.imwrite(
        original_out_path,
        cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    )

    # -----------------------------
    # Resize CAM to image size
    # -----------------------------
    cam_resized = cv2.resize(cam, (224, 224))

    # -----------------------------
    # Create heatmap
    # -----------------------------
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )

    heatmap = np.float32(heatmap) / 255
    original_float = np.float32(original_img) / 255

    # -----------------------------
    # Blend
    # -----------------------------
    gradcam_img = heatmap * 0.4 + original_float * 0.6
    gradcam_img = np.uint8(255 * gradcam_img)

    gradcam_out_path = os.path.join(OUTPUT_DIR, "gradcam_display.png")
    cv2.imwrite(
        gradcam_out_path,
        cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR)
    )

    return {
        "prediction": CLASSES[class_idx],
        "confidence": round(confidence * 100, 2),
        "original_image_path": original_out_path,
        "gradcam_image_path": gradcam_out_path
    }


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python gradcam.py <image_path>")
        sys.exit(1)

    result = generate_gradcam(sys.argv[1])
    print("Result:", result)
