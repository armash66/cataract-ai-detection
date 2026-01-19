import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import cv2
import numpy as np
import sys
import os

# -----------------------
# Config
# -----------------------
MODEL_PATH = "anterior_cataract_model_finetuned.pth"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["cataract", "normal"]

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
# Load Model
# -----------------------
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# -----------------------
# Grad-CAM Hooks
# -----------------------
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

# Hook the LAST convolution block
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# -----------------------
# Grad-CAM Function
# -----------------------
def generate_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    # Compute weights
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])

    # Weight activations
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(DEVICE)
    for i, w in enumerate(pooled_grads):
        cam += w * activations[0, i, :, :]

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()

    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (image.size[0], image.size[1]))

    # Overlay
    img_np = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Save result
    out_path = "gradcam_result.png"
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Prediction: {CLASS_NAMES[pred_class].upper()}")
    print(f"Grad-CAM saved as: {out_path}")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/gradcam.py <image_path>")
        sys.exit(1)

    generate_gradcam(sys.argv[1])
