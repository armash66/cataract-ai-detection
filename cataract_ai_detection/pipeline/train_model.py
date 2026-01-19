import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================
# 1. Config
# =========================
DATA_DIR = "dataset"
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. Transforms
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
# 3. Dataset & Loader
# =========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = dataset.classes
print("Classes:", class_names)

# =========================
# 4. Model (MobileNetV2)
# =========================
model = models.mobilenet_v2(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(DEVICE)

# =========================
# 5. Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# 6. Training Loop
# =========================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f} Accuracy: {acc:.2f}%")

# =========================
# 7. Save Model
# =========================
torch.save(model.state_dict(), "cataract_model.pth")
print("✅ Model saved as cataract_model.pth")
