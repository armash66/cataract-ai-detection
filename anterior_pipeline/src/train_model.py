import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
BATCH_SIZE = 16
EPOCHS = 10              # increased epochs
LR = 1e-5                # lower LR for fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORMS (Anterior-eye aware)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET
# =========================
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
class_names = full_dataset.classes
print("Classes:", class_names)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =========================
# MODEL (MobileNetV2)
# =========================
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

# Freeze all layers
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last 2 blocks (fine-tuning)
for param in model.features[-2:].parameters():
    param.requires_grad = True

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f} Accuracy: {acc:.2f}%")

# =========================
# EVALUATION
# =========================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu()

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "anterior_cataract_model_finetuned.pth")
print("✅ Fine-tuned model saved as anterior_cataract_model_finetuned.pth")
