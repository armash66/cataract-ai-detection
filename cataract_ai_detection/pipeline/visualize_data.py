import os
import random
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"

def show_images(label, n=5):
    folder = os.path.join(DATASET_DIR, label)
    images = random.sample(os.listdir(folder), n)

    plt.figure(figsize=(15, 3))
    for i, img_name in enumerate(images):
        img_path = os.path.join(folder, img_name)
        img = Image.open(img_path)

        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(label)

    plt.show()

# Show Normal eyes
show_images("normal", 5)

# Show Cataract eyes
show_images("cataract", 5)
