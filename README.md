# 🩺 Cataract AI Detection System

This project implements an AI-based cataract detection system using deep learning.
It is designed as a **clinical screening pipeline** that operates on retinal fundus images.

The repository follows a **clean, modular structure** with a single pipeline folder
containing all model-related code.

---

## 📌 Project Overview

Cataract is one of the leading causes of visual impairment worldwide.
Early screening can help prioritize patients for clinical examination.

This project uses a convolutional neural network (CNN) with transfer learning
to classify eye images into:
- **Cataract**
- **Normal**

---

## 🧠 Image Type & Medical Context

- **Image type:** Retinal fundus photographs  
- **Acquisition:** Ophthalmic fundus cameras (clinical environment)  
- **Dataset:** ODIR (Ocular Disease Intelligent Recognition)

⚠️ This system is intended for **academic and screening purposes only**  
and is **not a medical diagnostic tool**.

---

## 🧠 Model Details

- **Architecture:** MobileNetV2
- **Training:** Transfer learning (ImageNet pretrained)
- **Framework:** PyTorch
- **Output:** Class probabilities + predicted label

---

## 📂 Repository Structure

cataract-ai-detection/
├── pipeline/
│ ├── prepare_dataset.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ ├── predict_with_confidence.py
│ └── visualize_data.py
├── .gitignore
├── README.md
└── requirements.txt

---

## 🚀 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Prepare the Dataset
python pipeline/prepare_dataset.py

3️⃣ Train the Model
python pipeline/train_model.py

4️⃣ Evaluate Performance
python pipeline/evaluate_model.py

5️⃣ Run Prediction
python pipeline/predict_with_confidence.py

## 📈 Results

High accuracy on clinical fundus images

Confusion matrix included for performance analysis

Confidence-based predictions for screening support

## 🔮 Future Work

Smartphone / normal eye image pipeline

Domain adaptation between clinical and consumer images

Cataract severity grading

Explainable AI (Grad-CAM)

Mobile and web application deployment

## ⚠️ Disclaimer

This project is intended only for educational and research purposes.
It must not be used for real-world medical diagnosis or treatment.