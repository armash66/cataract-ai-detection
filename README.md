# 👁️ Cataract Detection System

An AI-based system for eye disease screening using deep learning.
This repository contains two independent pipelines designed for different eye imaging modalities:

Fundus-based cataract detection (clinical retinal images)

Anterior eye cataract detection (front-facing / smartphone images)

The project demonstrates how different medical image modalities require different preprocessing, modeling, and evaluation strategies, and serves as a foundation for an interactive assistant called EyeGPT.

## 🧩 Pipelines Included
### 1️⃣ Fundus Image Pipeline (Clinical)

Input: Retinal fundus images
Use case: Clinical and hospital environments

Features:

- Dataset preprocessing from clinical fundus datasets

- CNN-based cataract detection

- Model training and evaluation

- Confusion matrix and performance metrics

- Confidence-based predictions

This pipeline focuses on clinical-quality retinal images, where cataract indicators are derived from fundus characteristics.

### 2️⃣ Anterior Eye Pipeline (Front-Facing Images)

Input: Anterior (front-facing) eye images
Use case: Accessible screening, smartphone or camera-based images

Features:

- Binary cataract detection (Cataract vs Normal)

- Fine-tuned MobileNetV2 (ImageNet pretrained)

- Confidence-based predictions

- Grad-CAM explainability to visualize model attention

- Designed for real-world, non-clinical image conditions

This pipeline emphasizes accessibility and real-world usability, making it suitable for future consumer-facing applications.

## 📁 Repository Structure
eye-disease-detection/
├── fundus_pipeline/
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_with_confidence.py
│   └── visualize_data.py
│
├── anterior_pipeline/
│   ├── dataset/
│   │   ├── cataract/
│   │   └── normal/
│   ├── src/
│   │   ├── prepare_dataset.py
│   │   ├── train_model.py
│   │   ├── predict.py
│   │   └── gradcam.py
│
├── requirements.txt
├── .gitignore
└── README.md

## ⚙️ Installation
1️⃣ Clone the repository
```bash
git clone <repository-url>
```
```bash
cd eye-disease-detection
```

2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 🧪 How to Run
### ▶️ Fundus Pipeline
```bash
cd fundus_pipeline
```
```bash
python prepare_dataset.py
```
```bash
python train_model.py
```
```bash
python evaluate_model.py
```
```bash
python predict_with_confidence.py
```

### ▶️ Anterior Eye Pipeline
```bash
cd anterior_pipeline
```
```bash
python src/prepare_dataset.py
```
```bash
python src/train_model.py
```
```bash
python src/predict.py <path_to_image>
```
```bash
python src/gradcam.py <path_to_image>
```

This will generate a gradcam_result.png highlighting the image regions influencing the prediction.

## 🧠 Model Details (Anterior Pipeline)

* Architecture: MobileNetV2

* Pretraining: ImageNet

* Fine-tuning: Selective unfreezing of higher convolutional layers

* Loss Function: Cross-Entropy Loss

* Task: Binary classification (Cataract / Normal)

## 🔮 EyeGPT (Planned Extension)

EyeGPT is a proposed interactive AI assistant that combines:

- Vision-based eye disease detection

- Large Language Models (LLMs) for explanation and Q&A

- Planned capabilities:

- Image upload or camera capture

- Natural-language explanation of model predictions

- User questions about eye health and screening results

- Clear medical disclaimers and guidance

EyeGPT is designed to improve accessibility, interpretability, and user trust in AI-assisted eye screening.

## ⚠️ Disclaimer

- This project is intended strictly for educational and research purposes.

- It is not a medical diagnostic tool

- It does not replace professional ophthalmic evaluation

- Predictions should always be validated by a qualified medical professional

## 📌 Author Notes

This repository demonstrates:

- Modality-specific deep learning pipelines

- Practical application of transfer learning in medical imaging

- Explainable AI using Grad-CAM

- A scalable foundation for AI-assisted healthcare applications

## 🤝 Contributing

Contributions are welcome!

If you would like to:

- improve model performance

- add support for more eye diseases

- enhance explainability or evaluation

- help build the EyeGPT interface

- fix bugs or improve documentation

Feel free to fork this repository, make your changes, and submit a pull request. All contributions that improve the project’s quality, usability, or clarity are appreciated.
