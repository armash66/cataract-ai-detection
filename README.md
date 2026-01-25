# Cataract Eye Detection System

An AI-based **cataract eye detection system** exploring how deep learning models can analyze eye images across different imaging modalities and present interpretable screening results through both programmatic pipelines and a web-based interface (EyeGPT).

This repository contains **multiple independent pipelines**, each designed for a specific eye imaging modality, along with a **research-oriented UI application** for visualization and interaction.

---

## Overview

The project demonstrates how **different medical image modalities** require different preprocessing strategies, model architectures, and evaluation approaches.

Currently, the system includes:

- A **fundus image pipeline** for clinical retinal images  
- An **anterior eye image pipeline** for front-facing / smartphone-style images  
- A web-based research UI called **EyeGPT** for AI screening visualization  

**Important Clarification:**
- EyeGPT refers **only to the UI and web application layer**
- The AI models operate independently of the UI
- This project is strictly for **educational and research purposes**

---

## Pipelines Included

### 1. Fundus Image Pipeline (Clinical)

**Input:** Retinal fundus images  
**Use case:** Clinical and hospital environments  

**Features:**
- Dataset preprocessing for clinical fundus datasets  
- CNN-based cataract detection  
- Model training and evaluation  
- Confusion matrix and performance metrics  
- Confidence-based predictions  

This pipeline focuses on **clinical-quality retinal images**, where cataract indicators are inferred from fundus characteristics.

---

### 2. Anterior Eye Pipeline (Front-Facing Images)

**Input:** Anterior (front-facing) eye images  
**Use case:** Accessible screening (smartphone or camera-based images)  

**Features:**
- Binary cataract detection (Cataract vs Normal)  
- Fine-tuned **MobileNetV2** (ImageNet pretrained)  
- Confidence-based predictions  
- Grad-CAM explainability to visualize model attention  
- Designed for real-world, non-clinical image conditions  

This pipeline emphasizes **accessibility and real-world usability**, making it suitable for future consumer-facing or screening applications.

---

## EyeGPT Web Application (UI Layer)

EyeGPT is a **research-oriented web application** that visualizes AI screening outputs in an interpretable and transparent manner.

EyeGPT **does not perform diagnosis**. It only presents model outputs.

**Current UI Features:**
- Image upload–based AI screening  
- Confidence-based prediction display  
- Grad-CAM visual explanation toggle  
- Scan history and review functionality  
- Print-friendly review mode  
- Research-grade, non-consumer UI design  

⚠️ **Training Note:**  
The underlying model used in EyeGPT was trained using **fundus eye images**, but **fundus-specific UI workflows are not yet implemented**.

---

## Explainability (Grad-CAM)

- Allows toggling between original images and Grad-CAM heatmaps  
- Highlights regions influencing the model’s prediction  
- Explicitly labeled as **visual explanation**, not diagnosis  
- Intended to improve transparency and research interpretability  

---

## Camera Input (Experimental)

- Supports camera-based image capture  
- Explicit warnings about unreliability  
- Requires user acknowledgment before activation  
- Grad-CAM disabled for camera images  

This feature is experimental and **not clinically reliable**.

---

## Repository Structure

```bash
cataract-ai-detection/
├── app.py                      # Flask application entry point (EyeGPT)
├── templates/
│   ├── index.html              # Main screening UI
│   ├── history.html            # Scan history & review page
├── static/
│   ├── style.css               # Global UI styles
│   ├── history.css             # History page styles
│   ├── ui.js                   # UI interaction logic
│   ├── camera.js               # Camera handling logic
│   ├── images/                 # Stored scan images
│   └── gradcam/                # Generated Grad-CAM outputs
│
├── fundus_pipeline/
│   ├── dataset/
│   │   ├── cataract/
│   │   └── normal/
│   ├── src/
│   │   ├── prepare_dataset.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   ├── predict_with_confidence.py
│   │   └── visualize_data.py
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
├── model/
│   ├── model.h5                # Trained AI model (optional)
│   └── gradcam_utils.py        # Explainability utilities
│
├── requirements.txt
└── README.md
```
## Setup and Usage
### Clone the Repository
```bash
git clone https://github.com/armash66/cataract-ai-detection.git
cd cataract-ai-detection
```
### Install Dependencies
```bash

pip install -r requirements.txt
```
## Running the Pipelines
### Fundus Pipeline
```bash
cd fundus_pipeline
python prepare_dataset.py
python train_model.py
python evaluate_model.py
python predict_with_confidence.py
```
### Anterior Eye Pipeline
```bash
cd anterior_pipeline
python src/prepare_dataset.py
python src/train_model.py
python src/predict.py <path_to_image>
python src/gradcam.py <path_to_image>
```
This will generate a Grad-CAM visualization highlighting regions influencing the prediction.

### Running EyeGPT (Web UI)
```bash
python app.py
```
Open your browser at:
```bash
http://127.0.0.1:5000
```
## Limitations

- Dataset bias may affect predictions  
- Fundus-specific UI workflows are not yet implemented  
- Camera input is experimental and unreliable  
- Grad-CAM highlights model attention, not pathology  
- Performance metrics are not clinically benchmarked  

---

## Future Work

- Full fundus image workflow integration in EyeGPT  
- Multi-class eye disease screening  
- Improved dataset diversity  
- Model performance evaluation (ROC, sensitivity, specificity)  
- Exportable research reports  
- UI accessibility enhancements  
- Integration of LLM-based explanation (EyeGPT extension)  

---

## Medical and Ethical Disclaimer

This project is strictly for **educational and research purposes only**.

- Not a medical diagnostic tool  
- Not clinically validated  
- Not approved for patient use  
- Must not be used for treatment or clinical decisions  

AI confidence scores reflect **model behavior**, not disease severity or certainty.

---

## Author

**Armash Ansari**  

---

## Contributing

Contributions are welcome.

You may help by:
- Improving model performance  
- Adding support for additional eye diseases  
- Enhancing explainability or evaluation  
- Improving UI/UX  
- Fixing bugs or documentation  

Feel free to fork the repository and submit a pull request.
