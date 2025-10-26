
# 🌍 XAI-Based Earthquake Damage Detection

This project focuses on detecting earthquake damage from satellite images using deep learning and explainable AI (XAI) techniques.  
It is based on the **xBD (xView2)** disaster dataset, with emphasis on the **earthquake subset**.

---

## 🔹 Overview
This repository implements an explainable AI approach to classify and visualize earthquake damage severity using CNN-based feature extraction and traditional ML classification (ResNet + XGBoost).  

---

## 🔹 Data Preprocessing
- Extracted the large xBD dataset (~7.8 GB) into manageable chunks of **1000 images each**.  
- Applied preprocessing:  
  - Image resizing (224×224)  
  - Normalization and cleanup  
  - Filtering only earthquake-related data  
- Combined satellite image pairs (pre- and post-disaster).  
- Matched corresponding label JSONs to assign correct damage categories.  
- Final earthquake-only labeled data stored in `.npz` format for efficient loading in Colab.

---

## 📂 Dataset
The `data/` folder (xBD earthquake subset) is not included in this repository due to its size.  

However, you can **recreate the dataset locally** using the preprocessing scripts provided in `scripts/`:

| Script | Description |
|--------|--------------|
| `extract_tar.py` | Extracts raw `.tar` xBD files safely |
| `make_npz_chunks.py` | Converts images into 1000-image `.npz` chunks |
| `filter_earthquake_chunks.py` | Filters only earthquake images from all disasters |
| `match_labels_to_earthquake_npz.py` | Matches xBD label JSONs to filtered earthquake chunks |
| `verify_label.py` | Verifies loaded data integrity and label distribution |

The final processed dataset is saved under:  


*Dataset is also stored on Google Drive (link available on request).*

---

## 🔹 Model Architecture
- **Feature Extraction** → Pretrained **ResNet50** and **MobileNetV2**  
- **Classification** → **XGBoost** for final decision layer  
- **Explainable AI** → planned integration using **Grad-CAM**, **LIME**, and **SHAP**

---

## 🔹 Folder Structure
xai-earthquake-damage/
│
├── data/
│   └── earthquake_only_npz_labeled/
│
├── scripts/
│   ├── extract_tar.py
│   ├── make_npz_chunks.py
│   ├── filter_earthquake_chunks.py
│   ├── match_labels_to_earthquake_npz.py
│   └── verify_label.py
│
├── notebooks/
│   └── (Colab notebooks for ResNet, XGBoost, and XAI)
│
├── models/
│   └── 
│
├── .gitignore
└── README.md


---

## 🔹 Current Progress
✅ Dataset preprocessing and filtering completed  
✅ Earthquake-only labeled data verified  
🔄 CNN feature extraction (ResNet50) in progress  
🧠 Next: XGBoost classifier + XAI integration  

---
## ⚙️ Project Scope and Current Status

This repository implements a **working pipeline** for Explainable AI (XAI)-based Earthquake Damage Detection.

The current version focuses on:
- Building a **complete pipeline** — from data preprocessing to explainability (Grad-CAM, SHAP).
- Demonstrating the workflow of how satellite imagery is processed, labeled, and analyzed using **ResNet50 + XGBoost**.
- Showing **explainability visualizations** and interpretation logic.

> ⚠️ **Note:**  
> This project currently functions as a **prototype / proof-of-concept** rather than a final damage detection model.  
> It demonstrates the **working pipeline, model interpretability, and methodology**, suitable for academic presentation and future model improvements.


## 🔹 Future Work
- Improve label balance across damage categories  
- Extend to multiple disaster types (floods, fire, etc.)  
- Compare multiple XAI visualization techniques  
- Evaluate explainability metrics for better trust in model outputs  

---

## 💻 Tech Stack
- Python · NumPy · TensorFlow/Keras · XGBoost · Matplotlib  
- Google Colab · OpenCV · tqdm

---

*This project aims to contribute toward interpretable disaster impact assessment using AI — combining accuracy, transparency, and societal relevance.*


