#  Packaging Quality Classification using Deep Learning

##  Project Overview

This project is an end-to-end supervised machine learning system for **binary image classification of consumer product packaging**. It classifies packaging images into:

-  **Good Packaging (Class 0)**
-  **Defected Packaging (Class 1)**

The system is designed using multiple deep learning approaches including **custom CNNs**, **transfer learning models (VGG16, MobileNetV2, EfficientNetB0)**, and advanced techniques like **Grad-CAM visualization** and **incremental learning**.

---

##  Dataset Description

- Total Images: **657**
- Classes:
  - Good Packaging (~329 images)
  - Defected Packaging (~328 images)
- Image Size: **128 × 128**
- Format: RGB (converted and normalized)
- Collected from real-world retail environments in Karachi:
  - Supermarkets
  - Local stores
  - Home environments

### Key Characteristics
- Varying lighting conditions
- Multiple product types
- Real-world packaging defects (tears, dents, broken seals)
- Balanced dataset (no class imbalance)

---

##  Preprocessing Pipeline

- Image loading using TensorFlow / OpenCV
- Resizing to **128×128**
- Normalization to `[0, 1]`
- Label encoding:
  - Good → 0
  - Defected → 1
- Data augmentation:
  - Horizontal flip
  - Brightness adjustment
  - Contrast variation
  - Saturation variation
- Stratified train/validation/test split:
  - 70/15/15 (primary)
  - Additional splits for comparison

---

##  Models Implemented

### 1. Custom CNN (From Scratch)
- CNN-A (Lightweight)
- CNN-B (BatchNorm + deeper)
- CNN-C (Deep + L2 regularization)

### 2. Transfer Learning - VGG16
- VGG16-V1: Frozen base
- VGG16-V2: Partial fine-tuning (best performer)
- VGG16-V3: Full fine-tuning

### 3. Transfer Learning - MobileNetV2
- MobileNetV2-V1: Frozen base
- MobileNetV2-V2: Partial fine-tuning (best performer)
- MobileNetV2-V3: Full fine-tuning

### 4. Extension Model
- EfficientNetB0 (V1–V3 variants)

---

## Evaluation Metrics

All models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

##  Best Performing Models

| Model | Type | Performance |
|------|------|-------------|
| VGG16-V2 | Transfer Learning | ⭐ Best overall |
| MobileNetV2-V2 | Transfer Learning | ⭐ Very strong |
| CNN-C | Custom CNN | Moderate performance |

### Key Insight:
Partial fine-tuning consistently outperformed:
- Frozen models
- Fully fine-tuned models (due to overfitting on small dataset)

---

##  Key Findings

- Transfer learning significantly outperforms training CNNs from scratch
- Dataset size (~657 images) is sufficient for baseline learning but limits deep models
- Overfitting is a major challenge for:
  - VGG16-V3
  - MobileNetV2-V3
- Data augmentation improves generalization
- Grad-CAM improves model interpretability

---

##  Explainability (Grad-CAM)

Grad-CAM was implemented to highlight regions of packaging images responsible for predictions.

- Helps visualize defect areas
- Supports weak localization without bounding boxes
- Useful for industrial quality inspection

---

##  Incremental Learning

The system supports online updates:

- Add new labeled images
- Fine-tune model head only
- Save updated model automatically

This enables continuous improvement without full retraining.

---

##  Interactive Interface

Built using `ipywidgets` (Google Colab compatible):

Features:
- Image upload
- Real-time prediction
- Confidence score display
- Incremental learning interface

---

##  Test Cases

The model was tested on:
- Clear good packaging samples → Correct predictions
- Clear defected packaging samples → Correct predictions

---

##  Future Improvements

- Expand dataset (1000+ images per class)
- Multi-class classification (Good / Minor Defect / Major Defect)
- Object detection (YOLO-based defect localization)
- Real-time deployment (Flask / FastAPI)
- Model compression for edge devices
- Semi-supervised learning

---

##  Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV, PIL
- Matplotlib, Seaborn
- Scikit-learn
- Google Colab

---
