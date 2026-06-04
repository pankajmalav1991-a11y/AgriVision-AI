# 🌿 AgriVision AI

Plant Disease Classification using Vision Transformers (ViT)

## 📌 Project Overview

AgriVision AI is a deep learning project that identifies diseases in potato and tomato leaves using a Vision Transformer (ViT) model.

The model was fine-tuned on a subset of the PlantVillage dataset and can classify leaf images into six categories:

* Potato Early Blight
* Potato Late Blight
* Potato Healthy
* Tomato Early Blight
* Tomato Late Blight
* Tomato Healthy

The project also includes a Gradio web application for interactive disease prediction.

---

## 🚀 Features

* Vision Transformer (ViT) based image classification
* Transfer Learning using a pretrained ViT model
* Class imbalance handling with weighted loss
* Data augmentation for better generalization
* Interactive Gradio interface
* Real-world image testing
* Clean project structure for deployment and reproducibility

---

## 📂 Dataset

Dataset: PlantVillage

Classes used:

| Class                 | Images |
| --------------------- | -----: |
| Potato___Early_blight |  1000+ |
| Potato___Late_blight  |  1000+ |
| Potato___healthy      |    152 |
| Tomato_Early_blight   |  1000+ |
| Tomato_Late_blight    |  1900+ |
| Tomato_healthy        |  1500+ |

Only these six classes were used from the original PlantVillage dataset.

---

## 🏗️ Model Architecture

Base Model:

* google/vit-base-patch16-224

Modifications:

* Replaced original 1000-class classifier
* Added custom 6-class classification head
* Fine-tuned on PlantVillage dataset

---

## ⚙️ Training Configuration

* Image Size: 224 × 224
* Batch Size: 16
* Optimizer: AdamW
* Learning Rate: 2e-5
* Epochs: 5
* Loss Function: Weighted Cross Entropy Loss

Data Augmentation:

* Random Horizontal Flip
* Random Rotation
* Color Jitter

Dataset Split:

* Training: 70%
* Validation: 15%
* Test: 15%

---

## 📊 Results

### Test Accuracy

99.40%

### Classification Report

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 99.40% |
| Macro F1 Score    |   1.00 |
| Weighted F1 Score |   0.99 |

The model achieved excellent performance on the held-out test set.

---

## 🌎 Real-World Testing

The model was additionally tested on real-world images collected outside the PlantVillage dataset.

Challenges observed:

* Different lighting conditions
* Complex backgrounds
* Different camera qualities
* Leaf orientation variations
* Domain shift from laboratory-style images

This evaluation provided insights into the model's real-world robustness and limitations.

---

## 🖥️ Gradio Application

The project includes an interactive Gradio interface where users can:

1. Upload a leaf image
2. Receive disease prediction
3. View prediction confidence

Run locally:

```bash
python app.py
```

---

## 📁 Project Structure

```text
AgriVision-AI/
│
├── model/
│   ├── agrivision_vit_model.pth
│   └── best_vit_agrivision.pth
│
├── notebooks/
│   └── AgriVision_ViT.ipynb
│
├── sample_images/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔮 Future Improvements

* Collect real field images
* Improve robustness against domain shift
* Add additional crop diseases
* Deploy on Hugging Face Spaces
* Mobile-friendly interface

---

## 👨‍💻 Author

Pankaj Malav

Built as a computer vision project to explore Vision Transformers, transfer learning, model deployment, and practical agricultural AI applications.