# 🕵️‍♂️ Deep Fake Detection for Images and Videos

This project is an AI-based system that detects **deep fake content** in both images and videos using Convolutional Neural Networks (CNNs) and other deep learning models. It helps combat misinformation and manipulated media through accurate and fast detection techniques.

---

## 📌 Table of Contents

- [🔍 Overview](#-overview)  
- [🚀 Features](#-features)  
- [🛠️ Technologies Used](#️-technologies-used)  
- [📂 Project Structure](#-project-structure)  
- [⚙️ Installation](#️-installation)  
- [▶️ Usage](#️-usage)  
- [📈 Results](#-results)  
- [🧠 Model Details](#-model-details)  
- [📊 Dataset Used](#-dataset-used)

---

## 🔍 Overview

With the rise of AI-generated content, deep fakes pose a real threat in spreading false information and violating privacy. This system can analyze images and videos to determine if they are real or artificially generated using deep learning-based detection algorithms.

---

## 🚀 Features

- 🔍 Detect deep fakes in images with high accuracy  
- 🎥 Analyze video frames for fake detection  
- 🧠 Trained CNN model for feature learning  
- 📁 Modular and clean codebase  
- 💡 Optional web interface via Streamlit or Flask  
- 🗂️ Model file (`model.h5`) excluded from GitHub

---

## 🛠️ Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- OpenCV  
- NumPy / Pandas  
- Matplotlib / Seaborn  
- Streamlit or Flask (optional for deployment)

---

## 📂 Project Structure

```
Deep-Fake-Detection-For-images-and-video/
├── models/                  # Trained model (model.h5 not uploaded)
├── images/                  # Sample images for testing
├── videos/                  # Sample videos
├── utils/                   # Preprocessing scripts
├── notebooks/               # Jupyter notebooks for training
├── test/                    # Test prediction scripts
├── app.py                   # Streamlit or Flask app
├── train_model.py           # Model training script
├── README.md                # Project README
└── requirements.txt         # Python packages
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sagarpatel1024/Deep-Fake-Detection-For-images-and-video.git
cd Deep-Fake-Detection-For-images-and-video
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Add the Model

📥 Download `model.h5` (not on GitHub due to size) and place it inside the `models/` directory.

---

## ▶️ Usage

### ✅ Run Streamlit App

```bash
streamlit run app.py
```

### 🖼️ Predict from Image

```bash
python test/predict_image.py --image images/test1.jpg
```

### 🎞️ Predict from Video

```bash
python test/predict_video.py --video videos/sample.mp4
```

---

## 📈 Results

| Metric     | Value    |
|------------|----------|
| Accuracy   | 94.6%    |
| Precision  | 92.3%    |
| Recall     | 95.1%    |
| F1 Score   | 93.7%    |
| Dataset    | 140K Real & Fake Faces |

> Results may vary based on preprocessing and number of epochs.

---

## 🧠 Model Details

- Model: CNN (Custom)  
- Layers: Conv2D → MaxPooling → BatchNorm → Dropout  
- Activation: ReLU, Sigmoid  
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Input Size: 128x128 RGB  
- Epochs: 10–20

---

## 📊 Dataset Used

- **140K Real and Fake Faces**  
  Source: [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)  
  Description: Contains over 70K real human face images and 70K GAN-generated fake faces.

---
