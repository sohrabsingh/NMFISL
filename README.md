# 🤟 Detection of Non-Manual Features in Indian Sign Language Sentences

> **3rd Year B.Tech Computer Vision Project**

## 📌 Project Overview

A classical computer vision system that detects and analyzes **non-manual features** (facial expressions, head orientation, eye gaze regions) in Indian Sign Language (ISL) images, mapping them to corresponding text labels.

---

## 🗂️ Project Structure

```
isl_project/
│
├── ISL_NonManual_Features_Detection.ipynb   ← Main notebook (run this!)
├── requirements.txt                          ← Python dependencies
├── README.md                                 ← This file
│
└── models/                                   ← Auto-created after running
    ├── isl_best_model_*.pkl                  ← Saved best model
    ├── label_mapping.csv                     ← ISL label → text mapping
    └── results_summary.csv                   ← Accuracy comparison
```

---

## 📊 Dataset

**Source:** [`akritRihal/Indian_Sign_Language_dataset`](https://huggingface.co/datasets/akritRihal/Indian_Sign_Language_dataset)

| Property     | Value                  |
|--------------|------------------------|
| Total Images | 10,752                 |
| Train / Test | 9,141 / 1,611          |
| Classes      | 33 (digits 0–9 + letters A–Z minus some) |
| Format       | Parquet (HuggingFace)  |
| Image Size   | Variable (~365px)      |

---

## 🔬 Feature Extraction Methods

| Method | Library | What it captures |
|--------|---------|-----------------|
| **Contour Detection** | OpenCV | Boundary/shape of hand signs |
| **HOG** | scikit-image | Edge orientation — face/hand geometry |
| **Hu Moments** | OpenCV | Rotation-invariant shape descriptors |
| **Face Mesh** | MediaPipe | 468 facial keypoints (non-manual features) |
| **Region Intensity** | NumPy | Forehead / eye / nose / mouth zone stats |

---

## 🤖 Models Trained

| Model | Description |
|-------|-------------|
| **SVM (RBF kernel)** | Best for high-dim feature spaces |
| **Random Forest** | Ensemble, gives feature importance |

Both use: `StandardScaler → PCA (95% variance) → Classifier`

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
1. Upload `ISL_NonManual_Features_Detection.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Run all cells top to bottom (`Runtime → Run all`)
3. First run downloads the dataset (~516MB), takes ~5–10 min

### Option 2 — Local (Jupyter)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook ISL_NonManual_Features_Detection.ipynb
```

---

## 📈 Notebook Sections

| Section | Content |
|---------|---------|
| 1 | Install & Import |
| 2 | Load HuggingFace Dataset |
| 3 | EDA — class distribution, image samples, size analysis |
| 4 | Preprocessing — grayscale, resize, normalize |
| 5 | Feature Extraction — Contours, HOG, MediaPipe, Region Stats |
| 6 | PCA — dimensionality reduction & visualization |
| 7 | Model Training — SVM, Random Forest |
| 8 | Evaluation — confusion matrix, per-class F1, feature importance |
| 9 | Non-Manual Feature Analysis — facial region intensity profiles |
| 10 | Inference — single image + ISL sentence → text demo |
| 11 | Save model & results |
| 12 | Conclusions & Future Work |

---

## 🔭 Future Enhancements

- Real-time webcam detection using MediaPipe Holistic
- Deep learning baseline with MobileNetV3 / EfficientNet
- Optical flow for motion-based non-manual features (video input)
- NLP integration for ISL sentence → fluent text translation

---

## 👥 Team

3rd Year B.Tech Students  
Course Project — Computer Vision / Machine Learning
