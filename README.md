# 🧬 Breast Cancer Classification - CancerNet

## 📖 Overview
This project develops a Convolutional Neural Network (CNN) named **CancerNet** to classify breast cancer histology image patches (benign vs malignant) using the **IDC dataset**.
The aim is to aid in early detection of breast cancer and improve survival rates through AI-driven diagnostics.

## 🎯 Objectives
- Build a robust CNN for breast cancer classification.
- Evaluate model performance using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.
- Provide a reproducible pipeline for researchers and practitioners.

## 📊 Dataset
- **Dataset**: IDC_regular dataset from Kaggle
- **Patch Size**: 50×50 pixels (RGB)
- **Folder Structure (expected):**
```
data/
  IDC_regular/
    0/    # benign/negative class
    1/    # malignant/positive class
```
> Put the extracted dataset inside `data/IDC_regular/` before training.

## ⚙️ Setup & Installation
```bash
pip install -r requirements.txt
```

## 🚀 Training
```bash
python src/train.py --data_dir data/IDC_regular --img_size 50 --batch_size 256 --epochs 15
```
- The trained model and training curves will be saved to `results/`.

## 📈 Evaluation
```bash
python src/evaluate.py --data_dir data/IDC_regular --img_size 50 --batch_size 256 --model_path results/cancernet.h5
```
- Generates `confusion_matrix.png`, `roc_curve.png`, and prints a classification report.
- Metrics images are saved in `results/`.

## 🧩 Notes
- By default, we split the dataset into **train (80%)** and **validation (20%)** using `image_dataset_from_directory` with a fixed random seed. The evaluation script reuses the same deterministic split to approximate a held-out test set.
- For a *true* test set, place it in a separate folder and update the scripts accordingly.

## 🗂️ Project Structure
```
Breast-Cancer-Classification-CancerNet/
│── data/                         # Place dataset here (ignored by Git)
│── notebooks/
│   └── CancerNet_Training.ipynb  # Optional notebook (starter)
│── src/
│   ├── model.py                  # CNN model architecture
│   ├── train.py                  # Training pipeline
│   └── evaluate.py               # Evaluation & plots
│── results/                      # Saved model & figures
│── requirements.txt              # Dependencies
│── README.md                     # This file
│── .gitignore
```

## ✅ Reproducibility
- We fix random seeds in TensorFlow and NumPy where applicable.
- Ensure you use the same `--seed` across training and evaluation.

