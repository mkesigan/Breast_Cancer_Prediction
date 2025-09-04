# ✨ Breast Cancer Prediction

Predicts **breast cancer diagnosis** (malignant or benign) using **Python** and **Machine Learning**.  
Dataset from [Kaggle: Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

---

## 🚀 Project Overview

- **Goal:** Predict whether a tumor is malignant or benign based on features computed from cell nuclei images.  
- **Dataset:** 569 samples, 32 features (ID, diagnosis, 30 numeric features)  
- **ML Models Used:** Logistic Regression 
- **Evaluation Metrics:** Accuracy,Confusion Matrix

---

## 📂 Dataset Details

| Feature | Description |
|---------|-------------|
| id | Unique ID for each sample |
| diagnosis | M = Malignant, B = Benign |
| radius_mean | Mean of distances from center to points on the perimeter |
| texture_mean | Standard deviation of gray-scale values |
| perimeter_mean | Mean perimeter |
| area_mean | Mean area |
| smoothness_mean | Mean smoothness |
| ... | 30 total numeric features in dataset |

- Total Samples: **569**  
  - Benign: 357  
  - Malignant: 212  

---

## 🏗️ Project Structure
Breast-Cancer-Prediction/
│── notebook.ipynb # Jupyter notebook with full ML workflow
│── data/ # Dataset CSV files
│── README.md # This file
│── .gitignore # Ignores .env, kaggle.json, pycache, etc.
│── .env # Kaggle API credentials (ignored)


---

## ✅ Features Implemented

- **Data Loading & Preprocessing**
  - Handle missing values
  - Encode diagnosis labels (`M` → 1, `B` → 0)
  - Normalize numeric features

- **Exploratory Data Analysis (EDA)**
  - Count of malignant vs benign

- **Model Training & Evaluation**
  - Split dataset: 75% training, 25% testing
  - Train models: Logistic Regression
  - Evaluate accuracy
  - Display confusion matrix

- **Visualization**
  - Confusion matrix heatmaps

---

## 🧪 How to Run

1. **Clone repository**
```bash
git clone https://github.com/mkesigan/Breast_Cancer_Prediction.git
cd Breast_Cancer_Prediction

Install dependencies

pip install -r requirements.txt


Add Kaggle credentials

Place your kaggle.json in C:\Users\<YourUsername>\.kaggle\

OR create a .env file with your Kaggle API key
(Both are ignored by Git)

Run notebook

Open notebook.ipynb in VS Code Jupyter or Jupyter Notebook

Run all cells sequentially

Optional: manual dataset download

Kaggle Dataset Page

Place breast-cancer-wisconsin-data.csv in data/ folder

📦 Dependencies

Python 3.9+

pandas 1.5.3

numpy 1.25.0

scikit-learn 1.3.0

matplotlib 3.8.0

seaborn 0.12.2

kaggle 1.6.3 (optional if dataset downloaded manually)

📌 License / Terms

This project is for educational purposes only.
Dataset is from Kaggle and follows Kaggle Terms.


