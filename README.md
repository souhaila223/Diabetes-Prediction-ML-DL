# Diabetes Prediction Project

## 📚 Table of Contents

1. 📜 Project Overview
2. 🧩 Features
3. 🔑 Inputs
4. 🚀 Outputs
5. 📊 Models and Performance
6. 🛠️ How to Use
7. 📂 Dataset

---

## 1. 📜 Project Overview

### Overview

The goal of this project is to classify whether a person is diabetic (**1**) or non-diabetic (**0**) based on medical data. This beginner-friendly project demonstrates how to preprocess data, train models, and evaluate their performance using standard metrics.

### Scenario

This project uses the **Pima Indians Diabetes Database** to predict whether a person is diabetic based on features like glucose level, blood pressure, BMI, and more.

---

## 2. 🧩 Features

### Data Preprocessing

- Handling missing values
- Scaling and normalization

### Feature Selection/Dimensionality Reduction

- Principal Component Analysis (PCA)
- Correlation matrix
- Feature importance analysis

### Models Implemented

- Logistic Regression
- Random Forest
- Neural Network (using PyTorch)

### Performance Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

## 3. 🔑 Inputs

The input to the models consists of the following features derived from the dataset:

1. **Pregnancies**: Number of times the patient has been pregnant.
2. **Glucose**: Plasma glucose concentration (mg/dL).
3. **BloodPressure**: Diastolic blood pressure (mm Hg).
4. **SkinThickness**: Triceps skinfold thickness (mm).
5. **Insulin**: 2-hour serum insulin (mu U/ml).
6. **BMI**: Body Mass Index (weight in kg/(height in m)^2).
7. **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
8. **Age**: Patient's age (years).

---

## 4. 🚀 Outputs

The models predict whether a patient is diabetic (**1**) or not (**0**). Performance metrics such as accuracy, precision, recall, and F1 score are used to evaluate the models' effectiveness.

---

## 5. 📊 Models and Performance

The following models were implemented and compared based on their performance:

| **Model**                | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
| ------------------------ | ------------ | ------------- | ---------- | ------------ |
| Logistic Regression      | 0.75         | 0.65          | 0.67       | 0.66         |
| Random Forest            | 0.72         | 0.61          | 0.62       | 0.61         |
| Neural Network (PyTorch) | 0.77         | 0.67          | 0.69       | 0.68         |

---

## 6. 🛠️ How to Use

### Setup

1. **Clone the Repository:**

   ```
   git clone https://github.com/souhaila223/Diabetes-Prediction-ML-DL.git
   cd Diabetes-Prediction-ML-DL
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```
   python main.py
   ```

## 7. 📂 Dataset

The dataset used is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Ensure the dataset (diabetes.csv) is placed in the data/ folder.
The dataset consists of diagnostic measurements for females of Pima Indian heritage, containing the following attributes:

Total instances: 768
Number of features: 8
Target variable: 1 (diabetic) or 0 (non-diabetic)
Missing values: Yes (denoted by zeros in some fields)
