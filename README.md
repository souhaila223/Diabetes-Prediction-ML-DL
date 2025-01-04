# Diabetes Prediction Project

This project is a beginner-friendly implementation of machine learning and deep learning techniques to predict diabetes. It uses the **Pima Indians Diabetes Database** and implements models like Logistic Regression, Random Forest, and a PyTorch-based Neural Network.

---

## **1. Project Overview**
The goal of this project is to classify whether a person is diabetic (**1**) or non-diabetic (**0**) based on medical data. It demonstrates how to preprocess data, train models, and evaluate their performance using standard metrics.

---

## **2. Features**
### **Data Preprocessing**
- Handling missing values
- Scaling and normalization

### **Feature Selection/Dimensionality Reduction**
- Principal Component Analysis (PCA)
- Correlation matrix
- Feature importance analysis

### **Models Implemented**
- Logistic Regression
- Random Forest
- Neural Network (using PyTorch)

### **Performance Metrics**
- Accuracy
- Precision
- Recall
- F1 Score

---

## **3. Inputs**
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

## **4. Outputs**
The models predict whether a patient is diabetic (**1**) or not (**0**). Performance metrics are used to evaluate each model's effectiveness.

---

## **5. Models and Performance**
The following models were implemented and compared based on their performance:

| **Model**                | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|--------------------------|--------------|---------------|------------|--------------|
| Logistic Regression      | 0.75         | 0.65          | 0.67       | 0.66         |
| Random Forest            | 0.72         | 0.61          | 0.62       | 0.61         |
| Neural Network (PyTorch) | 0.77         | 0.67          | 0.69       | 0.68         |

---

## **6. How to Use**

### **Setup**
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project folder:
   ```bash
   cd project_folder
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Run the Project**
1. Train and evaluate models:
   ```bash
   python main.py
   ```
2. Outputs include performance metrics for each model and visualizations like the correlation matrix and PCA.

---

## **7. Dataset**
The dataset used is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Ensure the dataset (`diabetes.csv`) is placed in the `data/` folder.