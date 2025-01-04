import pandas as pd
from utils.data_preprocessing import load_and_preprocess_data, apply_pca, plot_correlation_matrix
from utils.evaluation import evaluate_traditional_model, evaluate_neural_network
from models.traditional_ml import get_feature_importance, train_logistic_regression, train_random_forest
from models.neural_networks import train_neural_network

# Filepath to the dataset
filepath = "data/diabetes.csv"

# Dictionary to store metrics for all models
metrics_dict = {}


# Step 1: Load and preprocess the data
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)

# Step 2: Apply PCA
print("\nApplying PCA...")
X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_components=2)

# Step 4: Load raw dataset for correlation analysis
raw_data = pd.read_csv(filepath)

# Step 5: Plot Correlation Matrix
print("\nPlotting Correlation Matrix...")
plot_correlation_matrix(raw_data)

# Step 6: Train and evaluate Logistic Regression model
print("Training Logistic Regression model...")
logistic_model = train_logistic_regression(X_train, y_train)
print("Evaluating Logistic Regression model...")

accuracy_lr, precision_lr, recall_lr, f1_lr = evaluate_traditional_model(logistic_model, X_test, y_test)
metrics_dict["Logistic Regression"] = (accuracy_lr, precision_lr, recall_lr, f1_lr)

# Step 7: Train and evaluate Random Forest model
print("\nTraining Random Forest model...")
random_forest_model = train_random_forest(X_train, y_train)
print("Evaluating Random Forest model...")

accuracy_rf, precision_rf, recall_rf, f1_rf = evaluate_traditional_model(random_forest_model, X_test, y_test)
metrics_dict["Random Forest"] = (accuracy_rf, precision_rf, recall_rf, f1_rf)

# Step 3: Feature Importance Analysis
print("\nFeature Importance Analysis (Random Forest):")
get_feature_importance(random_forest_model, feature_names)

# Step :8 Train and evaluate Neural Network model
print("\nTraining Neural Network model...")
input_size = X_train.shape[1]
nn_model = train_neural_network(X_train, y_train, input_size)
print("Evaluating Neural Network model...")

accuracy_nn, precision_nn, recall_nn, f1_nn = evaluate_neural_network(nn_model, X_test, y_test)
metrics_dict["Neural Network"] = (accuracy_nn, precision_nn, recall_nn, f1_nn)

def compare_models(metrics_dict):
    print("\nModel Comparison:")
    print("{:<20} {:<10} {:<10} {:<10} {:<10}".format("Model", "Accuracy", "Precision", "Recall", "F1 Score"))
    for model_name, metrics in metrics_dict.items():
        print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(model_name, *metrics))

# Display the comparison
compare_models(metrics_dict)
