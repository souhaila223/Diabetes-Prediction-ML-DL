from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_logistic_regression(X_train, y_train):
    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):
    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)
    return model

def get_feature_importance(model, feature_names):
    """
    Display feature importance for a model.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        for name, score in zip(feature_names, importance):
            print(f"Feature: {name}, Importance: {score:.4f}")
    else:
        print("Model does not support feature importance.")
