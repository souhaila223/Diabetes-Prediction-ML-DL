from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to evaluate traditional ML models
def evaluate_traditional_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return accuracy, precision, recall, f1

# Function to evaluate neural network models
def evaluate_neural_network(model, X_test, y_test):
    import torch
    # Convert test data to PyTorch tensors
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs > 0.5).float()

    # Convert tensors to NumPy arrays for sklearn metrics
    y_true = y_tensor.numpy().flatten()
    y_pred = predictions.numpy().flatten()

    # Calculate metrics
    accuracy = (y_pred == y_true).sum() / y_true.size
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Neural Network Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return accuracy, precision, recall, f1