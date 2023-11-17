import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd


# Train and evaluate a machine learning model
def train_and_evaluate(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return y_pred, accuracy, report

# Plot the results of the classification
def plot_results(X_test: pd.DataFrame, y_pred: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[y_pred == 0][:, 0], X_test[y_pred == 0][:, 1], c="r", label="Not Purchased")
    plt.scatter(X_test[y_pred == 1][:, 0], X_test[y_pred == 1][:, 1], c="g", label="Purchased")
    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()
    plt.show()

# Print algorithm evaluation results
def print_algorithm_results(accuracy: float, report: str) -> None:
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

# Generate and display a heatmap of the confusion matrix
def generate_report(y_pred: np.ndarray, y_real: np.ndarray, method: str) -> None:
    classes = np.unique(y_real)
    fig, ax = plt.subplots()
    conf_matrix = metrics.confusion_matrix(y_real, y_pred, labels=classes)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Reds, cbar=False)
    ax.set(xlabel="Predicted", ylabel="Actual", title=method)
    ax.set_yticklabels(labels=classes, rotation=0)
    plt.show()

# Evaluate a classifier using provided data and a machine learning model
def evaluate_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, model, method: str) -> None:
    y_pred, accuracy, report = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    print(method)
    print_algorithm_results(accuracy, report)
    generate_report(y_pred, y_test, method)
    plot_results(X_test, y_pred, method)
