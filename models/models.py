from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from visualization.visualization import evaluate_classifier
from sklearn.model_selection import GridSearchCV
import numpy as np


# Function to tune hyperparameters using GridSearchCV
def tune_hyperparameters(model, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray) -> object:
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters for {model.__class__.__name__}: {best_params}")
    return model.set_params(**best_params)

# Function to tune hyperparameters for Logistic Regression
def tune_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    lr_model = LogisticRegression()
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [100, 200, 300]
    }
    return tune_hyperparameters(lr_model, param_grid, X_train, y_train)

# Function to tune hyperparameters for Decision Tree
def tune_decision_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    dt_model = DecisionTreeClassifier()
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }
    return tune_hyperparameters(dt_model, param_grid, X_train, y_train)

# Function to tune hyperparameters for Random Forest
def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    rf_model = RandomForestClassifier()
    param_grid = {
        "n_estimators": [50, 100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }
    return tune_hyperparameters(rf_model, param_grid, X_train, y_train)

# Function to tune hyperparameters for Support Vector Machine
def tune_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    svm_model = SVC()
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", 0.001, 0.01, 0.1, 1],
        "degree": [2, 3, 4]
    }
    return tune_hyperparameters(svm_model, param_grid, X_train, y_train)

# Function to tune hyperparameters for K Nearest Neighbors
def tune_k_nearest_neighbors(X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    knn_model = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }
    return tune_hyperparameters(knn_model, param_grid, X_train, y_train)

# Function to perform Logistic Regression with tuned hyperparameters
def logistic_regression(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    lr_model = LogisticRegression(C=10, solver="liblinear", max_iter=100)
    #lr_model = tune_logistic_regression(X_train, y_train)
    evaluate_classifier(X_train, y_train, X_test, y_test, lr_model, "Logistic Regression")

# Function to perform Decision Tree with tuned hyperparameters
def decision_tree(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=20, min_samples_split=2, min_samples_leaf=4, max_features="sqrt")
    #dt_model = tune_decision_tree(X_train, y_train)
    evaluate_classifier(X_train, y_train, X_test, y_test, dt_model, "Decision Tree")

# Function to perform Random Forest with tuned hyperparameters
def random_forest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    rf_model = RandomForestClassifier(n_estimators=50, criterion="gini", max_depth=20, min_samples_split=10, min_samples_leaf=4, max_features="sqrt")
    #rf_model = tune_random_forest(X_train, y_train)
    evaluate_classifier(X_train, y_train, X_test, y_test, rf_model, "Random Forest")

# Function to perform Gaussian Naive Bayes
def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    nb_model = GaussianNB()
    evaluate_classifier(X_train, y_train, X_test, y_test, nb_model, "Gaussian Naive Bayes")

# Function to perform Support Vector Machine with tuned hyperparameters
def support_vector_machine(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    svm_model = SVC(C=0.1, kernel="rbf", gamma=1, degree=2)
    #svm_model = tune_svm(X_train, y_train)
    evaluate_classifier(X_train, y_train, X_test, y_test, svm_model, "Support Vector Machine (SVM)")

# Function to perform K Nearest Neighbors with tuned hyperparameters
def k_nearest_neighbors(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    knn_model = KNeighborsClassifier(n_neighbors=9, weights="uniform", p=1)
    #knn_model = tune_k_nearest_neighbors(X_train, y_train)
    evaluate_classifier(X_train, y_train, X_test, y_test, knn_model, "K Nearest Neighbors (KNN)")
