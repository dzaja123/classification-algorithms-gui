import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from data_loader.data_loader import load_data
from preproccessing.preproccessing import split_data, scale_data
from models.models import (logistic_regression, 
                           decision_tree, 
                           random_forest, 
                           gaussian_naive_bayes, 
                           support_vector_machine, 
                           k_nearest_neighbors)


# Function to execute the selected classification algorithm
def execute_algorithm(algorithm_func, X_train_scaled, y_train, X_test_scaled, y_test):
    try:
        algorithm_func(X_train_scaled, y_train, X_test_scaled, y_test)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during {algorithm_func.__name__}: {str(e)}")

# Function to prepare and split the dataset
def prepare_data():
    dataset = "Social_Network_Ads.csv"
    data = load_data(dataset)

    features = ["Age", "EstimatedSalary"]
    target = "Purchased"

    X_train, X_test, y_train, y_test = split_data(data, features, target)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test

# Function to create the graphical user interface (GUI)
def create_gui(root):
    root.title("Classification Algorithms")
    root.geometry("500x400")

    style = ttk.Style()
    style.configure("TButton", padding=(10, 5), font=("Helvetica", 12))
    style.configure("TLabel", font=("Helvetica", 14, "bold"))

    X_train_scaled, y_train, X_test_scaled, y_test = prepare_data()

    algorithms = {
        "Logistic Regression": logistic_regression,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "Gaussian Naive Bayes": gaussian_naive_bayes,
        "Support Vector Machine": support_vector_machine,
        "K Nearest Neighbors": k_nearest_neighbors,
    }

    label = ttk.Label(root, text="Select an Algorithm to Execute")
    label.grid(row=0, column=0, columnspan=2, pady=(20, 10))

    for i, (algorithm_name, algorithm_func) in enumerate(algorithms.items(), start=1):
        button = ttk.Button(root, text=algorithm_name, command=lambda func=algorithm_func: execute_algorithm(func, X_train_scaled, y_train, X_test_scaled, y_test))
        button.grid(row=i, column=0, pady=5, padx=5, sticky="ew")

    for i in range(1, len(algorithms) + 1):
        root.grid_rowconfigure(i, weight=1)

    root.grid_columnconfigure(0, weight=1)

# Main function to run the GUI application
def main():
    root = tk.Tk()
    create_gui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
