import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_dataset(dataset_name):
    save_dir = "generated_datasets"
    with np.load(os.path.join(save_dir, f"{dataset_name}.npz")) as data:
        return data['data'], data['labels']

def train_logistic_model(X_train, y_train, solver='lbfgs'):
    model = LogisticRegression(
        solver=solver,  
        max_iter=5000,  
        tol=1e-3,  
        random_state=12345  
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    scores = model.decision_function(X)  
    error_rate = np.mean(predictions != y)

    fpr, tpr, thresholds = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    min_error_idx = np.argmin(np.abs(fpr + (1 - tpr) - error_rate))
    min_error_fpr = fpr[min_error_idx]
    min_error_tpr = tpr[min_error_idx]

    print(f"Error Rate: {error_rate:.4f}, AUC: {roc_auc:.4f}")
    return fpr, tpr, roc_auc, (min_error_fpr, min_error_tpr)  


def create_quadratic_features(X):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    return poly.fit_transform(X)


def plot_roc_curves(roc_data, title):
    plt.figure()
    for label, (fpr, tpr, auc_value, min_error_point) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_value:.2f})')
        plt.scatter(*min_error_point, color='red', label=f'{label} Min Error')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def plot_decision_boundary(model, X, y, title, quadratic=False):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if quadratic:
        grid_points = create_quadratic_features(grid_points)

    predictions = np.array([
        model.predict([point])[0] for point in tqdm(grid_points, desc="Evaluating Grid Points")
    ])
    predictions = predictions.reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()


def main():
    data_20, labels_20 = load_dataset("D20_train")
    data_200, labels_200 = load_dataset("D200_train")
    data_2000, labels_2000 = load_dataset("D2000_train")
    data_validate, labels_validate = load_dataset("D10K_validate")

    print("Training linear logistic models with L-BFGS...")
    linear_roc_data = {}
    model_20 = train_logistic_model(data_20, labels_20)
    linear_roc_data["Linear (20 samples)"] = evaluate_model(model_20, data_validate, labels_validate)

    model_200 = train_logistic_model(data_200, labels_200)
    linear_roc_data["Linear (200 samples)"] = evaluate_model(model_200, data_validate, labels_validate)

    model_2000 = train_logistic_model(data_2000, labels_2000)
    linear_roc_data["Linear (2000 samples)"] = evaluate_model(model_2000, data_validate, labels_validate)

    plot_roc_curves(linear_roc_data, "ROC Curve for Linear Models")

    data_20_quad = create_quadratic_features(data_20)
    data_200_quad = create_quadratic_features(data_200)
    data_2000_quad = create_quadratic_features(data_2000)
    data_validate_quad = create_quadratic_features(data_validate)

    print("\nTraining quadratic logistic models with L-BFGS...")
    quadratic_roc_data = {}
    model_20_quad = train_logistic_model(data_20_quad, labels_20)
    quadratic_roc_data["Quadratic (20 samples)"] = evaluate_model(model_20_quad, data_validate_quad, labels_validate)

    model_200_quad = train_logistic_model(data_200_quad, labels_200)
    quadratic_roc_data["Quadratic (200 samples)"] = evaluate_model(model_200_quad, data_validate_quad, labels_validate)

    model_2000_quad = train_logistic_model(data_2000_quad, labels_2000)
    quadratic_roc_data["Quadratic (2000 samples)"] = evaluate_model(model_2000_quad, data_validate_quad, labels_validate)

    plot_roc_curves(quadratic_roc_data, "ROC Curve for Quadratic Models")

    print("\nPlotting decision boundaries...")
    plot_decision_boundary(model_20, data_validate, labels_validate, "Linear Decision Boundary (20 samples)")
    plot_decision_boundary(model_20_quad, data_validate_quad, labels_validate, "Quadratic Decision Boundary (20 samples)", quadratic=True)

if __name__ == "__main__":
    main()
