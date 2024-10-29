import numpy as np
from sklearn.metrics import mean_squared_error  # For MSE calculation
import matplotlib.pyplot as plt  # For visualization
from sklearn.preprocessing import PolynomialFeatures
from hw2q2 import hw2q2

# 1. Generate training and validation datasets
xTrain, yTrain, xValidate, yValidate = hw2q2()
print("Training Labels Range: min =", yTrain.min(), ", max =", yTrain.max())
print("Validation Labels Range: min =", yValidate.min(), ", max =", yValidate.max())
print(f"xTrain shape: {xTrain.shape}, yTrain shape: {yTrain.shape}")

# 2. Expand the features to cubic polynomial features
poly = PolynomialFeatures(degree=3)
xTrain_poly = poly.fit_transform(xTrain.T)
xValidate_poly = poly.transform(xValidate.T)

print(f"xTrain_poly shape: {xTrain_poly.shape}")  # Check the dimension of polynomial features

# 3. Implement the Maximum Likelihood Estimation (ML) function
def ml_estimator(X, y):
    """Compute the parameters w using Maximum Likelihood Estimation (ML)"""
    return np.linalg.pinv(X.T @ X) @ X.T @ y  # Use pseudo-inverse to handle non-invertible matrices

# 4. Implement the Maximum A Posteriori (MAP) Estimation function
def map_estimator(X, y, gamma):
    """Compute the parameters w using Maximum A Posteriori (MAP) Estimation"""
    n_features = X.shape[1]
    I = np.eye(n_features)  # Identity matrix for regularization
    return np.linalg.inv(X.T @ X + gamma * I) @ X.T @ y

# 5. Calculate ML estimated parameters and predict on the validation set
w_ml = ml_estimator(xTrain_poly, yTrain)
y_pred_ml = xValidate_poly @ w_ml
mse_ml = mean_squared_error(yValidate, y_pred_ml)

print(f"ML Estimated Parameters: {w_ml}")
print(f"ML MSE: {mse_ml}")

# 6. Define the range of gamma and compute MAP estimation
gammas = np.logspace(-10, 10, 100)
mse_values_map = []

best_mse_map = float('inf')
best_gamma = None
best_w_map = None

for gamma in gammas:
    w_map = map_estimator(xTrain_poly, yTrain, gamma)
    y_pred_map = xValidate_poly @ w_map
    mse_map = mean_squared_error(yValidate, y_pred_map)
    mse_values_map.append(mse_map)

    if mse_map < best_mse_map:
        best_mse_map = mse_map
        best_gamma = gamma
        best_w_map = w_map

print(f"Best MAP MSE: {best_mse_map}, Corresponding Gamma: {best_gamma}")
print(f"Best MAP Estimated Parameters: {best_w_map}")

# 7. Visualize the relationship between gamma and MSE
plt.plot(gammas, mse_values_map, label='MAP MSE')
plt.axhline(mse_ml, color='r', linestyle='--', label='ML MSE')
plt.xscale('log')
plt.xlabel('Gamma (log scale)')
plt.ylabel('MSE')
plt.title('MSE vs Gamma for MAP Estimator')
plt.legend()
plt.show()

# Compute baseline MSE using the mean of the training labels
y_baseline = np.mean(yTrain)
y_pred_baseline = np.full_like(yValidate, y_baseline)
mse_baseline = mean_squared_error(yValidate, y_pred_baseline)

print(f"Baseline MSE: {mse_baseline}")

