import numpy as np
import os
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm  

# 1. Define class-conditional Gaussian parameters
means = {
    (0, 1): np.array([-0.9, -1.1]),
    (0, 2): np.array([0.8, 0.75]),
    (1, 1): np.array([-1.1, 0.9]),
    (1, 2): np.array([0.9, -0.75])
}
covariances = {
    (0, 1): np.array([[0.75, 0], [0, 1.25]]),
    (0, 2): np.array([[0.75, 0], [0, 1.25]]),
    (1, 1): np.array([[0.75, 0], [0, 1.25]]),
    (1, 2): np.array([[0.75, 0], [0, 1.25]])
}

# 2. Define Gaussian PDF function
def gaussian_pdf(x, mean, cov):
    return multivariate_normal.pdf(x, mean=mean, cov=cov)

# 3. Compute class-conditional probabilities P(x | L)
def class_conditional_prob(x, label):
    if label == 0:
        return 0.5 * gaussian_pdf(x, means[(0, 1)], covariances[(0, 1)]) + \
               0.5 * gaussian_pdf(x, means[(0, 2)], covariances[(0, 2)])
    else:
        return 0.5 * gaussian_pdf(x, means[(1, 1)], covariances[(1, 1)]) + \
               0.5 * gaussian_pdf(x, means[(1, 2)], covariances[(1, 2)])

# 4. Bayes classifier with tqdm progress bar
def bayes_classifier(X):
    # Compute class-conditional probabilities for each sample in the dataset
    p_x_given_0 = np.array([class_conditional_prob(x, 0) * 0.6 for x in X])
    p_x_given_1 = np.array([class_conditional_prob(x, 1) * 0.4 for x in X])
    
    # Make predictions based on the Bayes decision rule
    predictions = np.where(p_x_given_0 > p_x_given_1, 0, 1)
    
    # Compute the discriminant scores for ROC curve plotting
    scores = p_x_given_0 - p_x_given_1  # Difference in class probabilities
    
    return predictions, scores


# 5. Load the validation dataset
def load_dataset(dataset_name):
    save_dir = "generated_datasets"
    with np.load(os.path.join(save_dir, f"{dataset_name}.npz")) as data:
        return data['data'], data['labels']

# 6. Apply the classifier and plot ROC curve
def plot_roc_curve(data, labels):
    predictions, scores = bayes_classifier(data)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[np.argmin(np.abs(thresholds))], 
                tpr[np.argmin(np.abs(thresholds))], color='red', label='Min P(Error)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Optimal Classifier')
    plt.legend(loc='lower right')
    plt.show()

    # Calculate and print minimum P(Error)
    min_p_error = np.mean(predictions != labels)
    print(f'Minimum P(Error) = {min_p_error:.4f}')

# 7. Plot decision boundary with tqdm progress
def plot_decision_boundary(data, labels):
    # 1. Create a meshgrid to cover the feature space
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # 2. Apply the classifier to each point in the grid with tqdm
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Use tqdm to show progress over all grid points in one progress bar
    predictions = np.array([bayes_classifier([point])[0] for point in tqdm(grid_points, desc="Evaluating Grid Points")])
    predictions = predictions.reshape(xx.shape)

    # 3. Plot decision boundary and data points
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.coolwarm)  # Decision boundary
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.coolwarm, edgecolors='k')  # Data points
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary for Bayes Optimal Classifier')
    plt.show()

# 8. Main execution: Load dataset and run the classifier
data, labels = load_dataset("D10K_validate")
plot_roc_curve(data, labels)  # Plot ROC curve
plot_decision_boundary(data, labels)  # Plot decision boundary
