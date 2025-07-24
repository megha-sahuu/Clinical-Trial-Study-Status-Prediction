import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    r2_score,
    mean_squared_error
)

def calculate_additional_metrics(y_test, y_prob):
    """
    Calculate R², Adjusted R², RMSE, and RAE.

    Args:
        y_test: True labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        r2: R-squared value.
        adj_r2: Adjusted R-squared value.
        rmse: Root Mean Squared Error.
        rae: Relative Absolute Error.
    """
    # R²
    r2 = r2_score(y_test, y_prob)

    # Adjusted R²
    n = len(y_test)  # Number of samples
    n_features = 1   # Assume 1 feature (update if dynamic)
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - n_features - 1))

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_prob))

    # RAE
    rae = np.sum(np.abs(y_test - y_prob)) / np.sum(np.abs(y_test - np.mean(y_test)))

    return r2, adj_r2, rmse, rae

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot the confusion matrix.
    """
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(y_test, y_prob):
    """
    Plot the ROC Curve.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(y_test, y_prob):
    """
    Plot the Precision-Recall Curve.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

def classification_metrics_table(y_test, y_pred):
    """
    Print classification metrics in tabular form.
    """
    print("=== Classification Metrics ===")
    print(classification_report(y_test, y_pred))

def plot_feature_importance(model, feature_names):
    """
    Plot the feature importance for the classifier.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        sorted_features = [feature_names[i] for i in indices]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, importance[indices], color='blue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print("Feature importance is not available for this model.")


import matplotlib.pyplot as plt

def plot_predicted_vs_actual(y_test, y_pred):
    """
    Plot a histogram of predicted vs actual values.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(
        [y_test, y_pred],
        bins=3,
        label=['Actual', 'Predicted'],
        color=['blue', 'orange'],
        alpha=0.7
    )
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Predicted vs Actual Distribution')
    plt.legend(loc='upper right')
    plt.show()
