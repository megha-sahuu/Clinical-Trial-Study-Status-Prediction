import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot the confusion matrix.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
    """
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(y_test, y_prob):
    """
    Plot the ROC Curve.

    Args:
        y_test: True labels.
        y_prob: Predicted probabilities for the positive class.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
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

    Args:
        y_test: True labels.
        y_prob: Predicted probabilities for the positive class.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plot the feature importance for the classifier.

    Args:
        model: Trained model.
        feature_names: Names of the features.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        sorted_features = [feature_names[i] for i in indices]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, importance[indices], color='blue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance (XGBoost)')
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print("Feature importance is not available for this model.")

def classification_metrics_table(y_test, y_pred):
    """
    Display classification metrics in tabular form.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
    """
    print("=== Classification Metrics ===")
    print(classification_report(y_test, y_pred))

def plot_predicted_vs_actual(y_test, y_pred):
    """
    Plot a histogram of predicted vs actual values.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
    """
    plt.figure(figsize=(8, 6))
    plt.hist([y_test, y_pred], bins=3, label=['Actual', 'Predicted'], color=['blue', 'orange'], alpha=0.7)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Predicted vs Actual Distribution')
    plt.legend(loc='upper right')
    plt.show()
