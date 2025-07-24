from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from evaluation import (
    calculate_additional_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_predicted_vs_actual,
    classification_metrics_table
)


def train_evaluate_model(X, y, pipeline, tune_model_func):
    """
    Train and evaluate the model, including additional metrics and visualizations.

    Args:
        X: Features DataFrame.
        y: Target Series.
        pipeline: Machine learning pipeline.
        tune_model_func: Function to tune hyperparameters.

    Returns:
        model: Trained and tuned model.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    # Tune the model
    model = tune_model_func(X_train, y_train, pipeline)

    # Check if the model supports predict_proba
    if not hasattr(model, "predict_proba"):
        raise ValueError("The model does not support predict_proba. Ensure it is a classifier.")

    # Generate predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Print classification metrics
    print("=== Classification Metrics ===")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    # Calculate additional metrics
    print("\n=== Additional Metrics ===")
    r2, rmse, rae = calculate_additional_metrics(y_test, y_prob)
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"RAE: {rae:.4f}")

    # Generate visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    plot_predicted_vs_actual(y_test, y_pred)
    plot_feature_importance(model.named_steps['clf'], ['Enrollment', 'combined_text'])

    return model
