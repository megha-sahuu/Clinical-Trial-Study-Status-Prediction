from sklearn.model_selection import RandomizedSearchCV


def tune_model(X_train, y_train, pipeline):
    """
    Perform hyperparameter tuning for XGBoost using RandomizedSearchCV.

    Arguments:
    - X_train: Training feature set.
    - y_train: Training target set.
    - pipeline: ML pipeline with preprocessing and classifier.

    Returns:
    - best_model: Tuned pipeline with the best parameters.
    """
    param_grid = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [5, 6, 8],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__scale_pos_weight': [1, 3, 5, 7]
    }

    # Use single-threaded execution to avoid serialization issues
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=5,
        cv=3,
        scoring='roc_auc',
        verbose=2,
        n_jobs=1,  # Avoid parallelism
        random_state=42
    )

    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    return search.best_estimator_
