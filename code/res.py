import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


# ---------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------
def load_data(csv_path):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    return df


# ---------------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------------
def preprocess_data(df):
    """
    Preprocess the dataset, including cleaning and feature engineering.
    """
    df['Study Status'] = df['Study Status'].str.upper().replace({
        'SUSPENDED': 'NOT_COMPLETED',
        'WITHDRAWN': 'NOT_COMPLETED',
        'TERMINATED': 'NOT_COMPLETED',
        'COMPLETED': 'COMPLETED'
    })

    df = df[df['Study Status'].isin(['COMPLETED', 'NOT_COMPLETED'])]

    if 'Enrollment' in df.columns:
        df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
        df['Enrollment'] = df['Enrollment'].fillna(0)

    text_cols = ['Study Title', 'Brief Summary', 'Conditions', 'Interventions']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    df['combined_text'] = (
        df['Study Title'] + ' ' + df.get('Brief Summary', '') + ' ' +
        df.get('Conditions', '') + ' ' + df.get('Interventions', '')
    )
    return df


# ---------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------
def create_features(df):
    """
    Create X (features) and y (target).
    """
    y = df['Study Status'].apply(lambda x: 1 if x == 'COMPLETED' else 0)
    X = df[['Enrollment', 'combined_text']]
    return X, y


# ---------------------------------------------------------------
# 4. Build Pipeline
# ---------------------------------------------------------------
def build_pipeline():
    """
    Build the preprocessing pipeline and classifier.
    """
    numeric_transformer = 'passthrough'
    text_transformer = TfidfVectorizer(max_features=500, stop_words='english')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['Enrollment']),
            ('text', text_transformer, 'combined_text')
        ]
    )
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', xgb)
    ])
    return pipeline


# ---------------------------------------------------------------
# 5. Hyperparameter Tuning
# ---------------------------------------------------------------
def tune_model(X_train, y_train, pipeline):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    """
    param_grid = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [5, 6, 8],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__scale_pos_weight': [1, 3, 5, 7]
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=5,
        cv=3,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    return search.best_estimator_


# ---------------------------------------------------------------
# 6. SHAP Explainability
# ---------------------------------------------------------------
def explain_model_with_shap(model, X_test):
    """
    Use SHAP to explain the predictions of the trained model.

    Parameters:
    - model: Trained XGBoost model.
    - X_test: Test data (features) after preprocessing.

    Generates:
    - SHAP Summary Plot (Bar and Beeswarm)
    - SHAP Force Plot (for a single prediction)
    - SHAP Decision Plot (cumulative feature contributions)
    """
    print("=== SHAP Explainability ===")

    # Initialize SHAP Explainer
    explainer = shap.Explainer(model)

    # Compute SHAP values
    print("Computing SHAP values...")
    shap_values = explainer(X_test)

    # SHAP Summary Plot (Bar)
    print("Generating SHAP Summary Plot (Bar)...")
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.show()

    # SHAP Summary Plot (Beeswarm)
    print("Generating SHAP Summary Plot (Beeswarm)...")
    shap.summary_plot(shap_values, X_test)
    plt.show()

    # SHAP Force Plot (Single Prediction)
    print("Generating SHAP Force Plot (Single Prediction)...")
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :], matplotlib=True)
    plt.show()

    # SHAP Decision Plot
    print("Generating SHAP Decision Plot...")
    shap.decision_plot(explainer.expected_value, shap_values[:100], X_test.iloc[:100])
    plt.show()


# ---------------------------------------------------------------
# 7. Train and Evaluate Model
# ---------------------------------------------------------------
def train_evaluate_model(X, y):
    """
    Train and evaluate the model, including SHAP explainability.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    model = tune_model(X_train, y_train, pipeline)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    print("=== Classification Metrics ===")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    # SHAP Explainability
    explain_model_with_shap(model.named_steps['clf'], X_test)


# ---------------------------------------------------------------
# 8. Main Script
# ---------------------------------------------------------------
def main():
    """
    Main script to load data, preprocess, train, and evaluate the model.
    """
    csv_path = r"C:\Users\megha\pythonProject2\Data\usecase_3_.csv"
    df = load_data(csv_path)
    df = preprocess_data(df)
    X, y = create_features(df)
    train_evaluate_model(X, y)


if __name__ == "__main__":
    main()
