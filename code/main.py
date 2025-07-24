from data_processing.data_loader import load_data
from data_processing.data_preprocessing import preprocess_data
from data_processing.feature_engineering import create_features
from model.pipeline import build_pipeline
from model.tuner import tune_model
from model.trainer import train_evaluate_model

def main():
    """
    Main function to execute the complete workflow.

    Steps:
    1. Load raw data from CSV.
    2. Preprocess the data for cleaning and standardization.
    3. Create features and target variables.
    4. Build a machine learning pipeline.
    5. Perform hyperparameter tuning and model training.
    6. Evaluate the model using various metrics.
    """
    csv_path = r"C:\Users\megha\pythonProject2\Data\usecase_3_.csv"

    # Step 1: Load Data
    df = load_data(csv_path)

    # Step 2: Preprocess Data
    df = preprocess_data(df)

    # Step 3: Create Features
    X, y = create_features(df)

    # Step 4-6: Train and Evaluate Model
    pipeline = build_pipeline()
    train_evaluate_model(X, y, pipeline, tune_model)


if __name__ == "__main__":
    main()
