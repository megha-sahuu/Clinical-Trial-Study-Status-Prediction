import pandas as pd  # Required import for pandas functions

def preprocess_data(df):
    """
    Preprocess the dataset, including data cleaning and standardization.

    Why this step:
    - To clean inconsistencies in the target variable `Study Status`.
    - Standardize categorical values for binary classification (Completed vs Not Completed).
    - Fill missing values to handle NaNs in numerical and text columns.
    - Combine textual data for better feature representation in modeling.

    Arguments:
    - df: Raw pandas DataFrame.

    Returns:
    - df: Cleaned and preprocessed DataFrame.
    """
    # Standardize target variable (Study Status) for binary classification
    df['Study Status'] = df['Study Status'].str.upper().replace({
        'SUSPENDED': 'NOT_COMPLETED',
        'WITHDRAWN': 'NOT_COMPLETED',
        'TERMINATED': 'NOT_COMPLETED',
        'COMPLETED': 'COMPLETED'
    })

    # Retain only relevant rows for binary classification
    df = df[df['Study Status'].isin(['COMPLETED', 'NOT_COMPLETED'])]

    # Convert 'Enrollment' column to numeric and fill missing values with 0
    if 'Enrollment' in df.columns:
        df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
        df['Enrollment'] = df['Enrollment'].fillna(0)

    # Fill missing values in text-based columns with empty strings
    text_cols = ['Study Title', 'Brief Summary', 'Conditions', 'Interventions']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    # Combine multiple text columns into one for unified feature representation
    df['combined_text'] = (
        df['Study Title'] + ' ' +
        df.get('Brief Summary', '') + ' ' +
        df.get('Conditions', '') + ' ' +
        df.get('Interventions', '')
    )
    return df
