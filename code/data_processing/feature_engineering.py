def create_features(df):
    """
    Create X (features) and y (target).
    """
    y = df['Study Status'].apply(lambda x: 1 if x == 'COMPLETED' else 0)
    X = df[['Enrollment', 'combined_text']]
    return X, y
