import pandas as pd

def load_data(csv_path):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    return df
