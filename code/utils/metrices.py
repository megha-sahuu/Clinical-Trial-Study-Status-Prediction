import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

def calculate_additional_metrics(y_test, y_prob):
    """
    Calculate R², Adjusted R², RMSE, and RAE.
    """
    r2 = r2_score(y_test, y_prob)
    rmse = np.sqrt(mean_squared_error(y_test, y_prob))
    rae = np.sum(np.abs(y_test - y_prob)) / np.sum(np.abs(y_test - np.mean(y_test)))
    return r2, rmse, rae
