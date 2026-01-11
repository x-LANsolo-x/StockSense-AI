"""
Metrics Module
Evaluation metrics for forecasting models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true: np.array, y_pred: np.array) -> float:
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_mse(y_true: np.array, y_pred: np.array) -> float:
    """Calculate Mean Squared Error."""
    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true: np.array, y_pred: np.array) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Note: Handles zero values by adding small epsilon.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    epsilon = 1e-10
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def calculate_r2(y_true: np.array, y_pred: np.array) -> float:
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)


def calculate_all_metrics(y_true: np.array, y_pred: np.array) -> dict:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of all metrics
    """
    return {
        'MAE': calculate_mae(y_true, y_pred),
        'MSE': calculate_mse(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred)
    }


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*40}")
    print(f"{model_name} Evaluation Metrics")
    print(f"{'='*40}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"{'='*40}\n")


def compare_models(results: dict) -> pd.DataFrame:
    """
    Compare multiple models based on their metrics.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        
    Returns:
        DataFrame comparing all models
    """
    comparison = pd.DataFrame(results).T
    comparison = comparison.round(4)
    return comparison
