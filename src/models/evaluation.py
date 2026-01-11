"""
Model Evaluation Module (Phase 2)
=================================
Scientific evaluation metrics to compare forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List, Optional, Any


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List]
) -> Dict[str, Union[float, str]]:
    """
    Calculate comprehensive evaluation metrics for forecasting models.
    
    Metrics calculated:
    - MAE (Mean Absolute Error): Average error magnitude
    - RMSE (Root Mean Squared Error): Penalizes large errors more heavily
    - MAPE (Mean Absolute Percentage Error): Error as percentage (easy to interpret)
    - R² (R-squared): Proportion of variance explained
    
    Args:
        y_true: Actual sales values
        y_pred: Forecasted/predicted sales values
        
    Returns:
        Dictionary with metrics: {'mae': 10.5, 'rmse': 12.3, 'mape': '5.2%', 'r2': 0.85}
        
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    # Convert to numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Arrays cannot be empty")
    
    # Calculate MAE (Mean Absolute Error)
    # Average of absolute differences - easy to interpret
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate RMSE (Root Mean Squared Error)
    # Square root of average squared errors - penalizes large errors
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Error as percentage - most intuitive for business users
    # Handle zero values to avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # Calculate R² (R-squared / Coefficient of Determination)
    # 1.0 = perfect, 0.0 = as good as mean, negative = worse than mean
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'mape': f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
        'mape_value': round(mape, 4) if not np.isnan(mape) else np.nan,
        'r2': round(r2, 4) if not np.isnan(r2) else np.nan
    }


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Interpretation:
    - On average, predictions are off by this amount
    - Same units as target variable (e.g., dollars, units)
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    RMSE = √[(1/n) * Σ(y_true - y_pred)²]
    
    Interpretation:
    - Penalizes large errors more than MAE
    - Same units as target variable
    - RMSE ≥ MAE always (equality when all errors are equal)
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
    
    Interpretation:
    - Error as percentage (easy for business users)
    - Scale-independent (can compare across different products)
    - Warning: Undefined when y_true = 0
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (Coefficient of Determination).
    
    R² = 1 - (SS_res / SS_tot)
    
    Interpretation:
    - 1.0 = Perfect predictions
    - 0.0 = As good as predicting the mean
    - < 0 = Worse than predicting the mean
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def print_metrics(metrics: Dict[str, Union[float, str]], model_name: str = "Model") -> None:
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary from calculate_metrics()
        model_name: Name of the model for display
    """
    print(f"\n📊 {model_name} Evaluation Metrics:")
    print("-" * 40)
    print(f"   MAE:  {metrics['mae']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAPE: {metrics['mape']}")
    print(f"   R²:   {metrics['r2']:.4f}" if metrics['r2'] is not np.nan else "   R²:   N/A")
    print("-" * 40)


def compare_model_metrics(
    models_results: Dict[str, Dict[str, Union[float, str]]]
) -> pd.DataFrame:
    """
    Compare metrics across multiple models.
    
    Args:
        models_results: Dictionary mapping model names to their metrics
                       e.g., {'ARIMA': metrics1, 'XGBoost': metrics2}
                       
    Returns:
        DataFrame comparing all models
    """
    comparison_data = []
    
    for model_name, metrics in models_results.items():
        comparison_data.append({
            'Model': model_name,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'MAPE': metrics.get('mape_value', metrics.get('mape', np.nan)),
            'R²': metrics['r2']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by RMSE (lower is better)
    df = df.sort_values('RMSE').reset_index(drop=True)
    
    # Add rank column
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df


def evaluate_forecast(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    model_name: str = "Model",
    verbose: bool = True
) -> Dict[str, Union[float, str]]:
    """
    Complete evaluation of a forecast with optional printing.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name for display
        verbose: Whether to print results
        
    Returns:
        Metrics dictionary
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    if verbose:
        print_metrics(metrics, model_name)
    
    return metrics


def get_best_model(
    models_results: Dict[str, Dict[str, Union[float, str]]],
    metric: str = 'rmse'
) -> str:
    """
    Determine the best model based on a specific metric.
    
    Args:
        models_results: Dictionary mapping model names to metrics
        metric: Metric to use for comparison ('mae', 'rmse', 'mape_value', 'r2')
        
    Returns:
        Name of the best performing model
    """
    if metric == 'r2':
        # Higher is better for R²
        best_model = max(models_results.keys(), 
                        key=lambda x: models_results[x].get(metric, -np.inf))
    else:
        # Lower is better for MAE, RMSE, MAPE
        best_model = min(models_results.keys(), 
                        key=lambda x: models_results[x].get(metric, np.inf))
    
    return best_model


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Evaluation Module")
    print("=" * 60)
    
    # Create sample actual and predicted values
    np.random.seed(42)
    y_true = np.array([100, 120, 115, 130, 125, 140, 135, 150, 145, 160])
    
    # Simulate predictions from different models
    y_pred_good = y_true + np.random.randn(10) * 5      # Good model (small errors)
    y_pred_medium = y_true + np.random.randn(10) * 15   # Medium model
    y_pred_bad = y_true + np.random.randn(10) * 30      # Bad model (large errors)
    
    print("\n📋 Sample Data:")
    print(f"   Actual values: {y_true[:5]}...")
    print(f"   Good predictions: {y_pred_good[:5].round(1)}...")
    print(f"   Bad predictions: {y_pred_bad[:5].round(1)}...")
    
    # Test calculate_metrics
    print("\n" + "=" * 60)
    print("Testing calculate_metrics()")
    print("=" * 60)
    
    metrics_good = calculate_metrics(y_true, y_pred_good)
    metrics_medium = calculate_metrics(y_true, y_pred_medium)
    metrics_bad = calculate_metrics(y_true, y_pred_bad)
    
    print_metrics(metrics_good, "Good Model")
    print_metrics(metrics_medium, "Medium Model")
    print_metrics(metrics_bad, "Bad Model")
    
    # Test model comparison
    print("\n" + "=" * 60)
    print("Testing compare_model_metrics()")
    print("=" * 60)
    
    all_results = {
        'Good Model': metrics_good,
        'Medium Model': metrics_medium,
        'Bad Model': metrics_bad
    }
    
    comparison_df = compare_model_metrics(all_results)
    print("\n📊 Model Comparison (sorted by RMSE):")
    print(comparison_df.to_string(index=False))
    
    # Test get_best_model
    print("\n" + "=" * 60)
    print("Testing get_best_model()")
    print("=" * 60)
    
    best_by_rmse = get_best_model(all_results, 'rmse')
    best_by_mape = get_best_model(all_results, 'mape_value')
    best_by_r2 = get_best_model(all_results, 'r2')
    
    print(f"   Best by RMSE: {best_by_rmse}")
    print(f"   Best by MAPE: {best_by_mape}")
    print(f"   Best by R²: {best_by_r2}")
    
    print("\n✅ Evaluation module tests complete!")
