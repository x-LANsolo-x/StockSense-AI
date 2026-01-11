"""
ARIMA Model Module
ARIMA (AutoRegressive Integrated Moving Average) forecasting model.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


def check_stationarity(series: pd.Series) -> dict:
    """
    Check if a time series is stationary using ADF test.
    
    Args:
        series: Time series data
        
    Returns:
        Dictionary with test results
    """
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05
    }


def fit_arima(series: pd.Series, order: tuple = (1, 1, 1)) -> ARIMA:
    """
    Fit an ARIMA model to the time series.
    
    Args:
        series: Time series data
        order: ARIMA order (p, d, q)
        
    Returns:
        Fitted ARIMA model
    """
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    return fitted_model


def predict_arima(model, steps: int) -> pd.Series:
    """
    Generate predictions from an ARIMA model.
    
    Args:
        model: Fitted ARIMA model
        steps: Number of steps to forecast
        
    Returns:
        Series of predictions
    """
    forecast = model.forecast(steps=steps)
    return forecast


def get_model_summary(model) -> str:
    """Get summary of the fitted ARIMA model."""
    return model.summary().as_text()


class ARIMAForecaster:
    """ARIMA Forecasting class for retail sales prediction."""
    
    def __init__(self, order: tuple = (1, 1, 1)):
        """
        Initialize ARIMA Forecaster.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.is_fitted = False
    
    def fit(self, series: pd.Series):
        """Fit the ARIMA model."""
        self.model = fit_arima(series, self.order)
        self.is_fitted = True
        return self
    
    def predict(self, steps: int) -> pd.Series:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return predict_arima(self.model, steps)
    
    def get_summary(self) -> str:
        """Get model summary."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return get_model_summary(self.model)
