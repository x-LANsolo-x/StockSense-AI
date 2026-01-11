"""
Moving Average Model Module
Simple and weighted moving average forecasting models.
"""

import pandas as pd
import numpy as np


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate simple moving average.
    
    Args:
        series: Time series data
        window: Window size for moving average
        
    Returns:
        Series of moving averages
    """
    return series.rolling(window=window).mean()


def weighted_moving_average(series: pd.Series, weights: list) -> pd.Series:
    """
    Calculate weighted moving average.
    
    Args:
        series: Time series data
        weights: List of weights (should sum to 1)
        
    Returns:
        Series of weighted moving averages
    """
    weights = np.array(weights)
    return series.rolling(window=len(weights)).apply(lambda x: np.sum(weights * x))


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate exponential moving average.
    
    Args:
        series: Time series data
        span: Span for EMA calculation
        
    Returns:
        Series of exponential moving averages
    """
    return series.ewm(span=span).mean()


def forecast_sma(series: pd.Series, window: int, steps: int) -> pd.Series:
    """
    Forecast using Simple Moving Average.
    
    Args:
        series: Time series data
        window: Window size
        steps: Number of steps to forecast
        
    Returns:
        Series of forecasted values
    """
    last_values = series.tail(window)
    forecast_value = last_values.mean()
    
    # Create forecast index
    last_index = series.index[-1]
    if isinstance(last_index, pd.Timestamp):
        forecast_index = pd.date_range(start=last_index + pd.Timedelta(days=1), periods=steps)
    else:
        forecast_index = range(last_index + 1, last_index + steps + 1)
    
    return pd.Series([forecast_value] * steps, index=forecast_index)


class MovingAverageForecaster:
    """Moving Average Forecasting class for retail sales prediction."""
    
    def __init__(self, window: int = 7, method: str = 'simple'):
        """
        Initialize Moving Average Forecaster.
        
        Args:
            window: Window size for moving average
            method: 'simple', 'weighted', or 'exponential'
        """
        self.window = window
        self.method = method
        self.series = None
        self.is_fitted = False
    
    def fit(self, series: pd.Series):
        """Fit the model (store the series)."""
        self.series = series
        self.is_fitted = True
        return self
    
    def predict(self, steps: int) -> pd.Series:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.method == 'simple':
            return forecast_sma(self.series, self.window, steps)
        elif self.method == 'exponential':
            # For EMA, use last EMA value as forecast
            ema = exponential_moving_average(self.series, self.window)
            forecast_value = ema.iloc[-1]
            last_index = self.series.index[-1]
            if isinstance(last_index, pd.Timestamp):
                forecast_index = pd.date_range(start=last_index + pd.Timedelta(days=1), periods=steps)
            else:
                forecast_index = range(last_index + 1, last_index + steps + 1)
            return pd.Series([forecast_value] * steps, index=forecast_index)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_moving_average(self) -> pd.Series:
        """Get the moving average series."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if self.method == 'simple':
            return simple_moving_average(self.series, self.window)
        elif self.method == 'exponential':
            return exponential_moving_average(self.series, self.window)
        else:
            raise ValueError(f"Unknown method: {self.method}")
