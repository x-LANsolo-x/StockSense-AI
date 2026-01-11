"""
Baseline Model Module
ARIMA (AutoRegressive Integrated Moving Average) baseline model for time series forecasting.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress convergence warnings during fitting
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def train_model(data: pd.Series, order: tuple = (5, 1, 0)) -> object:
    """
    Train an ARIMA model on time-series data.
    
    ARIMA Parameters (p, d, q):
    - p (5): Number of lag observations (autoregressive term)
    - d (1): Degree of differencing (to make series stationary)
    - q (0): Size of moving average window
    
    Args:
        data: Time series data (Sales column) as pandas Series
        order: ARIMA order tuple (p, d, q). Default: (5, 1, 0)
        
    Returns:
        Fitted ARIMA model object (ARIMAResults)
        
    Raises:
        ValueError: If data is empty or has insufficient observations
    """
    # Validate input
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    if len(data) < 10:
        raise ValueError(f"Insufficient data points ({len(data)}). Need at least 10 observations.")
    
    print(f"🔧 Training ARIMA model with order={order}...")
    print(f"   - Data points: {len(data)}")
    print(f"   - Date range: {data.index.min()} to {data.index.max()}" if hasattr(data.index, 'min') else "")
    
    # Initialize ARIMA model
    model = ARIMA(data, order=order)
    
    # Fit the model
    model_fit = model.fit()
    
    print(f"✅ Model trained successfully!")
    print(f"   - AIC: {model_fit.aic:.2f}")
    print(f"   - BIC: {model_fit.bic:.2f}")
    
    return model_fit


def make_forecast(model_fit: object, steps: int) -> pd.Series:
    """
    Generate forecasts using a trained ARIMA model.
    
    Args:
        model_fit: Trained ARIMA model (ARIMAResults object)
        steps: Number of steps (days) to forecast
        
    Returns:
        Pandas Series with forecasted values
        
    Raises:
        ValueError: If steps is not positive
    """
    if steps <= 0:
        raise ValueError(f"Steps must be positive. Got: {steps}")
    
    print(f"🔮 Forecasting {steps} steps ahead...")
    
    # Generate forecast
    forecast = model_fit.forecast(steps=steps)
    
    print(f"✅ Forecast complete!")
    print(f"   - Forecast range: {forecast.min():.2f} to {forecast.max():.2f}")
    print(f"   - Mean forecast: {forecast.mean():.2f}")
    
    return forecast


def train_and_forecast(data: pd.Series, steps: int, order: tuple = (5, 1, 0)) -> tuple:
    """
    Convenience function to train model and generate forecast in one step.
    
    Args:
        data: Time series data (Sales column)
        steps: Number of steps to forecast
        order: ARIMA order tuple. Default: (5, 1, 0)
        
    Returns:
        Tuple of (model_fit, forecast)
    """
    model_fit = train_model(data, order=order)
    forecast = make_forecast(model_fit, steps=steps)
    return model_fit, forecast


def get_model_summary(model_fit: object) -> str:
    """
    Get a summary of the trained ARIMA model.
    
    Args:
        model_fit: Trained ARIMA model
        
    Returns:
        String containing model summary
    """
    return model_fit.summary().as_text()


def get_forecast_with_confidence(model_fit: object, steps: int, alpha: float = 0.05) -> pd.DataFrame:
    """
    Generate forecast with confidence intervals.
    
    Args:
        model_fit: Trained ARIMA model
        steps: Number of steps to forecast
        alpha: Significance level for confidence interval (default: 0.05 for 95% CI)
        
    Returns:
        DataFrame with columns: forecast, lower_ci, upper_ci
    """
    # Get forecast with confidence intervals
    forecast_result = model_fit.get_forecast(steps=steps)
    
    # Extract values
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=alpha)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'forecast': forecast,
        'lower_ci': conf_int.iloc[:, 0],
        'upper_ci': conf_int.iloc[:, 1]
    })
    
    confidence_pct = int((1 - alpha) * 100)
    print(f"✅ Generated forecast with {confidence_pct}% confidence intervals")
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Baseline ARIMA Model")
    print("=" * 60)
    
    # Create sample time series data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Simulate sales data with trend and noise
    trend = np.linspace(100, 150, 100)
    noise = np.random.normal(0, 10, 100)
    sales = trend + noise
    
    # Create Series with date index
    data = pd.Series(sales, index=dates, name='Sales')
    
    print("\n📋 Sample Data:")
    print(f"   - Total observations: {len(data)}")
    print(f"   - Date range: {data.index.min()} to {data.index.max()}")
    print(f"   - Sales range: {data.min():.2f} to {data.max():.2f}")
    
    # Test train_model
    print("\n" + "=" * 60)
    print("Testing train_model()")
    print("=" * 60)
    model_fit = train_model(data)
    
    # Test make_forecast
    print("\n" + "=" * 60)
    print("Testing make_forecast()")
    print("=" * 60)
    forecast = make_forecast(model_fit, steps=7)
    
    print("\n📋 7-Day Forecast:")
    print(forecast)
    
    # Test forecast with confidence intervals
    print("\n" + "=" * 60)
    print("Testing get_forecast_with_confidence()")
    print("=" * 60)
    forecast_ci = get_forecast_with_confidence(model_fit, steps=7)
    print("\n📋 Forecast with 95% Confidence Intervals:")
    print(forecast_ci)
