"""
Machine Learning Models Module (Phase 2)
========================================
XGBoost and Prophet models for advanced sales forecasting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# DATA PREPARATION UTILITIES
# =============================================================================

def prepare_features_target(
    df: pd.DataFrame,
    target_column: str = 'y',
    date_column: str = 'ds',
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix (X) and target vector (y) for ML models.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column (default: 'y')
        date_column: Name of date column to exclude (default: 'ds')
        exclude_columns: Additional columns to exclude from features
        
    Returns:
        Tuple of (X features DataFrame, y target Series)
        
    Note:
        Automatically excludes non-numeric columns (strings, objects, etc.)
        as XGBoost and other ML models require numeric input.
    """
    exclude = [target_column, date_column]
    if exclude_columns:
        exclude.extend(exclude_columns)
    
    # Get feature columns (exclude target, date, and non-numeric columns)
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']
    feature_cols = [
        col for col in df.columns 
        if col not in exclude and df[col].dtype.name in numeric_types
    ]
    
    # Also exclude any remaining object/string columns
    feature_cols = [
        col for col in feature_cols 
        if df[col].dtype != 'object' and not pd.api.types.is_categorical_dtype(df[col])
    ]
    
    X = df[feature_cols]
    y = df[target_column]
    
    return X, y


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into training and validation sets.
    
    IMPORTANT: Time series data must NOT be shuffled randomly!
    We use the oldest data for training and newest for validation.
    
    Args:
        df: DataFrame sorted by date
        train_ratio: Proportion of data for training (default: 0.8 = 80%)
        
    Returns:
        Tuple of (train_df, val_df)
    """
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"📊 Time Series Split:")
    print(f"   - Training set: {len(train_df)} rows ({train_ratio*100:.0f}%)")
    print(f"   - Validation set: {len(val_df)} rows ({(1-train_ratio)*100:.0f}%)")
    
    return train_df, val_df


# =============================================================================
# XGBOOST MODEL
# =============================================================================

def train_xgboost(
    df: pd.DataFrame,
    target_column: str = 'y',
    date_column: str = 'ds',
    train_ratio: float = 0.8,
    params: Optional[Dict[str, Any]] = None,
    return_validation: bool = True
) -> Tuple[Any, Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    Train an XGBoost model for sales forecasting.
    
    XGBoost is a powerful gradient boosting algorithm that works well
    with tabular data and engineered features.
    
    Args:
        df: DataFrame with features (lags, rolling, date parts) and target
        target_column: Name of target column (default: 'y')
        date_column: Name of date column (default: 'ds')
        train_ratio: Proportion for training (default: 0.8)
        params: XGBoost hyperparameters (optional)
        return_validation: Whether to return validation data and metrics
        
    Returns:
        Tuple of (trained model, validation_df, metrics_dict)
        - If return_validation=False, returns (model, None, None)
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print("\n" + "=" * 50)
    print("🌲 TRAINING XGBOOST MODEL")
    print("=" * 50)
    
    # Step 1: Split data (IMPORTANT: no random shuffle for time series!)
    train_df, val_df = time_series_split(df, train_ratio=train_ratio)
    
    # Step 2: Prepare features and target
    X_train, y_train = prepare_features_target(train_df, target_column, date_column)
    X_val, y_val = prepare_features_target(val_df, target_column, date_column)
    
    print(f"\n📋 Features used ({len(X_train.columns)}):")
    print(f"   {list(X_train.columns)}")
    
    # Step 3: Initialize XGBRegressor with default or custom params
    default_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    if params:
        default_params.update(params)
    
    model = XGBRegressor(**default_params)
    
    # Step 4: Fit model
    print(f"\n🔧 Training with {len(X_train)} samples...")
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")
    
    # Step 5: Evaluate on validation set (if requested)
    metrics = None
    if return_validation:
        y_pred = model.predict(X_val)
        
        metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
        
        print(f"\n📊 Validation Metrics:")
        print(f"   - MAE:  {metrics['mae']:.4f}")
        print(f"   - RMSE: {metrics['rmse']:.4f}")
        print(f"   - R²:   {metrics['r2']:.4f}")
        
        # Add predictions to validation df
        val_df = val_df.copy()
        val_df['y_pred'] = y_pred
    
    print("=" * 50)
    
    if return_validation:
        return model, val_df, metrics
    return model, None, None


def predict_xgboost(
    model: Any,
    df: pd.DataFrame,
    date_column: str = 'ds',
    target_column: str = 'y'
) -> np.ndarray:
    """
    Generate predictions using a trained XGBoost model.
    
    Args:
        model: Trained XGBRegressor
        df: DataFrame with features
        date_column: Date column to exclude
        target_column: Target column to exclude
        
    Returns:
        Array of predictions
    """
    X, _ = prepare_features_target(df, target_column, date_column)
    predictions = model.predict(X)
    return predictions


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from trained XGBoost model.
    
    Args:
        model: Trained XGBRegressor
        feature_names: List of feature names
        
    Returns:
        DataFrame with features sorted by importance
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance


# =============================================================================
# PROPHET MODEL
# =============================================================================

def train_prophet(
    df: pd.DataFrame,
    date_column: str = 'ds',
    target_column: str = 'y',
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    additional_params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Train a Prophet model for sales forecasting.
    
    Prophet is Facebook's open-source library designed specifically for
    time series forecasting with strong seasonality.
    
    Args:
        df: DataFrame with date and target columns
        date_column: Name of date column (default: 'ds')
        target_column: Name of target column (default: 'y')
        yearly_seasonality: Enable yearly patterns (default: True)
        weekly_seasonality: Enable weekly patterns (default: True)
        daily_seasonality: Enable daily patterns (default: False)
        additional_params: Additional Prophet parameters
        
    Returns:
        Trained Prophet model
        
    Note:
        Prophet requires exactly two columns named 'ds' (date) and 'y' (target)
    """
    from prophet import Prophet
    
    print("\n" + "=" * 50)
    print("📈 TRAINING PROPHET MODEL")
    print("=" * 50)
    
    # Step 1: Prepare DataFrame with exactly ds and y columns
    prophet_df = df[[date_column, target_column]].copy()
    
    # Rename columns if needed (Prophet requires 'ds' and 'y')
    if date_column != 'ds':
        prophet_df = prophet_df.rename(columns={date_column: 'ds'})
    if target_column != 'y':
        prophet_df = prophet_df.rename(columns={target_column: 'y'})
    
    print(f"📋 Data prepared:")
    print(f"   - Rows: {len(prophet_df)}")
    print(f"   - Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    
    # Step 2: Initialize Prophet with seasonality settings
    model_params = {
        'yearly_seasonality': yearly_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'daily_seasonality': daily_seasonality
    }
    
    if additional_params:
        model_params.update(additional_params)
    
    model = Prophet(**model_params)
    
    print(f"\n🔧 Training Prophet model...")
    print(f"   - Yearly seasonality: {yearly_seasonality}")
    print(f"   - Weekly seasonality: {weekly_seasonality}")
    print(f"   - Daily seasonality: {daily_seasonality}")
    
    # Step 3: Fit model
    model.fit(prophet_df)
    print("✅ Model trained successfully!")
    print("=" * 50)
    
    return model


def predict_prophet(
    model: Any,
    periods: int,
    freq: str = 'D',
    include_history: bool = False
) -> pd.DataFrame:
    """
    Generate forecasts using a trained Prophet model.
    
    Args:
        model: Trained Prophet model
        periods: Number of future periods to forecast
        freq: Frequency of predictions ('D' for daily, 'W' for weekly)
        include_history: Whether to include historical predictions
        
    Returns:
        DataFrame with forecast, trend, and seasonality components
    """
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    if not include_history:
        # Only keep future dates
        future = future.tail(periods)
    
    # Generate forecast
    forecast = model.predict(future)
    
    print(f"🔮 Generated {periods}-step forecast")
    
    return forecast


def get_prophet_components(model: Any, forecast: pd.DataFrame) -> None:
    """
    Plot Prophet forecast components (trend, seasonality).
    
    Args:
        model: Trained Prophet model
        forecast: Forecast DataFrame from predict_prophet
    """
    try:
        from prophet.plot import plot_components
        import matplotlib.pyplot as plt
        
        fig = model.plot_components(forecast)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("⚠️ matplotlib required for plotting")


# =============================================================================
# MODEL COMPARISON UTILITY
# =============================================================================

def compare_models(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    forecast_steps: int = 7
) -> pd.DataFrame:
    """
    Train and compare XGBoost and Prophet models.
    
    Args:
        df: DataFrame with features and target
        train_ratio: Training data proportion
        forecast_steps: Number of steps to forecast
        
    Returns:
        DataFrame comparing model performance
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print("\n" + "=" * 60)
    print("🏆 MODEL COMPARISON")
    print("=" * 60)
    
    results = []
    
    # Split data
    train_df, val_df = time_series_split(df, train_ratio)
    
    # Train XGBoost
    try:
        xgb_model, xgb_val_df, xgb_metrics = train_xgboost(df, train_ratio=train_ratio)
        results.append({
            'Model': 'XGBoost',
            'MAE': xgb_metrics['mae'],
            'RMSE': xgb_metrics['rmse'],
            'R²': xgb_metrics['r2']
        })
    except Exception as e:
        print(f"⚠️ XGBoost failed: {e}")
    
    # Train Prophet
    try:
        prophet_model = train_prophet(train_df)
        prophet_forecast = predict_prophet(prophet_model, periods=len(val_df))
        
        y_true = val_df['y'].values
        y_pred = prophet_forecast['yhat'].values[-len(val_df):]
        
        results.append({
            'Model': 'Prophet',
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R²': r2_score(y_true, y_pred)
        })
    except Exception as e:
        print(f"⚠️ Prophet failed: {e}")
    
    comparison_df = pd.DataFrame(results)
    
    print("\n📊 Comparison Results:")
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    
    return comparison_df


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from features.engineering import create_advanced_features
    
    print("=" * 60)
    print("Testing ML Models Module")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
    trend = np.linspace(100, 150, 120)
    weekly = 15 * np.sin(np.arange(120) * 2 * np.pi / 7)
    noise = np.random.randn(120) * 5
    sales = trend + weekly + noise
    
    df = pd.DataFrame({'ds': dates, 'y': sales})
    print(f"\n📋 Sample data: {len(df)} rows")
    
    # Create features
    df_features = create_advanced_features(df, lags=[1, 7, 30], rolling_windows=[7], drop_na=True)
    print(f"📋 After feature engineering: {len(df_features)} rows")
    
    # Test XGBoost
    print("\n" + "=" * 60)
    print("Testing XGBoost")
    print("=" * 60)
    xgb_model, val_df, metrics = train_xgboost(df_features)
    
    # Show feature importance
    feature_cols = [col for col in df_features.columns if col not in ['ds', 'y']]
    importance = get_feature_importance(xgb_model, feature_cols)
    print("\n📊 Top 5 Important Features:")
    print(importance.head())
    
    # Test Prophet
    print("\n" + "=" * 60)
    print("Testing Prophet")
    print("=" * 60)
    try:
        prophet_model = train_prophet(df)
        forecast = predict_prophet(prophet_model, periods=7)
        print("\n📋 7-Day Forecast:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))
    except ImportError:
        print("⚠️ Prophet not installed. Install with: pip install prophet")
    
    print("\n✅ ML Models tests complete!")
