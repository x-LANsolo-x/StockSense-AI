"""
Feature Engineering Module
Create features for time series forecasting.
"""

import pandas as pd
import numpy as np


def create_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create date-based features from a datetime column.
    
    Args:
        df: Input DataFrame
        date_column: Name of the datetime column
        
    Returns:
        DataFrame with new date features
    """
    df = df.copy()
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['quarter'] = df[date_column].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, target_column: str, lags: list = [1, 7, 14, 30]) -> pd.DataFrame:
    """
    Create lag features for the target variable.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, target_column: str, windows: list = [7, 14, 30]) -> pd.DataFrame:
    """
    Create rolling window statistics features.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    for window in windows:
        df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
        df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
        df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window=window).min()
        df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window=window).max()
    
    return df


def create_expanding_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Create expanding window features.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        DataFrame with expanding features
    """
    df = df.copy()
    
    df[f'{target_column}_expanding_mean'] = df[target_column].expanding().mean()
    df[f'{target_column}_expanding_std'] = df[target_column].expanding().std()
    
    return df


def engineer_features(df: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.
    
    Args:
        df: Input DataFrame
        date_column: Name of the datetime column
        target_column: Name of the target column
        
    Returns:
        DataFrame with all engineered features
    """
    # Create date features
    df = create_date_features(df, date_column)
    
    # Create lag features
    df = create_lag_features(df, target_column, lags=[1, 7, 14, 30])
    
    # Create rolling features
    df = create_rolling_features(df, target_column, windows=[7, 14, 30])
    
    # Create expanding features
    df = create_expanding_features(df, target_column)
    
    return df
