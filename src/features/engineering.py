"""
Feature Engineering Module
Extract useful time-based features from the date column for forecasting.

Phase 1: Time-based features (year, month, day_of_week, etc.)
Phase 2: Lag features, rolling window features for ML models
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def create_features(df: pd.DataFrame, date_column: str = 'ds') -> pd.DataFrame:
    """
    Create time-based features from the date column.
    
    This function extracts useful temporal features that help:
    - Verify data understanding
    - Prepare data for ML models (Phase 2)
    - Capture seasonality patterns (weekly, monthly, yearly)
    
    Args:
        df: Input DataFrame with a datetime column
        date_column: Name of the date column (default: 'ds')
        
    Returns:
        DataFrame with new time-based feature columns added
        
    Raises:
        KeyError: If date column doesn't exist
        TypeError: If date column is not datetime type
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Validate date column exists
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found. Available columns: {list(df.columns)}")
    
    # Ensure column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise TypeError(f"Column '{date_column}' must be datetime type. Got: {df[date_column].dtype}")
    
    # Extract time-based features
    print(f"📅 Creating time-based features from '{date_column}'...")
    
    # Basic temporal features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek  # Monday=0, Sunday=6
    
    # Additional useful features
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    
    # Boolean features for patterns
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Saturday=5, Sunday=6
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    
    # Summary of features created
    new_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 
                    'week_of_year', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end']
    
    print(f"✅ Created {len(new_features)} features: {new_features}")
    
    return df


def create_basic_features(df: pd.DataFrame, date_column: str = 'ds') -> pd.DataFrame:
    """
    Create only the basic time-based features (minimal version).
    
    Creates: year, month, day_of_week
    
    Args:
        df: Input DataFrame with a datetime column
        date_column: Name of the date column (default: 'ds')
        
    Returns:
        DataFrame with basic time features added
    """
    df = df.copy()
    
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found.")
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise TypeError(f"Column '{date_column}' must be datetime type.")
    
    # Basic features as specified in requirements
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day_of_week'] = df[date_column].dt.dayofweek
    
    print(f"✅ Created basic features: ['year', 'month', 'day_of_week']")
    
    return df


# =============================================================================
# PHASE 2: ADVANCED FEATURE ENGINEERING
# =============================================================================

def create_lag_features(
    df: pd.DataFrame, 
    target_column: str = 'y', 
    lags: List[int] = [1, 7, 30]
) -> pd.DataFrame:
    """
    Create lag features representing past sales values.
    
    Lag features help ML models learn from historical patterns:
    - lag_1: Sales yesterday (short-term trend)
    - lag_7: Sales same day last week (weekly seasonality)
    - lag_30: Sales last month (monthly patterns)
    
    Args:
        df: Input DataFrame with target column
        target_column: Name of the target column (default: 'y')
        lags: List of lag periods to create (default: [1, 7, 30])
        
    Returns:
        DataFrame with lag feature columns added
        
    Note:
        Lag features create NaN values at the start of the data.
        Use drop_na_rows() to remove these before training.
    """
    df = df.copy()
    
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found.")
    
    print(f"📊 Creating lag features from '{target_column}'...")
    
    lag_features = []
    for lag in lags:
        col_name = f'sales_lag_{lag}'
        df[col_name] = df[target_column].shift(lag)
        lag_features.append(col_name)
    
    print(f"✅ Created {len(lag_features)} lag features: {lag_features}")
    
    return df


def create_rolling_features(
    df: pd.DataFrame, 
    target_column: str = 'y', 
    windows: List[int] = [7]
) -> pd.DataFrame:
    """
    Create rolling window features to capture trends and volatility.
    
    Rolling features help ML models understand:
    - rolling_mean: Average trend over the window
    - rolling_std: Volatility/variability over the window
    
    Args:
        df: Input DataFrame with target column
        target_column: Name of the target column (default: 'y')
        windows: List of window sizes (default: [7])
        
    Returns:
        DataFrame with rolling feature columns added
        
    Note:
        Rolling features create NaN values at the start of the data.
        Use drop_na_rows() to remove these before training.
    """
    df = df.copy()
    
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found.")
    
    print(f"📈 Creating rolling window features from '{target_column}'...")
    
    rolling_features = []
    for window in windows:
        # Rolling mean - captures average trend
        mean_col = f'rolling_mean_{window}'
        df[mean_col] = df[target_column].rolling(window=window).mean()
        rolling_features.append(mean_col)
        
        # Rolling std - captures volatility
        std_col = f'rolling_std_{window}'
        df[std_col] = df[target_column].rolling(window=window).std()
        rolling_features.append(std_col)
    
    print(f"✅ Created {len(rolling_features)} rolling features: {rolling_features}")
    
    return df


def drop_na_rows(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Drop rows with NaN values created by lag/rolling features.
    
    IMPORTANT: This must be called before training ML models!
    Lag and rolling features create NaN at the start of data.
    
    Args:
        df: Input DataFrame with potential NaN values
        verbose: Whether to print information (default: True)
        
    Returns:
        DataFrame with NaN rows removed
    """
    original_len = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    dropped = original_len - len(df_clean)
    
    if verbose:
        print(f"🧹 Dropped {dropped} rows with NaN values ({original_len} → {len(df_clean)} rows)")
    
    return df_clean


def create_advanced_features(
    df: pd.DataFrame,
    date_column: str = 'ds',
    target_column: str = 'y',
    lags: List[int] = [1, 7, 30],
    rolling_windows: List[int] = [7],
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Create all advanced features for ML models (Phase 2).
    
    This function combines:
    1. Time-based features (year, month, day_of_week, etc.)
    2. Lag features (sales_lag_1, sales_lag_7, sales_lag_30)
    3. Rolling features (rolling_mean_7, rolling_std_7)
    4. Optional NaN removal
    
    Args:
        df: Input DataFrame with date and target columns
        date_column: Name of date column (default: 'ds')
        target_column: Name of target column (default: 'y')
        lags: Lag periods to create (default: [1, 7, 30])
        rolling_windows: Rolling window sizes (default: [7])
        drop_na: Whether to drop NaN rows (default: True)
        
    Returns:
        DataFrame with all features, ready for ML training
    """
    print("\n" + "=" * 50)
    print("🚀 ADVANCED FEATURE ENGINEERING (Phase 2)")
    print("=" * 50)
    
    # Step 1: Time-based features
    df = create_features(df, date_column=date_column)
    
    # Step 2: Lag features
    df = create_lag_features(df, target_column=target_column, lags=lags)
    
    # Step 3: Rolling window features
    df = create_rolling_features(df, target_column=target_column, windows=rolling_windows)
    
    # Step 4: Handle NaN artifacts (IMPORTANT!)
    if drop_na:
        df = drop_na_rows(df)
    else:
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️  Warning: DataFrame has {nan_count} NaN values. Set drop_na=True before training!")
    
    # Summary
    feature_cols = [col for col in df.columns if col not in [date_column, target_column]]
    print(f"\n📊 Total features created: {len(feature_cols)}")
    print(f"   Final dataset size: {len(df)} rows")
    print("=" * 50)
    
    return df


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of the time-based features.
    
    Args:
        df: DataFrame with time features
        
    Returns:
        Summary DataFrame with statistics for each feature
    """
    feature_cols = ['year', 'month', 'day', 'day_of_week', 'day_of_year',
                    'week_of_year', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end']
    
    # Filter to only existing columns
    existing_features = [col for col in feature_cols if col in df.columns]
    
    if not existing_features:
        print("⚠️ No time features found in DataFrame")
        return pd.DataFrame()
    
    summary = df[existing_features].describe().T
    summary['unique_values'] = df[existing_features].nunique()
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Feature Engineering Module (Phase 1 & 2)")
    print("=" * 60)
    
    # Create sample data (already cleaned format with ds and y columns)
    # Need at least 30+ rows to demonstrate lag_30
    sample_data = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=60, freq='D'),
        'y': 100 + np.cumsum(np.random.randn(60) * 5)  # Random walk sales
    })
    
    print("\n📋 Input Data:")
    print(f"   Rows: {len(sample_data)}")
    print(f"   Date range: {sample_data['ds'].min()} to {sample_data['ds'].max()}")
    print(sample_data.head())
    
    # Test Phase 1: Basic time features
    print("\n" + "=" * 60)
    print("PHASE 1: Testing create_features()")
    print("=" * 60)
    df_phase1 = create_features(sample_data.copy())
    print(f"\n📋 Columns after Phase 1: {list(df_phase1.columns)}")
    
    # Test Phase 2: Advanced features
    print("\n" + "=" * 60)
    print("PHASE 2: Testing Advanced Features")
    print("=" * 60)
    
    # Test individual functions
    print("\n--- Testing create_lag_features() ---")
    df_lags = create_lag_features(sample_data.copy(), lags=[1, 7, 30])
    print(f"NaN count after lags: {df_lags.isna().sum().sum()}")
    
    print("\n--- Testing create_rolling_features() ---")
    df_rolling = create_rolling_features(sample_data.copy(), windows=[7])
    print(f"NaN count after rolling: {df_rolling.isna().sum().sum()}")
    
    # Test combined function
    print("\n--- Testing create_advanced_features() (ALL-IN-ONE) ---")
    df_advanced = create_advanced_features(
        sample_data.copy(),
        lags=[1, 7, 30],
        rolling_windows=[7],
        drop_na=True
    )
    
    print(f"\n📋 Final DataFrame:")
    print(f"   Columns: {list(df_advanced.columns)}")
    print(f"   Shape: {df_advanced.shape}")
    print(f"   NaN values: {df_advanced.isna().sum().sum()}")
    print(f"\n📋 Sample rows:")
    print(df_advanced.head())
