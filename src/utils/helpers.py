"""
Helper Functions Module
Utility functions for the forecasting project.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def save_dataframe(df: pd.DataFrame, file_path: str, index: bool = False):
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file
        index: Whether to include index
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index)
    print(f"Saved: {file_path}")


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    return pd.read_csv(file_path)


def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Split time series data into train and test sets.
    
    Args:
        df: DataFrame (should be sorted by date)
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def print_data_summary(df: pd.DataFrame, name: str = "Dataset"):
    """Print a summary of the DataFrame."""
    print(f"\n{'='*50}")
    print(f"{name} Summary")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nBasic Statistics:")
    print(df.describe())
    print(f"{'='*50}\n")


def create_date_range(start_date: str, periods: int, freq: str = 'D') -> pd.DatetimeIndex:
    """
    Create a date range for forecasting.
    
    Args:
        start_date: Start date string
        periods: Number of periods
        freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        
    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def format_predictions(predictions: pd.Series, date_column: str = 'date') -> pd.DataFrame:
    """
    Format predictions into a DataFrame.
    
    Args:
        predictions: Series of predictions
        date_column: Name for the date column
        
    Returns:
        DataFrame with date and prediction columns
    """
    return pd.DataFrame({
        date_column: predictions.index,
        'predicted_sales': predictions.values
    })
