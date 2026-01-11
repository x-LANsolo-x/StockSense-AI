"""
Data Cleaner Module
Clean and preprocess raw data for analysis.
"""

import pandas as pd
import numpy as np


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop', 'mean', 'median', 'ffill', 'bfill'
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'ffill':
        return df.fillna(method='ffill')
    elif strategy == 'bfill':
        return df.fillna(method='bfill')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates()


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from a specific column.
    
    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        method: 'iqr' (Interquartile Range) or 'zscore'
        
    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column]))
        return df[z_scores < 3]
    
    else:
        raise ValueError(f"Unknown method: {method}")


def convert_date_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert a column to datetime format."""
    df = df.copy()
    df[column] = pd.to_datetime(df[column])
    return df


def clean_data(df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
    """
    Perform full data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column (optional)
        
    Returns:
        Cleaned DataFrame
    """
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Convert date column if specified
    if date_column:
        df = convert_date_column(df, date_column)
        df = df.sort_values(by=date_column)
    
    # Handle missing values
    df = handle_missing_values(df, strategy='ffill')
    
    return df.reset_index(drop=True)
