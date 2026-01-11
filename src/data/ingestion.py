"""
Data Ingestion Module
Load CSV files and clean data for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file to load
        
    Returns:
        pandas DataFrame with the loaded data
        
    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} rows from {file_path}")
    return df


def clean_data(
    df: pd.DataFrame,
    date_column: str = 'Date',
    sales_column: str = 'Sales',
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Clean and prepare data for time series forecasting.
    
    This function:
    1. Converts the date column to datetime objects
    2. Sorts data by date (crucial for time series)
    3. Fills missing sales values
    4. Renames columns to standard format (ds for date, y for sales)
    
    Args:
        df: Input DataFrame with sales data
        date_column: Name of the date column (default: 'Date')
        sales_column: Name of the sales column (default: 'Sales')
        fill_method: Method to fill missing values - 'ffill' (forward fill), 
                     'bfill' (backward fill), 'zero', or 'mean' (default: 'ffill')
        
    Returns:
        Cleaned DataFrame with columns renamed to 'ds' (date) and 'y' (sales)
        
    Raises:
        KeyError: If specified columns don't exist in the DataFrame
        ValueError: If date column cannot be converted to datetime
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Validate columns exist
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found. Available columns: {list(df.columns)}")
    if sales_column not in df.columns:
        raise KeyError(f"Sales column '{sales_column}' not found. Available columns: {list(df.columns)}")
    
    # Step 1: Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    print(f"✅ Converted '{date_column}' to datetime")
    
    # Step 2: Sort by date (crucial for time series!)
    df = df.sort_values(by=date_column).reset_index(drop=True)
    print(f"✅ Sorted data by '{date_column}'")
    
    # Step 3: Handle missing values in sales column
    missing_count = df[sales_column].isna().sum()
    if missing_count > 0:
        if fill_method == 'ffill':
            # Forward fill: use previous day's value
            df[sales_column] = df[sales_column].ffill()
            # If first values are NaN, backfill them
            df[sales_column] = df[sales_column].bfill()
        elif fill_method == 'bfill':
            # Backward fill: use next day's value
            df[sales_column] = df[sales_column].bfill()
            df[sales_column] = df[sales_column].ffill()
        elif fill_method == 'zero':
            # Fill with 0
            df[sales_column] = df[sales_column].fillna(0)
        elif fill_method == 'mean':
            # Fill with column mean
            df[sales_column] = df[sales_column].fillna(df[sales_column].mean())
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}. Use 'ffill', 'bfill', 'zero', or 'mean'")
        
        print(f"✅ Filled {missing_count} missing values using '{fill_method}' method")
    else:
        print("✅ No missing values in sales column")
    
    # Step 4: Rename columns to standard format (ds for date, y for sales)
    # Keep other columns as they are
    df = df.rename(columns={
        date_column: 'ds',
        sales_column: 'y'
    })
    print(f"✅ Renamed columns: '{date_column}' → 'ds', '{sales_column}' → 'y'")
    
    # Final summary
    print(f"\n📊 Data Summary:")
    print(f"   - Total rows: {len(df)}")
    print(f"   - Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"   - Sales range: {df['y'].min():.2f} to {df['y'].max():.2f}")
    print(f"   - Columns: {list(df.columns)}")
    
    return df


def load_and_clean_data(
    file_path: str,
    date_column: str = 'Date',
    sales_column: str = 'Sales',
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Convenience function to load and clean data in one step.
    
    Args:
        file_path: Path to the CSV file to load
        date_column: Name of the date column (default: 'Date')
        sales_column: Name of the sales column (default: 'Sales')
        fill_method: Method to fill missing values (default: 'ffill')
        
    Returns:
        Cleaned DataFrame ready for forecasting
    """
    df = load_data(file_path)
    df_clean = clean_data(df, date_column, sales_column, fill_method)
    return df_clean


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    print("=" * 60)
    print("Testing Data Ingestion Module")
    print("=" * 60)
    
    # Create a sample CSV file for testing
    sample_data = pd.DataFrame({
        'Date': ['2024-01-05', '2024-01-01', '2024-01-03', '2024-01-02', '2024-01-04'],
        'Sales': [150.0, 100.0, np.nan, 120.0, 140.0],
        'Store': ['A', 'A', 'A', 'A', 'A']
    })
    
    # Save sample data
    sample_path = 'data/raw/sample_sales.csv'
    import os
    os.makedirs('data/raw', exist_ok=True)
    sample_data.to_csv(sample_path, index=False)
    print(f"\n📝 Created sample data at {sample_path}")
    print(f"   Original data (unsorted, with missing values):")
    print(sample_data.to_string(index=False))
    
    # Test load_data
    print("\n" + "=" * 60)
    print("Testing load_data()")
    print("=" * 60)
    df = load_data(sample_path)
    
    # Test clean_data
    print("\n" + "=" * 60)
    print("Testing clean_data()")
    print("=" * 60)
    df_clean = clean_data(df, date_column='Date', sales_column='Sales')
    
    print("\n📋 Final cleaned data:")
    print(df_clean.to_string(index=False))
    
    # Cleanup test file
    os.remove(sample_path)
    print(f"\n🧹 Removed test file: {sample_path}")
