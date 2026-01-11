"""
Data Loader Module
Load CSV/Excel files and perform initial data loading operations.
"""

import pandas as pd
from pathlib import Path


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def load_excel(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """Load an Excel file into a pandas DataFrame."""
    return pd.read_excel(file_path, sheet_name=sheet_name)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV or Excel file based on extension.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        pandas DataFrame with loaded data
    """
    path = Path(file_path)
    
    if path.suffix.lower() == '.csv':
        return load_csv(file_path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return load_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def get_data_info(df: pd.DataFrame) -> dict:
    """Get basic information about the dataset."""
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
