"""
StockSense-AI Configuration Settings
====================================
Centralized configuration for the StockSense-AI system.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "champion_model.pkl"
ARIMA_MODEL_PATH = MODELS_DIR / "arima_model.pkl"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
PROPHET_MODEL_PATH = MODELS_DIR / "prophet_model.pkl"

# Default data file
DEFAULT_DATA_FILE = RAW_DATA_PATH / "sales.csv"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Forecasting settings
FORECAST_HORIZON = 30  # Default number of days to forecast
TEST_DAYS = 30         # Days to reserve for testing

# ARIMA parameters
ARIMA_ORDER = (5, 1, 0)

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0
}

# Feature engineering settings
LAG_FEATURES = [1, 7, 30]       # Lag periods for feature engineering
ROLLING_WINDOWS = [7]           # Rolling window sizes
TRAIN_RATIO = 0.8               # Train/test split ratio

# Prophet parameters
PROPHET_PARAMS = {
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False
}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Default column names
DEFAULT_DATE_COLUMN = 'Date'
DEFAULT_SALES_COLUMN = 'Sales'

# Standard column names (after cleaning)
STANDARD_DATE_COLUMN = 'ds'
STANDARD_TARGET_COLUMN = 'y'

# Missing value handling
DEFAULT_FILL_METHOD = 'ffill'  # Forward fill

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Dashboard settings
APP_TITLE = "StockSense-AI"
APP_DESCRIPTION = "Intelligent Sales Forecasting & Inventory Optimization"
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        MODELS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return True


def get_model_path(model_name: str) -> Path:
    """Get the path for a specific model."""
    model_paths = {
        'champion': MODEL_PATH,
        'arima': ARIMA_MODEL_PATH,
        'xgboost': XGBOOST_MODEL_PATH,
        'prophet': PROPHET_MODEL_PATH
    }
    return model_paths.get(model_name.lower(), MODEL_PATH)


# =============================================================================
# PRINT CONFIG (for debugging)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("APPLICATION CONFIGURATION")
    print("=" * 60)
    print(f"\nPaths:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  MODELS_DIR: {MODELS_DIR}")
    print(f"  MODEL_PATH: {MODEL_PATH}")
    
    print(f"\nModel Settings:")
    print(f"  FORECAST_HORIZON: {FORECAST_HORIZON}")
    print(f"  ARIMA_ORDER: {ARIMA_ORDER}")
    print(f"  LAG_FEATURES: {LAG_FEATURES}")
    print(f"  ROLLING_WINDOWS: {ROLLING_WINDOWS}")
    
    print(f"\nData Settings:")
    print(f"  DEFAULT_DATE_COLUMN: {DEFAULT_DATE_COLUMN}")
    print(f"  DEFAULT_SALES_COLUMN: {DEFAULT_SALES_COLUMN}")
    
    print("\n" + "=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    print("✅ All directories created/verified")
