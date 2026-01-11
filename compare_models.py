"""
StockSense-AI - Model Comparison
================================
Run ARIMA, XGBoost, and Prophet models and compare their performance.

Usage:
    py compare_models.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import our modules
from src.data.ingestion import load_data, clean_data
from src.features.engineering import create_advanced_features, create_features
from src.models.baseline import train_model as train_arima, make_forecast as arima_forecast
from src.models.ml_models import train_xgboost, predict_xgboost, train_prophet, predict_prophet
from src.models.evaluation import (
    calculate_metrics, 
    print_metrics, 
    compare_model_metrics, 
    get_best_model
)


def create_sample_data(output_path: str, n_days: int = 365) -> None:
    """
    Create sample sales data with realistic patterns.
    
    Args:
        output_path: Path to save the CSV
        n_days: Number of days of data to generate
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    # Create realistic sales patterns
    trend = np.linspace(100, 150, n_days)  # Upward trend
    yearly_seasonality = 20 * np.sin(np.arange(n_days) * 2 * np.pi / 365)  # Yearly cycle
    weekly_seasonality = 15 * np.sin(np.arange(n_days) * 2 * np.pi / 7)   # Weekly cycle
    noise = np.random.randn(n_days) * 8  # Random noise
    
    sales = trend + yearly_seasonality + weekly_seasonality + noise
    sales = np.maximum(sales, 10)  # Ensure positive values
    
    # Add some missing values
    sales[50] = np.nan
    sales[150] = np.nan
    sales[250] = np.nan
    
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Sales': sales.round(2)
    })
    
    df.to_csv(output_path, index=False)
    print(f"✅ Created sample data: {n_days} days at {output_path}")


def split_train_test(df: pd.DataFrame, test_days: int = 30):
    """
    Split data into training and test sets.
    Reserve the last N days for testing.
    
    Args:
        df: Full dataset
        test_days: Number of days to reserve for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = len(df) - test_days
    
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    print(f"📊 Train/Test Split:")
    print(f"   - Training: {len(train_df)} days ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
    print(f"   - Testing:  {len(test_df)} days ({test_df['ds'].min().date()} to {test_df['ds'].max().date()})")
    
    return train_df, test_df


def run_arima_model(train_df: pd.DataFrame, test_days: int) -> np.ndarray:
    """
    Train ARIMA and predict.
    """
    print("\n" + "=" * 60)
    print("📈 MODEL 1: ARIMA (Baseline)")
    print("=" * 60)
    
    # Prepare time series
    ts_data = train_df.set_index('ds')['y']
    
    # Train model
    model = train_arima(ts_data, order=(5, 1, 0))
    
    # Forecast
    predictions = arima_forecast(model, steps=test_days)
    
    return predictions.values


def run_xgboost_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Train XGBoost and predict.
    """
    print("\n" + "=" * 60)
    print("🌲 MODEL 2: XGBoost")
    print("=" * 60)
    
    # Create features for training data
    print("\n📋 Preparing training features...")
    train_features = create_advanced_features(
        train_df.copy(), 
        lags=[1, 7, 30], 
        rolling_windows=[7],
        drop_na=True
    )
    
    # Train model (use all training data, no validation split)
    from xgboost import XGBRegressor
    from src.models.ml_models import prepare_features_target
    
    X_train, y_train = prepare_features_target(train_features)
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    print(f"🔧 Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    print("✅ Model trained!")
    
    # For XGBoost prediction, we need to create features for test data
    # This requires combining train + test to calculate lags correctly
    print("\n📋 Preparing test features...")
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_features = create_advanced_features(
        full_df.copy(),
        lags=[1, 7, 30],
        rolling_windows=[7],
        drop_na=False  # Keep all rows
    )
    
    # Get only test rows (last N rows that have complete features)
    test_features = full_features.tail(len(test_df)).dropna()
    
    if len(test_features) < len(test_df):
        print(f"⚠️ Only {len(test_features)} test samples have complete features")
    
    # Predict
    X_test, _ = prepare_features_target(test_features)
    predictions = model.predict(X_test)
    
    # Pad with NaN if needed to match test length
    if len(predictions) < len(test_df):
        pad_length = len(test_df) - len(predictions)
        predictions = np.concatenate([np.full(pad_length, np.nan), predictions])
    
    return predictions


def run_prophet_model(train_df: pd.DataFrame, test_days: int) -> np.ndarray:
    """
    Train Prophet and predict.
    """
    print("\n" + "=" * 60)
    print("🔮 MODEL 3: Prophet")
    print("=" * 60)
    
    # Train model
    model = train_prophet(
        train_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Forecast
    forecast = predict_prophet(model, periods=test_days)
    predictions = forecast['yhat'].values
    
    return predictions


def run_comparison(
    input_file: str = 'data/raw/sales.csv',
    date_column: str = 'Date',
    sales_column: str = 'Sales',
    test_days: int = 30
):
    """
    Run the full model comparison pipeline.
    
    Args:
        input_file: Path to input CSV
        date_column: Name of date column
        sales_column: Name of sales column
        test_days: Number of days to reserve for testing
    """
    print("=" * 70)
    print("STOCKSENSE-AI - MODEL COMPARISON")
    print("=" * 70)
    print(f"   Input file: {input_file}")
    print(f"   Test period: Last {test_days} days")
    print(f"   Models: ARIMA, XGBoost, Prophet")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Load and clean data
    # =========================================================================
    print("\n" + "=" * 60)
    print("📥 STEP 1: Load and Clean Data")
    print("=" * 60)
    
    if not os.path.exists(input_file):
        print(f"⚠️ File not found. Creating sample data...")
        create_sample_data(input_file, n_days=365)
    
    df_raw = load_data(input_file)
    df_clean = clean_data(df_raw, date_column=date_column, sales_column=sales_column)
    
    # =========================================================================
    # Step 2: Split data (reserve last 30 days for testing)
    # =========================================================================
    print("\n" + "=" * 60)
    print("✂️ STEP 2: Split Data")
    print("=" * 60)
    
    train_df, test_df = split_train_test(df_clean, test_days=test_days)
    y_true = test_df['y'].values
    
    # =========================================================================
    # Step 3: Run all models
    # =========================================================================
    print("\n" + "=" * 60)
    print("🤖 STEP 3: Train and Predict with All Models")
    print("=" * 60)
    
    results = {}
    
    # Model 1: ARIMA
    try:
        arima_pred = run_arima_model(train_df, test_days)
        results['ARIMA'] = arima_pred
    except Exception as e:
        print(f"❌ ARIMA failed: {e}")
    
    # Model 2: XGBoost
    try:
        xgb_pred = run_xgboost_model(train_df, test_df)
        results['XGBoost'] = xgb_pred
    except Exception as e:
        print(f"❌ XGBoost failed: {e}")
    
    # Model 3: Prophet
    try:
        prophet_pred = run_prophet_model(train_df, test_days)
        results['Prophet'] = prophet_pred
    except Exception as e:
        print(f"❌ Prophet failed: {e}")
    
    # =========================================================================
    # Step 4: Evaluate all models
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 STEP 4: Evaluate Models")
    print("=" * 60)
    
    all_metrics = {}
    
    for model_name, predictions in results.items():
        # Handle NaN values in predictions
        valid_mask = ~np.isnan(predictions) & ~np.isnan(y_true)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = predictions[valid_mask]
        
        if len(y_true_valid) > 0:
            metrics = calculate_metrics(y_true_valid, y_pred_valid)
            all_metrics[model_name] = metrics
            print_metrics(metrics, model_name)
        else:
            print(f"⚠️ {model_name}: No valid predictions to evaluate")
    
    # =========================================================================
    # Step 5: Print Leaderboard
    # =========================================================================
    print("\n" + "=" * 70)
    print("🏆 LEADERBOARD")
    print("=" * 70)
    
    if all_metrics:
        comparison_df = compare_model_metrics(all_metrics)
        print("\n📊 Model Comparison (sorted by RMSE - lower is better):\n")
        print(comparison_df.to_string(index=False))
        
        # Determine winners
        print("\n" + "-" * 70)
        best_by_mae = get_best_model(all_metrics, 'mae')
        best_by_rmse = get_best_model(all_metrics, 'rmse')
        best_by_mape = get_best_model(all_metrics, 'mape_value')
        best_by_r2 = get_best_model(all_metrics, 'r2')
        
        print(f"   🥇 Best by MAE:  {best_by_mae}")
        print(f"   🥇 Best by RMSE: {best_by_rmse}")
        print(f"   🥇 Best by MAPE: {best_by_mape}")
        print(f"   🥇 Best by R²:   {best_by_r2}")
        
        # Overall winner (by RMSE)
        print("\n" + "=" * 70)
        print(f"🏆 OVERALL WINNER: {best_by_rmse}")
        print("=" * 70)
        
        # Save results
        os.makedirs('data/processed', exist_ok=True)
        comparison_df.to_csv('data/processed/model_comparison.csv', index=False)
        print(f"\n💾 Results saved to: data/processed/model_comparison.csv")
    else:
        print("❌ No models successfully evaluated!")
    
    return all_metrics, comparison_df if all_metrics else None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_comparison(
        input_file='data/raw/sales.csv',
        date_column='Date',
        sales_column='Sales',
        test_days=30
    )
