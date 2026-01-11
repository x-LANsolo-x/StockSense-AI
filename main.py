"""
StockSense-AI - Main Pipeline
=============================
This script runs the complete forecasting pipeline:
1. Load raw sales data
2. Clean and prepare data
3. Create time-based features
4. Train ARIMA baseline model
5. Generate 7-day forecast
6. Save results to processed folder
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Step A: Import functions from our modules
from src.data.ingestion import load_data, clean_data
from src.features.engineering import create_features
from src.models.baseline import train_model, make_forecast, get_forecast_with_confidence


def run_pipeline(
    input_file: str = 'data/raw/sales.csv',
    date_column: str = 'Date',
    sales_column: str = 'Sales',
    forecast_days: int = 7,
    save_results: bool = True
):
    """
    Run the complete sales forecasting pipeline.
    
    Args:
        input_file: Path to raw CSV file
        date_column: Name of date column in CSV
        sales_column: Name of sales column in CSV
        forecast_days: Number of days to forecast (default: 7)
        save_results: Whether to save results to CSV (default: True)
        
    Returns:
        Dictionary containing cleaned data, features, model, and forecast
    """
    print("=" * 70)
    print("STOCKSENSE-AI FORECASTING PIPELINE")
    print("=" * 70)
    print(f"   Input file: {input_file}")
    print(f"   Forecast horizon: {forecast_days} days")
    print("=" * 70)
    
    # =========================================================================
    # Step B: Define path to raw CSV file
    # =========================================================================
    if not os.path.exists(input_file):
        print(f"\n⚠️  File not found: {input_file}")
        print("   Creating sample data for demonstration...")
        create_sample_data(input_file)
    
    # =========================================================================
    # Step C: Load and clean data
    # =========================================================================
    print("\n" + "=" * 70)
    print("📥 STEP 1: Loading Data")
    print("=" * 70)
    df_raw = load_data(input_file)
    print(f"\n📋 Raw data preview:")
    print(df_raw.head())
    
    print("\n" + "=" * 70)
    print("🧹 STEP 2: Cleaning Data")
    print("=" * 70)
    df_clean = clean_data(df_raw, date_column=date_column, sales_column=sales_column)
    
    # =========================================================================
    # Step C.5: Create features (optional but useful)
    # =========================================================================
    print("\n" + "=" * 70)
    print("⚙️  STEP 3: Creating Features")
    print("=" * 70)
    df_features = create_features(df_clean)
    print(f"\n📋 Data with features preview:")
    print(df_features.head())
    
    # =========================================================================
    # Step D: Train ARIMA model
    # =========================================================================
    print("\n" + "=" * 70)
    print("🤖 STEP 4: Training ARIMA Model")
    print("=" * 70)
    
    # Prepare time series data with date index for proper forecasting
    ts_data = df_clean.set_index('ds')['y']
    
    # Train the model
    model_fit = train_model(ts_data, order=(5, 1, 0))
    
    # =========================================================================
    # Step E: Make forecast for next 7 days
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"🔮 STEP 5: Forecasting Next {forecast_days} Days")
    print("=" * 70)
    
    # Get forecast with confidence intervals
    forecast_df = get_forecast_with_confidence(model_fit, steps=forecast_days)
    
    # Add date index to forecast
    last_date = df_clean['ds'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df.index = forecast_dates
    forecast_df.index.name = 'date'
    
    print(f"\n📋 {forecast_days}-Day Forecast:")
    print(forecast_df.round(2))
    
    # =========================================================================
    # Step F: Save results
    # =========================================================================
    if save_results:
        print("\n" + "=" * 70)
        print("💾 STEP 6: Saving Results")
        print("=" * 70)
        
        # Ensure output directory exists
        os.makedirs('data/processed', exist_ok=True)
        
        # Save cleaned data with features
        cleaned_path = 'data/processed/cleaned_sales.csv'
        df_features.to_csv(cleaned_path, index=False)
        print(f"✅ Saved cleaned data to: {cleaned_path}")
        
        # Save forecast
        forecast_path = 'data/processed/forecast.csv'
        forecast_df.to_csv(forecast_path)
        print(f"✅ Saved forecast to: {forecast_path}")
        
        # Save summary report
        summary_path = 'data/processed/summary_report.txt'
        save_summary_report(df_clean, forecast_df, summary_path)
        print(f"✅ Saved summary report to: {summary_path}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   - Historical data points: {len(df_clean)}")
    print(f"   - Date range: {df_clean['ds'].min().date()} to {df_clean['ds'].max().date()}")
    print(f"   - Average historical sales: {df_clean['y'].mean():.2f}")
    print(f"   - Forecast days: {forecast_days}")
    print(f"   - Average forecasted sales: {forecast_df['forecast'].mean():.2f}")
    print("=" * 70)
    
    return {
        'raw_data': df_raw,
        'cleaned_data': df_clean,
        'features_data': df_features,
        'model': model_fit,
        'forecast': forecast_df
    }


def create_sample_data(output_path: str):
    """
    Create sample sales data for demonstration.
    
    Args:
        output_path: Path to save the sample CSV
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate realistic sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    
    # Create sales with trend, weekly seasonality, and noise
    trend = np.linspace(100, 130, 90)  # Upward trend
    weekly_pattern = 20 * np.sin(np.arange(90) * 2 * np.pi / 7)  # Weekly cycle
    noise = np.random.normal(0, 8, 90)  # Random noise
    
    sales = trend + weekly_pattern + noise
    sales = np.maximum(sales, 0)  # Ensure non-negative
    
    # Add some missing values for realism
    sales[15] = np.nan
    sales[45] = np.nan
    
    # Create DataFrame
    sample_df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Sales': sales.round(2),
        'Store': 'Store_A'
    })
    
    # Save to CSV
    sample_df.to_csv(output_path, index=False)
    print(f"✅ Created sample data with {len(sample_df)} rows at {output_path}")


def save_summary_report(df_clean: pd.DataFrame, forecast_df: pd.DataFrame, output_path: str):
    """
    Save a summary report of the forecasting results.
    
    Args:
        df_clean: Cleaned historical data
        forecast_df: Forecast results
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("STOCKSENSE-AI - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("HISTORICAL DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total observations: {len(df_clean)}\n")
        f.write(f"Date range: {df_clean['ds'].min().date()} to {df_clean['ds'].max().date()}\n")
        f.write(f"Min sales: {df_clean['y'].min():.2f}\n")
        f.write(f"Max sales: {df_clean['y'].max():.2f}\n")
        f.write(f"Mean sales: {df_clean['y'].mean():.2f}\n")
        f.write(f"Std sales: {df_clean['y'].std():.2f}\n\n")
        
        f.write("FORECAST SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Forecast horizon: {len(forecast_df)} days\n")
        f.write(f"Forecast period: {forecast_df.index.min().date()} to {forecast_df.index.max().date()}\n")
        f.write(f"Mean forecast: {forecast_df['forecast'].mean():.2f}\n")
        f.write(f"Forecast range: {forecast_df['forecast'].min():.2f} to {forecast_df['forecast'].max():.2f}\n\n")
        
        f.write("DETAILED FORECAST\n")
        f.write("-" * 40 + "\n")
        for idx, row in forecast_df.iterrows():
            f.write(f"{idx.date()}: {row['forecast']:.2f} (CI: {row['lower_ci']:.2f} - {row['upper_ci']:.2f})\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF REPORT\n")


# =========================================================================
# Main entry point
# =========================================================================
if __name__ == "__main__":
    # Run the pipeline with default settings
    results = run_pipeline(
        input_file='data/raw/sales.csv',
        date_column='Date',
        sales_column='Sales',
        forecast_days=7,
        save_results=True
    )
    
    print("\n💡 Tip: Access results programmatically:")
    print("   results['forecast']      - View forecast DataFrame")
    print("   results['model']         - Access trained ARIMA model")
    print("   results['cleaned_data']  - View cleaned historical data")
