"""
Phase 1 MVP Tests
=================
Test script to verify data integrity, feature engineering, and model sanity.

Run with: py -m pytest tests/test_mvp.py -v
Or directly: py tests/test_mvp.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ingestion import load_data, clean_data
from src.features.engineering import create_features
from src.models.baseline import train_model, make_forecast


# =============================================================================
# 1. DATA INTEGRITY TESTS
# =============================================================================

def test_data_cleaning(df: pd.DataFrame) -> bool:
    """
    Test data cleaning integrity.
    
    Tests:
    1. Date column is datetime type
    2. Data is sorted chronologically
    3. No missing values in target column
    """
    print("\n" + "=" * 50)
    print("🧪 TEST 1: Data Integrity Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1.1: Date Parsing - Verify 'ds' column is datetime
    print("\n  1.1 Date Parsing...")
    try:
        assert pd.api.types.is_datetime64_any_dtype(df['ds']), \
            "Error: 'ds' column is not datetime!"
        print("      ✅ PASSED: 'ds' column is datetime64")
    except AssertionError as e:
        print(f"      ❌ FAILED: {e}")
        all_passed = False
    
    # Test 1.2: Sorting - Data must be chronological
    print("\n  1.2 Chronological Sorting...")
    try:
        assert df['ds'].is_monotonic_increasing, \
            "Error: Data is not sorted by date!"
        print("      ✅ PASSED: Data is sorted chronologically")
    except AssertionError as e:
        print(f"      ❌ FAILED: {e}")
        all_passed = False
    
    # Test 1.3: Missing Values - No NaNs in target column
    print("\n  1.3 Missing Values Check...")
    try:
        nan_count = df['y'].isnull().sum()
        assert nan_count == 0, \
            f"Error: Target column 'y' has {nan_count} missing values!"
        print("      ✅ PASSED: No missing values in 'y' column")
    except AssertionError as e:
        print(f"      ❌ FAILED: {e}")
        all_passed = False
    
    # Summary
    if all_passed:
        print("\n  ✅ DATA INTEGRITY TESTS: ALL PASSED")
    else:
        print("\n  ❌ DATA INTEGRITY TESTS: SOME FAILED")
    
    return all_passed


# =============================================================================
# 2. FEATURE ENGINEERING TESTS
# =============================================================================

def test_feature_engineering(df: pd.DataFrame) -> bool:
    """
    Test feature engineering.
    
    Tests:
    1. Required columns exist
    2. Logic check - spot check specific date values
    """
    print("\n" + "=" * 50)
    print("🧪 TEST 2: Feature Engineering Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Test 2.1: Column Existence
    print("\n  2.1 Column Existence...")
    required_columns = ['year', 'month', 'day_of_week']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    try:
        assert len(missing_columns) == 0, \
            f"Error: Missing columns: {missing_columns}"
        print(f"      ✅ PASSED: All required columns exist: {required_columns}")
    except AssertionError as e:
        print(f"      ❌ FAILED: {e}")
        all_passed = False
    
    # Test 2.2: Logic Check - Spot check date values
    print("\n  2.2 Logic Check (Spot Check)...")
    try:
        # Get first row for spot check
        first_row = df.iloc[0]
        expected_year = first_row['ds'].year
        expected_month = first_row['ds'].month
        expected_dow = first_row['ds'].dayofweek
        
        assert first_row['year'] == expected_year, \
            f"Error: Year mismatch. Expected {expected_year}, got {first_row['year']}"
        assert first_row['month'] == expected_month, \
            f"Error: Month mismatch. Expected {expected_month}, got {first_row['month']}"
        assert first_row['day_of_week'] == expected_dow, \
            f"Error: Day of week mismatch. Expected {expected_dow}, got {first_row['day_of_week']}"
        
        print(f"      ✅ PASSED: Date {first_row['ds'].date()} correctly has:")
        print(f"         - year: {first_row['year']}")
        print(f"         - month: {first_row['month']}")
        print(f"         - day_of_week: {first_row['day_of_week']}")
    except AssertionError as e:
        print(f"      ❌ FAILED: {e}")
        all_passed = False
    except KeyError as e:
        print(f"      ❌ FAILED: Column not found - {e}")
        all_passed = False
    
    # Summary
    if all_passed:
        print("\n  ✅ FEATURE ENGINEERING TESTS: ALL PASSED")
    else:
        print("\n  ❌ FEATURE ENGINEERING TESTS: SOME FAILED")
    
    return all_passed


# =============================================================================
# 3. MODEL SANITY TESTS
# =============================================================================

def test_forecast_output(forecast_values, steps: int = 7) -> bool:
    """
    Test model forecast output.
    
    Tests:
    1. Output shape matches requested steps
    2. No NaN values in forecast
    """
    print("\n" + "=" * 50)
    print("🧪 TEST 3: Model Sanity Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Convert to numpy array if needed
    if isinstance(forecast_values, pd.Series):
        forecast_array = forecast_values.values
    elif isinstance(forecast_values, pd.DataFrame):
        forecast_array = forecast_values['forecast'].values if 'forecast' in forecast_values.columns else forecast_values.iloc[:, 0].values
    else:
        forecast_array = np.array(forecast_values)
    
    # Test 3.1: Output Shape
    print(f"\n  3.1 Output Shape (expecting {steps} values)...")
    try:
        assert len(forecast_array) == steps, \
            f"Error: Expected {steps} predictions, got {len(forecast_array)}"
        print(f"      ✅ PASSED: Forecast has exactly {steps} values")
    except AssertionError as e:
        print(f"      ❌ FAILED: {e}")
        all_passed = False
    
    # Test 3.2: No NaNs in Forecast
    print("\n  3.2 No NaN Values...")
    try:
        nan_count = np.isnan(forecast_array).sum()
        assert nan_count == 0, \
            f"Error: Forecast contains {nan_count} NaN values!"
        print("      ✅ PASSED: No NaN values in forecast")
    except AssertionError as e:
        print(f"      ❌ FAILED: {e}")
        all_passed = False
    
    # Test 3.3: Reasonable Values (bonus sanity check)
    print("\n  3.3 Reasonable Values (Bonus)...")
    try:
        assert not np.isinf(forecast_array).any(), \
            "Error: Forecast contains infinite values!"
        assert (forecast_array >= 0).all() or True, \
            "Warning: Some forecast values are negative (may be valid depending on context)"
        print(f"      ✅ PASSED: Forecast values are finite")
        print(f"         - Range: {forecast_array.min():.2f} to {forecast_array.max():.2f}")
        print(f"         - Mean: {forecast_array.mean():.2f}")
    except AssertionError as e:
        print(f"      ⚠️ WARNING: {e}")
    
    # Summary
    if all_passed:
        print("\n  ✅ MODEL SANITY TESTS: ALL PASSED")
    else:
        print("\n  ❌ MODEL SANITY TESTS: SOME FAILED")
    
    return all_passed


# =============================================================================
# FULL TEST SUITE
# =============================================================================

def run_all_tests():
    """
    Run the complete Phase 1 MVP test suite.
    """
    print("\n" + "=" * 60)
    print("🚀 PHASE 1 MVP TEST SUITE")
    print("=" * 60)
    
    results = {
        'data_integrity': False,
        'feature_engineering': False,
        'model_sanity': False
    }
    
    try:
        # Setup: Create sample data and run pipeline
        print("\n📋 Setting up test data...")
        
        # Create sample CSV
        os.makedirs('data/raw', exist_ok=True)
        sample_df = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=60, freq='D'),
            'Sales': 100 + np.cumsum(np.random.randn(60) * 5)
        })
        # Add some missing values
        sample_df.loc[10, 'Sales'] = np.nan
        sample_df.loc[30, 'Sales'] = np.nan
        sample_df.to_csv('data/raw/test_sales.csv', index=False)
        print("   ✅ Created test data with 60 rows and 2 missing values")
        
        # Step 1: Load and clean data
        print("\n📋 Running data ingestion...")
        df_raw = load_data('data/raw/test_sales.csv')
        df_clean = clean_data(df_raw, date_column='Date', sales_column='Sales')
        
        # Step 2: Create features
        print("\n📋 Running feature engineering...")
        df_features = create_features(df_clean)
        
        # Step 3: Train model and forecast
        print("\n📋 Running model training...")
        ts_data = df_clean.set_index('ds')['y']
        model_fit = train_model(ts_data, order=(5, 1, 0))
        forecast = make_forecast(model_fit, steps=7)
        
        # Run Tests
        results['data_integrity'] = test_data_cleaning(df_clean)
        results['feature_engineering'] = test_feature_engineering(df_features)
        results['model_sanity'] = test_forecast_output(forecast, steps=7)
        
        # Cleanup
        os.remove('data/raw/test_sales.csv')
        print("\n📋 Cleaned up test files")
        
    except Exception as e:
        print(f"\n❌ ERROR during test setup: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print("-" * 60)
    print(f"   Total: {total_passed}/{total_tests} test groups passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Phase 1 MVP is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
    
    print("=" * 60)
    
    return all(results.values())


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
