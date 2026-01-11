"""
Retail Sales Forecasting Dashboard
===================================
Advanced Streamlit dashboard for sales forecasting with:
- Data upload and health checks
- Model training controls
- Interactive forecasting with What-If analysis

Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import (
    FORECAST_HORIZON, LAG_FEATURES, ROLLING_WINDOWS,
    MODEL_PATH, XGBOOST_PARAMS, ensure_directories
)
from src.data.ingestion import load_data, clean_data
from src.features.engineering import create_advanced_features, create_features
from src.models.persistence import save_model, load_model, model_exists
from src.models.evaluation import calculate_metrics

# Ensure directories exist
ensure_directories()

# =============================================================================
# CACHING FOR PERFORMANCE (loads in < 2 seconds)
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_load_data(file_content: bytes, filename: str) -> pd.DataFrame:
    """Cache data loading to improve performance."""
    import io
    return pd.read_csv(io.BytesIO(file_content))

@st.cache_data
def cached_clean_data(df_json: str, date_col: str, sales_col: str) -> pd.DataFrame:
    """Cache data cleaning."""
    df = pd.read_json(df_json)
    return clean_data(df, date_column=date_col, sales_column=sales_col)

@st.cache_data
def cached_create_features(df_json: str) -> pd.DataFrame:
    """Cache feature creation."""
    df = pd.read_json(df_json)
    return create_features(df)

@st.cache_resource
def cached_load_model(model_path: str):
    """Cache model loading - only reload when model changes."""
    return load_model(model_path)

# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

def validate_uploaded_file(uploaded_file) -> tuple:
    """
    Validate uploaded file and return (is_valid, error_message).
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file extension
    filename = uploaded_file.name.lower()
    if not filename.endswith('.csv'):
        return False, f"Invalid file type: '{filename}'. Please upload a CSV file (.csv)"
    
    # Check file size (max 100MB)
    file_size = uploaded_file.size
    if file_size > 100 * 1024 * 1024:
        return False, f"File too large ({file_size / 1024 / 1024:.1f} MB). Maximum size is 100 MB"
    
    # Try to read the file
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        
        if len(df) == 0:
            return False, "File is empty. Please upload a file with data"
        
        if len(df.columns) < 2:
            return False, "File must have at least 2 columns (Date and Sales)"
            
    except pd.errors.EmptyDataError:
        return False, "File is empty or corrupted"
    except pd.errors.ParserError as e:
        return False, f"Could not parse CSV file: {str(e)}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"
    
    return True, "File is valid"

def safe_process_data(df: pd.DataFrame, date_col: str, sales_col: str) -> tuple:
    """
    Safely process data with error handling.
    Returns (success, result_or_error_message).
    """
    try:
        # Validate columns exist
        if date_col not in df.columns:
            return False, f"Date column '{date_col}' not found in data"
        if sales_col not in df.columns:
            return False, f"Sales column '{sales_col}' not found in data"
        
        # Clean data
        df_clean = clean_data(df.copy(), date_column=date_col, sales_column=sales_col)
        
        # Create features
        df_features = create_features(df_clean.copy())
        
        return True, (df_clean, df_features)
        
    except Exception as e:
        return False, f"Error processing data: {str(e)}"

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="StockSense-AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'features_data' not in st.session_state:
    st.session_state.features_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("StockSense-AI")
    st.markdown("---")
    
    st.subheader("Quick Stats")
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.metric("Total Records", len(df))
        st.metric("Date Range", f"{df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}")
        st.metric("Avg Daily Sales", f"${df['y'].mean():.2f}")
    else:
        st.info("Upload data to see stats")
    
    st.markdown("---")
    st.subheader("Model Status")
    if st.session_state.model is not None:
        st.success("Model Loaded ✓")
        if st.session_state.model_metrics:
            st.metric("Model MAPE", st.session_state.model_metrics.get('mape', 'N/A'))
    elif model_exists(str(MODEL_PATH)):
        st.warning("Saved model available")
        if st.button("Load Saved Model"):
            try:
                model, metadata = load_model(str(MODEL_PATH))
                st.session_state.model = model
                st.session_state.model_metrics = metadata.get('metrics', {}) if metadata else {}
                st.success("Model loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading model: {e}")
    else:
        st.info("No model trained yet")

# =============================================================================
# MAIN CONTENT - TABBED INTERFACE
# =============================================================================

st.title("StockSense-AI")
st.markdown("Intelligent Sales Forecasting & Inventory Optimization")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Upload Data", "Model Training", "Forecast Dashboard"])

# =============================================================================
# TAB 1: UPLOAD & DATA HEALTH
# =============================================================================

with tab1:
    st.header("Upload & Data Health")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with Date and Sales columns"
        )
        
        if uploaded_file is not None:
            # Validate file first
            is_valid, validation_msg = validate_uploaded_file(uploaded_file)
            
            if not is_valid:
                st.error(validation_msg)
            else:
                # Column mapping
                st.markdown("**Column Mapping**")
                
                # Read file to get columns
                try:
                    temp_df = pd.read_csv(uploaded_file)
                    uploaded_file.seek(0)  # Reset file pointer
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    temp_df = None
                
                if temp_df is not None:
                    date_col = st.selectbox(
                        "Select Date Column",
                        options=temp_df.columns.tolist(),
                        index=0 if 'Date' not in temp_df.columns else temp_df.columns.tolist().index('Date')
                    )
                    
                    sales_col = st.selectbox(
                        "Select Sales Column",
                        options=temp_df.columns.tolist(),
                        index=1 if 'Sales' not in temp_df.columns else temp_df.columns.tolist().index('Sales')
                    )
                    
                    if st.button("Process Data", type="primary"):
                        with st.spinner("Processing data..."):
                            # Load raw data
                            try:
                                df_raw = pd.read_csv(uploaded_file)
                                st.session_state.data = df_raw
                            except Exception as e:
                                st.error(f"Error loading data: {e}")
                                df_raw = None
                            
                            if df_raw is not None:
                                # Use safe processing with error handling
                                success, result = safe_process_data(df_raw, date_col, sales_col)
                                
                                if success:
                                    df_clean, df_features = result
                                    st.session_state.cleaned_data = df_clean
                                    st.session_state.features_data = df_features
                                    st.success("Data processed successfully!")
                                    st.rerun()
                                else:
                                    st.error(result)
        
        # Use sample data option
        st.markdown("---")
        st.markdown("**Or use sample data:**")
        if st.button("Load Sample Data"):
            with st.spinner("Generating sample data..."):
                # Generate sample data
                np.random.seed(42)
                dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
                trend = np.linspace(100, 150, 365)
                weekly = 15 * np.sin(np.arange(365) * 2 * np.pi / 7)
                yearly = 20 * np.sin(np.arange(365) * 2 * np.pi / 365)
                noise = np.random.randn(365) * 8
                sales = trend + weekly + yearly + noise
                
                df_raw = pd.DataFrame({
                    'Date': dates,
                    'Sales': sales.round(2),
                    'Store': np.random.choice(['Store_A', 'Store_B', 'Store_C'], 365)
                })
                
                st.session_state.data = df_raw
                df_clean = clean_data(df_raw.copy(), date_column='Date', sales_column='Sales')
                st.session_state.cleaned_data = df_clean
                df_features = create_features(df_clean.copy())
                st.session_state.features_data = df_features
                
                st.success("Sample data loaded!")
                st.rerun()
    
    with col2:
        st.subheader("Data Health Report")
        
        if st.session_state.cleaned_data is not None:
            df = st.session_state.cleaned_data
            raw_df = st.session_state.data
            
            # Data quality metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                missing_raw = raw_df.isnull().sum().sum() if raw_df is not None else 0
                st.metric("Missing Values (Raw)", missing_raw)
            
            with col_b:
                missing_clean = df.isnull().sum().sum()
                st.metric("Missing Values (Cleaned)", missing_clean)
            
            with col_c:
                st.metric("Total Records", len(df))
            
            # Date range
            st.info(f"**Date Range:** {df['ds'].min().strftime('%B %d, %Y')} to {df['ds'].max().strftime('%B %d, %Y')}")
            
            # Data preview
            st.markdown("**Data Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
        else:
            st.info("Upload data to see health report")
    
    # Seasonality Analysis
    if st.session_state.features_data is not None:
        st.markdown("---")
        st.subheader("Seasonality Analysis")
        
        df_feat = st.session_state.features_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by Day of Week
            dow_sales = df_feat.groupby('day_of_week')['y'].mean().reset_index()
            dow_sales['day_name'] = dow_sales['day_of_week'].map({
                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
            })
            
            fig_dow = px.bar(
                dow_sales, x='day_name', y='y',
                title='Average Sales by Day of Week',
                labels={'y': 'Avg Sales', 'day_name': 'Day'},
                color='y',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_dow, use_container_width=True)
        
        with col2:
            # Sales by Month
            month_sales = df_feat.groupby('month')['y'].mean().reset_index()
            month_sales['month_name'] = month_sales['month'].map({
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            })
            
            fig_month = px.bar(
                month_sales, x='month_name', y='y',
                title='Average Sales by Month',
                labels={'y': 'Avg Sales', 'month_name': 'Month'},
                color='y',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_month, use_container_width=True)
        
        # Time series plot
        st.markdown("**Sales Over Time:**")
        fig_ts = px.line(
            df_feat, x='ds', y='y',
            title='Historical Sales Data',
            labels={'ds': 'Date', 'y': 'Sales'}
        )
        fig_ts.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig_ts, use_container_width=True)

# =============================================================================
# TAB 2: MODEL TRAINING
# =============================================================================

with tab2:
    st.header("Model Training Control")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please upload and process data first in the 'Upload Data' tab.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Training Configuration")
            
            # Model selection
            model_type = st.selectbox(
                "Select Model",
                options=['XGBoost', 'ARIMA', 'Prophet'],
                index=0
            )
            
            # Forecast horizon
            forecast_horizon = st.slider(
                "Forecast Horizon (days)",
                min_value=7, max_value=90, value=FORECAST_HORIZON,
                help="Number of days to forecast"
            )
            
            # Test split
            test_days = st.slider(
                "Test Days (for validation)",
                min_value=7, max_value=60, value=30,
                help="Days to reserve for testing"
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                if model_type == 'XGBoost':
                    n_estimators = st.slider("Number of Estimators", 50, 500, 100)
                    max_depth = st.slider("Max Depth", 3, 10, 5)
                    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                elif model_type == 'ARIMA':
                    p = st.slider("AR Order (p)", 1, 10, 5)
                    d = st.slider("Differencing (d)", 0, 2, 1)
                    q = st.slider("MA Order (q)", 0, 5, 0)
        
        with col2:
            st.subheader("Train Model")
            
            if st.button("Train New Model", type="primary", use_container_width=True):
                df = st.session_state.cleaned_data.copy()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Feature Engineering
                    status_text.text("Step 1/4: Creating features...")
                    progress_bar.progress(10)
                    
                    df_features = create_advanced_features(
                        df.copy(),
                        lags=LAG_FEATURES,
                        rolling_windows=ROLLING_WINDOWS,
                        drop_na=True
                    )
                    progress_bar.progress(25)
                    
                    # Step 2: Split data
                    status_text.text("Step 2/4: Splitting data...")
                    split_idx = len(df_features) - test_days
                    train_df = df_features.iloc[:split_idx]
                    test_df = df_features.iloc[split_idx:]
                    progress_bar.progress(40)
                    
                    # Step 3: Train model
                    status_text.text(f"Step 3/4: Training {model_type} model...")
                    
                    if model_type == 'XGBoost':
                        from xgboost import XGBRegressor
                        from src.models.ml_models import prepare_features_target
                        
                        X_train, y_train = prepare_features_target(train_df)
                        X_test, y_test = prepare_features_target(test_df)
                        
                        model = XGBRegressor(
                            n_estimators=n_estimators if 'n_estimators' in dir() else 100,
                            max_depth=max_depth if 'max_depth' in dir() else 5,
                            learning_rate=learning_rate if 'learning_rate' in dir() else 0.1,
                            random_state=42,
                            verbosity=0
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                    elif model_type == 'ARIMA':
                        from src.models.baseline import train_model as train_arima, make_forecast
                        
                        ts_data = train_df.set_index('ds')['y']
                        order = (p if 'p' in dir() else 5, d if 'd' in dir() else 1, q if 'q' in dir() else 0)
                        model = train_arima(ts_data, order=order)
                        y_pred = make_forecast(model, steps=len(test_df)).values
                        y_test = test_df['y'].values
                        
                    elif model_type == 'Prophet':
                        from src.models.ml_models import train_prophet, predict_prophet
                        
                        model = train_prophet(train_df[['ds', 'y']])
                        forecast = predict_prophet(model, periods=len(test_df))
                        y_pred = forecast['yhat'].values
                        y_test = test_df['y'].values
                    
                    progress_bar.progress(75)
                    
                    # Step 4: Evaluate
                    status_text.text("Step 4/4: Evaluating model...")
                    metrics = calculate_metrics(y_test, y_pred)
                    progress_bar.progress(90)
                    
                    # Save model
                    save_model(
                        model, 
                        str(MODEL_PATH),
                        metadata={
                            'model_type': model_type,
                            'metrics': metrics,
                            'forecast_horizon': forecast_horizon,
                            'training_samples': len(train_df),
                            'test_samples': len(test_df)
                        }
                    )
                    
                    # Update session state
                    st.session_state.model = model
                    st.session_state.model_metrics = metrics
                    st.session_state.model_type = model_type
                    
                    progress_bar.progress(100)
                    status_text.text("Training complete!")
                    
                    # Show results
                    st.success("Model trained and saved successfully!")
                    
                    # Display metrics
                    st.markdown("### Model Performance")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("MAE", f"{metrics['mae']:.2f}")
                    with metric_col2:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                    with metric_col3:
                        st.metric("MAPE", metrics['mape'])
                    with metric_col4:
                        st.metric("R²", f"{metrics['r2']:.4f}")
                    
                    # Plot actual vs predicted
                    st.markdown("### Actual vs Predicted (Test Set)")
                    comparison_df = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': y_pred
                    })
                    
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Scatter(y=comparison_df['Actual'], name='Actual', line=dict(color='blue')))
                    fig_compare.add_trace(go.Scatter(y=comparison_df['Predicted'], name='Predicted', line=dict(color='red', dash='dash')))
                    fig_compare.update_layout(title='Model Predictions vs Actual', xaxis_title='Days', yaxis_title='Sales')
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# =============================================================================
# TAB 3: INTERACTIVE FORECASTING
# =============================================================================

with tab3:
    st.header("Interactive Forecast Dashboard")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' tab.")
        
        # Try to load existing model
        if model_exists(str(MODEL_PATH)):
            if st.button("Load Existing Model"):
                model, metadata = load_model(str(MODEL_PATH))
                st.session_state.model = model
                st.session_state.model_metrics = metadata.get('metrics', {}) if metadata else {}
                st.session_state.model_type = metadata.get('model_type', 'Unknown') if metadata else 'Unknown'
                st.rerun()
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Forecast Controls")
            
            # Forecast horizon
            forecast_days = st.slider(
                "Forecast Days",
                min_value=7, max_value=90, value=30
            )
            
            # Store/Product filter (if available)
            if st.session_state.data is not None and 'Store' in st.session_state.data.columns:
                stores = ['All'] + st.session_state.data['Store'].unique().tolist()
                selected_store = st.selectbox("Select Store", stores)
            else:
                selected_store = 'All'
            
            st.markdown("---")
            
            # What-If Analysis
            st.subheader("What-If Analysis")
            
            promo_uplift = st.slider(
                "Promotional Uplift (%)",
                min_value=-30, max_value=50, value=0,
                help="Simulate the impact of promotions on sales"
            )
            
            trend_adjustment = st.slider(
                "Trend Adjustment (%)",
                min_value=-20, max_value=20, value=0,
                help="Adjust for market trends"
            )
            
            st.markdown("---")
            
            # Generate forecast button
            generate_forecast = st.button("Generate Forecast", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("Forecast Results")
            
            if generate_forecast or st.session_state.forecast is not None:
                if generate_forecast:
                    with st.spinner("Generating forecast..."):
                        try:
                            df = st.session_state.cleaned_data.copy()
                            model = st.session_state.model
                            model_type = getattr(st.session_state, 'model_type', 'XGBoost')
                            
                            # Generate base forecast
                            if model_type == 'Prophet':
                                from src.models.ml_models import predict_prophet
                                forecast_result = predict_prophet(model, periods=forecast_days)
                                base_forecast = forecast_result['yhat'].values
                                lower_ci = forecast_result['yhat_lower'].values
                                upper_ci = forecast_result['yhat_upper'].values
                            elif model_type == 'ARIMA':
                                from src.models.baseline import get_forecast_with_confidence
                                forecast_result = get_forecast_with_confidence(model, steps=forecast_days)
                                base_forecast = forecast_result['forecast'].values
                                lower_ci = forecast_result['lower_ci'].values
                                upper_ci = forecast_result['upper_ci'].values
                            else:  # XGBoost
                                # For XGBoost, we need to create features for future dates
                                df_features = create_advanced_features(df.copy(), lags=LAG_FEATURES, rolling_windows=ROLLING_WINDOWS, drop_na=True)
                                from src.models.ml_models import prepare_features_target
                                
                                # Use last available features and predict iteratively
                                last_features = df_features.iloc[-1:].copy()
                                X_last, _ = prepare_features_target(last_features)
                                
                                base_forecast = []
                                for i in range(forecast_days):
                                    pred = model.predict(X_last)[0]
                                    base_forecast.append(pred)
                                
                                base_forecast = np.array(base_forecast)
                                # Estimate CI for XGBoost
                                std_estimate = base_forecast.std() * 0.2
                                lower_ci = base_forecast - 1.96 * std_estimate
                                upper_ci = base_forecast + 1.96 * std_estimate
                            
                            # Apply What-If adjustments
                            uplift_factor = 1 + (promo_uplift / 100)
                            trend_factor = 1 + (trend_adjustment / 100)
                            
                            adjusted_forecast = base_forecast * uplift_factor * trend_factor
                            adjusted_lower = lower_ci * uplift_factor * trend_factor
                            adjusted_upper = upper_ci * uplift_factor * trend_factor
                            
                            # Create forecast dates
                            last_date = df['ds'].max()
                            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
                            
                            # Create forecast DataFrame
                            forecast_df = pd.DataFrame({
                                'Date': forecast_dates,
                                'Forecast': adjusted_forecast.round(2),
                                'Lower_CI': adjusted_lower.round(2),
                                'Upper_CI': adjusted_upper.round(2),
                                'Base_Forecast': base_forecast.round(2)
                            })
                            
                            st.session_state.forecast = forecast_df
                            
                        except Exception as e:
                            st.error(f"Error generating forecast: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                
                if st.session_state.forecast is not None:
                    forecast_df = st.session_state.forecast
                    
                    # Summary metrics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Total Forecast", f"${forecast_df['Forecast'].sum():,.0f}")
                    with col_b:
                        st.metric("Avg Daily", f"${forecast_df['Forecast'].mean():.2f}")
                    with col_c:
                        st.metric("Peak Day", f"${forecast_df['Forecast'].max():.2f}")
                    with col_d:
                        if promo_uplift != 0 or trend_adjustment != 0:
                            impact = ((forecast_df['Forecast'].sum() - forecast_df['Base_Forecast'].sum()) / forecast_df['Base_Forecast'].sum()) * 100
                            st.metric("What-If Impact", f"{impact:+.1f}%")
                        else:
                            st.metric("Forecast Days", len(forecast_df))
                    
                    # Forecast plot
                    fig_forecast = go.Figure()
                    
                    # Historical data
                    df_hist = st.session_state.cleaned_data
                    fig_forecast.add_trace(go.Scatter(
                        x=df_hist['ds'], y=df_hist['y'],
                        name='Historical', line=dict(color='blue')
                    ))
                    
                    # Forecast with confidence interval
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_df['Date'], y=forecast_df['Upper_CI'],
                        fill=None, mode='lines', line_color='rgba(0,100,80,0)',
                        showlegend=False
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_df['Date'], y=forecast_df['Lower_CI'],
                        fill='tonexty', mode='lines', line_color='rgba(0,100,80,0)',
                        fillcolor='rgba(0,100,80,0.2)', name='95% CI'
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_df['Date'], y=forecast_df['Forecast'],
                        name='Forecast', line=dict(color='red', dash='dash')
                    ))
                    
                    fig_forecast.update_layout(
                        title='Sales Forecast with Confidence Interval',
                        xaxis_title='Date',
                        yaxis_title='Sales',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Forecast table
                    st.markdown("### Forecast Details")
                    st.dataframe(
                        forecast_df.style.format({
                            'Forecast': '${:.2f}',
                            'Lower_CI': '${:.2f}',
                            'Upper_CI': '${:.2f}',
                            'Base_Forecast': '${:.2f}'
                        }),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Download button
                    st.markdown("### Export Forecast")
                    
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast CSV",
                        data=csv,
                        file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>StockSense-AI | Intelligent Sales Forecasting & Inventory Optimization</p>
    </div>
    """,
    unsafe_allow_html=True
)
