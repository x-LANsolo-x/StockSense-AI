"""
StockSense-AI API
=================
RESTful API for intelligent sales forecasting and inventory optimization.

Run with: uvicorn src.api.main:app --reload
Or: py -m uvicorn src.api.main:app --reload

API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import (
    MODEL_PATH, FORECAST_HORIZON, LAG_FEATURES, ROLLING_WINDOWS,
    ensure_directories
)
from src.models.persistence import load_model, model_exists, get_model_info

# Ensure directories exist
ensure_directories()

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="StockSense-AI API",
    description="Intelligent Sales Forecasting & Inventory Optimization API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class PredictRequest(BaseModel):
    """Request schema for single prediction."""
    date: str = Field(..., description="Date for prediction (YYYY-MM-DD)", example="2024-01-01")
    store_id: Optional[int] = Field(None, description="Store ID (optional)", example=1)
    product_id: Optional[int] = Field(None, description="Product ID (optional)", example=101)
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-01-01",
                "store_id": 1,
                "product_id": 101
            }
        }


class PredictResponse(BaseModel):
    """Response schema for single prediction."""
    date: str
    prediction: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    store_id: Optional[int] = None
    product_id: Optional[int] = None


class ForecastRequest(BaseModel):
    """Request schema for multi-day forecast."""
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)", example="2024-01-01")
    days: int = Field(default=30, ge=1, le=365, description="Number of days to forecast")
    store_id: Optional[int] = Field(None, description="Store ID (optional)")
    promotional_uplift: Optional[float] = Field(0.0, ge=-50, le=100, description="Promotional uplift percentage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2024-01-01",
                "days": 30,
                "store_id": 1,
                "promotional_uplift": 10.0
            }
        }


class ForecastResponse(BaseModel):
    """Response schema for multi-day forecast."""
    forecasts: List[Dict[str, Any]]
    total_predicted_sales: float
    average_daily_sales: float
    model_type: str
    generated_at: str


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_exists: bool
    model_type: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    trained_at: Optional[str] = None
    file_size_mb: Optional[float] = None


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    timestamp: str


# =============================================================================
# GLOBAL MODEL CACHE
# =============================================================================

class ModelCache:
    """Cache for loaded model to avoid reloading on every request."""
    _model = None
    _metadata = None
    _model_type = None
    
    @classmethod
    def get_model(cls):
        """Get cached model or load from disk."""
        if cls._model is None:
            cls.load_model()
        return cls._model, cls._metadata, cls._model_type
    
    @classmethod
    def load_model(cls):
        """Load model from disk."""
        if not model_exists(str(MODEL_PATH)):
            raise HTTPException(
                status_code=503,
                detail="No trained model found. Please train a model first."
            )
        
        cls._model, cls._metadata = load_model(str(MODEL_PATH))
        cls._model_type = cls._metadata.get('model_type', 'Unknown') if cls._metadata else 'Unknown'
        return cls._model
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache."""
        cls._model = None
        cls._metadata = None
        cls._model_type = None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "StockSense-AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status."""
    model_loaded = ModelCache._model is not None or model_exists(str(MODEL_PATH))
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_information():
    """Get information about the current model."""
    if not model_exists(str(MODEL_PATH)):
        return ModelInfoResponse(model_exists=False)
    
    info = get_model_info(str(MODEL_PATH))
    metadata = info.get('metadata', {})
    
    return ModelInfoResponse(
        model_exists=True,
        model_type=metadata.get('model_type'),
        metrics=metadata.get('metrics'),
        trained_at=metadata.get('saved_at'),
        file_size_mb=info.get('size_mb')
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload the model from disk (useful after retraining)."""
    try:
        ModelCache.clear_cache()
        ModelCache.load_model()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["Predictions"])
async def predict_single(request: PredictRequest):
    """
    Generate a single sales prediction.
    
    - **date**: Date for prediction (YYYY-MM-DD format)
    - **store_id**: Optional store identifier
    - **product_id**: Optional product identifier
    """
    try:
        # Validate date
        try:
            pred_date = datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Load model
        model, metadata, model_type = ModelCache.get_model()
        
        # Generate prediction based on model type
        prediction = _generate_prediction(model, model_type, pred_date, 1)
        
        return PredictResponse(
            date=request.date,
            prediction=round(prediction[0], 2),
            confidence_lower=round(prediction[0] * 0.9, 2),  # Simplified CI
            confidence_upper=round(prediction[0] * 1.1, 2),
            store_id=request.store_id,
            product_id=request.product_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast", response_model=ForecastResponse, tags=["Predictions"])
async def forecast_multi_day(request: ForecastRequest):
    """
    Generate multi-day sales forecast.
    
    - **start_date**: Start date for forecast
    - **days**: Number of days to forecast (1-365)
    - **store_id**: Optional store filter
    - **promotional_uplift**: Percentage uplift for promotions (-50 to 100)
    """
    try:
        # Validate date
        try:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Load model
        model, metadata, model_type = ModelCache.get_model()
        
        # Generate predictions
        predictions = _generate_prediction(model, model_type, start_date, request.days)
        
        # Apply promotional uplift
        uplift_factor = 1 + (request.promotional_uplift / 100)
        adjusted_predictions = predictions * uplift_factor
        
        # Create forecast list
        forecasts = []
        for i, pred in enumerate(adjusted_predictions):
            forecast_date = start_date + timedelta(days=i)
            forecasts.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "prediction": round(pred, 2),
                "confidence_lower": round(pred * 0.85, 2),
                "confidence_upper": round(pred * 1.15, 2),
                "day_of_week": forecast_date.strftime("%A")
            })
        
        return ForecastResponse(
            forecasts=forecasts,
            total_predicted_sales=round(sum(adjusted_predictions), 2),
            average_daily_sales=round(np.mean(adjusted_predictions), 2),
            model_type=model_type,
            generated_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/quick", tags=["Predictions"])
async def quick_forecast(
    days: int = Query(default=7, ge=1, le=90, description="Days to forecast"),
    uplift: float = Query(default=0, ge=-50, le=100, description="Promotional uplift %")
):
    """
    Quick forecast endpoint using query parameters.
    
    Example: /forecast/quick?days=7&uplift=10
    """
    request = ForecastRequest(
        start_date=datetime.now().strftime("%Y-%m-%d"),
        days=days,
        promotional_uplift=uplift
    )
    return await forecast_multi_day(request)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_prediction(model, model_type: str, start_date: datetime, days: int) -> np.ndarray:
    """
    Generate predictions using the loaded model.
    
    Args:
        model: Trained model object
        model_type: Type of model ('XGBoost', 'ARIMA', 'Prophet')
        start_date: Start date for predictions
        days: Number of days to predict
        
    Returns:
        Array of predictions
    """
    if model_type == 'Prophet':
        from prophet import Prophet
        
        # Create future dataframe
        future = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=days, freq='D')})
        forecast = model.predict(future)
        return forecast['yhat'].values
        
    elif model_type == 'ARIMA':
        # ARIMA forecast
        forecast = model.forecast(steps=days)
        return np.array(forecast)
        
    else:  # XGBoost or other sklearn-like models
        # For XGBoost, we need to create features
        # This is a simplified version - in production, you'd want historical data
        
        # Generate date features for prediction
        predictions = []
        for i in range(days):
            pred_date = start_date + timedelta(days=i)
            
            # Create basic features
            features = {
                'year': pred_date.year,
                'month': pred_date.month,
                'day': pred_date.day,
                'day_of_week': pred_date.weekday(),
                'day_of_year': pred_date.timetuple().tm_yday,
                'week_of_year': pred_date.isocalendar()[1],
                'quarter': (pred_date.month - 1) // 3 + 1,
                'is_weekend': 1 if pred_date.weekday() >= 5 else 0,
                'is_month_start': 1 if pred_date.day == 1 else 0,
                'is_month_end': 1 if pred_date.day >= 28 else 0,
            }
            
            # Add placeholder lag/rolling features (in production, use actual historical data)
            features['sales_lag_1'] = 120.0  # Placeholder
            features['sales_lag_7'] = 115.0  # Placeholder
            features['sales_lag_30'] = 110.0  # Placeholder
            features['rolling_mean_7'] = 118.0  # Placeholder
            features['rolling_std_7'] = 10.0  # Placeholder
            
            # Create DataFrame for prediction
            X = pd.DataFrame([features])
            
            # Get expected feature order from model if available
            if hasattr(model, 'feature_names_in_'):
                X = X.reindex(columns=model.feature_names_in_, fill_value=0)
            
            pred = model.predict(X)[0]
            predictions.append(pred)
        
        return np.array(predictions)


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": True,
        "status_code": exc.status_code,
        "message": exc.detail
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Starting StockSense-AI API")
    print("=" * 60)
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs:  http://localhost:8000/redoc")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
