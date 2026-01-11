# StockSense-AI

**Intelligent Sales Forecasting & Inventory Optimization**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51+-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

StockSense-AI is a production-ready machine learning system that helps retail businesses predict future sales, optimize inventory levels, and make data-driven decisions. Built with Python, it combines multiple forecasting models with an interactive dashboard and RESTful API.

---

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Capabilities

- **Multi-Model Forecasting**: Choose from ARIMA (baseline), XGBoost (gradient boosting), or Prophet (Facebook's time series library)
- **Interactive Dashboard**: User-friendly Streamlit interface for data upload, model training, and visualization
- **RESTful API**: FastAPI backend for programmatic access and integration with other systems
- **What-If Analysis**: Simulate promotional impacts and market trends on forecasts
- **Automated Feature Engineering**: Lag features, rolling statistics, and temporal features generated automatically
- **Model Comparison**: Side-by-side evaluation with MAE, RMSE, MAPE, and R² metrics
- **Export Functionality**: Download forecasts as CSV for further analysis

### Dashboard Features

| Tab | Functionality |
|-----|---------------|
| **Upload Data** | CSV upload, data validation, quality metrics, seasonality analysis |
| **Model Training** | Model selection, hyperparameter tuning, progress tracking, performance metrics |
| **Forecast Dashboard** | Interactive forecasting, What-If sliders, confidence intervals, CSV export |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and model status |
| `GET` | `/model/info` | Model metadata and metrics |
| `POST` | `/predict` | Single-day prediction |
| `POST` | `/forecast` | Multi-day forecast with confidence intervals |
| `GET` | `/forecast/quick` | Quick forecast via query parameters |

---

## Demo

### Dashboard Interface

The StockSense-AI dashboard provides a clean, professional interface for:

1. **Data Upload & Validation**
   - Upload CSV files with Date and Sales columns
   - Automatic data quality checks (missing values, date parsing)
   - Seasonality analysis (by day of week, by month)

2. **Model Training**
   - Select from XGBoost, ARIMA, or Prophet
   - Configure hyperparameters via intuitive sliders
   - Real-time training progress with performance metrics

3. **Interactive Forecasting**
   - Adjustable forecast horizon (7-90 days)
   - What-If analysis with promotional uplift (-30% to +50%)
   - Trend adjustment controls
   - Export results to CSV

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        StockSense-AI                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Streamlit │    │   FastAPI   │    │    CLI      │         │
│  │  Dashboard  │    │     API     │    │  Pipeline   │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│  ┌─────────────────────────┴─────────────────────────┐         │
│  │                  Core Engine                       │         │
│  ├────────────────────────────────────────────────────┤         │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │         │
│  │  │   Data   │  │ Feature  │  │     Models       │ │         │
│  │  │Ingestion │→ │Engineering│→ │ ARIMA/XGB/Prophet│ │         │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │         │
│  └───────────────────────────────────────────────────┘         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Data Layer                            │   │
│  │  ┌─────────┐  ┌────────────┐  ┌─────────────────────┐   │   │
│  │  │ Raw CSV │  │ Processed  │  │   Trained Models    │   │   │
│  │  │  Data   │  │   Data     │  │      (.pkl)         │   │   │
│  │  └─────────┘  └────────────┘  └─────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/StockSense-AI.git
cd StockSense-AI
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Option 1: Run the Dashboard (Recommended)

```bash
streamlit run src/dashboard/app.py
```

Open http://localhost:8501 in your browser.

### Option 2: Run the API

```bash
uvicorn src.api.main:app --reload
```

API available at http://localhost:8000. Documentation at http://localhost:8000/docs.

### Option 3: Run the Pipeline (CLI)

```bash
python main.py
```

---

## Usage Guide

### 1. Prepare Your Data

Your CSV file should have at minimum:
- **Date column**: Dates in `YYYY-MM-DD` format
- **Sales column**: Numeric sales values

Example:
```csv
Date,Sales,Store
2024-01-01,150.50,Store_A
2024-01-02,142.30,Store_A
2024-01-03,168.75,Store_A
```

### 2. Upload and Process Data

1. Launch the dashboard: `streamlit run src/dashboard/app.py`
2. Go to **Upload Data** tab
3. Upload your CSV file
4. Select the Date and Sales columns
5. Click **Process Data**

### 3. Train a Model

1. Go to **Model Training** tab
2. Select model type (XGBoost, ARIMA, or Prophet)
3. Configure parameters (or use defaults)
4. Click **Train New Model**
5. View performance metrics (MAE, RMSE, MAPE, R²)

### 4. Generate Forecasts

1. Go to **Forecast Dashboard** tab
2. Set forecast horizon (days)
3. Adjust What-If parameters (optional)
4. Click **Generate Forecast**
5. Download results as CSV

### 5. API Usage

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"date": "2024-06-01", "store_id": 1}
)
print(response.json())
# {"date": "2024-06-01", "prediction": 156.78, ...}

# Multi-day forecast
response = requests.post(
    "http://localhost:8000/forecast",
    json={
        "start_date": "2024-06-01",
        "days": 30,
        "promotional_uplift": 10.0
    }
)
print(response.json())
```

---

## API Reference

### Health Check
```
GET /health
```
Returns API status and model availability.

### Model Information
```
GET /model/info
```
Returns model type, metrics, and training timestamp.

### Single Prediction
```
POST /predict
Content-Type: application/json

{
    "date": "2024-06-01",
    "store_id": 1,
    "product_id": 101
}
```

### Multi-Day Forecast
```
POST /forecast
Content-Type: application/json

{
    "start_date": "2024-06-01",
    "days": 30,
    "store_id": 1,
    "promotional_uplift": 10.0
}
```

### Quick Forecast
```
GET /forecast/quick?days=7&uplift=10
```

Full API documentation available at `/docs` (Swagger) or `/redoc` (ReDoc).

---

## Model Performance

Benchmark results on sample dataset (365 days, 30-day test period):

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| **Prophet** | 6.67 | 8.59 | 4.59% | 0.60 |
| **XGBoost** | 8.12 | 10.77 | 5.42% | 0.37 |
| **ARIMA** | 10.45 | 13.15 | 7.05% | 0.06 |

### Metric Definitions

- **MAE** (Mean Absolute Error): Average prediction error in original units
- **RMSE** (Root Mean Square Error): Penalizes large errors more heavily
- **MAPE** (Mean Absolute Percentage Error): Error as percentage (intuitive for business)
- **R²** (R-squared): Proportion of variance explained (1.0 = perfect)

---

## Testing

### Test Suite

The project includes comprehensive tests covering:

1. **Data Integrity Tests**
   - Date parsing verification
   - Chronological sorting
   - Missing value handling

2. **Feature Engineering Tests**
   - Column existence validation
   - Logic verification (spot checks)
   - Data leakage prevention

3. **Model Sanity Tests**
   - Output shape validation
   - NaN detection in forecasts
   - Reasonable value ranges

### Running Tests

```bash
# Run MVP test suite
python tests/test_mvp.py

# Run model comparison
python compare_models.py
```

### Test Results

```
=== DATA INTEGRITY TESTS ===
[PASS] Date column is datetime64
[PASS] Data is sorted chronologically
[PASS] No missing values in target column

=== FEATURE ENGINEERING TESTS ===
[PASS] Required columns exist: ['year', 'month', 'day_of_week']
[PASS] Logic check passed for date features

=== MODEL SANITY TESTS ===
[PASS] Forecast has exactly 7 values
[PASS] No NaN values in forecast
[PASS] Forecast values are finite

ALL TESTS PASSED!
```

### Data Leakage Prevention

We ensure no data leakage by:
- Using `shift()` for lag features (yesterday's data, not today's)
- Time-based train/test split (oldest 80% train, newest 20% test)
- No random shuffling of time series data

---

## Project Structure

```
StockSense-AI/
│
├── config/
│   ├── __init__.py
│   └── settings.py           # Centralized configuration
│
├── data/
│   ├── raw/                  # Input CSV files
│   │   └── sales.csv         # Sample data
│   └── processed/            # Generated outputs
│
├── models/                   # Saved trained models (.pkl)
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py           # FastAPI application
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py            # Streamlit dashboard
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py      # Data loading & cleaning
│   │   ├── loader.py         # CSV/Excel loaders
│   │   └── cleaner.py        # Data cleaning utilities
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py    # Feature engineering (lags, rolling)
│   │   └── engineer.py       # Additional feature utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py       # ARIMA model
│   │   ├── ml_models.py      # XGBoost & Prophet
│   │   ├── evaluation.py     # Metrics (MAE, RMSE, MAPE)
│   │   ├── persistence.py    # Model save/load
│   │   ├── arima.py          # ARIMA utilities
│   │   └── moving_average.py # Moving average models
│   │
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py        # Utility functions
│       └── metrics.py        # Evaluation metrics
│
├── tests/
│   ├── __init__.py
│   └── test_mvp.py           # MVP test suite
│
├── notebooks/
│   └── 01_getting_started.ipynb
│
├── main.py                   # CLI pipeline entry point
├── compare_models.py         # Model comparison script
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **ML/Data** | Pandas, NumPy, Scikit-learn, XGBoost, Prophet, Statsmodels |
| **Dashboard** | Streamlit, Plotly |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Persistence** | Joblib |
| **Data Format** | CSV, Parquet |

### Why These Technologies?

- **XGBoost**: State-of-the-art gradient boosting, excellent for tabular data
- **Prophet**: Handles seasonality and holidays automatically, robust to missing data
- **ARIMA**: Classic time series baseline, interpretable results
- **Streamlit**: Rapid prototyping, pure Python, interactive widgets
- **FastAPI**: High performance, automatic documentation, async support

---

## Configuration

Key settings in `config/settings.py`:

```python
# Forecasting
FORECAST_HORIZON = 30        # Default days to forecast
TEST_DAYS = 30               # Validation period

# Feature Engineering
LAG_FEATURES = [1, 7, 30]    # Lag periods
ROLLING_WINDOWS = [7]        # Rolling window sizes

# Model Parameters
ARIMA_ORDER = (5, 1, 0)
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1
}
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with modern Python ML/Data stack
- Inspired by real-world retail forecasting challenges
- Thanks to the open-source community for the amazing libraries

---

## Contact

For questions or feedback, please open an issue on GitHub.

---

**StockSense-AI** - Intelligent Sales Forecasting & Inventory Optimization
