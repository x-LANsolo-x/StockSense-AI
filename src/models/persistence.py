"""
Model Persistence Module
========================
Save and load trained models using joblib for production deployment.

Why persist models?
- Avoid re-training on every page refresh
- Load pre-trained "brain" instantly
- Enable model versioning and rollback
"""

import joblib
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict] = None,
    overwrite: bool = True
) -> str:
    """
    Save a trained model to disk using joblib.
    
    Args:
        model: Trained model object (ARIMA, XGBoost, Prophet, etc.)
        filepath: Path to save the model (.pkl file)
        metadata: Optional metadata dictionary (metrics, training date, etc.)
        overwrite: Whether to overwrite existing file (default: True)
        
    Returns:
        Path where model was saved
        
    Example:
        save_model(trained_xgb, 'models/xgboost_model.pkl', 
                   metadata={'mae': 5.2, 'trained_on': '2024-01-15'})
    """
    filepath = Path(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"Model file already exists: {filepath}. Set overwrite=True to replace.")
    
    # Add default metadata
    if metadata is None:
        metadata = {}
    
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_type'] = type(model).__name__
    
    # Save model
    joblib.dump(model, filepath)
    print(f"✅ Model saved to: {filepath}")
    
    # Save metadata alongside model
    metadata_path = filepath.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    return str(filepath)


def load_model(filepath: str, load_metadata: bool = True) -> Tuple[Any, Optional[Dict]]:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model (.pkl file)
        load_metadata: Whether to also load metadata (default: True)
        
    Returns:
        Tuple of (model, metadata) or (model, None) if no metadata
        
    Example:
        model, metadata = load_model('models/xgboost_model.pkl')
        predictions = model.predict(X_test)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load model
    model = joblib.load(filepath)
    print(f"✅ Model loaded from: {filepath}")
    
    # Load metadata if requested
    metadata = None
    if load_metadata:
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded from: {metadata_path}")
    
    return model, metadata


def model_exists(filepath: str) -> bool:
    """Check if a saved model exists."""
    return Path(filepath).exists()


def get_model_info(filepath: str) -> Dict:
    """
    Get information about a saved model without loading it.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Dictionary with model information
    """
    filepath = Path(filepath)
    
    info = {
        'exists': filepath.exists(),
        'filepath': str(filepath),
        'size_mb': None,
        'metadata': None
    }
    
    if filepath.exists():
        info['size_mb'] = round(filepath.stat().st_size / (1024 * 1024), 2)
        
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                info['metadata'] = json.load(f)
    
    return info


def save_champion_model(
    model: Any,
    model_name: str,
    metrics: Dict[str, float],
    config: Optional[Dict] = None
) -> str:
    """
    Save a model as the "champion" (best performing model).
    
    This is the model that will be used in production/dashboard.
    
    Args:
        model: Trained model object
        model_name: Name of the model (e.g., 'XGBoost', 'Prophet')
        metrics: Performance metrics dictionary
        config: Model configuration/hyperparameters
        
    Returns:
        Path where champion model was saved
    """
    from config.settings import MODEL_PATH, ensure_directories
    
    ensure_directories()
    
    metadata = {
        'model_name': model_name,
        'metrics': metrics,
        'config': config or {},
        'is_champion': True
    }
    
    return save_model(model, MODEL_PATH, metadata=metadata)


def load_champion_model() -> Tuple[Any, Optional[Dict]]:
    """
    Load the champion model for production use.
    
    Returns:
        Tuple of (model, metadata)
    """
    from config.settings import MODEL_PATH
    
    return load_model(MODEL_PATH)


def save_all_models(
    models: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]]
) -> Dict[str, str]:
    """
    Save multiple models at once.
    
    Args:
        models: Dictionary mapping model names to model objects
        metrics: Dictionary mapping model names to their metrics
        
    Returns:
        Dictionary mapping model names to their save paths
    """
    from config.settings import get_model_path, ensure_directories
    
    ensure_directories()
    saved_paths = {}
    
    for name, model in models.items():
        filepath = get_model_path(name)
        model_metrics = metrics.get(name, {})
        
        save_model(model, filepath, metadata={'metrics': model_metrics})
        saved_paths[name] = str(filepath)
    
    return saved_paths


def list_saved_models(models_dir: str = None) -> list:
    """
    List all saved models in the models directory.
    
    Args:
        models_dir: Path to models directory (uses config default if None)
        
    Returns:
        List of dictionaries with model information
    """
    if models_dir is None:
        from config.settings import MODELS_DIR
        models_dir = MODELS_DIR
    
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        return []
    
    models = []
    for pkl_file in models_dir.glob('*.pkl'):
        info = get_model_info(pkl_file)
        models.append(info)
    
    return models


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from config.settings import ensure_directories, MODELS_DIR
    
    print("=" * 60)
    print("Testing Model Persistence Module")
    print("=" * 60)
    
    ensure_directories()
    
    # Create a simple mock model for testing
    class MockModel:
        def __init__(self):
            self.trained = True
            self.params = {'n_estimators': 100}
        
        def predict(self, X):
            return np.ones(len(X)) * 100
    
    mock_model = MockModel()
    
    # Test save_model
    print("\n--- Testing save_model() ---")
    test_path = MODELS_DIR / "test_model.pkl"
    save_model(
        mock_model, 
        test_path, 
        metadata={'mae': 5.5, 'rmse': 7.2, 'mape': '4.5%'}
    )
    
    # Test load_model
    print("\n--- Testing load_model() ---")
    loaded_model, metadata = load_model(test_path)
    print(f"   Model type: {type(loaded_model).__name__}")
    print(f"   Metadata: {metadata}")
    
    # Test model_exists
    print("\n--- Testing model_exists() ---")
    print(f"   Test model exists: {model_exists(test_path)}")
    print(f"   Fake model exists: {model_exists('models/fake.pkl')}")
    
    # Test get_model_info
    print("\n--- Testing get_model_info() ---")
    info = get_model_info(test_path)
    print(f"   Info: {info}")
    
    # Test list_saved_models
    print("\n--- Testing list_saved_models() ---")
    models = list_saved_models()
    print(f"   Found {len(models)} saved model(s)")
    
    # Cleanup test file
    import os
    os.remove(test_path)
    os.remove(test_path.with_suffix('.json'))
    print("\n✅ Cleaned up test files")
    
    print("\n" + "=" * 60)
    print("✅ All persistence tests passed!")
    print("=" * 60)
