import os
import tempfile
import pandas as pd
import numpy as np
import joblib
import pytest
from sklearn.ensemble import RandomForestRegressor
from src.model import train, predict, evaluate

def test_train_model_creates_feature_importances():
    # Create dummy data
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
    y = pd.Series([1, 2, 3, 4, 5])
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'model_data'), exist_ok=True)
        X.to_csv(os.path.join(tmpdir, 'model_data', 'selected_features.csv'), index=False)
        y.to_csv(os.path.join(tmpdir, 'model_data', 'target.csv'), index=False)
        # Patch file paths in train.py if needed
        # Run training
        # This test assumes train.main() uses relative paths
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            train.main()
            assert os.path.exists('model_data/feature_importances.csv')
        finally:
            os.chdir(cwd)

def test_predict_model_missing_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'model_data'), exist_ok=True)
        # Create dummy features file
        pd.DataFrame({'a': [1, 2], 'b': [3, 4]}).to_csv(os.path.join(tmpdir, 'model_data', 'selected_features.csv'), index=False)
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with pytest.raises(FileNotFoundError):
                predict.predict_model()
        finally:
            os.chdir(cwd)

def test_evaluate_model_metrics():
    # Create dummy model and data
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    y = np.array([1, 2, 3])
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.joblib')
        joblib.dump(model, model_path)
        # Evaluate
        metrics = evaluate.evaluate_model(model, X, y)
        assert 'mse' in metrics and 'rmse' in metrics and 'mae' in metrics and 'r2' in metrics
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert -1 <= metrics['r2'] <= 1

def test_evaluate_load_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        evaluate.load_model('nonexistent_model.joblib')

def test_evaluate_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        evaluate.load_data('nonexistent_data.csv')
