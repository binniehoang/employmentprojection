import numpy as np
import pandas as pd
import tempfile
import os
import joblib
import pytest
from sklearn.ensemble import RandomForestRegressor
from src.model import evaluate

def test_evaluate_model_metrics():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    y = np.array([1, 2, 3])
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.joblib')
        joblib.dump(model, model_path)
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
