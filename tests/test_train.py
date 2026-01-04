import os
import tempfile
import pandas as pd
import pytest
from src.model import train

def test_train_model_creates_feature_importances():
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
    y = pd.Series([1, 2, 3, 4, 5])
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'model_data'), exist_ok=True)
        X.to_csv(os.path.join(tmpdir, 'model_data', 'selected_features.csv'), index=False)
        y.to_csv(os.path.join(tmpdir, 'model_data', 'target.csv'), index=False)
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            train.main()
            assert os.path.exists('model_data/feature_importances.csv')
        finally:
            os.chdir(cwd)
