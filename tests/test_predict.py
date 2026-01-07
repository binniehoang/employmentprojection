import os
import tempfile
import pandas as pd
import pytest
from src.model import predict

def test_predict_model_missing_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'model_data'), exist_ok=True)
        pd.DataFrame({'a': [1, 2], 'b': [3, 4]}).to_csv(os.path.join(tmpdir, 'model_data', 'selected_features.csv'), index=False)
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with pytest.raises(FileNotFoundError):
                predict.predict_model()
        finally:
            os.chdir(cwd)
