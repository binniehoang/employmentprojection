import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from src.features import (
    select_important_features,
    scale_features,
    load_cleaned_data,
    get_features_and_target,
    encode_categorical_features,
    handle_missing_values
)

def test_select_important_features():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
    y = np.random.rand(100)
    X_selected = select_important_features(X, y, n_features=3)
    assert X_selected.shape[1] == 3
    assert all(col in X.columns for col in X_selected.columns)

def test_scale_features():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    X_scaled = scale_features(X)
    np.testing.assert_almost_equal(X_scaled.mean().values, 0, decimal=6)
    np.testing.assert_almost_equal(X_scaled.std(ddof=0).values, 1, decimal=6)

def test_load_cleaned_data(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    file = tmp_path / 'test.csv'
    df.to_csv(file, index=False)
    loaded = load_cleaned_data(str(file))
    pd.testing.assert_frame_equal(df, loaded)

def test_get_features_and_target():
    df = pd.DataFrame({'Employment 2034': [1, 2], 'f1': [3, 4], 'f2': [5, 6]})
    X, y = get_features_and_target(df, target_column='Employment 2034')
    assert 'Employment 2034' not in X.columns
    assert all(y == df['Employment 2034'])

def test_encode_categorical_features():
    X = pd.DataFrame({'cat': ['a', 'b', 'a'], 'num': [1, 2, 3]})
    X_encoded = encode_categorical_features(X)
    assert 'cat_b' in X_encoded.columns
    assert 'cat' not in X_encoded.columns
    assert X_encoded.shape[1] == 2

def test_handle_missing_values():
    X = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
    # mean
    X_mean = handle_missing_values(X, strategy='mean')
    assert not X_mean.isnull().any().any()
    # median
    X_median = handle_missing_values(X, strategy='median')
    assert not X_median.isnull().any().any()
    # drop
    X_drop = handle_missing_values(X, strategy='drop')
    assert X_drop.shape[0] < X.shape[0]
