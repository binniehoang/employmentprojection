import pandas as pd
import pytest
from src.data_preprocessing import df as processed_df
import os

def test_no_missing_values_in_essential_columns():
    essential_cols = [
        'Employment 2024',
        'Employment 2034',
        'Median Annual Wage 2024'
    ]
    for col in essential_cols:
        assert processed_df[col].isnull().sum() == 0, f"Missing values found in column: {col}"
def test_no_duplicates():
    assert processed_df.duplicated().sum() == 0, "Duplicates found in the processed DataFrame"
def test_categorical_columns_encoded():
    categorical_cols = [
        'Occupation Title',
        'Typical Entry-Level Education',
        'Work Experience in a Related Occupation',
        'Typical on-the-job Training'
    ]
    for col in categorical_cols:
        encoded_cols = [c for c in processed_df.columns if c.startswith(col + '_')]
        assert len(encoded_cols) > 0, f"Categorical column {col} not encoded properly"
def test_numeric_columns_type():
    numeric_cols = [
        'Employment 2024',
        'Employment 2034',
        'Median Annual Wage 2024'
    ]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(processed_df[col]), f"Column {col} is not numeric"
def test_cleaned_data_saved():
    assert os.path.exists('data/cleaned_employment_projections.csv'), "Cleaned data file not found"
