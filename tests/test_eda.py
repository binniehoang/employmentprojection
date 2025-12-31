import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pytest

@pytest.fixture
def df():
    return pd.read_csv('data/cleaned_employment_projections.csv')
def test_dataframe_info(df):
    assert df is not None
    assert not df.empty
    assert 'Employment 2034' in df.columns
def test_missing_values(df):
    missing_values = df.isnull().sum()
    assert missing_values.sum() == 0  # just check that the sum is non-negative
def test_numeric_columns_distribution(df):
    numeric_cols = [
        'Employment 2024',
        'Employment 2034',
        'Employment Change, 2024-2034',
        'Employment Percent Change, 2024-2034',
        'Occupational Openings, 2024-2034 Annual Average',
        'Median Annual Wage 2024'
    ]
    for col in numeric_cols:
        assert col in df.columns
        assert df[col].dtype in [float, int]
def test_correlation_heatmap(df):
    numeric_cols = [
        'Employment 2024',
        'Employment 2034',
        'Employment Change, 2024-2034',
        'Employment Percent Change, 2024-2034',
        'Occupational Openings, 2024-2034 Annual Average',
        'Median Annual Wage 2024'
    ]
    corr_matrix = df[numeric_cols].corr()
    assert corr_matrix.shape == (len(numeric_cols), len(numeric_cols))
    assert not corr_matrix.isnull().values.any()
def test_plots_creation():
    os.makedirs('plots', exist_ok=True)
    numeric_cols = [
        'Employment 2024',
        'Employment 2034',
        'Employment Change, 2024-2034',
        'Employment Percent Change, 2024-2034',
        'Occupational Openings, 2024-2034 Annual Average',
        'Median Annual Wage 2024'
    ]
    plot_files = []
    # Generate and save distribution plots for each numeric column
    for col in numeric_cols:
        plot_path = f'plots/{col}_distribution.png'
        plt.figure()
        sns.histplot(df[col].dropna())
        plt.title(f'{col} distribution')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plot_files.append(plot_path)
    # Generate and save correlation heatmap
    corr_matrix = df[numeric_cols].corr()
    plt.figure()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.tight_layout()
    corr_plot_path = 'plots/correlation_heatmap.png'
    plt.savefig(corr_plot_path)
    plt.close()
    plot_files.append(corr_plot_path)
    for plot_file in plot_files:
        assert os.path.exists(plot_file)

# Note: These tests check for the existence of data, correct data types, and the creation of plots.
# They do not generate plots themselves; that would be part of the EDA script.

