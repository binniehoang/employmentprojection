# feature selection for employment projections dataset
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def load_cleaned_data(filepath='data/cleaned_employment_projections.csv'):
	"""
	Loads the cleaned employment projections dataset from the specified CSV file.
	Returns a pandas DataFrame.
	"""
	df = pd.read_csv(filepath)
	return df

# def get_features_and_target(df, target_column='Employment 2034'):
# 	X = df.drop(columns=[target_column])
# 	y = df[target_column]
# 	return X, y

def show_column_names():
    """
    Prints the column names of the cleaned employment projections dataset.
    """
    df = load_cleaned_data()
    # print("Column names in the dataset:")
    # for col in df.columns:
    #     print(col)
    print(df.columns.tolist())