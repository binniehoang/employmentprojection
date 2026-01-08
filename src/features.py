# feature selection for employment projections dataset
import os
def select_important_features(X, y, n_features=10, random_state=42):
	"""
	Select the top n most important features using a RandomForestRegressor.
    
	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series or np.ndarray): Target variable.
		n_features (int): Number of top features to select.
		random_state (int): Random seed for reproducibility.
    
	Returns:
		pd.DataFrame: DataFrame containing only the selected features.
	"""
	model = RandomForestRegressor(n_estimators=100, random_state=random_state)
	model.fit(X, y)
	importances = model.feature_importances_
	indices = importances.argsort()[::-1][:n_features]
	selected_columns = X.columns[indices]
	return X[selected_columns]

# feature selection for employment projections dataset
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
def scale_features(X):
	"""
	Scale numeric features in the DataFrame using StandardScaler.
    
	Args:
		X (pd.DataFrame): Input DataFrame with numeric features.
    
	Returns:
		pd.DataFrame: DataFrame with scaled numeric features.
	"""
	scaler = StandardScaler()
	numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
	X_scaled = X.copy()
	X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
	return X_scaled

def load_cleaned_data(filepath='data/cleaned_employment_projections.csv'):
	"""
	Load the cleaned employment projections dataset from a CSV file.
    
	Args:
		filepath (str): Path to the cleaned CSV file.
    
	Returns:
		pd.DataFrame: Loaded DataFrame.
    
	Raises:
		FileNotFoundError: If the file does not exist.
		pd.errors.ParserError: If the CSV cannot be parsed.
	"""
	try:
		df = pd.read_csv(filepath)
	except FileNotFoundError as e:
		raise FileNotFoundError(f"File not found: {filepath}") from e
	except pd.errors.ParserError as e:
		raise pd.errors.ParserError(f"Error parsing CSV file at path '{filepath}': {e}") from e
	return df
	

def get_features_and_target(df, target_column='Employment 2034'):
	"""
	Split a DataFrame into features and target.
    
	Args:
		df (pd.DataFrame): Input DataFrame.
		target_column (str): Name of the target column.
    
	Returns:
		Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).
	"""
	X = df.drop(columns=[target_column])
	y = df[target_column]
	return X, y


# encode categorical features in the input dataframe using one-hot encoding
def encode_categorical_features(X):
	"""
	Encode categorical features in the DataFrame using one-hot encoding.
    
	Args:
		X (pd.DataFrame): Input DataFrame containing categorical and numeric features.
    
	Returns:
		pd.DataFrame: DataFrame with categorical features one-hot encoded.
	"""
	categorical_cols = X.select_dtypes(include=['object', 'category']).columns
	X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
	return X_encoded

def handle_missing_values(X, strategy='mean'):
	"""
	Handle missing values in the DataFrame X based on the specified strategy.
    
	Args:
		X (pd.DataFrame): Input DataFrame.
		strategy (str): Strategy to handle missing values ('mean', 'median', 'drop').
    
	Returns:
		pd.DataFrame: DataFrame with missing values handled.
	"""
	if strategy == 'drop':
		return X.dropna()
	elif strategy == 'median':
		return X.fillna(X.median())
	else:  # default to mean
		return X.fillna(X.mean())

# Add script entry point for standalone execution
if __name__ == "__main__":
	

	df = load_cleaned_data()
	# Remove commas and convert target column to float
	df['Employment 2034'] = df['Employment 2034'].replace(',', '', regex=True).astype(float)
	X, y = get_features_and_target(df)
	X_encoded = encode_categorical_features(X)
	X_encoded = handle_missing_values(X_encoded, strategy='mean')
	X_scaled = scale_features(X_encoded)
	X_selected = select_important_features(X_scaled, y, n_features=10)

	# Save selected features and target to files for modeling
	os.makedirs('model_data', exist_ok=True)
	X_selected.to_csv('model_data/selected_features.csv', index=False)
	y.to_csv('model_data/target.csv', index=False)
	print("Top 10 selected features saved to model_data/selected_features.csv")
	print("Target saved to model_data/target.csv")
	X_encoded.to_csv('data/encoded_features.csv', index=False)
	print("Encoded features saved to data/encoded_features.csv")
	X_scaled.to_csv('data/scaled_features.csv', index=False)
	print("Scaled features saved to data/scaled_features.csv")