import os
def select_important_features(X, y, n_features=10, random_state=42):
	
	# selects top n_features
	
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
	# returns dataframe with scaled numeric features
	scaler = StandardScaler()
	numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
	X_scaled = X.copy()
	X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
	return X_scaled

def load_cleaned_data(filepath='data/cleaned_employment_projections.csv'):
	# loads the cleaned employment projections dataset from the specified CSV file
	# returns a pandas DataFrame
	try:
		df = pd.read_csv(filepath)
	except FileNotFoundError as e:
		raise FileNotFoundError(f"File not found: {filepath}") from e
	except pd.errors.ParserError as e:
		raise pd.errors.ParserError(f"Error parsing CSV file at path '{filepath}': {e}") from e
	return df
	

def get_features_and_target(df, target_column='Employment 2034'):
	X = df.drop(columns=[target_column])
	y = df[target_column]
	return X, y


# encode categorical features in the input dataframe using one-hot encoding
def encode_categorical_features(X):
	categorical_cols = X.select_dtypes(include=['object', 'category']).columns # parameters: input dataframe containing categorical and numeric features
	X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
	return X_encoded

def handle_missing_values(X, strategy='mean'):
	# handles missing values in the DataFrame X based on the specified strategy
	if strategy == 'drop':
		return X.dropna()
	elif strategy == 'median':
		return X.fillna(X.median())
	else:  # default to mean
		return X.fillna(X.mean())



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
