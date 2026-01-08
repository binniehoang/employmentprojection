"""
Data preprocessing script for the employment projections dataset.

This script loads the raw employment projections data, handles missing values, encodes categorical variables, removes duplicates, and saves the cleaned data to a CSV file.
"""

import pandas as pd

def main():
	"""
	Main preprocessing workflow for the employment projections dataset.
	Loads raw data, cleans and encodes it, and saves the cleaned dataset.
	"""
	df = pd.read_csv('data/Employment Projections.csv')
	print(f"Initial shape: {df.shape}")

	# Inspect the dataframe
	print(df.info())
	print(df.head())

	# Handle missing values
	essential_cols = [
		'Employment 2024',
		'Employment 2034',
		'Median Annual Wage 2024'
	]

	# essential numeric columns are floats
	for col in essential_cols:
		df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

	# Only drop rows missing in essential columns
	before_drop = df.shape[0]
	df = df.dropna(subset=essential_cols)
	after_drop = df.shape[0]
	print(f"Dropped {before_drop - after_drop} rows due to non-numeric or missing values in essential columns.")

	# Remove duplicates
	df = df.drop_duplicates()
	print(f"Shape after dropping duplicates: {df.shape}")


	# Do NOT encode categorical columns here. Encoding will be handled in main.py for consistency.
	print(f"Shape after cleaning (no encoding): {df.shape}")

	# Save cleaned data
	df.to_csv('data/cleaned_employment_projections.csv', index=False)

if __name__ == "__main__":
	main()