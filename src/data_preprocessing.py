"""
Data preprocessing script for the employment projections dataset.

This script loads the raw employment projections data, handles missing values, 
removes duplicates, and saves the cleaned data to a CSV file. Categorical encoding
is deliberately not performed here to maintain consistency with the ML pipeline.
"""

import pandas as pd

def get_preprocessing_requirements(data_path='data/Employment Projections.csv'):
	"""
	Analyze the dataset to determine preprocessing requirements without performing the preprocessing.
	
	Args:
		data_path (str): Path to the raw data file
	
	Returns:
		dict: Information about what preprocessing is needed, including categorical columns
	"""
	df = pd.read_csv(data_path)
	
	categorical_cols = [
		'Typical Entry-Level Education',
		'Work Experience in a Related Occupation', 
		'Typical on-the-job Training',
		'Occupation Title',
		'Occupation Code'
	]
	
	existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
	target_col = 'Employment 2034'
	excluded_cols = existing_categorical_cols + [target_col]
	numeric_cols = [col for col in df.columns if col not in excluded_cols]
	
	return {
		'categorical_columns': existing_categorical_cols,
		'numeric_columns': numeric_cols,
		'target_column': target_col,
		'total_rows': len(df),
		'encoding_required': len(existing_categorical_cols) > 0
	}

def main():
	"""
	Main preprocessing workflow for the employment projections dataset.
	Loads raw data, performs basic cleaning, and saves the cleaned dataset.
	
	IMPORTANT: This function performs basic data cleaning (handling missing values,
	removing duplicates) but does NOT encode categorical variables. Categorical
	encoding must be handled separately by the calling code to maintain consistency
	with the model training pipeline.
	
	Returns:
		dict: Preprocessing metadata including:
			- categorical_columns: List of categorical columns that need encoding
			- numeric_columns: List of numeric columns ready for use
			- output_path: Path to the saved cleaned data
			- rows_processed: Number of rows in the cleaned dataset
	
	Note: If this function is called independently, the caller must handle
	categorical encoding before using the data for ML modeling.
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

	# Identify categorical and numeric columns for caller reference
	categorical_cols = [
		'Typical Entry-Level Education',
		'Work Experience in a Related Occupation', 
		'Typical on-the-job Training',
		'Occupation Title',  # Note: Usually dropped but listed for completeness
		'Occupation Code'    # Note: Usually dropped but listed for completeness
	]
	
	# Filter to only include categorical columns that actually exist in the data
	existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
	
	# Identify numeric columns (excluding categorical and target columns)
	target_col = 'Employment 2034'
	excluded_cols = existing_categorical_cols + [target_col]
	numeric_cols = [col for col in df.columns if col not in excluded_cols]

	# IMPORTANT: Categorical columns are NOT encoded here. 
	# This preserves the raw categorical values for consistent encoding in the ML pipeline.
	# The calling code must handle categorical encoding to maintain consistency with
	# saved encoders and feature selections.
	print(f"Shape after cleaning (categorical encoding required): {df.shape}")
	print(f"Categorical columns requiring encoding: {existing_categorical_cols}")
	print(f"Numeric columns ready for use: {numeric_cols}")

	# Save cleaned data
	output_path = 'data/cleaned_employment_projections.csv'
	df.to_csv(output_path, index=False)
	
	# Return preprocessing metadata for the caller
	preprocessing_metadata = {
		'categorical_columns': existing_categorical_cols,
		'numeric_columns': numeric_cols, 
		'output_path': output_path,
		'rows_processed': len(df),
		'target_column': target_col,
		'encoding_required': len(existing_categorical_cols) > 0
	}
	
	print("Data preprocessing completed. Categorical encoding required before ML modeling.")
	return preprocessing_metadata

if __name__ == "__main__":
	main()