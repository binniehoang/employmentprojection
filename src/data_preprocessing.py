import pandas as pd
import numpy as np

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

# Only drop rows missing in essential columns
df = df.dropna(subset=essential_cols)
print(f"Shape after dropping rows with missing values in essential columns: {df.shape}")

# Optionally fill missing values in other columns
# df = df.fillna({'SomeColumn': value})

# Remove duplicates
df = df.drop_duplicates()
print(f"Shape after dropping duplicates: {df.shape}")


# Identify categorical columns to encode (exclude numeric columns)
categorical_cols = [
	'Occupation Title',
	'Typical Entry-Level Education',
	'Work Experience in a Related Occupation',
	'Typical on-the-job Training'
]

# Only encode categorical columns that exist in the DataFrame
cols_to_encode = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
print(f"Shape after encoding categorical variables: {df.shape}")

# Save cleaned data
df.to_csv('data/cleaned_employment_projections.csv', index=False)