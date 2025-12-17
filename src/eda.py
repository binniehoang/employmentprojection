import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# exploratory Data Analysis (EDA) on cleaned employment projections data
df = pd.read_csv('data/cleaned_employment_projections.csv')

#show basic info
df = pd.read_csv('data/cleaned_employment_projections.csv')
print(df.info())
print(df.describe())

#check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Plot distributions of key numeric columns
numeric_cols = [
    'Employment 2024',
    'Employment 2034',
    'Employment Change, 2024-2034',
    'Employment Percent Change, 2024-2034',
    'Occupational Openings, 2024-2034 Annual Average',
    'Median Annual Wage 2024'
]
os.makedirs('plots', exist_ok=True)
for col in numeric_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'plots/{col}_distribution.png')
        plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Columns')
plt.show()

