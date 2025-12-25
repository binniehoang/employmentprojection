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
column_labels = {
    'Employment 2024': 'Employment (2024)',
    'Employment 2034': 'Employment (2034)',
    'Employment Change, 2024-2034': 'Employment Change (2024-2034)',
    'Employment Percent Change, 2024-2034': 'Employment % Change (2024-2034)',
    'Occupational Openings, 2024-2034 Annual Average': 'Annual Avg Openings (2024-2034)',
    'Median Annual Wage 2024': 'Median Annual Wage (2024)'
}
for col in numeric_cols:
    if col in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col].dropna(), kde=True)
        label = column_labels.get(col, col)
        plt.title(f'Distribution of {label}')
        plt.xlabel(label, fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'plots/{col}_distribution.png')
        plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 10))
heatmap_labels = [column_labels.get(col, col) for col in numeric_cols]
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', annot_kws={"size": 9},
            xticklabels=heatmap_labels, yticklabels=heatmap_labels)
plt.title('Correlation Heatmap (2024-2034)')
plt.tight_layout()
plt.show()

