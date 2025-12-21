# Model training template for employment projections
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


# Load processed features and target
X = pd.read_csv('model_data/selected_features.csv')
y = pd.read_csv('model_data/target.csv').squeeze()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
print(f"Test R^2: {r2:.2f}")

# Save the trained model
os.makedirs('model_data', exist_ok=True)
joblib.dump(model, 'model_data/random_forest_model.joblib')
print("Model saved to model_data/random_forest_model.joblib")

# Example prediction
example_data = X_test.iloc[:5]
example_predictions = model.predict(example_data)
print("Example predictions on test data:")
print(example_predictions)

