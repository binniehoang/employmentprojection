import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
import os


def load_model(model_path):
    return joblib.load(model_path)

def load_data(data_path):
    return pd.read_csv(data_path)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # regression
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

if __name__ == '__main__':
    model = load_model("model_data/random_forest_model.joblib")
    data = load_data("data/cleaned_employment_projections.csv")
    X_test = data.drop("Employment 2034", axis=1)
    y_test = data["Employment 2034"]
    evaluate_model(model, X_test, y_test)