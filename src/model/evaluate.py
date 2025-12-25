import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score



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

    # Load feature columns used during training
    feature_cols = pd.read_csv("model_data/selected_features.csv", nrows=0).columns.tolist()

    # Add missing columns as 0, drop extra columns, and ensure correct order
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_cols]

    evaluate_model(model, X_test, y_test)

    # Show feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        print("\nTop Feature Importances:")
        print(feature_importance_df.head(20).to_string(index=False))
    else:
        print("\nModel does not provide feature importances.")
