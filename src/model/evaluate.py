import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score



def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        raise
    except Exception as e:
        print(f"Error: Failed to load model from '{model_path}': {e}")
        raise
    
def load_data(data_path):
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_path}'.")
        raise
    except Exception as e:
        print(f"Error: Failed to load data from '{data_path}': {e}")
        raise

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # regression
    mse = mean_squared_error(y_test, y_pred)
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R^2 Score: {r2:.4f}")
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

if __name__ == '__main__':
    model = load_model("model_data/random_forest_model.joblib")
    data = load_data("data/cleaned_employment_projections.csv")
    X_test = data.drop("Employment 2034", axis=1)
    y_test = data["Employment 2034"]

    # Load feature columns used during training
    try:
        feature_cols = pd.read_csv("model_data/selected_features.csv", nrows=0).columns.tolist()
    except FileNotFoundError:
        print("Error: Feature columns file not found at 'model_data/selected_features.csv'.")
        raise
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse feature columns CSV: {e}")
        raise
    except Exception as e:
        print(f"Error: Failed to load feature columns from 'model_data/selected_features.csv': {e}")
        raise

    # Add missing columns as 0, drop extra columns, and ensure correct order
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_cols]

    metrics = evaluate_model(model, X_test, y_test)
    # metrics dict now contains mse, rmse, mae, r2

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
