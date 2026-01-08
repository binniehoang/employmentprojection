"""
Main pipeline for employment projections project.
Runs data preprocessing, feature selection, model training, prediction, and evaluation.
"""

print("Starting main.py execution...")

import os
import pandas as pd
from src import config
from src import data_preprocessing
from src import features
from src.model import train, predict, evaluate
from sklearn.preprocessing import OneHotEncoder
import joblib


def main():
    print("Starting data preprocessing...")
    data_preprocessing.main()

    print("Loading cleaned data and encoding features with consistent columns...")
    raw_data = pd.read_csv(config.PROCESSED_DATA_PATH)
    target = raw_data['Employment 2034']  # Adjust target column as needed
    non_numeric_cols = ['Occupation Title', 'Occupation Code']
    drop_cols = [col for col in non_numeric_cols if col in raw_data.columns]
    categorical_cols = [
        'Typical Entry-Level Education',
        'Work Experience in a Related Occupation',
        'Typical on-the-job Training'
    ]
    # Only encode categorical columns that exist
    cols_to_encode = [col for col in categorical_cols if col in raw_data.columns]
    X = raw_data.drop(drop_cols + ['Employment 2034'], axis=1)
    X_categorical = raw_data[cols_to_encode] if cols_to_encode else pd.DataFrame()
    X_numeric = X.drop(cols_to_encode, axis=1) if cols_to_encode else X

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if not X_categorical.empty:
        encoder.fit(X_categorical)
        X_encoded = pd.DataFrame(encoder.transform(X_categorical), columns=encoder.get_feature_names_out(cols_to_encode))
        X_full = pd.concat([X_numeric.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    else:
        X_full = X_numeric
    # Save encoder for future use (optional)
    encoder_path = os.path.join(os.path.dirname(config.SELECTED_FEATURES_PATH), 'onehot_encoder.joblib')
    joblib.dump(encoder, encoder_path)

    # Save the full set of columns for alignment
    full_feature_columns_path = os.path.join(os.path.dirname(config.SELECTED_FEATURES_PATH), 'full_feature_columns.txt')
    with open(full_feature_columns_path, 'w') as f:
        for col in X_full.columns:
            f.write(f"{col}\n")

    print("Selecting important features...")
    selected_X = features.select_important_features(X_full, target, n_features=10)
    selected_X.to_csv(config.SELECTED_FEATURES_PATH, index=False)
    target.to_csv(config.TARGET_PATH, index=False)
    # Save feature names for later use
    feature_names_path = os.path.join(os.path.dirname(config.SELECTED_FEATURES_PATH), 'selected_feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for col in selected_X.columns:
            f.write(f"{col}\n")

    print("Training model...")
    train.main()

    print("Generating predictions...")
    # Use the same encoded data for prediction as for training
    pred_X_full = X_full.copy()
    # Align columns
    with open(full_feature_columns_path, 'r') as f:
        full_columns = [line.strip() for line in f.readlines()]
    for col in full_columns:
        if col not in pred_X_full.columns:
            pred_X_full[col] = 0
    pred_X_full = pred_X_full[full_columns]
    # Align to selected features
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    for col in feature_names:
        if col not in pred_X_full.columns:
            pred_X_full[col] = 0
    pred_X_full = pred_X_full[feature_names]
    pred_X_full.to_csv(config.SELECTED_FEATURES_PATH, index=False)
    predict.predict_model()

    print("Evaluating model...")
    model_path = config.MODEL_PATH
    test_data_path = config.SELECTED_FEATURES_PATH
    predictions_path = config.PREDICTIONS_PATH
    try:
        evaluate.load_model(model_path)
        test_data = evaluate.load_data(test_data_path)
        predictions = pd.read_csv(predictions_path)
        print(f"Loaded test data with {len(test_data)} rows for evaluation.")
        print(f"Loaded predictions with {len(predictions)} rows for evaluation.")
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
