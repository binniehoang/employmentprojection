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

    # Check if model and features already exist
    model_exists = os.path.exists(config.MODEL_PATH)
    features_exist = os.path.exists(config.SELECTED_FEATURES_PATH)
    encoder_path = os.path.join(os.path.dirname(config.SELECTED_FEATURES_PATH), 'onehot_encoder.joblib')
    encoder_exists = os.path.exists(encoder_path)
    
    if model_exists and features_exist and encoder_exists:
        print("Model and features already exist. Using existing preprocessing pipeline for consistency.")
        # Load the existing encoder and feature names
        encoder = joblib.load(encoder_path)
        
        # Load feature names that were used during training
        feature_names_path = os.path.join(os.path.dirname(config.SELECTED_FEATURES_PATH), 'selected_feature_names.txt')
        with open(feature_names_path, 'r') as f:
            expected_features = [line.strip() for line in f.readlines()]
        
        print(f"Expected features for prediction: {expected_features}")
        print("Skipping feature re-processing to maintain consistency.")
        
        # Jump directly to prediction
        print("Generating predictions...")
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
            
            # Add comprehensive visualization
            print("Creating comprehensive model visualizations...")
            from src import visualization
            visualization.create_model_summary_report()
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
        return
    
    print("Training new model with consistent preprocessing pipeline...")
    print("Loading cleaned data and encoding features with consistent columns...")
    raw_data = pd.read_csv(config.PROCESSED_DATA_PATH)
    target = raw_data['Employment 2034'].replace({',': ''}, regex=True).astype(float) # Ensure numeric target values 
    non_numeric_cols = ['Occupation Title', 'Occupation Code']
    drop_cols = [col for col in non_numeric_cols if col in raw_data.columns]
    categorical_cols = [
        'Typical Entry-Level Education',
        'Work Experience in a Related Occupation',
        'Typical on-the-job Training'
    ]
    # Only encode categorical columns that exist
    cols_to_encode = [col for col in categorical_cols if col in raw_data.columns]
    cols_to_drop = [col for col in drop_cols + ['Employment 2034'] if col in raw_data.columns]
    X_categorical = raw_data[cols_to_encode] if cols_to_encode else pd.DataFrame()
    X_numeric = raw_data.drop(cols_to_encode, axis=1) if cols_to_encode else raw_data

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if not X_categorical.empty:
        encoder.fit(X_categorical)
        X_encoded = pd.DataFrame(encoder.transform(X_categorical), columns=encoder.get_feature_names_out(cols_to_encode))
        X_full = pd.concat([X_numeric.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    else:
        X_full = X_numeric
    # Save encoder for future use
    encoder_path = os.path.join(os.path.dirname(config.SELECTED_FEATURES_PATH), 'onehot_encoder.joblib')
    joblib.dump(encoder, encoder_path)
    if not X_categorical.empty:
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
        
        # Add comprehensive visualization
        print("Creating comprehensive model visualizations...")
        from src import visualization
        visualization.create_model_summary_report()
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
