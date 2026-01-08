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


def main():
    print("Starting data preprocessing...")
    data_preprocessing.main()

    print("Selecting important features...")
    raw_data = pd.read_csv(config.PROCESSED_DATA_PATH)
    target = raw_data['Employment 2034']  # Adjust target column as needed
    X = raw_data.drop(['Employment 2034'], axis=1)
    selected_X = features.select_important_features(X, target, n_features=10)
    selected_X.to_csv(config.SELECTED_FEATURES_PATH, index=False)
    target.to_csv(config.TARGET_PATH, index=False)

    print("Training model...")
    train.main()

    print("Generating predictions...")
    predict.predict_model()

    print("Evaluating model...")
    model_path = config.MODEL_PATH
    test_data_path = config.SELECTED_FEATURES_PATH
    predictions_path = config.PREDICTIONS_PATH
    try:
        model = evaluate.load_model(model_path)
        test_data = evaluate.load_data(test_data_path)
        predictions = pd.read_csv(predictions_path)
        # Add your evaluation logic here
        # For example: print regression metrics
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
