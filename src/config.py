import os

# base directory for project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#data file paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'Employment Projections.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'cleaned_employment_projections.csv')

# model data paths
MODEL_DATA_DIR = os.path.join(BASE_DIR, 'model_data')
SELECTED_FEATURES_PATH = os.path.join(MODEL_DATA_DIR, 'selected_features.csv')
TARGET_PATH = os.path.join(MODEL_DATA_DIR, 'target.csv')
FEATURE_IMPORTANCES_PATH = os.path.join(MODEL_DATA_DIR, 'feature_importances.csv')
MODEL_PATH = os.path.join(MODEL_DATA_DIR, 'random_forest_model.joblib')
PREDICTIONS_PATH = os.path.join(MODEL_DATA_DIR, 'predictions.csv')
TRAINING_LOG_PATH = os.path.join(MODEL_DATA_DIR, 'logs', 'training_log.json')

# plot output directory
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# model parameters
MODEL_PARAMS = {
    'learning_rate': 0.01,
    'n_estimators': 100,
    'random_state': 42
}

# output directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# logging configuration
LOG_LEVEL = 'INFO'
# log file path
LOG_FILE_PATH = os.path.join(BASE_DIR, 'logs', 'app.log')

