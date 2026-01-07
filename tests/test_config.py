from src import config
import os
import pytest
def test_config_loading():
    cfg = config.load_config("tests/test_config.yaml")
    assert cfg["setting1"] == "value1"
    assert cfg["setting2"]["subsetting"] == 42
def test_config_paths():
    assert config.DATA_DIR.endswith('data')
    assert config.MODEL_PATH.endswith('random_forest_model.joblib')
def test_model_params():
    params = config.MODEL_PARAMS
    assert params['learning_rate'] == 0.01
    assert params['n_estimators'] == 100
    assert params['random_state'] == 42
def test_log_file_path():
    assert config.LOG_FILE_PATH.endswith('logs/app.log')
def test_output_directory():
    assert config.OUTPUT_DIR.endswith('output')

print("BASE_DIR:", config.BASE_DIR)
print("RAW_DATA_PATH exists:", os.path.exists(config.RAW_DATA_PATH))
print("PROCESSED_DATA_PATH exists:", os.path.exists(config.PROCESSED_DATA_PATH))
print("MODEL_PATH:", config.MODEL_PATH)
print("PLOTS_DIR:", config.PLOTS_DIR)
print("LOG_FILE_PATH:", config.LOG_FILE_PATH)
print("MODEL_PARAMS:", config.MODEL_PARAMS)

print("All config tests passed.")