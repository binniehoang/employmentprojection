from src import config
import os


def test_config_loading(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("setting1: value1\n"
                           "setting2:\n" \
                           "subsetting: 42\n")
    cfg = config.load_config(str(config_file))
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

