# import required modules
import os
import json
import tempfile
import pandas as pd
import numpy as np
import logging
import pytest
from src import utils

def test_load_and_save_csv():
	df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
	with tempfile.TemporaryDirectory() as tmpdir:
		path = os.path.join(tmpdir, 'test.csv')
		utils.save_csv(df, path)
		loaded = utils.load_csv(path)
		pd.testing.assert_frame_equal(df, loaded)

def test_load_csv_file_not_found():
	with pytest.raises(FileNotFoundError):
		utils.load_csv('nonexistent_file.csv')

def test_save_csv_oserror(monkeypatch):
	df = pd.DataFrame({'a': [1]})
	def raise_oserror(*args, **kwargs):
		raise OSError('Mocked OSError')
	monkeypatch.setattr(pd.DataFrame, 'to_csv', raise_oserror)
	with pytest.raises(OSError):
		utils.save_csv(df, 'dummy.csv')

def test_load_and_save_json():
	obj = {'x': 1, 'y': [1, 2, 3]}
	with tempfile.TemporaryDirectory() as tmpdir:
		path = os.path.join(tmpdir, 'test.json')
		utils.save_json(obj, path)
		loaded = utils.load_json(path)
		assert obj == loaded

def test_load_json_file_not_found():
	with pytest.raises(FileNotFoundError):
		utils.load_json('nonexistent.json')

def test_load_json_decode_error():
	with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
		tmp.write('{bad json}')
		tmp_path = tmp.name
	try:
		with pytest.raises(json.JSONDecodeError):
			utils.load_json(tmp_path)
	finally:
		os.remove(tmp_path)

def test_save_json_oserror(monkeypatch):
	def raise_oserror(*args, **kwargs):
		raise OSError('Mocked OSError')
	monkeypatch.setattr('builtins.open', lambda *a, **k: (_ for _ in ()).throw(OSError('Mocked OSError')))
	with pytest.raises(OSError):
		utils.save_json({'a': 1}, 'dummy.json')

def test_setup_logging_creates_log_file():
	with tempfile.TemporaryDirectory() as tmpdir:
		log_path = os.path.join(tmpdir, 'logs', 'app.log')
		utils.setup_logging(log_path)
		logger = logging.getLogger()
		logger.info('Test log message')
		assert os.path.exists(log_path)

def test_rmse_and_mae():
	y_true = np.array([1, 2, 3])
	y_pred = np.array([1, 2, 4])
	assert np.isclose(utils.rmse(y_true, y_pred), 0.577350269)
	assert np.isclose(utils.mae(y_true, y_pred), 0.333333333)
