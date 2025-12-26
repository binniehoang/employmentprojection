# utils.py - Starter template for common utility functions
import pandas as pd
import json
import logging
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

def load_csv(path):
	"""Load a CSV file into a DataFrame.
	Note:
		This function may raise exceptions from :func:`pandas.read_csv`
		(such as ``FileNotFoundError``, ``pandas.errors.EmptyDataError``,
		or ``pandas.errors.ParserError``). Callers should handle these
		exceptions as appropriate.
	"""
	try:
		return pd.read_csv(path)
	except FileNotFoundError as e:
		logger.error(f'File not found: {path}')
		raise
	except pd.errors.EmptyDataError as e:
		logger.error(f'No data: {path} is empty')
		raise
	except pd.errors.ParserError as e:
		logger.error(f'Parsing error: {path} could not be parsed')
		raise
	except Exception as exc:
		logger.error(f'Unexpected error loading {path}: {exc}')
		raise


def save_csv(df, path, index=False):
	"""Save a DataFrame to a CSV file."""
	try:
		df.to_csv(path, index=index)
	except (OSError, IOError) as e:
		logging.error("Failed to save CSV to %s: %s", path, e)
		raise

def load_json(path):
	try:
		with open(path, 'r') as f:
			return json.load(f)
	except FileNotFoundError as e:
		logging.getLogger(__name__).error("JSON file not found: %s", path)
		raise
	except json.JSONDecodeError as e:
		logging.getLogger(__name__).error("Failed to decode JSON from file: %s", path)
		raise
	except OSError as e:
		logging.getLogger(__name__).error("OS error while reading JSON file %s: %s", path, e)
		raise

def save_json(obj, path):
	"""Save a dict as a JSON file."""
	try:
		with open(path, 'w') as f:
			json.dump(obj, f, indent=2)
	except OSError as e:
		logging.getLogger(__name__).error("OS error while writing JSON file %s: %s", path, e)
		raise

def setup_logging(log_path='logs/app.log'):
	"""Set up logging to a file and console."""
	os.makedirs(os.path.dirname(log_path), exist_ok=True)
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s %(levelname)s %(message)s',
		handlers=[
			logging.FileHandler(log_path),
			logging.StreamHandler()
		]
	)

def rmse(y_true, y_pred):
	"""Calculate Root Mean Squared Error."""
	return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
	"""Calculate Mean Absolute Error."""
	return mean_absolute_error(y_true, y_pred)
