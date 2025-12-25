# utils.py - Starter template for common utility functions
import pandas as pd
import json
import logging
import os

def load_csv(path):
	"""Load a CSV file into a DataFrame."""
	return pd.read_csv(path)

def save_csv(df, path, index=False):
	"""Save a DataFrame to a CSV file."""
	df.to_csv(path, index=index)

def load_json(path):
	"""Load a JSON file as a dict."""
	with open(path, 'r') as f:
		return json.load(f)

def save_json(obj, path):
	"""Save a dict as a JSON file."""
	with open(path, 'w') as f:
		json.dump(obj, f, indent=2)

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
	from sklearn.metrics import mean_squared_error
	import numpy as np
	return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
	"""Calculate Mean Absolute Error."""
	from sklearn.metrics import mean_absolute_error
	return mean_absolute_error(y_true, y_pred)
