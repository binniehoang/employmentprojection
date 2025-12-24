import pandas as pd
import joblib
import os

def predictmodel():
	# Paths
	model_path = os.path.join('model_data', 'random_forest_model.joblib')
	features_path = os.path.join('model_data', 'selected_features.csv')
	output_path = os.path.join('model_data', 'predictions.csv')

	# Load model
	try:
		model = joblib.load(model_path)
	except (FileNotFoundError, OSError, IOError) as e:
		print(f'Error loading model from {model_path}: {e}')
		raise
	except Exception as e:
		print(f'Unexpected error loading model from {model_path}: {e}')
		raise
	# Load features
	try:
		X = pd.read_csv(features_path)
	except (FileNotFoundError, OSError, IOError) as e:
		print(f'Error loading features from {features_path}: {e}')
		raise
	except pd.errors.ParserError as e:
		print(f'Error parsing features CSV at {features_path}: {e}')
		raise
	except Exception as e:
		print(f'Unexpected error loading features from {features_path}: {e}')
		raise

	# Predict
	predictions = model.predict(X)

	# Print predictions
	print('Predictions:')
	print(predictions)

	# Save predictions to CSV
	result_df = X.copy()
	result_df['Predicted Employment 2034'] = predictions
	try:
		result_df.to_csv(output_path, index=False)
		print(f'Predictions saved to {output_path}')
	except OSError as e:
		print(f'Error saving predictions to {output_path}: {e}')
		raise

predictmodel()
