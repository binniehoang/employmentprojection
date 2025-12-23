import pandas as pd
import joblib
import os

def main():
	# Paths
	model_path = os.path.join('model_data', 'random_forest_model.joblib')
	features_path = os.path.join('model_data', 'selected_features.csv')
	output_path = os.path.join('model_data', 'predictions.csv')

	# Load model
	model = joblib.load(model_path)

	# Load features
	X = pd.read_csv(features_path)

	# Predict
	predictions = model.predict(X)

	# Print predictions
	print('Predictions:')
	print(predictions)

	# Save predictions to CSV
	result_df = X.copy()
	result_df['Predicted Employment 2034'] = predictions
	result_df.to_csv(output_path, index=False)
	try:
		result_df.to_csv(output_path, index=False)
		print(f'Predictions saved to {output_path}')
	except OSError as e:
		print(f'Error saving predictions to {output_path}: {e}')
		raise
if __name__ == '__main__':
	main()
