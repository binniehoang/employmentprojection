# model training script
"""
Model training script for employment projections.

Trains a RandomForestRegressor on selected features and target, evaluates performance, and saves feature importances.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
from sklearn.model_selection import cross_val_score

def main():
	"""
	Main function to train a RandomForestRegressor on employment projections data.
	Loads features and target, splits data, trains model, evaluates, and saves results.
	"""
	import os
	# Accept base_dir as a parameter for testability
	def _main(base_dir="."):
		"""
		Internal function to allow specifying base directory for testability.
		"""
		# load processed features and targets
		X = pd.read_csv(os.path.join(base_dir, 'model_data', 'selected_features.csv'))
		y = pd.read_csv(os.path.join(base_dir, 'model_data', 'target.csv')).squeeze()

		# split data and train
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# train randomforestregressor
		model_params = {
			'n_estimators': 100,
			'random_state': 42
		}

		model = RandomForestRegressor(**model_params)

		# cross validation
		cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
		print(f"5-fold CV R^2 scores: {cv_scores}")
		print(f"Average CV R^2 score: {cv_scores.mean():.2f}")

		model.fit(X_train, y_train)

		# feature importance
		feature_importances = model.feature_importances_
		importance_df = pd.DataFrame({
			'feature': X.columns,
			'importance': feature_importances
		}).sort_values(by='importance', ascending=False)
		# added error handling for file writing
		try:
			importance_path = os.path.join(base_dir, 'model_data', 'feature_importances.csv')
			importance_df.to_csv(importance_path, index=False)
		except OSError as e:
			print(f"Failed to write feature importances to '{importance_path}': {e}")
			raise

		print("\nTop 10 Feature Importances:")
		print(importance_df.head(10))

		# evaluate model
		y_pred = model.predict(X_test)
		mse = mean_squared_error(y_test, y_pred)
		r2 = r2_score(y_test, y_pred)
		print(f"Test MSE: {mse:.2f}")
		print(f"Test R^2: {r2:.2f}")

		# log parameters and metrics
		log = {
			'model_params': model_params,
			'test_mse': mse,
			'test_r2': r2,
			'cv_r2_scores': cv_scores.tolist(),
			'cv_r2_mean': float(cv_scores.mean())
		}
		# added error handling for file writing
		try:
			logs_dir = os.path.join(base_dir, 'model_data', 'logs')
			os.makedirs(logs_dir, exist_ok=True)
			log_path = os.path.join(logs_dir, 'training_log.json')
			with open(log_path, 'w') as f:
				json.dump(log, f, indent=2)
				print(f"Training log saved to {log_path}")
		except OSError as e:
			print(f"Failed to write training log to {log_path}: {e}")
			raise

		# save trained model
		try:
			model_path = os.path.join(base_dir, 'model_data', 'random_forest_model.joblib')
			joblib.dump(model, model_path)
			print(f"Model saved to {model_path}")
		except OSError as e:
			print(f"Failed to save model to '{model_path}': {e}")
			raise

		# predict on some example data
		example_data = X_test.iloc[:5]
		example_predictions = model.predict(example_data)
		print("Example predictions on test data:")
		print(example_predictions)

	return _main


if __name__ == '__main__':
	main()()

