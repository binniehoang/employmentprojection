# Employment Projection

This project uses machine learning algorithms to analyze and visualize employment trends based on U.S. employment projections data.

## Project Structure

- **src/**: Source code for data processing, feature engineering, modeling, and utilities
- **model_data/**: Model artifacts, selected features, predictions, and logs
- **data/**: Raw and cleaned datasets
- **plots/**: Generated plots and visualizations
- **tests/**: Unit and integration tests

## Setup

1. Clone the repository:
	```sh
	git clone <repo-url>
	cd employmentprojection
	```
2. (Optional) Create and activate a virtual environment:
	```sh
	python -m venv venv
	source venv/bin/activate  # On Windows: venv\Scripts\activate
	```
3. Install dependencies:
	```sh
	pip install -r requirements.txt
	```

## Usage

1. **Data Preprocessing:**
	```sh
	python src/data_preprocessing.py
	```
2. **Exploratory Data Analysis:**
	```sh
	python src/eda.py
	```
3. **Model Training:**
	```sh
	python src/model/train.py
	```
4. **Model Prediction:**
	```sh
	python src/model/predict.py
	```
5. **Model Evaluation:**
	```sh
	python src/model/evaluate.py
	```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, discuss them first by opening an issue.

## License

This project is licensed under the MIT License.