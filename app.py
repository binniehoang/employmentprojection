"""
Flask web application for Employment Projection ML model.
Provides a user-friendly interface to interact with the trained model.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import joblib
import os
import numpy as np
from src import config
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json

app = Flask(__name__)
app.secret_key = 'employment_projection_secret_key'  # Change this in production

class EmploymentPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.encoder = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and necessary artifacts"""
        try:
            # Load model
            self.model = joblib.load(config.MODEL_PATH)
            
            # Load feature names
            feature_names_path = os.path.join(config.MODEL_DATA_DIR, 'selected_feature_names.txt')
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            # Load encoder if exists
            encoder_path = os.path.join(config.MODEL_DATA_DIR, 'onehot_encoder.joblib')
            if os.path.exists(encoder_path):
                self.encoder = joblib.load(encoder_path)
                
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict_single(self, input_data):
        """Make a single prediction"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Convert input to DataFrame with proper feature names
            input_df = pd.DataFrame([input_data])
            
            # Ensure all expected features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Default value for missing features
            
            # Select only the features used in training
            input_df = input_df[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            
            return prediction, None
            
        except Exception as e:
            return None, str(e)
    
    def predict_batch(self, df):
        """Make batch predictions on a DataFrame"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Ensure all expected features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Select only the features used in training
            X = df[self.feature_names]
            
            # Make predictions
            predictions = self.model.predict(X)
            
            return predictions, None
            
        except Exception as e:
            return None, str(e)

# Initialize predictor
predictor = EmploymentPredictor()

@app.route('/')
def home():
    """Home page with overview and navigation"""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """Individual prediction form page"""
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prediction, error = predictor.predict_single(data)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'prediction': float(prediction),
            'input_data': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results_page():
    """View existing predictions and results"""
    try:
        # Load existing predictions
        predictions_df = pd.read_csv(config.PREDICTIONS_PATH)
        
        # Create visualizations
        fig1 = px.histogram(
            predictions_df, 
            x='Predicted Employment 2034',
            title='Distribution of Predicted Employment 2034',
            nbins=30
        )
        
        fig2 = px.scatter(
            predictions_df,
            x='Employment 2024',
            y='Predicted Employment 2034',
            title='Employment 2024 vs Predicted Employment 2034',
            trendline='ols'
        )
        
        # Convert plots to JSON
        graph1_json = json.dumps(fig1, cls=PlotlyJSONEncoder)
        graph2_json = json.dumps(fig2, cls=PlotlyJSONEncoder)
        
        # Get summary statistics
        stats = {
            'total_occupations': len(predictions_df),
            'avg_predicted_employment': predictions_df['Predicted Employment 2034'].mean(),
            'max_predicted_employment': predictions_df['Predicted Employment 2034'].max(),
            'min_predicted_employment': predictions_df['Predicted Employment 2034'].min()
        }
        
        return render_template('results.html',
                             graph1_json=graph1_json,
                             graph2_json=graph2_json,
                             stats=stats,
                             predictions=predictions_df.head(20).to_dict('records'))
        
    except Exception as e:
        flash(f'Error loading results: {str(e)}', 'error')
        return render_template('results.html', error=str(e))

@app.route('/upload')
def upload_page():
    """Upload page for batch predictions"""
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for batch predictions via file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Read uploaded CSV
            df = pd.read_csv(file)
            
            # Make predictions
            predictions, error = predictor.predict_batch(df)
            
            if error:
                return jsonify({'error': error}), 500
            
            # Add predictions to dataframe
            df['Predicted Employment 2034'] = predictions
            
            # Save results
            output_path = os.path.join(config.MODEL_DATA_DIR, 'uploaded_predictions.csv')
            df.to_csv(output_path, index=False)
            
            return jsonify({
                'message': f'Successfully processed {len(df)} records',
                'predictions_count': len(predictions),
                'output_file': output_path,
                'sample_predictions': predictions[:5].tolist()
            })
        
        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
def model_info():
    """Get information about the loaded model"""
    if predictor.model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Load feature importance if available
        feature_importance = None
        if os.path.exists(config.FEATURE_IMPORTANCES_PATH):
            fi_df = pd.read_csv(config.FEATURE_IMPORTANCES_PATH)
            feature_importance = fi_df.to_dict('records')
        
        return jsonify({
            'model_type': type(predictor.model).__name__,
            'feature_count': len(predictor.feature_names),
            'feature_names': predictor.feature_names,
            'feature_importance': feature_importance,
            'model_loaded': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("Starting Employment Projection Web App...")
    print("Navigate to http://127.0.0.1:5000 to access the application")
    app.run(debug=True, host='127.0.0.1', port=5000)