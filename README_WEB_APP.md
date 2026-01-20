# Employment Projection Web Application

A Flask-based web application that provides an interactive interface for your machine learning employment projection model. This application allows users to make predictions, upload batch data, and visualize results through a user-friendly web interface.

## 🌟 Features

### 🏠 Home Dashboard
- Overview of application capabilities
- Model information and statistics
- Navigation to all features

### 📊 Individual Predictions
- Interactive form to input occupation details
- Real-time predictions for employment in 2034
- Input validation and error handling

### 📤 Batch Processing
- CSV file upload for multiple predictions
- Sample CSV download
- Progress tracking and results summary

### 📈 Results & Visualization
- Interactive charts using Plotly
- Statistical summaries
- Data tables with detailed results
- Export capabilities

### 🎯 Model Information
- Feature importance display
- Model metrics and performance
- Technical specifications

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- All packages from the original employment projection project
- Web browser (Chrome, Firefox, Safari, etc.)

### Installation

1. **Install web application dependencies:**
   ```bash
   pip install -r requirements_web.txt
   ```

2. **Ensure your model is trained:**
   ```bash
   python main.py  # If not already done
   ```

### Running the Application

#### Option 1: Using the startup script (Windows)
```bash
./start_web_app.bat
```

#### Option 2: Direct Python execution
```bash
python app.py
```

#### Option 3: With custom settings
```bash
# For development with debug mode
export FLASK_ENV=development
python app.py

# For production
export FLASK_ENV=production
python app.py
```

### Accessing the Application

Once started, open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## 📱 Application Structure

```
Employment Projection Web App
├── 🏠 Home (/)
│   ├── Feature overview
│   └── Model information modal
├── 🔮 Predict (/predict)
│   ├── Interactive prediction form
│   └── Real-time results display
├── 📤 Upload (/upload)
│   ├── CSV batch processing
│   └── Sample file download
└── 📊 Results (/results)
    ├── Interactive visualizations
    ├── Summary statistics
    └── Data export options
```

## 🔧 API Endpoints

The application also provides REST API endpoints:

### Prediction Endpoints
- `POST /api/predict` - Make single predictions
- `POST /api/upload` - Batch predictions via CSV upload
- `GET /api/model-info` - Get model information and feature details

### Example API Usage

**Single Prediction:**
```python
import requests

data = {
    "Employment 2024": 1579.8,
    "Occupational Openings, 2024-2034 Annual Average": 124.2,
    "Employment Percent Change, 2024-2034": 4.6,
    "Employment Change, 2024-2034": 72.8,
    "Median Annual Wage 2024": 81680,
    "trCode": 6,
    "Typical Entry-Level Education_Bachelor's degree": 1,
    "Typical Entry-Level Education_High school diploma or equivalent": 0,
    "Typical on-the-job Training_Short-term on-the-job training": 0,
    "Typical on-the-job Training_nan": 1
}

response = requests.post('http://127.0.0.1:5000/api/predict', json=data)
prediction = response.json()
print(f"Predicted Employment 2034: {prediction['prediction']}")
```

## 📊 Input Features

The model accepts the following input features:

### Numeric Features
- **Employment 2024** - Current employment in thousands
- **Occupational Openings, 2024-2034 Annual Average** - Average annual job openings
- **Employment Percent Change, 2024-2034** - Projected percentage change
- **Employment Change, 2024-2034** - Projected employment change in thousands
- **Median Annual Wage 2024** - Median annual wage in dollars
- **trCode** - Training code (3-6)

### Categorical Features (Binary: 0 or 1)
- **Typical Entry-Level Education_Bachelor's degree**
- **Typical Entry-Level Education_High school diploma or equivalent**
- **Typical on-the-job Training_Short-term on-the-job training**
- **Typical on-the-job Training_nan**

## 📁 File Structure

```
employmentprojection/
├── app.py                 # Main Flask application
├── start_web_app.bat     # Windows startup script
├── requirements_web.txt  # Web app dependencies
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── index.html       # Home page
│   ├── predict.html     # Prediction form
│   ├── upload.html      # Upload page
│   └── results.html     # Results page
├── static/              # Static assets
│   ├── css/
│   │   └── style.css    # Custom styles
│   └── js/
│       └── app.js       # JavaScript functionality
└── ... (existing project files)
```

## 🎨 Technologies Used

### Backend
- **Flask 3.0** - Web framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Joblib** - Model serialization
- **Plotly** - Interactive visualizations

### Frontend
- **Bootstrap 5** - UI framework
- **Font Awesome 6** - Icons
- **Plotly.js** - Interactive charts
- **Vanilla JavaScript** - Client-side functionality

## 🛠️ Customization

### Styling
Modify [`static/css/style.css`](static/css/style.css) to customize the appearance:
- Colors and themes
- Layout and spacing
- Component styling

### Functionality
Extend [`static/js/app.js`](static/js/app.js) to add:
- New interactive features
- Enhanced validation
- Additional visualizations

### Templates
Update HTML templates in the `templates/` directory:
- Add new pages
- Modify existing layouts
- Customize content and structure

## 🚨 Troubleshooting

### Common Issues

**1. Model Not Found Error**
```
Error: Model not loaded
```
**Solution:** Ensure you have run `python main.py` to train the model first.

**2. Feature Mismatch Error**
```
Error: Feature names don't match
```
**Solution:** Retrain the model or check feature consistency.

**3. Port Already in Use**
```
Error: Address already in use
```
**Solution:** Change the port in `app.py` or kill existing processes.

### Debug Mode
Enable debug mode for detailed error messages:
```python
app.run(debug=True)
```

## 📈 Performance Considerations

### For Large Datasets
- Implement pagination for results tables
- Add progress bars for long-running uploads
- Consider using background tasks for heavy processing

### For Production Deployment
- Use a production WSGI server (e.g., Gunicorn)
- Implement proper error handling and logging
- Add authentication and security measures
- Configure SSL/HTTPS

## 🔒 Security Notes

For production deployment, consider:
- Changing the Flask secret key
- Implementing user authentication
- Adding rate limiting
- Validating file uploads more strictly
- Using HTTPS

## 🤝 Contributing

To extend this web application:

1. Fork the project
2. Create a feature branch
3. Add your enhancements
4. Test thoroughly
5. Submit a pull request

## 📜 License

This project is part of the Employment Projection analysis and follows the same licensing terms.

---

**Enjoy exploring employment predictions through your new web interface!** 🎉