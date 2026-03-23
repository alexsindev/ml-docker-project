"""
Iris Species Classification Flask API
Student ID: st126112
Serving Random Forest Classifier predictions
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model once at startup
try:
    model = joblib.load('model.pkl')
    logger.info(f"Model loaded successfully: {type(model).__name__}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Iris species names
SPECIES_NAMES = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    """Render the main prediction interface"""
    logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': type(model).__name__,
        'timestamp': datetime.now().isoformat(),
        'student_id': 'st126112'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict iris species based on flower measurements
    Expects JSON payload with sepal and petal measurements
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract and validate features
        features = np.array([[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]])

        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        species = SPECIES_NAMES[prediction]
        confidence = float(probabilities[prediction])

        logger.info(f"Prediction made: {species}, Confidence: {confidence:.4f}")

        return jsonify({
            'species': species,
            'species_index': int(prediction),
            'confidence': round(confidence, 4),
            'probabilities': {
                'Setosa': round(float(probabilities[0]), 4),
                'Versicolor': round(float(probabilities[1]), 4),
                'Virginica': round(float(probabilities[2]), 4)
            }
        })

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    logger.info("Starting Iris Species Classifier API - Student ID: st126112")
    logger.info(f"Model type: {type(model).__name__}")
    app.run(host='0.0.0.0', port=5001, debug=False)
