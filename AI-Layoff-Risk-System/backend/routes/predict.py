"""Prediction API routes."""

import time
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
import joblib

from config import (
    CATBOOST_MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS, RISK_MAPPING,
    INDUSTRY_LIST, JOB_ROLE_LIST, COMPANY_SIZE_LIST, JOB_LEVEL_LIST, AI_ADOPTION_LIST
)
from database import save_prediction, log_query

predict_bp = Blueprint('predict', __name__)

# Load models
try:
    model = joblib.load(CATBOOST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    model = None
    scaler = None


def build_feature_vector(data):
    """Build complete feature vector from input data."""
    features = []
    
    # Numeric features
    numeric_features = [
        'Age', 'Years_of_Experience', 'Routine_Task_Percentage',
        'Creativity_Requirement', 'Human_Interaction_Level',
        'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
        'Tasks_Automated_Percentage', 'AI_Training_Hours'
    ]
    
    for col in numeric_features:
        features.append(float(data.get(col, 0)))
    
    # Education (default to Bachelor's)
    edu_level = data.get('Education_Level', "Bachelor's")
    features.append(1 if edu_level == 'High School' else 0)
    features.append(1 if edu_level == "Master's" else 0)
    features.append(1 if edu_level == 'PhD' else 0)
    
    # Industry
    industry = data.get('Industry', '')
    for ind in INDUSTRY_LIST:
        features.append(1 if industry == ind else 0)
    
    # Job Role
    job_role = data.get('Job_Role', '')
    for job in JOB_ROLE_LIST:
        features.append(1 if job_role == job else 0)
    
    # Company Size
    company_size = data.get('Company_Size', 'Medium')
    features.append(1 if company_size == 'Medium' else 0)
    features.append(1 if company_size == 'Small' else 0)
    
    # Job Level
    job_level = data.get('Job_Level', 'Entry')
    features.append(1 if job_level == 'Mid' else 0)
    features.append(1 if job_level == 'Senior' else 0)
    
    # AI Adoption Level
    ai_adoption = data.get('AI_Adoption', 'Low')
    features.append(1 if ai_adoption == 'Low' else 0)
    features.append(1 if ai_adoption == 'Medium' else 0)
    
    return features


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint."""
    start_time = time.time()
    
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503
    
    try:
        data = request.json
        
        # Build feature vector
        features = build_feature_vector(data)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))
        
        # Get readable risk level
        risk_level = RISK_MAPPING[prediction]
        
        # Save to database
        prediction_id = save_prediction(data, risk_level, confidence)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'risk_level': risk_level,
            'confidence': confidence,
            'probabilities': {
                'Low': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'High': float(probabilities[2])
            },
            'prediction_id': prediction_id
        }
        
        # Log query
        elapsed_ms = (time.time() - start_time) * 1000
        log_query('/predict', elapsed_ms)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@predict_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded
    })