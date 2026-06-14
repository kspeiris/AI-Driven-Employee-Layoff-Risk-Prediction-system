"""Prediction API routes."""

import time
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
import joblib

from config import (
    ML_MODEL_PATH, LEGACY_ML_MODEL_PATH, ANN_MODEL_PATH, DNN_MODEL_PATH,
    SCALER_PATH, FEATURE_COLUMNS, RISK_MAPPING,
    INDUSTRY_LIST, JOB_ROLE_LIST, COMPANY_SIZE_LIST, JOB_LEVEL_LIST, AI_ADOPTION_LIST
)
from database import save_prediction, log_query

predict_bp = Blueprint('predict', __name__)

# Load saved model artifacts. There is no rule-based fallback: if an artifact is
# unavailable, that model option is reported as unavailable.
try:
    model_path = ML_MODEL_PATH if ML_MODEL_PATH.exists() else LEGACY_ML_MODEL_PATH
    ml_model = joblib.load(model_path)
    if hasattr(ml_model, "n_jobs"):
        ml_model.n_jobs = 1
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    ml_model = None
    scaler = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

ann_model = load_model(ANN_MODEL_PATH) if load_model and ANN_MODEL_PATH.exists() else None
dnn_model = load_model(DNN_MODEL_PATH) if load_model and DNN_MODEL_PATH.exists() else None
model_loaded = ml_model is not None and scaler is not None


def _selected_model(model_type):
    """Return the requested saved model artifact."""
    normalized = (model_type or "ml").lower()
    if normalized == "ann":
        return ann_model, "ANN"
    if normalized == "dnn":
        return dnn_model, "DNN"
    return ml_model, "Random Forest"


def _encoded_value(data, column, raw_value=None, prefix=None):
    """Read one-hot values from direct columns or raw categorical fields."""
    if column in data:
        return int(bool(data.get(column)))
    if raw_value is None or prefix is None:
        return 0
    return int(column == f"{prefix}_{raw_value}")


def build_feature_vector(data):
    """Build complete feature vector from input data."""
    feature_values = {}
    
    # Numeric features
    numeric_features = [
        'Age', 'Years_of_Experience', 'Routine_Task_Percentage',
        'Creativity_Requirement', 'Human_Interaction_Level',
        'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
        'Tasks_Automated_Percentage', 'AI_Training_Hours'
    ]
    
    for col in numeric_features:
        feature_values[col] = float(data.get(col, 0))
    
    # Education (Bachelor's is the dropped baseline from training)
    edu_level = data.get('Education_Level', "Bachelor's")
    feature_values['Education_Level_High School'] = _encoded_value(
        data, 'Education_Level_High School', edu_level, 'Education_Level'
    )
    feature_values["Education_Level_Master's"] = _encoded_value(
        data, "Education_Level_Master's", edu_level, 'Education_Level'
    )
    feature_values['Education_Level_PhD'] = _encoded_value(
        data, 'Education_Level_PhD', edu_level, 'Education_Level'
    )
    
    # Industry
    industry = data.get('Industry', '')
    for ind in INDUSTRY_LIST:
        col = f'Industry_{ind}'
        feature_values[col] = _encoded_value(data, col, industry, 'Industry')
    
    # Job Role
    job_role = data.get('Job_Role', '')
    for job in JOB_ROLE_LIST:
        col = f'Job_Role_{job}'
        feature_values[col] = _encoded_value(data, col, job_role, 'Job_Role')
    
    # Company Size
    company_size = data.get('Company_Size', 'Medium')
    feature_values['Company_Size_Medium'] = _encoded_value(
        data, 'Company_Size_Medium', company_size, 'Company_Size'
    )
    feature_values['Company_Size_Small'] = _encoded_value(
        data, 'Company_Size_Small', company_size, 'Company_Size'
    )
    
    # Job Level
    job_level = data.get('Job_Level', 'Entry')
    feature_values['Job_Level_Mid'] = _encoded_value(data, 'Job_Level_Mid', job_level, 'Job_Level')
    feature_values['Job_Level_Senior'] = _encoded_value(data, 'Job_Level_Senior', job_level, 'Job_Level')
    
    # AI Adoption Level
    ai_adoption = data.get('AI_Adoption', 'Low')
    feature_values['AI_Adoption_Level_Low'] = _encoded_value(
        data, 'AI_Adoption_Level_Low', ai_adoption, 'AI_Adoption_Level'
    )
    feature_values['AI_Adoption_Level_Medium'] = _encoded_value(
        data, 'AI_Adoption_Level_Medium', ai_adoption, 'AI_Adoption_Level'
    )
    
    return [feature_values.get(col, 0) for col in FEATURE_COLUMNS]


def run_model_prediction(data):
    """Run the selected saved ML/DL model and return prediction details."""
    if scaler is None:
        raise RuntimeError('Scaler not loaded')

    model_type = data.get('model_type', 'ml')
    selected_model, model_name = _selected_model(model_type)
    if selected_model is None:
        raise RuntimeError(f'{model_name} model is not loaded')

    features = build_feature_vector(data)
    feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    if model_name in {'ANN', 'DNN'}:
        features_scaled = scaler.transform(feature_frame)
        probabilities = selected_model.predict(features_scaled, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
    else:
        prediction = int(selected_model.predict(feature_frame)[0])
        probabilities = selected_model.predict_proba(feature_frame)[0]

    confidence = float(np.max(probabilities))
    risk_level = RISK_MAPPING[prediction]

    return {
        'prediction': prediction,
        'risk_level': risk_level,
        'confidence': confidence,
        'probabilities': {
            'Low': float(probabilities[0]),
            'Medium': float(probabilities[1]),
            'High': float(probabilities[2])
        },
        'model_type': model_type,
        'model_name': model_name
    }


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint."""
    start_time = time.time()
    
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503
    
    try:
        data = request.json
        result = run_model_prediction(data)
        
        # Save to database
        prediction_id = save_prediction(data, result['risk_level'], result['confidence'])
        
        # Prepare response
        response = {
            'success': True,
            **result,
            'prediction_id': prediction_id
        }
        
        # Log query
        elapsed_ms = (time.time() - start_time) * 1000
        log_query('/predict', elapsed_ms)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@predict_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint using the selected saved ML/DL model."""
    start_time = time.time()

    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503

    try:
        records = request.json.get('records', [])
        predictions = []

        for record in records:
            result = run_model_prediction(record)
            prediction_id = save_prediction(record, result['risk_level'], result['confidence'])
            predictions.append({
                **result,
                'prediction_id': prediction_id
            })

        elapsed_ms = (time.time() - start_time) * 1000
        log_query('/predict/batch', elapsed_ms)

        return jsonify({
            'success': True,
            'count': len(predictions),
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@predict_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'models': {
            'ml': ml_model is not None,
            'ann': ann_model is not None,
            'dnn': dnn_model is not None
        }
    })
