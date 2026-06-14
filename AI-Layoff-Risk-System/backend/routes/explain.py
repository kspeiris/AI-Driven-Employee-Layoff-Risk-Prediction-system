"""Explainability API routes."""

import time
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
import joblib

from config import FEATURE_COLUMNS, SCALER_PATH
from explainability import explainer
from database import log_query

explain_bp = Blueprint('explain', __name__)

# Load scaler
scaler = joblib.load(SCALER_PATH)


@explain_bp.route('/explain/shap', methods=['POST'])
def get_shap_explanation():
    """Get SHAP explanation for a prediction."""
    start_time = time.time()
    
    try:
        data = request.json
        features = [data.get(col, 0) for col in FEATURE_COLUMNS]
        features_scaled = scaler.transform([features])
        
        shap_values = explainer.get_shap_values(features_scaled)
        
        # Get top contributing features
        shap_df = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'shap_value': shap_values
        })
        
        positive = shap_df[shap_df['shap_value'] > 0].sort_values('shap_value', ascending=False).head(10)
        negative = shap_df[shap_df['shap_value'] < 0].sort_values('shap_value', ascending=True).head(5)
        
        elapsed_ms = (time.time() - start_time) * 1000
        log_query('/explain/shap', elapsed_ms)
        
        return jsonify({
            'success': True,
            'positive_contributors': positive.to_dict('records'),
            'negative_contributors': negative.to_dict('records'),
            'all_features': shap_df.to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@explain_bp.route('/explain/lime', methods=['POST'])
def get_lime_explanation():
    """Get LIME explanation for a prediction."""
    start_time = time.time()
    
    try:
        data = request.json
        features = [data.get(col, 0) for col in FEATURE_COLUMNS]
        
        lime_explanation = explainer.get_lime_explanation(features)
        
        elapsed_ms = (time.time() - start_time) * 1000
        log_query('/explain/lime', elapsed_ms)
        
        return jsonify({
            'success': True,
            'explanation': lime_explanation
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@explain_bp.route('/explain/shap/summary', methods=['POST'])
def get_shap_summary():
    """Get SHAP summary for sample data."""
    start_time = time.time()
    
    try:
        data = request.json
        sample_data = data.get('sample_data', [])
        
        if not sample_data:
            return jsonify({'success': False, 'error': 'No sample data provided'}), 400
        
        sample_array = np.array(sample_data)
        importance = explainer.get_shap_summary(sample_array)
        
        elapsed_ms = (time.time() - start_time) * 1000
        log_query('/explain/shap/summary', elapsed_ms)
        
        return jsonify({
            'success': True,
            'feature_importance': importance
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500