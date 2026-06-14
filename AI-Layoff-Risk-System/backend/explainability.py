"""SHAP and LIME explainability utilities."""

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from config import CATBOOST_MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS


class ModelExplainer:
    """Handles model explainability using SHAP and LIME."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = FEATURE_COLUMNS
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Try to load model
        try:
            self.model = joblib.load(CATBOOST_MODEL_PATH)
            if hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1
            self.scaler = joblib.load(SCALER_PATH)
            self.shap_explainer = shap.TreeExplainer(self.model)
        except FileNotFoundError:
            print("Warning: Model files not found. Please train and save the model first.")
    
    def set_training_data(self, X_train):
        """Set training data for LIME explainer."""
        if X_train is not None:
            self.lime_explainer = LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                class_names=['Low', 'Medium', 'High'],
                mode='classification',
                discretize_continuous=True
            )
    
    def get_shap_values(self, features):
        """Get SHAP values for a single prediction."""
        if self.model is None:
            return []
        
        features_frame = pd.DataFrame([features], columns=self.feature_names)
        shap_values = self.shap_explainer.shap_values(features_frame)
        
        # For multiclass, take the predicted class
        if isinstance(shap_values, list):
            prediction = self.model.predict(features_frame)[0]
            shap_values = shap_values[prediction][0]
        else:
            shap_values = shap_values[0]
        
        return shap_values.tolist()
    
    def get_shap_summary(self, X_sample):
        """Generate SHAP summary for a sample of data."""
        if self.model is None or len(X_sample) == 0:
            return []
        
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            # For multiclass, aggregate importance
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        return importance_df.to_dict('records')
    
    def get_lime_explanation(self, features):
        """Get LIME explanation for a single prediction."""
        if self.lime_explainer is None or self.model is None:
            return None
        
        features_array = np.array(features).reshape(1, -1)
        
        explanation = self.lime_explainer.explain_instance(
            features_array[0],
            self.model.predict_proba,
            num_features=10,
            num_samples=5000
        )
        
        return [
            {'feature': feat, 'weight': weight}
            for feat, weight in explanation.as_list()
        ]


# Global instance
explainer = ModelExplainer()
