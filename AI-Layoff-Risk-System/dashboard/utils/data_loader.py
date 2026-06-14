"""Data loading utilities for the dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Path to data
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed_layoff_dataset.csv"


@st.cache_data
def load_data():
    """Load and prepare the dataset with decoded categorical columns."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        # Return sample data if file not found
        st.error(f"Data file not found at {DATA_PATH}. Please ensure the processed dataset exists.")
        # Create minimal dataframe for demo
        df = pd.DataFrame({
            'Age': [30, 40, 50],
            'Years_of_Experience': [5, 15, 25],
            'Layoff_Risk': [0, 1, 2]
        })
    
    # Decode target
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    df['Risk_Level'] = df['Layoff_Risk'].map(risk_mapping)
    
    # Decode Industry
    industry_cols = [c for c in df.columns if c.startswith('Industry_')]
    if industry_cols:
        df['Industry'] = df[industry_cols].idxmax(axis=1).str.replace('Industry_', '')
    else:
        df['Industry'] = 'Unknown'
    
    # Decode Job Role
    job_cols = [c for c in df.columns if c.startswith('Job_Role_')]
    if job_cols:
        df['Job_Role'] = df[job_cols].idxmax(axis=1).str.replace('Job_Role_', '')
    else:
        df['Job_Role'] = 'Unknown'
    
    # Decode Company Size
    size_cols = [c for c in df.columns if c.startswith('Company_Size_')]
    if size_cols:
        df['Company_Size'] = df[size_cols].idxmax(axis=1).str.replace('Company_Size_', '')
    else:
        df['Company_Size'] = 'Medium'
    
    # Decode Job Level
    level_cols = [c for c in df.columns if c.startswith('Job_Level_')]
    if level_cols:
        df['Job_Level'] = df[level_cols].idxmax(axis=1).str.replace('Job_Level_', '')
    else:
        df['Job_Level'] = 'Entry'
    
    # Decode AI Adoption Level
    ai_cols = [c for c in df.columns if c.startswith('AI_Adoption_Level_')]
    if ai_cols:
        df['AI_Adoption'] = df[ai_cols].idxmax(axis=1).str.replace('AI_Adoption_Level_', '')
    else:
        df['AI_Adoption'] = 'Low'
    
    return df


@st.cache_data
def get_feature_ranges(df):
    """Get min/max ranges for numeric features."""
    numeric_cols = [
        'Age', 'Years_of_Experience', 'Routine_Task_Percentage',
        'Creativity_Requirement', 'Human_Interaction_Level',
        'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
        'Tasks_Automated_Percentage', 'AI_Training_Hours'
    ]
    
    ranges = {}
    for col in numeric_cols:
        if col in df.columns:
            ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean())
            }
        else:
            ranges[col] = {'min': 0, 'max': 100, 'mean': 50}
    
    return ranges


@st.cache_data
def get_categorical_options(df):
    """Get unique values for categorical features."""
    return {
        'Industry': sorted(df['Industry'].unique()) if 'Industry' in df.columns else ['Finance', 'IT', 'Healthcare'],
        'Job_Role': sorted(df['Job_Role'].unique()) if 'Job_Role' in df.columns else ['Analyst', 'Engineer', 'Manager'],
        'Company_Size': sorted(df['Company_Size'].unique()) if 'Company_Size' in df.columns else ['Small', 'Medium', 'Large'],
        'Job_Level': sorted(df['Job_Level'].unique()) if 'Job_Level' in df.columns else ['Entry', 'Mid', 'Senior'],
        'AI_Adoption': sorted(df['AI_Adoption'].unique()) if 'AI_Adoption' in df.columns else ['Low', 'Medium', 'High']
    }