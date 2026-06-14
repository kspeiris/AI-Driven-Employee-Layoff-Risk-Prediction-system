"""Configuration settings for the backend service."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR.parent / "data"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Database
DATABASE_PATH = BASE_DIR / "ai_layoff.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Model files
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True

# Feature columns (must match training)
FEATURE_COLUMNS = [
    'Age', 'Years_of_Experience', 'Routine_Task_Percentage',
    'Creativity_Requirement', 'Human_Interaction_Level',
    'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
    'Tasks_Automated_Percentage', 'AI_Training_Hours',
    'Education_Level_High School', "Education_Level_Master's",
    'Education_Level_PhD', 'Industry_Finance', 'Industry_Healthcare',
    'Industry_IT', 'Industry_Logistics', 'Industry_Manufacturing',
    'Industry_Retail', 'Industry_Telecom', 'Job_Role_Accountant',
    'Job_Role_Auditor', 'Job_Role_Data Analyst', 'Job_Role_Dispatcher',
    'Job_Role_Financial Analyst', 'Job_Role_Health Analyst',
    'Job_Role_Inventory Analyst', 'Job_Role_ML Engineer',
    'Job_Role_Medical Assistant', 'Job_Role_Network Engineer',
    'Job_Role_Nurse', 'Job_Role_Operations Analyst', 'Job_Role_Operator',
    'Job_Role_Production Supervisor', 'Job_Role_Quality Engineer',
    'Job_Role_Research Assistant', 'Job_Role_Sales Associate',
    'Job_Role_Software Engineer', 'Job_Role_Store Manager',
    'Job_Role_Supply Chain Analyst', 'Job_Role_Support Specialist',
    'Job_Role_Teacher', 'Job_Role_Warehouse Manager',
    'Company_Size_Medium', 'Company_Size_Small', 'Job_Level_Mid',
    'Job_Level_Senior', 'AI_Adoption_Level_Low', 'AI_Adoption_Level_Medium'
]

# Categorical feature mappings for API
INDUSTRY_LIST = [
    'Finance', 'Healthcare', 'IT', 'Logistics', 'Manufacturing', 'Retail', 'Telecom', 'Education'
]

JOB_ROLE_LIST = [
    'Accountant', 'Auditor', 'Data Analyst', 'Dispatcher', 'Financial Analyst',
    'Health Analyst', 'Inventory Analyst', 'ML Engineer', 'Medical Assistant',
    'Network Engineer', 'Nurse', 'Operations Analyst', 'Operator',
    'Production Supervisor', 'Quality Engineer', 'Research Assistant',
    'Sales Associate', 'Software Engineer', 'Store Manager',
    'Supply Chain Analyst', 'Support Specialist', 'Teacher', 'Warehouse Manager'
]

COMPANY_SIZE_LIST = ['Small', 'Medium', 'Large']
JOB_LEVEL_LIST = ['Entry', 'Mid', 'Senior']
AI_ADOPTION_LIST = ['Low', 'Medium', 'High']

# Risk mapping
RISK_MAPPING = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}