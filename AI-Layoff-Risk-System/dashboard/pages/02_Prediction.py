"""Prediction Center - Single and batch predictions."""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_data, get_feature_ranges, get_categorical_options

st.set_page_config(
    page_title="Prediction Center | AI Workforce Intelligence",
    page_icon="🤖",
    layout="wide"
)

# Load data for reference
df = load_data()
feature_ranges = get_feature_ranges(df)
cat_options = get_categorical_options(df)

# API endpoint
API_URL = "http://localhost:5000"

st.title("🎯 Layoff Risk Prediction Center")
st.markdown("### Predict employee layoff risk using AI models")

# Mode selection
mode = st.radio("Select Prediction Mode", ["Single Prediction", "Batch Prediction"], horizontal=True)

if mode == "Single Prediction":
    st.markdown("---")
    st.markdown("### 📝 Enter Employee Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.slider(
            "Age",
            min_value=int(feature_ranges['Age']['min']),
            max_value=int(feature_ranges['Age']['max']),
            value=int(feature_ranges['Age']['mean'])
        )
        
        experience = st.slider(
            "Years of Experience",
            min_value=int(feature_ranges['Years_of_Experience']['min']),
            max_value=int(feature_ranges['Years_of_Experience']['max']),
            value=int(feature_ranges['Years_of_Experience']['mean'])
        )
        
        st.markdown("#### 🏢 Employment")
        industry = st.selectbox("Industry", cat_options['Industry'])
        job_role = st.selectbox("Job Role", cat_options['Job_Role'])
        company_size = st.selectbox("Company Size", cat_options['Company_Size'])
        job_level = st.selectbox("Job Level", cat_options['Job_Level'])
    
    with col2:
        st.markdown("#### 🤖 AI & Automation")
        routine_task = st.slider(
            "Routine Task Percentage",
            min_value=0.0, max_value=100.0,
            value=round(float(feature_ranges['Routine_Task_Percentage']['mean']), 1),
            step=0.1
        )
        
        creativity = st.slider(
            "Creativity Requirement",
            min_value=0.0, max_value=100.0,
            value=round(float(feature_ranges['Creativity_Requirement']['mean']), 1),
            step=0.1
        )
        
        human_interaction = st.slider(
            "Human Interaction Level",
            min_value=0.0, max_value=100.0,
            value=round(float(feature_ranges['Human_Interaction_Level']['mean']), 1),
            step=0.1
        )
        
        ai_adoption = st.selectbox("AI Adoption Level", cat_options['AI_Adoption'])
        
        ai_tools = st.number_input(
            "Number of AI Tools Used",
            min_value=0, max_value=20,
            value=int(feature_ranges['Number_of_AI_Tools_Used']['mean'])
        )
        
        ai_usage_hours = st.slider(
            "AI Usage Hours Per Week",
            min_value=0.0, max_value=30.0,
            value=round(float(feature_ranges['AI_Usage_Hours_Per_Week']['mean']), 1),
            step=0.1
        )
        
        tasks_automated = st.slider(
            "Tasks Automated Percentage",
            min_value=0.0, max_value=100.0,
            value=round(float(feature_ranges['Tasks_Automated_Percentage']['mean']), 1),
            step=0.1
        )
        
        ai_training = st.slider(
            "AI Training Hours",
            min_value=0.0, max_value=80.0,
            value=round(float(feature_ranges['AI_Training_Hours']['mean']), 1),
            step=0.1
        )
    
    # Prepare features for API
    features = {
        'Age': age,
        'Years_of_Experience': experience,
        'Routine_Task_Percentage': routine_task,
        'Creativity_Requirement': creativity,
        'Human_Interaction_Level': human_interaction,
        'Number_of_AI_Tools_Used': ai_tools,
        'AI_Usage_Hours_Per_Week': ai_usage_hours,
        'Tasks_Automated_Percentage': tasks_automated,
        'AI_Training_Hours': ai_training,
        f'Education_Level_Bachelor\'s': 0,  # Default
        f'Education_Level_Master\'s': 0,
        'Education_Level_PhD': 0,
        f'Industry_{industry}': 1,
        f'Job_Role_{job_role}': 1,
        f'Company_Size_{company_size}': 1,
        f'Job_Level_{job_level}': 1,
        f'AI_Adoption_Level_{ai_adoption}': 1
    }
    
    # Add all other one-hot encoded columns as 0
    for col in cat_options.keys():
        if col != 'Industry' and col != 'Job_Role' and col != 'Company_Size' and col != 'Job_Level' and col != 'AI_Adoption':
            for opt in cat_options[col]:
                features[f'{col}_{opt}'] = 1 if opt == eval(col.lower().replace(' ', '_')) else 0
    
    if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing employee data..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=features)
                result = response.json()
                
                if result['success']:
                    st.success("Prediction complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    risk_color = {
                        'High': '🔴',
                        'Medium': '🟡',
                        'Low': '🟢'
                    }
                    
                    with col1:
                        st.metric("Predicted Risk", f"{risk_color[result['risk_level']]} {result['risk_level']}")
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    
                    with col3:
                        st.metric("Risk Score", f"{result['prediction']}")
                    
                    # Gauge chart
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['confidence'] * 100,
                        title={"text": "Confidence Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#1f3a93"},
                            'steps': [
                                {'range': [0, 50], 'color': "#ff9999"},
                                {'range': [50, 80], 'color': "#ffeb99"},
                                {'range': [80, 100], 'color': "#99ff99"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Probability breakdown
                    st.subheader("📊 Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Risk Level': ['Low', 'Medium', 'High'],
                        'Probability': [result['probabilities']['Low'], 
                                      result['probabilities']['Medium'], 
                                      result['probabilities']['High']]
                    })
                    
                    fig = px.bar(
                        prob_df, x='Risk Level', y='Probability',
                        color='Risk Level',
                        color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
                        title="Prediction Probabilities"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
                st.info("Make sure the backend server is running: python backend/app.py")

else:  # Batch Prediction
    st.markdown("---")
    st.markdown("### 📁 Batch Prediction")
    st.info("Upload a CSV file with employee data to get bulk predictions")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())
        
        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner(f"Processing {len(batch_df)} records..."):
                # Prepare batch data
                records = batch_df.to_dict('records')
                
                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json={'records': records}
                    )
                    results = response.json()
                    
                    if results['success']:
                        # Add predictions to dataframe
                        batch_df['Predicted_Risk'] = [p['risk_level'] for p in results['predictions']]
                        batch_df['Confidence'] = [p['confidence'] for p in results['predictions']]
                        
                        st.success(f"Processed {len(batch_df)} records")
                        
                        # Summary
                        st.subheader("Batch Summary")
                        summary = batch_df['Predicted_Risk'].value_counts()
                        
                        fig = px.pie(
                            values=summary.values,
                            names=summary.index,
                            title="Batch Prediction Distribution",
                            color=summary.index,
                            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results CSV",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                        
                        st.dataframe(batch_df)
                    else:
                        st.error(f"Error: {results.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"Failed to connect to API: {str(e)}")
