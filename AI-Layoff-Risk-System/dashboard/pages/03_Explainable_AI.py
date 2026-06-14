"""Explainable AI page - SHAP and LIME explanations."""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_data

st.set_page_config(
    page_title="Explainable AI | AI Workforce Intelligence",
    page_icon="🔍",
    layout="wide"
)

df = load_data()
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")
REQUEST_TIMEOUT = 30

st.title("🔍 Explainable AI Center")
st.markdown("### Understand why employees are at risk using SHAP and LIME")

# Select an employee for explanation
st.markdown("---")
st.markdown("### 👤 Select Employee for Analysis")

# Create sample selection
sample_df = df.sample(min(100, len(df)))[['Age', 'Years_of_Experience', 'Industry', 'Job_Role', 'Routine_Task_Percentage', 'Creativity_Requirement', 'Risk_Level']]
sample_df = sample_df.reset_index(drop=True)
sample_df.index = sample_df.index + 1

selected_idx = st.selectbox(
    "Choose an employee record",
    options=sample_df.index,
    format_func=lambda x: f"Employee #{x} - {sample_df.loc[x, 'Industry']} - {sample_df.loc[x, 'Job_Role']}"
)

if selected_idx:
    selected_record = sample_df.loc[selected_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Employee Details")
        st.json(selected_record.to_dict())
    
    with col2:
        risk_level = selected_record['Risk_Level']
        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
            <h4>Current Risk Assessment</h4>
            <p style="font-size: 2rem; margin: 0;">{risk_color[risk_level]} {risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get SHAP explanation
    st.markdown("---")
    st.subheader("📊 SHAP Analysis (Global and Local)")
    
    # Prepare features for API (simplified for demo)
    features = {
        'Age': float(selected_record['Age']),
        'Years_of_Experience': float(selected_record['Years_of_Experience']),
        'Routine_Task_Percentage': float(df[df['Industry'] == selected_record['Industry']]['Routine_Task_Percentage'].mean()),
        'Creativity_Requirement': float(df[df['Industry'] == selected_record['Industry']]['Creativity_Requirement'].mean()),
        'Human_Interaction_Level': 50.0,
        'Number_of_AI_Tools_Used': 2.0,
        'AI_Usage_Hours_Per_Week': 5.0,
        'Tasks_Automated_Percentage': 30.0,
        'AI_Training_Hours': 10.0
    }
    
    with st.spinner("Computing SHAP values..."):
        try:
            response = requests.post(f"{API_URL}/explain/shap", json=features, timeout=REQUEST_TIMEOUT)
            shap_result = response.json()
            
            if shap_result['success']:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ⬆️ Factors Increasing Risk")
                    if shap_result['positive_contributors']:
                        pos_df = pd.DataFrame(shap_result['positive_contributors'])
                        st.dataframe(pos_df, use_container_width=True)
                    else:
                        st.info("No positive contributors found")
                
                with col2:
                    st.markdown("#### ⬇️ Factors Reducing Risk")
                    if shap_result['negative_contributors']:
                        neg_df = pd.DataFrame(shap_result['negative_contributors'])
                        st.dataframe(neg_df, use_container_width=True)
                    else:
                        st.info("No negative contributors found")
                
                # Feature importance visualization
                st.subheader("Feature Importance (SHAP Values)")
                
                all_features = pd.DataFrame(shap_result['all_features'])
                all_features['abs_shap'] = abs(all_features['shap_value'])
                top_features = all_features.nlargest(15, 'abs_shap')
                
                fig = px.bar(
                    top_features,
                    x='shap_value',
                    y='feature',
                    orientation='h',
                    title="Top 15 Features by SHAP Impact",
                    color='shap_value',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Could not fetch SHAP explanation: {str(e)}")
    
    # LIME explanation
    st.subheader("🔬 LIME Local Explanation")
    
    if st.button("Generate LIME Explanation", type="secondary"):
        with st.spinner("Computing LIME explanation..."):
            try:
                response = requests.post(f"{API_URL}/explain/lime", json=features, timeout=REQUEST_TIMEOUT)
                lime_result = response.json()
                
                if lime_result['success'] and lime_result['explanation']:
                    lime_df = pd.DataFrame(lime_result['explanation'])
                    lime_df['abs_weight'] = abs(lime_df['weight'])
                    lime_df = lime_df.nlargest(10, 'abs_weight')
                    
                    fig = px.bar(
                        lime_df,
                        x='weight',
                        y='feature',
                        orientation='h',
                        title="LIME Feature Weights",
                        color='weight',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No LIME explanation available for this prediction")
            
            except Exception as e:
                st.error(f"Could not fetch LIME explanation: {str(e)}")

# Global SHAP Summary
st.markdown("---")
st.subheader("📊 Global SHAP Analysis")

if st.button("Generate Global Feature Importance"):
    with st.spinner("Analyzing global feature importance..."):
        try:
            # Take a sample for global analysis
            sample_features = df.sample(500)[[
                'Age', 'Years_of_Experience', 'Routine_Task_Percentage',
                'Creativity_Requirement', 'Human_Interaction_Level',
                'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
                'Tasks_Automated_Percentage', 'AI_Training_Hours'
            ]].values.tolist()
            
            response = requests.post(
                f"{API_URL}/explain/shap/summary",
                json={'sample_data': sample_features},
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                importance = response.json()['feature_importance']
                imp_df = pd.DataFrame(importance)
                
                fig = px.bar(
                    imp_df.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Global Feature Importance (SHAP)",
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Could not generate global SHAP summary: {str(e)}")
