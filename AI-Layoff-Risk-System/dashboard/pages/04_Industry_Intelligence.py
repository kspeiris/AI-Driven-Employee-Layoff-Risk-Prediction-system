"""Industry Intelligence page - Industry-specific risk analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_data

st.set_page_config(
    page_title="Industry Intelligence | AI Workforce Intelligence",
    page_icon="🏭",
    layout="wide"
)

df = load_data()

st.title("🏭 Industry Intelligence")
st.markdown("### In-depth analysis of layoff risk by industry")

# Industry selection
industries = sorted(df['Industry'].unique())
selected_industry = st.selectbox("Select Industry", industries)

if selected_industry:
    industry_df = df[df['Industry'] == selected_industry]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", f"{len(industry_df):,}")
    
    with col2:
        high_risk_pct = (industry_df['Layoff_Risk'] == 2).sum() / len(industry_df) * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    
    with col3:
        avg_ai_usage = industry_df['AI_Usage_Hours_Per_Week'].mean()
        st.metric("Avg AI Hours/Week", f"{avg_ai_usage:.1f}")
    
    with col4:
        avg_automation = industry_df['Tasks_Automated_Percentage'].mean()
        st.metric("Avg Automation %", f"{avg_automation:.1f}%")
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        risk_counts = industry_df['Risk_Level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title=f"Risk Profile - {selected_industry}",
            color=risk_counts.index,
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Job Role Risk Heatmap")
        
        # Get top job roles
        job_roles = industry_df['Job_Role'].value_counts().head(10).index
        job_risk_data = []
        
        for job in job_roles:
            job_data = industry_df[industry_df['Job_Role'] == job]
            high_pct = (job_data['Layoff_Risk'] == 2).sum() / len(job_data) * 100
            job_risk_data.append({'Job Role': job, 'High Risk %': high_pct, 'Count': len(job_data)})
        
        job_risk_df = pd.DataFrame(job_risk_data).sort_values('High Risk %', ascending=False)
        
        fig = px.bar(
            job_risk_df,
            x='High Risk %',
            y='Job Role',
            orientation='h',
            title="High Risk Percentage by Job Role",
            color='High Risk %',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Industry comparison
    st.subheader("Industry Comparison")
    
    # Calculate metrics for all industries
    industry_metrics = []
    for ind in industries:
        ind_df = df[df['Industry'] == ind]
        industry_metrics.append({
            'Industry': ind,
            'High Risk %': (ind_df['Layoff_Risk'] == 2).sum() / len(ind_df) * 100,
            'Avg AI Usage': ind_df['AI_Usage_Hours_Per_Week'].mean(),
            'Avg Automation': ind_df['Tasks_Automated_Percentage'].mean(),
            'Avg Creativity': ind_df['Creativity_Requirement'].mean()
        })
    
    comp_df = pd.DataFrame(industry_metrics)
    
    # Comparison charts
    metric_choice = st.selectbox(
        "Select Metric to Compare",
        ['High Risk %', 'Avg AI Usage', 'Avg Automation', 'Avg Creativity']
    )
    
    fig = px.bar(
        comp_df.sort_values(metric_choice, ascending=False),
        x='Industry',
        y=metric_choice,
        title=f"Industry Comparison - {metric_choice}",
        color=metric_choice,
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlations for this industry
    st.subheader(f"Risk Drivers in {selected_industry}")
    
    numeric_cols = [
        'Routine_Task_Percentage', 'Creativity_Requirement',
        'Human_Interaction_Level', 'AI_Usage_Hours_Per_Week',
        'Tasks_Automated_Percentage', 'AI_Training_Hours'
    ]
    
    correlations = industry_df[numeric_cols + ['Layoff_Risk']].corr()['Layoff_Risk'].drop('Layoff_Risk').sort_values(ascending=False)
    
    fig = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        title="Feature Correlations with Risk",
        color=correlations.values,
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Treemap view
st.markdown("---")
st.subheader("🏗️ Industry Structure Heatmap")

fig = px.treemap(
    df,
    path=['Industry', 'Job_Role'],
    values='Layoff_Risk',
    color='Layoff_Risk',
    color_continuous_scale='RdYlGn_r',
    title="Industry and Job Role Risk Heatmap"
)
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)