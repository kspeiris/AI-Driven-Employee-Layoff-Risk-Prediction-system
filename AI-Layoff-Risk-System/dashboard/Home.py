"""Home page - Main dashboard with KPIs and overview."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.data_loader import load_data

# Page config
st.set_page_config(
    page_title="AI Workforce Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open(Path(__file__).parent / "assets" / "style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load data
df = load_data()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.markdown("## AI Workforce Intelligence")
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("Use the menu above to explore different sections.")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This platform uses machine learning to predict employee layoff risk based on:
    - Demographics
    - Job characteristics
    - AI adoption metrics
    - Task automation levels
    """)
    st.markdown("---")
    st.markdown("**Version:** 2.0.0")
    st.markdown("**Deployed Models:** CatBoost, ANN, DNN")

# Main content
st.title("🤖 AI Workforce Intelligence Platform")
st.markdown("### Real-time Workforce Risk Analytics Dashboard")

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Employees</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    high_risk = len(df[df['Layoff_Risk'] == 2])
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #dc3545, #c82333);">
        <div class="metric-value">{high_risk:,}</div>
        <div class="metric-label">High Risk Employees</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    medium_risk = len(df[df['Layoff_Risk'] == 1])
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #ffc107, #e0a800);">
        <div class="metric-value">{medium_risk:,}</div>
        <div class="metric-label">Medium Risk Employees</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    low_risk = len(df[df['Layoff_Risk'] == 0])
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #28a745, #1e7e34);">
        <div class="metric-value">{low_risk:,}</div>
        <div class="metric-label">Low Risk Employees</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    avg_ai_usage = df['AI_Usage_Hours_Per_Week'].mean()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #17a2b8, #117a8b);">
        <div class="metric-value">{avg_ai_usage:.1f} hrs</div>
        <div class="metric-label">Avg AI Usage/Week</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Two column layout for main charts
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📊 Risk Distribution")
    
    risk_counts = df['Risk_Level'].value_counts()
    colors = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
    
    fig = go.Figure(data=[
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker_colors=[colors[x] for x in risk_counts.index],
            hole=0.4,
            textinfo='label+percent',
            textposition='auto'
        )
    ])
    fig.update_layout(
        height=400,
        showlegend=False,
        annotations=[dict(text='Risk Profile', x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("🏭 Top Industries by Risk")
    
    industry_risk = df.groupby('Industry').apply(
        lambda x: (x['Layoff_Risk'] == 2).sum() / len(x) * 100
    ).sort_values(ascending=False).head(10)
    
    fig = px.bar(
        x=industry_risk.values,
        y=industry_risk.index,
        orientation='h',
        color=industry_risk.values,
        color_continuous_scale='RdYlGn_r',
        title="High Risk Percentage by Industry"
    )
    fig.update_layout(height=400, xaxis_title="High Risk (%)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Second row
left_col2, right_col2 = st.columns(2)

with left_col2:
    st.subheader("📈 AI Adoption vs Layoff Risk")
    
    ai_risk = df.groupby('AI_Adoption')['Layoff_Risk'].mean().reset_index()
    ai_risk['Risk_Level'] = ai_risk['Layoff_Risk'].map({0: 'Low', 1: 'Medium', 2: 'High'})
    
    fig = px.bar(
        ai_risk,
        x='AI_Adoption',
        y='Layoff_Risk',
        color='Risk_Level',
        color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
        title="Average Risk Score by AI Adoption Level"
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with right_col2:
    st.subheader("🎯 Key Insights")
    
    st.markdown("""
    <div class="info-box">
        <strong>📌 Key Finding #1</strong><br>
        <code>Routine_Task_Percentage</code> is the strongest predictor of layoff risk.
    </div>
    
    <div class="info-box">
        <strong>📌 Key Finding #2</strong><br>
        Higher creativity requirements significantly reduce layoff vulnerability.
    </div>
    
    <div class="info-box">
        <strong>📌 Key Finding #3</strong><br>
        Industries with high AI adoption show elevated risk levels.
    </div>
    
    <div class="success-box">
        <strong>✅ Model Performance</strong><br>
        CatBoost is deployed as the primary high-accuracy ML model; ANN and DNN are also available.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <hr>
    <p>© 2024 AI Workforce Intelligence Platform | Powered by CatBoost, DL models &amp; SHAP</p>
</div>
""", unsafe_allow_html=True)
