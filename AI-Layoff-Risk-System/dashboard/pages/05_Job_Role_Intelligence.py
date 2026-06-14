"""Job Role Intelligence page - Role-specific risk analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_data

st.set_page_config(
    page_title="Job Role Intelligence | AI Workforce Intelligence",
    page_icon="💼",
    layout="wide"
)

df = load_data()

st.title("💼 Job Role Intelligence")
st.markdown("### Comprehensive analysis of job role vulnerability")

# Job role selection
job_roles = sorted(df['Job_Role'].unique())
selected_role = st.selectbox("Select Job Role", job_roles)

if selected_role:
    role_df = df[df['Job_Role'] == selected_role]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Positions", f"{len(role_df):,}")
    
    with col2:
        high_risk_pct = (role_df['Layoff_Risk'] == 2).sum() / len(role_df) * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    
    with col3:
        avg_routine = role_df['Routine_Task_Percentage'].mean()
        st.metric("Avg Routine %", f"{avg_routine:.1f}%")
    
    with col4:
        avg_creativity = role_df['Creativity_Requirement'].mean()
        st.metric("Avg Creativity", f"{avg_creativity:.1f}")
    
    # Risk by industry for this role
    st.subheader(f"Risk Distribution by Industry for {selected_role}")
    
    industry_risk = role_df.groupby('Industry').apply(
        lambda x: (x['Layoff_Risk'] == 2).sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    fig = px.bar(
        x=industry_risk.values,
        y=industry_risk.index,
        orientation='h',
        title=f"High Risk Percentage by Industry - {selected_role}",
        color=industry_risk.values,
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Impact Analysis
    st.subheader("🤖 AI Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AI Adoption vs Risk
        ai_risk = role_df.groupby('AI_Adoption')['Layoff_Risk'].mean().reset_index()
        fig = px.bar(
            ai_risk,
            x='AI_Adoption',
            y='Layoff_Risk',
            title="Average Risk by AI Adoption Level",
            color='AI_Adoption'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Automation vs Risk
        automation_bins = pd.cut(role_df['Tasks_Automated_Percentage'], bins=5)
        automation_risk = role_df.groupby(automation_bins)['Layoff_Risk'].mean().reset_index()
        automation_risk['Automation_Range'] = automation_risk['Tasks_Automated_Percentage'].astype(str)
        
        fig = px.line(
            automation_risk,
            x='Automation_Range',
            y='Layoff_Risk',
            title="Risk vs Tasks Automated",
            markers=True
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Creativity Protection Effect
    st.subheader("🎨 Creativity Protection Effect")
    
    creativity_bins = pd.cut(role_df['Creativity_Requirement'], bins=5)
    creativity_risk = role_df.groupby(creativity_bins)['Layoff_Risk'].mean().reset_index()
    creativity_risk['Creativity_Range'] = creativity_risk['Creativity_Requirement'].astype(str)
    
    fig = px.line(
        creativity_risk,
        x='Creativity_Range',
        y='Layoff_Risk',
        title="Risk vs Creativity Requirement",
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Role comparison radar chart
    st.subheader("📊 Role Comparison Radar")
    
    # Select roles to compare
    compare_roles = st.multiselect(
        "Select roles to compare",
        job_roles,
        default=[selected_role] + (job_roles[:3] if len(job_roles) > 1 else [])
    )
    
    if compare_roles:
        metrics = ['Routine_Task_Percentage', 'Creativity_Requirement', 
                   'AI_Usage_Hours_Per_Week', 'Tasks_Automated_Percentage']
        
        radar_data = []
        for role in compare_roles:
            role_data = df[df['Job_Role'] == role]
            radar_data.append({
                'Job Role': role,
                'Routine Task %': role_data['Routine_Task_Percentage'].mean(),
                'Creativity': role_data['Creativity_Requirement'].mean(),
                'AI Usage Hours': role_data['AI_Usage_Hours_Per_Week'].mean(),
                'Tasks Automated %': role_data['Tasks_Automated_Percentage'].mean()
            })
        
        radar_df = pd.DataFrame(radar_data)
        
        fig = go.Figure()
        
        for role in compare_roles:
            role_metrics = radar_df[radar_df['Job Role'] == role].iloc[0]
            fig.add_trace(go.Scatterpolar(
                r=[role_metrics['Routine Task %'], role_metrics['Creativity'],
                   role_metrics['AI Usage Hours'], role_metrics['Tasks Automated %'],
                   role_metrics['Routine Task %']],
                theta=['Routine', 'Creativity', 'AI Usage', 'Automation', 'Routine'],
                fill='toself',
                name=role
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Job Role Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

# Most vulnerable roles
st.markdown("---")
st.subheader("⚠️ Most Vulnerable Job Roles")

vulnerable_roles = df.groupby('Job_Role').apply(
    lambda x: (x['Layoff_Risk'] == 2).sum() / len(x) * 100
).sort_values(ascending=False).head(10)

fig = px.bar(
    x=vulnerable_roles.values,
    y=vulnerable_roles.index,
    orientation='h',
    title="Top 10 High-Risk Job Roles",
    color=vulnerable_roles.values,
    color_continuous_scale='Reds'
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Safest roles
st.subheader("✅ Safest Job Roles")

safe_roles = df.groupby('Job_Role').apply(
    lambda x: (x['Layoff_Risk'] == 0).sum() / len(x) * 100
).sort_values(ascending=False).head(10)

fig = px.bar(
    x=safe_roles.values,
    y=safe_roles.index,
    orientation='h',
    title="Top 10 Low-Risk Job Roles",
    color=safe_roles.values,
    color_continuous_scale='Greens'
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)