"""Analytics page - Dataset exploration and feature analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_data

st.set_page_config(
    page_title="Analytics | AI Workforce Intelligence",
    page_icon="📊",
    layout="wide"
)

df = load_data()

st.title("📊 Workforce Analytics")
st.markdown("### Explore dataset insights and feature distributions")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Dataset Overview", 
    "📈 Feature Analysis", 
    "🔗 Correlations", 
    "📊 Distributions"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.markdown(f"""
        - **Total Records:** `{len(df):,}`
        - **Features:** `{df.shape[1]}`
        - **Classes:** Low, Medium, High
        - **Missing Values:** `{df.isnull().sum().sum()}`
        - **Duplicates:** `{df.duplicated().sum()}`
        """)
    
    with col2:
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Data Types")
    dtype_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Type': df.dtypes.values
    })
    st.dataframe(dtype_df, use_container_width=True)

with tab2:
    st.subheader("Numeric Features Summary")
    
    numeric_cols = [
        'Age', 'Years_of_Experience', 'Routine_Task_Percentage',
        'Creativity_Requirement', 'Human_Interaction_Level',
        'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
        'Tasks_Automated_Percentage', 'AI_Training_Hours'
    ]
    
    summary_df = df[numeric_cols].describe().round(2)
    st.dataframe(summary_df, use_container_width=True)
    
    st.subheader("Feature Explorer")
    
    selected_feature = st.selectbox("Select a feature to analyze", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x=selected_feature, color='Risk_Level',
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
            title=f"{selected_feature} Distribution by Risk Level",
            barmode='overlay', opacity=0.7
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df, x='Risk_Level', y=selected_feature,
            color='Risk_Level',
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
            title=f"{selected_feature} Boxplot by Risk Level"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Feature Correlation with Layoff Risk")
    
    numeric_df = df.select_dtypes(include='number')
    correlations = numeric_df.corr()['Layoff_Risk'].drop('Layoff_Risk').sort_values(ascending=False)
    
    # Create color mapping
    colors = ['#dc3545' if x > 0 else '#28a745' for x in correlations.values[:20]]
    
    fig = px.bar(
        x=correlations.values[:20],
        y=correlations.index[:20],
        orientation='h',
        color=correlations.values[:20],
        color_continuous_scale='RdYlGn_r',
        title="Top 20 Feature Correlations"
    )
    fig.update_layout(height=600, xaxis_title="Correlation", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Full Correlation Matrix")
    
    corr_matrix = df[numeric_cols + ['Layoff_Risk']].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Categorical Feature Distributions")
    
    categorical_cols = ['Industry', 'Job_Role', 'Company_Size', 'Job_Level', 'AI_Adoption']
    
    for col in categorical_cols:
        st.markdown(f"#### {col}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            counts = df[col].value_counts().head(10)
            fig = px.bar(
                x=counts.values, y=counts.index, orientation='h',
                title=f"Top 10 {col}s by Count"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            risk_by_cat = df.groupby(col)['Layoff_Risk'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=risk_by_cat.values, y=risk_by_cat.index, orientation='h',
                title=f"Average Risk by {col}",
                color=risk_by_cat.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
