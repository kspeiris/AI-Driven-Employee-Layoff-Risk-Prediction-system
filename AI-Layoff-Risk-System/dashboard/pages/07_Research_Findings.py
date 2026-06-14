"""Research Findings page - Key insights and conclusions."""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_data

st.set_page_config(
    page_title="Research Findings | AI Workforce Intelligence",
    page_icon="📄",
    layout="wide"
)

df = load_data()

st.title("📄 Research Findings")
st.markdown("### Key insights from AI-driven workforce risk analysis")

st.markdown("---")

# Executive Summary
st.header("🎯 Executive Summary")

st.markdown("""
<div class="info-box" style="background-color: #e8f4f8;">
    <p>This research analyzed <strong>20,000 employees</strong> across <strong>8 industries</strong> to identify factors that predict layoff risk in the age of AI. 
    Using machine learning models (CatBoost achieving <strong>95.02% accuracy</strong>), we identified key risk drivers and protective factors.</p>
</div>
""", unsafe_allow_html=True)

# Key Findings
st.header("🔑 Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Risk Factors</h4>
        <ul>
            <li><strong>Routine_Task_Percentage</strong> is the strongest predictor (correlation: +0.78)</li>
            <li><strong>Tasks_Automated_Percentage</strong> strongly correlates with risk (+0.75)</li>
            <li><strong>AI_Usage_Hours</strong> shows moderate positive correlation (+0.50)</li>
            <li><strong>Manufacturing</strong> and <strong>Finance</strong> industries show elevated risk</li>
            <li><strong>Operational roles</strong> (Production Supervisor, Operator) are most vulnerable</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="success-box">
        <h4>🛡️ Protective Factors</h4>
        <ul>
            <li><strong>Creativity_Requirement</strong> strongly reduces risk (correlation: -0.75)</li>
            <li><strong>Human_Interaction_Level</strong> provides moderate protection (-0.16)</li>
            <li><strong>Higher education levels</strong> correlate with lower risk</li>
            <li><strong>Healthcare</strong> and <strong>Education</strong> industries show lower risk</li>
            <li><strong>Creative roles</strong> (ML Engineer, Nurse) are more resilient</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Statistical Analysis
st.header("📊 Statistical Analysis")

# Correlation chart
st.subheader("Feature Correlations with Layoff Risk")

numeric_df = df.select_dtypes(include='number')
correlations = numeric_df.corr()['Layoff_Risk'].drop('Layoff_Risk').sort_values(ascending=False).head(15)

fig = px.bar(
    x=correlations.values,
    y=correlations.index,
    orientation='h',
    title="Feature Correlation with Layoff Risk",
    color=correlations.values,
    color_continuous_scale='RdYlGn_r',
    labels={'x': 'Correlation Coefficient', 'y': ''}
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> **Note:** Positive correlations indicate features that increase layoff risk. 
> Negative correlations indicate protective factors.
""")

# Industry Risk Ranking
st.subheader("🏭 Industry Risk Ranking")

industry_risk = df.groupby('Industry').apply(
    lambda x: (x['Layoff_Risk'] == 2).sum() / len(x) * 100
).sort_values(ascending=False)

fig = px.bar(
    x=industry_risk.values,
    y=industry_risk.index,
    orientation='h',
    title="High Risk Percentage by Industry",
    color=industry_risk.values,
    color_continuous_scale='Reds'
)
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Model Performance Summary
st.header("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Best Model", "CatBoost")
    st.metric("Accuracy", "95.02%")

with col2:
    st.metric("Precision", "95.02%")
    st.metric("Recall", "95.02%")

with col3:
    st.metric("F1 Score", "95.02%")
    st.metric("AUC-ROC", "0.98")

# Recommendations
st.header("💡 Recommendations")

st.markdown("""
### For Organizations

1. **Invest in Creativity and Innovation**
   - Redesign routine-heavy roles to incorporate creative elements
   - Provide training for creative problem-solving
   - Creativity_Requirement shows strong protective effect (correlation: -0.75)

2. **Manage AI Implementation Strategically**
   - High AI adoption correlates with increased risk (r = +0.50)
   - Balance automation with human oversight
   - Provide adequate reskilling opportunities

3. **Focus on High-Risk Roles**
   - Target intervention programs for roles with >80% routine tasks
   - Prioritize manufacturing and finance sectors
   - Implement career transition programs for vulnerable roles

4. **Leverage Predictive Analytics**
   - Use the CatBoost model for early risk identification
   - Monitor key risk indicators quarterly
   - Develop personalized retention strategies

### For Employees

1. **Develop Creative Skills**: Creativity reduces layoff risk by up to 75%
2. **Reduce Routine Work**: Automate routine tasks, focus on strategic work
3. **Embrace AI Training**: Continuous learning reduces vulnerability
4. **Seek Human-Interactive Roles**: Higher interaction levels provide protection
""")

# Conclusion
st.header("📝 Conclusion")

st.markdown("""
<div class="info-box">
    <p>The AI Workforce Intelligence Platform demonstrates that machine learning can effectively 
    predict layoff risk with high accuracy (95%). Key findings reveal that routine task percentage 
    and creativity requirement are the most influential factors. Organizations can use these insights 
    to implement targeted interventions, while employees can focus on developing creative skills 
    to remain competitive in an AI-driven economy.</p>
    <br>
    <p><strong>Future Research Directions:</strong></p>
    <ul>
        <li>Longitudinal studies on AI adoption impact</li>
        <li>Industry-specific risk models</li>
        <li>Integration of economic indicators</li>
        <li>Real-time risk monitoring systems</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Report generated from analysis of 20,000 employee records across 8 industries*")
