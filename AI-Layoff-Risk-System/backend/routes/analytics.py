"""Analytics API routes."""

import pandas as pd
from flask import Blueprint, jsonify
from pathlib import Path

analytics_bp = Blueprint('analytics', __name__)

# Load dataset
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed_layoff_dataset.csv"
df = pd.read_csv(DATA_PATH)


def risk_correlations():
    """Calculate correlations using only numeric columns."""
    numeric_df = df.select_dtypes(include='number')
    return numeric_df.corr()['Layoff_Risk'].drop('Layoff_Risk').sort_values(ascending=False)


@analytics_bp.route('/analytics/summary', methods=['GET'])
def get_summary():
    """Get dataset summary statistics."""
    
    # Target distribution
    risk_counts = df['Layoff_Risk'].value_counts().to_dict()
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_distribution = {risk_mapping[k]: v for k, v in risk_counts.items()}
    
    # Industry risk analysis
    industry_cols = [c for c in df.columns if c.startswith('Industry_')]
    industry_risk = {}
    for col in industry_cols:
        industry = col.replace('Industry_', '')
        high_risk = df[(df[col] == 1) & (df['Layoff_Risk'] == 2)].shape[0]
        total = df[df[col] == 1].shape[0]
        if total > 0:
            industry_risk[industry] = round(high_risk / total * 100, 1)
    
    # Top 5 industries by risk
    top_risk_industries = dict(sorted(industry_risk.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Correlation with target
    correlations = risk_correlations()
    top_correlations = correlations.head(10).to_dict()
    
    return jsonify({
        'total_records': int(df.shape[0]),
        'total_features': int(df.shape[1]),
        'risk_distribution': risk_distribution,
        'top_risk_industries': top_risk_industries,
        'top_correlations': {k: round(v, 3) for k, v in top_correlations.items()}
    })


@analytics_bp.route('/analytics/industry/<industry>', methods=['GET'])
def get_industry_analysis(industry):
    """Get detailed analysis for a specific industry."""
    
    industry_col = f'Industry_{industry}'
    if industry_col not in df.columns:
        return jsonify({'error': f'Industry {industry} not found'}), 404
    
    industry_df = df[df[industry_col] == 1]
    
    # Risk distribution
    risk_counts = industry_df['Layoff_Risk'].value_counts().to_dict()
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_distribution = {risk_mapping[k]: v for k, v in risk_counts.items()}
    
    # Top job roles in this industry
    job_cols = [c for c in df.columns if c.startswith('Job_Role_')]
    job_risks = []
    for col in job_cols:
        job = col.replace('Job_Role_', '')
        job_data = industry_df[industry_df[col] == 1]
        if len(job_data) > 0:
            high_risk_pct = (job_data['Layoff_Risk'] == 2).sum() / len(job_data) * 100
            job_risks.append({'job_role': job, 'high_risk_percentage': round(high_risk_pct, 1), 'count': len(job_data)})
    
    job_risks = sorted(job_risks, key=lambda x: x['high_risk_percentage'], reverse=True)[:10]
    
    # Average metrics
    metrics = {
        'avg_routine_task': round(industry_df['Routine_Task_Percentage'].mean(), 1),
        'avg_creativity': round(industry_df['Creativity_Requirement'].mean(), 1),
        'avg_ai_usage': round(industry_df['AI_Usage_Hours_Per_Week'].mean(), 1),
        'avg_tasks_automated': round(industry_df['Tasks_Automated_Percentage'].mean(), 1)
    }
    
    return jsonify({
        'industry': industry,
        'total_employees': len(industry_df),
        'risk_distribution': risk_distribution,
        'top_risk_jobs': job_risks,
        'metrics': metrics
    })


@analytics_bp.route('/analytics/job/<job_role>', methods=['GET'])
def get_job_analysis(job_role):
    """Get detailed analysis for a specific job role."""
    
    job_col = f'Job_Role_{job_role}'
    if job_col not in df.columns:
        return jsonify({'error': f'Job role {job_role} not found'}), 404
    
    job_df = df[df[job_col] == 1]
    
    # Risk distribution
    risk_counts = job_df['Layoff_Risk'].value_counts().to_dict()
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_distribution = {risk_mapping[k]: v for k, v in risk_counts.items()}
    
    # Industry distribution for this job
    industry_cols = [c for c in df.columns if c.startswith('Industry_')]
    industries = []
    for col in industry_cols:
        industry = col.replace('Industry_', '')
        count = job_df[job_df[col] == 1].shape[0]
        if count > 0:
            industries.append({'industry': industry, 'count': count})
    
    industries = sorted(industries, key=lambda x: x['count'], reverse=True)
    
    # AI adoption impact
    ai_adoption_cols = ['AI_Adoption_Level_Low', 'AI_Adoption_Level_Medium']
    ai_impact = {}
    for col in ai_adoption_cols:
        level = col.replace('AI_Adoption_Level_', '')
        level_df = job_df[job_df[col] == 1]
        high_risk_pct = (level_df['Layoff_Risk'] == 2).sum() / len(level_df) * 100 if len(level_df) > 0 else 0
        ai_impact[level] = round(high_risk_pct, 1)
    
    return jsonify({
        'job_role': job_role,
        'total_employees': len(job_df),
        'risk_distribution': risk_distribution,
        'industries': industries[:10],
        'ai_impact': ai_impact,
        'avg_routine_task': round(job_df['Routine_Task_Percentage'].mean(), 1),
        'avg_creativity': round(job_df['Creativity_Requirement'].mean(), 1)
    })


@analytics_bp.route('/analytics/correlations', methods=['GET'])
def get_correlations():
    """Get feature correlations with layoff risk."""
    
    correlations = risk_correlations()
    
    return jsonify({
        'correlations': {k: round(v, 3) for k, v in correlations.head(20).items()}
    })
