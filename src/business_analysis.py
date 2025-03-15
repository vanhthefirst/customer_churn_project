"""
Business analysis module for customer churn prediction.
"""
import pandas as pd
import numpy as np

def add_churn_probability(df, model, X_train_columns):
    """Add churn probability predictions to dataframe"""
    # Prepare features for prediction
    X = df.drop('Churn', axis=1) if 'Churn' in df.columns else df.copy()
    
    # Remove customerID if present
    if 'customerID' in X.columns:
        X = X.drop('customerID', axis=1)
        
    # Apply one-hot encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Align columns with training data
    X_aligned = X_encoded.reindex(columns=X_train_columns, fill_value=0)
    
    # Add predictions
    df_with_proba = df.copy()
    df_with_proba['churn_probability'] = model.predict_proba(X_aligned)[:, 1]
    
    # Create risk segments
    df_with_proba['risk_segment'] = pd.qcut(
        df_with_proba['churn_probability'], 
        q=[0, 0.25, 0.5, 0.75, 1.0], 
        labels=['Low Risk', 'Medium-Low Risk', 'Medium-High Risk', 'High Risk']
    )
    
    return df_with_proba

def get_segment_insights(segment_df, full_df):
    """Compare segment characteristics with overall population"""
    insights = {}
    
    # Compare numerical features
    for col in segment_df.select_dtypes(include=[np.number]).columns:
        if col in full_df.columns:  # Check if column exists in both DataFrames
            segment_mean = segment_df[col].mean()
            overall_mean = full_df[col].mean()
            difference = (segment_mean - overall_mean) / overall_mean * 100 if overall_mean != 0 else 0
            if abs(difference) > 10:  # Only show significant differences
                insights[col] = f"{difference:.1f}% {'higher' if difference > 0 else 'lower'} than average"
    
    # Compare categorical features
    for col in segment_df.select_dtypes(include=['object']).columns:
        if col in full_df.columns:  # Check if column exists in both DataFrames
            segment_top = segment_df[col].value_counts(normalize=True).index[0]
            overall_top = full_df[col].value_counts(normalize=True).index[0]
            if segment_top != overall_top:
                insights[col] = f"Most common: {segment_top} (vs {overall_top} overall)"
    
    return insights

def calculate_roi(segment_df, intervention_cost_per_customer, estimated_retention_rate):
    """Calculate ROI for retention interventions"""
    num_customers = len(segment_df)
    total_cost = num_customers * intervention_cost_per_customer
    monthly_revenue = segment_df['MonthlyCharges'].sum()
    annual_revenue_saved = monthly_revenue * 12 * estimated_retention_rate
    roi = (annual_revenue_saved - total_cost) / total_cost if total_cost > 0 else 0
    
    return {
        'customers_targeted': num_customers,
        'intervention_cost': total_cost,
        'potential_annual_revenue_saved': annual_revenue_saved,
        'estimated_roi': roi
    }

def get_revenue_metrics(df):
    """Calculate revenue-related metrics"""
    # Overall metrics
    total_customers = len(df)
    monthly_revenue = df['MonthlyCharges'].sum()
    annual_revenue = monthly_revenue * 12
    churn_rate = df['Churn'].mean() * 100
    
    # High-risk segment metrics
    high_risk = df[df['risk_segment'] == 'High Risk']
    high_risk_count = len(high_risk)
    high_risk_percentage = (high_risk_count / total_customers) * 100
    
    # Revenue at risk
    revenue_at_risk = high_risk['MonthlyCharges'].sum() * 12
    revenue_at_risk_percentage = (revenue_at_risk / annual_revenue) * 100
    
    return {
        'total_customers': total_customers,
        'monthly_revenue': monthly_revenue,
        'annual_revenue': annual_revenue,
        'churn_rate': churn_rate,
        'high_risk_customers': high_risk_count,
        'high_risk_percentage': high_risk_percentage,
        'revenue_at_risk': revenue_at_risk,
        'revenue_at_risk_percentage': revenue_at_risk_percentage
    }