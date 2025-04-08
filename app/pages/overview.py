"""
Overview page for the Customer Churn Prediction dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_overview(df, metrics):
    """Display overview dashboard with key metrics and visualizations"""
    st.header("Churn Analysis Overview")
    st.markdown("""
    This dashboard provides an overview of customer churn analysis, including key metrics,
    churn distribution across different segments, and risk analysis.
    """)
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Churn Rate", 
            f"{metrics['churn_rate']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Total Customers", 
            f"{metrics['total_customers']:,}"
        )
    
    with col3:
        st.metric(
            "Monthly Revenue", 
            f"${metrics['monthly_revenue']:,.2f}"
        )
    
    with col4:
        revenue_at_risk = metrics['revenue_at_risk']
        formatted_revenue = f"${revenue_at_risk/1000000:.2f}M" if revenue_at_risk >= 1000000 else f"${revenue_at_risk/1000:.2f}K"
        st.metric(
            "Annual Revenue at Risk", 
            formatted_revenue,
            f"{metrics['revenue_at_risk_percentage']:.1f}% of total"
        )

    st.subheader("Customer Risk Segments")

    risk_dist = df['risk_segment'].value_counts().reset_index()
    risk_dist.columns = ['Risk Segment', 'Count']
    
    col5, col6 = st.columns([3, 2])
    
    with col5:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']  # Green to red
        sns.barplot(x='Risk Segment', y='Count', hue='Risk Segment', data=risk_dist, palette=colors, ax=ax)
        ax.set_title('Customer Distribution by Risk Segment')
        ax.set_ylabel('Number of Customers')
        st.pyplot(fig)
    
    with col6:
        st.subheader("Risk Segment Details")
        for segment in ['High Risk', 'Medium-High Risk', 'Medium-Low Risk', 'Low Risk']:
            segment_count = len(df[df['risk_segment'] == segment])
            segment_percent = segment_count / len(df) * 100
            segment_churn_rate = df[df['risk_segment'] == segment]['Churn'].mean() * 100
            
            if segment == 'High Risk':
                st.markdown(f"**{segment}**: {segment_count:,} customers ({segment_percent:.1f}%)")
                st.markdown(f"- Average churn rate: {segment_churn_rate:.1f}%")
                st.markdown(f"- Monthly revenue: ${df[df['risk_segment'] == segment]['MonthlyCharges'].sum():,.2f}")

    st.subheader("Churn Analysis by Key Dimensions")

    dimension = st.selectbox(
        "Select Dimension",
        ["Contract", "TenureGroup", "InternetService", "PaymentMethod"]
    )

    if dimension == "TenureGroup":
        if df[dimension].dtype in [np.int64, np.float64]:
            tenure_labels = {
                0: '0-1 year',
                1: '1-2 years',
                2: '2-3 years',
                3: '3-4 years',
                4: '4-5 years',
                5: '5+ years'
            }

            churn_by_dimension = df.groupby(dimension)['Churn'].mean().reset_index()
            churn_by_dimension['Churn Percentage'] = churn_by_dimension['Churn'] * 100
            churn_by_dimension['Display Label'] = churn_by_dimension[dimension].map(tenure_labels)
            churn_by_dimension['Customer Count'] = df.groupby(dimension).size().values
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x='Display Label', y='Churn Percentage', data=churn_by_dimension, ax=ax)
        else:
            churn_by_dimension = df.groupby(dimension)['Churn'].mean().reset_index()
            churn_by_dimension['Churn Percentage'] = churn_by_dimension['Churn'] * 100
            churn_by_dimension['Customer Count'] = df.groupby(dimension).size().values

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x=dimension, y='Churn Percentage', data=churn_by_dimension, ax=ax)
    else:

        churn_by_dimension = df.groupby(dimension)['Churn'].mean().reset_index()
        churn_by_dimension['Churn Percentage'] = churn_by_dimension['Churn'] * 100
        churn_by_dimension['Customer Count'] = df.groupby(dimension).size().values

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(x=dimension, y='Churn Percentage', data=churn_by_dimension, ax=ax)

    for i, bar in enumerate(bars.patches):
        count = churn_by_dimension.iloc[i]['Customer Count']
        bars.text(
            bar.get_x() + bar.get_width()/2.,
            0.1,
            f'n={count:,}',
            ha='center',
            color='white',
            fontweight='bold'
        )
    
    ax.set_title(f'Churn Rate by {dimension}')
    ax.set_ylabel('Churn Rate (%)')
    st.pyplot(fig)

    st.subheader("Feature Correlation with Churn")

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 
                   'TotalServices', 'CLV', 'ContractRiskFactor']
    
    valid_cols = [col for col in numeric_cols if col in df.columns]

    corr_matrix = df[numeric_cols].corr()
    churn_corr = corr_matrix['Churn'].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    churn_corr = churn_corr.drop('Churn')
    sns.barplot(x=churn_corr.values, y=churn_corr.index, ax=ax)
    ax.set_title('Feature Correlation with Churn')
    ax.set_xlabel('Correlation Coefficient')
    st.pyplot(fig)