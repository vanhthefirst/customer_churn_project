"""
Segment analysis page for the Churn Prediction dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import copy

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.business_analysis import get_segment_insights

def show_segment_analysis(df):
    """Display segment analysis dashboard"""
    # Make a copy of the dataframe to avoid mutations
    df = copy.deepcopy(df)
    
    # Header
    st.header("Customer Segment Analysis")
    st.write("Analyze different customer segments to understand their characteristics, churn patterns, and business impact.")
    
    # Segment Type Selection
    segment_type = st.selectbox(
        "Select Segment Type",
        ["Risk Level", "Contract Type", "Internet Service", "Tenure Group", "Payment Method"],
        key="segment_type_selector"
    )
    
    # Map segment type to dataframe column
    segment_column_map = {
        "Risk Level": "risk_segment",
        "Contract Type": "Contract",
        "Internet Service": "InternetService",
        "Tenure Group": "TenureGroup",
        "Payment Method": "PaymentMethod"
    }
    
    segment_column = segment_column_map[segment_type]
    
    # Handle segment selection based on type
    if segment_column == "TenureGroup" and df[segment_column].dtype in [np.int64, np.float64]:
        # Create labels for display
        tenure_map = {
            0: '0-1 year',
            1: '1-2 years', 
            2: '2-3 years',
            3: '3-4 years',
            4: '4-5 years',
            5: '5+ years'
        }
        
        # Get unique values and map to display labels
        unique_values = sorted(df[segment_column].unique())
        segment_values = [tenure_map.get(val, f"Group {val}") for val in unique_values]
        
        # Allow user to select segment
        selected_label = st.selectbox(
            f"Select {segment_type}",
            segment_values,
            key="tenure_selector"
        )
        
        # Map back to numeric value
        reverse_map = {v: k for k, v in tenure_map.items()}
        selected_value = reverse_map.get(selected_label, 0)
        
        # Filter for selected segment
        segment_df = df[df[segment_column] == selected_value]
        
        # Display title with label name
        segment_title = f"{segment_type}: {selected_label}"
    else:
        # Get unique values for the segment
        segment_values = sorted(df[segment_column].unique().tolist())
        
        # For risk segment, use specific order
        if segment_column == "risk_segment":
            segment_values = ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"]
        
        # Allow user to select segment
        selected_segment = st.selectbox(
            f"Select {segment_type}",
            segment_values,
            key="segment_selector"
        )
    
        # Filter for selected segment
        segment_df = df[df[segment_column] == selected_segment]
        
        # Display title with segment name
        segment_title = f"{segment_type}: {selected_segment}"
    
    # Display segment heading
    st.subheader(segment_title)
    
    # Calculate metrics
    segment_size = len(segment_df)
    segment_pct = (segment_size / len(df)) * 100
    segment_churn = segment_df['Churn'].mean() * 100
    overall_churn = df['Churn'].mean() * 100
    churn_diff = segment_churn - overall_churn
    segment_monthly_revenue = segment_df['MonthlyCharges'].sum()
    
    # Display metrics (without nesting)
    st.write("### Segment Metrics")
    
    # First row of metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Segment Size", f"{segment_size:,}", f"{segment_pct:.1f}% of total")
    col2.metric("Churn Rate", f"{segment_churn:.1f}%", f"{churn_diff:+.1f}% vs overall", delta_color="inverse")
    col3.metric("Monthly Revenue", f"${segment_monthly_revenue:,.2f}")
    
    # Get insights
    insights = get_segment_insights(segment_df, df)
    
    # Group insights by category
    demographic_insights = {}
    service_insights = {}
    financial_insights = {}
    
    # Categorize insights
    for key, value in insights.items():
        if key in ['gender', 'SeniorCitizen', 'Partner', 'Dependents']:
            demographic_insights[key] = value
        elif key in ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 
                   'StreamingMovies', 'PhoneService', 'MultipleLines',
                   'TotalServices']:
            service_insights[key] = value
        elif key in ['MonthlyCharges', 'TotalCharges', 'Contract', 
                    'PaymentMethod', 'PaperlessBilling', 'tenure',
                    'CLV', 'AvgMonthlySpend']:
            financial_insights[key] = value
    
    # Display insights section heading
    st.write("### Segment Insights")
    
    # Display insights in separate columns
    ins_col1, ins_col2, ins_col3 = st.columns(3)
    
    ins_col1.write("**Demographic Insights**")
    if demographic_insights:
        for key, value in demographic_insights.items():
            ins_col1.write(f"- **{key}**: {value}")
    else:
        ins_col1.write("No significant demographic differences")
    
    ins_col2.write("**Service Insights**")
    if service_insights:
        for key, value in service_insights.items():
            ins_col2.write(f"- **{key}**: {value}")
    else:
        ins_col2.write("No significant service differences")
    
    ins_col3.write("**Financial Insights**")
    if financial_insights:
        for key, value in financial_insights.items():
            ins_col3.write(f"- **{key}**: {value}")
    else:
        ins_col3.write("No significant financial differences")
    
    # Tab-based visualizations section
    st.write("### Segment Visualizations")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Service Distribution", "Demographics", "Financial Metrics"])
    
    # SERVICE TAB
    with tab1:
        st.write("#### Service Adoption")
        
        # Services to analyze
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Calculate service adoption rates
        service_data = []
        for service in services:
            if service in segment_df.columns:
                segment_adoption = (segment_df[service] == 'Yes').mean() * 100
                overall_adoption = (df[service] == 'Yes').mean() * 100
                service_data.append({
                    'Service': service,
                    'Segment Adoption (%)': segment_adoption,
                    'Overall Adoption (%)': overall_adoption
                })
        
        if service_data:
            service_df = pd.DataFrame(service_data)
            
            # Plot service adoption comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            services = service_df['Service'].tolist()
            x = np.arange(len(services))
            width = 0.35
            
            segment_bars = ax.bar(
                x - width/2, 
                service_df['Segment Adoption (%)'], 
                width, 
                label=f'Selected Segment'
            )
            
            overall_bars = ax.bar(
                x + width/2, 
                service_df['Overall Adoption (%)'], 
                width, 
                label='Overall Population',
                alpha=0.7
            )
            
            ax.set_ylabel('Adoption Rate (%)')
            ax.set_title('Service Adoption Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(services)
            ax.legend()
            
            # Rotate x labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
        else:
            st.info("Service data not available for this segment.")
    
    # DEMOGRAPHICS TAB
    with tab2:
        st.write("#### Demographic Distribution")
        
        try:
            # Create 2x2 grid of demographic charts
            demo_fig, demo_ax = plt.subplots(2, 2, figsize=(12, 10))
            demo_ax = demo_ax.flatten()
            
            # Gender distribution
            if 'gender' in segment_df.columns:
                gender_segment = segment_df['gender'].value_counts(normalize=True) * 100
                gender_overall = df['gender'].value_counts(normalize=True) * 100
                
                gender_data = pd.DataFrame({
                    'Gender': gender_segment.index,
                    'Segment (%)': gender_segment.values,
                    'Overall (%)': [gender_overall.get(gender, 0) for gender in gender_segment.index]
                })
                
                sns.barplot(x='Gender', y='Segment (%)', data=gender_data, ax=demo_ax[0])
                demo_ax[0].set_title('Gender Distribution')
            
            # Senior Citizen
            if 'SeniorCitizen' in segment_df.columns:
                senior_segment = segment_df['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).value_counts(normalize=True) * 100
                senior_overall = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).value_counts(normalize=True) * 100
                
                senior_data = pd.DataFrame({
                    'Senior Citizen': senior_segment.index,
                    'Segment (%)': senior_segment.values,
                    'Overall (%)': [senior_overall.get(senior, 0) for senior in senior_segment.index]
                })
                
                sns.barplot(x='Senior Citizen', y='Segment (%)', data=senior_data, ax=demo_ax[1])
                demo_ax[1].set_title('Senior Citizen Distribution')
            
            # Partner
            if 'Partner' in segment_df.columns and isinstance(segment_df['Partner'].iloc[0], (int, float, np.integer, np.floating)):
                partner_segment = segment_df['Partner'].map({0: 'No', 1: 'Yes'}).value_counts(normalize=True) * 100
                partner_overall = df['Partner'].map({0: 'No', 1: 'Yes'}).value_counts(normalize=True) * 100
                
                partner_data = pd.DataFrame({
                    'Partner': partner_segment.index,
                    'Segment (%)': partner_segment.values,
                    'Overall (%)': [partner_overall.get(partner, 0) for partner in partner_segment.index]
                })
                
                sns.barplot(x='Partner', y='Segment (%)', data=partner_data, ax=demo_ax[2])
                demo_ax[2].set_title('Partner Distribution')
            elif 'Partner' in segment_df.columns:
                partner_segment = segment_df['Partner'].value_counts(normalize=True) * 100
                partner_overall = df['Partner'].value_counts(normalize=True) * 100
                
                partner_data = pd.DataFrame({
                    'Partner': partner_segment.index,
                    'Segment (%)': partner_segment.values,
                    'Overall (%)': [partner_overall.get(partner, 0) for partner in partner_segment.index]
                })
                
                sns.barplot(x='Partner', y='Segment (%)', data=partner_data, ax=demo_ax[2])
                demo_ax[2].set_title('Partner Distribution')
            
            # Dependents
            if 'Dependents' in segment_df.columns and isinstance(segment_df['Dependents'].iloc[0], (int, float, np.integer, np.floating)):
                dependents_segment = segment_df['Dependents'].map({0: 'No', 1: 'Yes'}).value_counts(normalize=True) * 100
                dependents_overall = df['Dependents'].map({0: 'No', 1: 'Yes'}).value_counts(normalize=True) * 100
                
                dependents_data = pd.DataFrame({
                    'Dependents': dependents_segment.index,
                    'Segment (%)': dependents_segment.values,
                    'Overall (%)': [dependents_overall.get(dependent, 0) for dependent in dependents_segment.index]
                })
                
                sns.barplot(x='Dependents', y='Segment (%)', data=dependents_data, ax=demo_ax[3])
                demo_ax[3].set_title('Dependents Distribution')
            elif 'Dependents' in segment_df.columns:
                dependents_segment = segment_df['Dependents'].value_counts(normalize=True) * 100
                dependents_overall = df['Dependents'].value_counts(normalize=True) * 100
                
                dependents_data = pd.DataFrame({
                    'Dependents': dependents_segment.index,
                    'Segment (%)': dependents_segment.values,
                    'Overall (%)': [dependents_overall.get(dependent, 0) for dependent in dependents_segment.index]
                })
                
                sns.barplot(x='Dependents', y='Segment (%)', data=dependents_data, ax=demo_ax[3])
                demo_ax[3].set_title('Dependents Distribution')
            
            plt.tight_layout()
            st.pyplot(demo_fig)
        except Exception as e:
            st.error(f"Error creating demographics plots: {str(e)}")
            st.info("Some demographic visualizations may not be available for this segment.")
    
    # FINANCIAL TAB
    with tab3:
        st.write("#### Financial Metrics")
        
        try:
            # Create 2x1 grid of financial charts
            fin_fig, fin_ax = plt.subplots(2, 1, figsize=(10, 10))
            
            # Monthly charges distribution
            if 'MonthlyCharges' in segment_df.columns:
                sns.histplot(
                    data=segment_df, x='MonthlyCharges', 
                    kde=True, color='blue', 
                    ax=fin_ax[0], label='Selected Segment'
                )
                
                sns.histplot(
                    data=df, x='MonthlyCharges', 
                    kde=True, color='gray', alpha=0.5, 
                    ax=fin_ax[0], label='Overall Population'
                )
                
                fin_ax[0].set_title('Monthly Charges Distribution')
                fin_ax[0].set_xlabel('Monthly Charges ($)')
                fin_ax[0].legend()
            
            # Tenure distribution
            if 'TenureGroup' in segment_df.columns:
                # Handle numeric TenureGroup
                segment_df_copy = segment_df.copy()
                df_copy = df.copy()
                
                if segment_df_copy['TenureGroup'].dtype in [np.int64, np.float64]:
                    # Create labels for display
                    tenure_map = {
                        0: '0-1 year',
                        1: '1-2 years', 
                        2: '2-3 years',
                        3: '3-4 years',
                        4: '4-5 years',
                        5: '5+ years'
                    }
                    
                    # Map numeric values to display labels
                    segment_df_copy['TenureGroupDisplay'] = segment_df_copy['TenureGroup'].map(tenure_map)
                    df_copy['TenureGroupDisplay'] = df_copy['TenureGroup'].map(tenure_map)
                    
                    segment_tenure = segment_df_copy['TenureGroupDisplay'].value_counts(normalize=True) * 100
                    overall_tenure = df_copy['TenureGroupDisplay'].value_counts(normalize=True) * 100
                else:
                    segment_tenure = segment_df_copy['TenureGroup'].value_counts(normalize=True) * 100
                    overall_tenure = df_copy['TenureGroup'].value_counts(normalize=True) * 100
                
                # Combine for display
                tenure_data = pd.DataFrame({
                    'Tenure Group': segment_tenure.index,
                    'Segment (%)': segment_tenure.values,
                    'Overall (%)': [overall_tenure.get(tenure, 0) for tenure in segment_tenure.index]
                })
                
                # Sort by tenure group to ensure chronological order
                if segment_df['TenureGroup'].dtype not in [np.int64, np.float64]:
                    tenure_order = ['0-1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5+ years']
                    tenure_data['Tenure Group'] = pd.Categorical(
                        tenure_data['Tenure Group'], 
                        categories=tenure_order, 
                        ordered=True
                    )
                    tenure_data = tenure_data.sort_values('Tenure Group')
                
                sns.barplot(x='Tenure Group', y='Segment (%)', data=tenure_data, ax=fin_ax[1])
                fin_ax[1].set_title('Tenure Distribution')
                fin_ax[1].set_xticklabels(fin_ax[1].get_xticklabels(), rotation=45)
            elif 'tenure' in segment_df.columns:
                # Use tenure if TenureGroup is unavailable
                segment_df_copy = segment_df.copy()
                df_copy = df.copy()
                bins = [0, 12, 24, 36, 48, 60, 72]
                labels = ['0-1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-6 years']
                
                segment_df_copy['tenure_group'] = pd.cut(segment_df_copy['tenure'], bins=bins, labels=labels)
                df_copy['tenure_group'] = pd.cut(df_copy['tenure'], bins=bins, labels=labels)
                
                segment_tenure = segment_df_copy['tenure_group'].value_counts(normalize=True) * 100
                overall_tenure = df_copy['tenure_group'].value_counts(normalize=True) * 100
                
                tenure_data = pd.DataFrame({
                    'Tenure Group': segment_tenure.index,
                    'Segment (%)': segment_tenure.values,
                    'Overall (%)': [overall_tenure.get(tenure, 0) for tenure in segment_tenure.index]
                })
                
                sns.barplot(x='Tenure Group', y='Segment (%)', data=tenure_data, ax=fin_ax[1])
                fin_ax[1].set_title('Tenure Distribution')
                fin_ax[1].set_xticklabels(fin_ax[1].get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fin_fig)
        except Exception as e:
            st.error(f"Error creating financial plots: {str(e)}")
            st.info("Some financial visualizations may not be available for this segment.")
    
    # Churn analysis section
    st.write("### Churn Analysis Within Segment")
    
    # Internal segment analysis
    try:
        # Allow user to select internal segment to analyze churn
        internal_segment_options = ["Contract", "InternetService", "PaymentMethod"]
        if 'TenureGroup' in segment_df.columns:
            internal_segment_options.insert(2, "TenureGroup")
        
        # Only keep columns that exist in the dataframe
        internal_segment_options = [opt for opt in internal_segment_options if opt in segment_df.columns]
        
        if internal_segment_options:
            internal_segment = st.selectbox(
                "Select factor to analyze churn within this segment",
                internal_segment_options,
                key="internal_segment_selector"
            )
            
            # Special handling for numeric TenureGroup
            if internal_segment == "TenureGroup" and segment_df[internal_segment].dtype in [np.int64, np.float64]:
                segment_df_copy = segment_df.copy()
                
                # Map to display values for visualization
                tenure_map = {
                    0: '0-1 year',
                    1: '1-2 years', 
                    2: '2-3 years',
                    3: '3-4 years',
                    4: '4-5 years',
                    5: '5+ years'
                }
                
                # Group by TenureGroup and calculate churn rate
                churn_groups = segment_df_copy.groupby(internal_segment)['Churn'].mean() * 100
                counts = segment_df_copy.groupby(internal_segment).size()
                
                # Create dataframe for visualization
                internal_churn_data = pd.DataFrame({
                    'TenureGroup': [tenure_map.get(idx, f"Group {idx}") for idx in churn_groups.index],
                    'Churn Rate (%)': churn_groups.values,
                    'Count': counts.values
                })
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x='TenureGroup', y='Churn Rate (%)', data=internal_churn_data, ax=ax)
                title = 'Churn Rate by Tenure Group within Selected Segment'
            else:
                segment_df_copy = segment_df.copy()
                
                # Calculate churn rate by internal segment
                internal_churn = segment_df_copy.groupby(internal_segment)['Churn'].mean() * 100
                internal_counts = segment_df_copy.groupby(internal_segment).size()
                
                internal_churn_data = pd.DataFrame({
                    internal_segment: internal_churn.index,
                    'Churn Rate (%)': internal_churn.values,
                    'Count': internal_counts.values
                })
                
                # Create bar chart with customer counts
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x=internal_segment, y='Churn Rate (%)', data=internal_churn_data, ax=ax)
                title = f'Churn Rate by {internal_segment} within Selected Segment'
            
            # Add customer count as text on bars
            for i, bar in enumerate(bars.patches):
                count = internal_churn_data.iloc[i]['Count']
                bars.text(
                    bar.get_x() + bar.get_width()/2.,
                    5,
                    f'n={count}',
                    ha='center',
                    color='white',
                    fontweight='bold'
                )
            
            ax.set_title(title)
            ax.set_ylabel('Churn Rate (%)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No appropriate dimensions available for internal segment analysis.")
    except Exception as e:
        st.error(f"Error in churn analysis: {str(e)}")
        st.info("Churn analysis is not available for this segment combination.")