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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.business_analysis import overall_business_insights

def get_segment_insights(segment_df, full_df):
    """
    Analyze segment data and return insights compared to the overall population.
    
    Parameters:
    segment_df (pandas.DataFrame): DataFrame for the selected segment
    full_df (pandas.DataFrame): DataFrame for the full dataset
    
    Returns:
    dict: Dictionary of insights for the segment
    """
    insights = {}

    columns_to_analyze = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'PhoneService', 'MultipleLines',
        'MonthlyCharges', 'TotalCharges', 'Contract', 
        'PaymentMethod', 'PaperlessBilling', 'tenure',
        'TotalServices'
    ]

    valid_columns = [col for col in columns_to_analyze if col in segment_df.columns and col in full_df.columns]
    
    for col in valid_columns:
        if segment_df[col].nunique() <= 1:
            continue

        if segment_df[col].dtype in [np.int64, np.float64]:
            segment_mean = segment_df[col].mean()
            overall_mean = full_df[col].mean()

            if overall_mean > 0:
                diff_pct = (segment_mean - overall_mean) / overall_mean * 100

                if abs(diff_pct) >= 10:
                    if diff_pct > 0:
                        insights[col] = f"{diff_pct:.1f}% higher than average ({segment_mean:.1f} vs {overall_mean:.1f})"
                    else:
                        insights[col] = f"{abs(diff_pct):.1f}% lower than average ({segment_mean:.1f} vs {overall_mean:.1f})"
        else:
            segment_counts = segment_df[col].value_counts(normalize=True) * 100
            overall_counts = full_df[col].value_counts(normalize=True) * 100

            most_common_value = segment_counts.index[0]
            segment_pct = segment_counts.iloc[0]
            overall_pct = overall_counts.get(most_common_value, 0)

            diff_pct = segment_pct - overall_pct

            if abs(diff_pct) >= 10:
                if diff_pct > 0:
                    insights[col] = f"{most_common_value} is {diff_pct:.1f}% more common ({segment_pct:.1f}% vs {overall_pct:.1f}%)"
                else:
                    insights[col] = f"{most_common_value} is {abs(diff_pct):.1f}% less common ({segment_pct:.1f}% vs {overall_pct:.1f}%)"
    
    return insights

def show_segment_analysis(df):
    df = copy.deepcopy(df)

    st.header("Customer Segment Analysis")
    st.write("Analyze different customer segments to understand their characteristics, churn patterns, and business impact.")

    segment_type = st.selectbox(
        "Select Segment Type",
        ["Risk Level", "Contract Type", "Internet Service", "Tenure Group", "Payment Method"],
        key="segment_type_selector"
    )

    segment_column_map = {
        "Risk Level": "risk_segment",
        "Contract Type": "Contract",
        "Internet Service": "InternetService",
        "Tenure Group": "TenureGroup",
        "Payment Method": "PaymentMethod"
    }
    
    segment_column = segment_column_map[segment_type]

    if segment_column == "TenureGroup" and df[segment_column].dtype in [np.int64, np.float64]:
        tenure_map = {
            0: '0-1 year',
            1: '1-2 years', 
            2: '2-3 years',
            3: '3-4 years',
            4: '4-5 years',
            5: '5+ years'
        }

        unique_values = sorted(df[segment_column].unique())
        segment_values = [tenure_map.get(val, f"Group {val}") for val in unique_values]

        selected_label = st.selectbox(
            f"Select {segment_type}",
            segment_values,
            key="tenure_selector"
        )

        reverse_map = {v: k for k, v in tenure_map.items()}
        selected_value = reverse_map.get(selected_label, 0)

        segment_df = df[df[segment_column] == selected_value]

        segment_title = f"{segment_type}: {selected_label}"
    else:
        segment_values = sorted(df[segment_column].unique().tolist())

        if segment_column == "risk_segment":
            segment_values = ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"]

        selected_segment = st.selectbox(
            f"Select {segment_type}",
            segment_values,
            key="segment_selector"
        )

        segment_df = df[df[segment_column] == selected_segment]
        segment_title = f"{segment_type}: {selected_segment}"

    st.subheader(segment_title)

    segment_size = len(segment_df)
    segment_pct = (segment_size / len(df)) * 100
    segment_churn = segment_df['Churn'].mean() * 100
    overall_churn = df['Churn'].mean() * 100
    churn_diff = segment_churn - overall_churn
    segment_monthly_revenue = segment_df['MonthlyCharges'].sum()

    st.write("### Segment Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Segment Size", f"{segment_size:,}", f"{segment_pct:.1f}% of total")
    col2.metric("Churn Rate", f"{segment_churn:.1f}%", f"{churn_diff:+.1f}% vs overall", delta_color="inverse")
    col3.metric("Monthly Revenue", f"${segment_monthly_revenue:,.2f}")

    insights = get_segment_insights(segment_df, df)

    demographic_insights = {}
    service_insights = {}
    financial_insights = {}

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

    st.write("### Segment Insights")

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

    st.write("### Segment Visualizations")

    tab1, tab2, tab3 = st.tabs(["Service Distribution", "Demographics", "Financial Metrics"])

    with tab1:
        st.write("#### Service Adoption")
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']

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

            locs = ax.get_xticks()
            labels = [item for item in services]
            ax.set_xticks(locs)
            ax.set_xticklabels(labels, rotation=45)
            fig.tight_layout(pad=2.0)
            
            st.pyplot(fig)
        else:
            st.info("Service data not available for this segment.")

    with tab2:
        st.write("#### Demographic Distribution")
        
        try:
            demo_fig, demo_ax = plt.subplots(2, 2, figsize=(14, 12))
            demo_ax = demo_ax.flatten()

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

            demo_fig.subplots_adjust(hspace=0.3, wspace=0.3)
            demo_fig.set_constrained_layout(True)
            
            st.pyplot(demo_fig)
        except Exception as e:
            st.error(f"Error creating demographics plots: {str(e)}")
            st.info("Some demographic visualizations may not be available for this segment.")

    with tab3:
        st.write("#### Financial Metrics")
        
        try:
            fin_fig, fin_ax = plt.subplots(2, 1, figsize=(10, 12))

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

            if 'TenureGroup' in segment_df.columns:
                segment_df_copy = segment_df.copy()
                df_copy = df.copy()
                
                if segment_df_copy['TenureGroup'].dtype in [np.int64, np.float64]:
                    tenure_map = {
                        0: '0-1 year',
                        1: '1-2 years', 
                        2: '2-3 years',
                        3: '3-4 years',
                        4: '4-5 years',
                        5: '5+ years'
                    }

                    segment_df_copy['TenureGroupDisplay'] = segment_df_copy['TenureGroup'].map(tenure_map)
                    df_copy['TenureGroupDisplay'] = df_copy['TenureGroup'].map(tenure_map)
                    
                    segment_tenure = segment_df_copy['TenureGroupDisplay'].value_counts(normalize=True) * 100
                    overall_tenure = df_copy['TenureGroupDisplay'].value_counts(normalize=True) * 100
                else:
                    segment_tenure = segment_df_copy['TenureGroup'].value_counts(normalize=True) * 100
                    overall_tenure = df_copy['TenureGroup'].value_counts(normalize=True) * 100

                tenure_data = pd.DataFrame({
                    'Tenure Group': segment_tenure.index,
                    'Segment (%)': segment_tenure.values,
                    'Overall (%)': [overall_tenure.get(tenure, 0) for tenure in segment_tenure.index]
                })

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

                locs = fin_ax[1].get_xticks()
                labels = [str(item) for item in tenure_data['Tenure Group']]
                fin_ax[1].set_xticks(locs[:len(labels)])
                fin_ax[1].set_xticklabels(labels, rotation=45)
                
            elif 'tenure' in segment_df.columns:
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

                locs = fin_ax[1].get_xticks()
                labels = [str(item) for item in tenure_data['Tenure Group']]
                fin_ax[1].set_xticks(locs[:len(labels)])
                fin_ax[1].set_xticklabels(labels, rotation=45)

            fin_fig.subplots_adjust(hspace=0.4)
            
            st.pyplot(fin_fig)
        except Exception as e:
            st.error(f"Error creating financial plots: {str(e)}")
            st.info("Some financial visualizations may not be available for this segment.")

    st.write("### Churn Analysis Within Segment")

    try:
        internal_segment_options = ["Contract", "InternetService", "PaymentMethod"]
        if 'TenureGroup' in segment_df.columns:
            internal_segment_options.insert(2, "TenureGroup")

        internal_segment_options = [opt for opt in internal_segment_options if opt in segment_df.columns]
        
        if internal_segment_options:
            internal_segment = st.selectbox(
                "Select factor to analyze churn within this segment",
                internal_segment_options,
                key="internal_segment_selector"
            )

            if internal_segment == "TenureGroup" and segment_df[internal_segment].dtype in [np.int64, np.float64]:
                segment_df_copy = segment_df.copy()

                tenure_map = {
                    0: '0-1 year',
                    1: '1-2 years', 
                    2: '2-3 years',
                    3: '3-4 years',
                    4: '4-5 years',
                    5: '5+ years'
                }

                churn_groups = segment_df_copy.groupby(internal_segment)['Churn'].mean() * 100
                counts = segment_df_copy.groupby(internal_segment).size()

                internal_churn_data = pd.DataFrame({
                    'TenureGroup': [tenure_map.get(idx, f"Group {idx}") for idx in churn_groups.index],
                    'Churn Rate (%)': churn_groups.values,
                    'Count': counts.values
                })

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x='TenureGroup', y='Churn Rate (%)', data=internal_churn_data, ax=ax)
                title = 'Churn Rate by Tenure Group within Selected Segment'
            else:
                segment_df_copy = segment_df.copy()
                internal_churn = segment_df_copy.groupby(internal_segment)['Churn'].mean() * 100
                internal_counts = segment_df_copy.groupby(internal_segment).size()
                
                internal_churn_data = pd.DataFrame({
                    internal_segment: internal_churn.index,
                    'Churn Rate (%)': internal_churn.values,
                    'Count': internal_counts.values
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x=internal_segment, y='Churn Rate (%)', data=internal_churn_data, ax=ax)
                title = f'Churn Rate by {internal_segment} within Selected Segment'

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

            locs = ax.get_xticks()
            if internal_segment == "TenureGroup" and segment_df[internal_segment].dtype in [np.int64, np.float64]:
                labels = internal_churn_data['TenureGroup'].tolist()
            else:
                labels = internal_churn_data[internal_segment].tolist()
            
            ax.set_xticks(locs[:len(labels)])
            ax.set_xticklabels(labels, rotation=45)

            fig.subplots_adjust(bottom=0.2)
            
            st.pyplot(fig)
        else:
            st.info("No appropriate dimensions available for internal segment analysis.")
    except Exception as e:
        st.error(f"Error in churn analysis: {str(e)}")
        st.info("Churn analysis is not available for this segment combination.")