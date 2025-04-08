"""
Business analysis module for customer churn prediction.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def add_churn_probability(df, model, X_train_columns):
    X = df.drop(['Churn', 'customerID'] if 'customerID' in df.columns and 'Churn' in df.columns 
            else ['Churn'] if 'Churn' in df.columns 
            else ['customerID'] if 'customerID' in df.columns 
            else [], axis=1, errors='ignore')

    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    missing_cols = set(X_train_columns) - set(X_encoded.columns)
    for col in missing_cols:
        X_encoded[col] = 0

    X_aligned = X_encoded[X_train_columns]

    df_with_proba = df.copy()
    df_with_proba['churn_probability'] = model.predict_proba(X_aligned)[:, 1]

    df_with_proba['risk_segment'] = pd.cut(
        df_with_proba['churn_probability'], 
        bins=[0, 0.25, 0.5, 0.75, 1.0], 
        labels=['Low Risk', 'Medium-Low Risk', 'Medium-High Risk', 'High Risk'],
    )
    
    return df_with_proba

def geographic_churn_analysis(df, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    geo_insights = {}

    df_copy = df.copy()
    if df_copy['Churn'].dtype == 'object':
        df_copy['Churn'] = pd.to_numeric(df_copy['Churn'], errors='coerce').fillna(0)

    required_cols = ['City', 'zip_code']
    if not all(col in df_copy.columns for col in required_cols):
        print("Geographic analysis requires City and zip_code columns")
        return geo_insights
    
    # Churn by city
    city_churn = df_copy.groupby('City')['Churn'].agg(['mean', 'count']).reset_index()
    city_churn.columns = ['City', 'Churn Rate', 'Customer Count']
    city_churn['Churn Rate'] = city_churn['Churn Rate'] * 100
    
    # Filter to cities with sufficient customers for statistical significance
    significant_cities = city_churn[city_churn['Customer Count'] >= 30].sort_values('Churn Rate', ascending=False)
    significant_cities.to_csv(os.path.join(output_dir, 'churn_by_city.csv'), index=False)

    plt.figure(figsize=(12, 10))

    top_cities = significant_cities.head(10)
    plt.subplot(2, 1, 1)
    sns.barplot(x='Churn Rate', y='City', data=top_cities)
    plt.title('Top 10 Cities by Churn Rate')
    plt.xlabel('Churn Rate (%)')

    bottom_cities = significant_cities.tail(10)
    plt.subplot(2, 1, 2)
    sns.barplot(x='Churn Rate', y='City', data=bottom_cities)
    plt.title('Bottom 10 Cities by Churn Rate')
    plt.xlabel('Churn Rate (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'city_churn_comparison.png'))
    plt.close()

    if 'Population' in df.columns:
        df['Population Segment'] = pd.qcut(
            df['Population'], 
            q=4, 
            labels=['Low Population', 'Medium-Low Population', 'Medium-High Population', 'High Population']
        )

        pop_churn = df.groupby('Population Segment')['Churn'].agg(['mean', 'count']).reset_index()
        pop_churn.columns = ['Population Segment', 'Churn Rate', 'Customer Count']
        pop_churn['Churn Rate'] = pop_churn['Churn Rate'] * 100
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Population Segment', y='Churn Rate', data=pop_churn)
        plt.title('Churn Rate by Population Density')
        plt.xlabel('Population Segment')
        plt.ylabel('Churn Rate (%)')
        plt.savefig(os.path.join(output_dir, 'population_churn_analysis.png'))
        plt.close()

        geo_insights['population_churn'] = pop_churn.to_dict('records')

    geo_insights['high_churn_cities'] = top_cities.to_dict('records')
    geo_insights['low_churn_cities'] = bottom_cities.to_dict('records')
    
    return geo_insights

def service_usage_analysis(df, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    usage_insights = {}
    usage_cols = ['Avg Monthly GB Download', 'Avg Monthly Long Distance Charges',
                  'TotalServices', 'StreamingServices']
    
    available_cols = [col for col in usage_cols if col in df.columns]
    if not available_cols:
        print("Service usage analysis requires usage-related columns")
        return usage_insights

    ts_mapping = {
        0: 'No Services', 1: 'Very Low (1)', 2: 'Low (2)', 3: 'Medium (3)',
        4: 'Medium-High (4)', 5: 'High (5)', 6: 'Very High (6)'
    }
        
    for col in available_cols:
        if df[col].dtype not in [np.int64, np.float64]:
            continue
            
        try:
            unique_values = df[col].nunique()

            if unique_values == 1:
                continue  # Skip single-value columns

            elif unique_values == 2:
                # Binary variables
                distinct_values = sorted(df[col].unique())
                df[f'{col}_Group'] = df[col].map({
                    distinct_values[0]: f'Low ({distinct_values[0]})',
                    distinct_values[1]: f'High ({distinct_values[1]})'
                })
            
            elif col == 'TotalServices':
                df[f'{col}_Group'] = df[col].map(lambda x: ts_mapping.get(x, f'Premium ({x})'))
            
            elif unique_values <= 10:
                distinct_values = sorted(df[col].unique())
                labels = ['Very Low', 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High', 'Very High']
                value_mapping = {val: labels[i % len(labels)] for i, val in enumerate(distinct_values)}
                df[f'{col}_Group'] = df[col].map(value_mapping)
            
            else:
                try:
                    df[f'{col}_Group'] = pd.qcut(
                        df[col], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                        duplicates='drop'
                    )
                except ValueError:
                    min_val, max_val = df[col].min(), df[col].max()
                    bin_edges = np.linspace(min_val, max_val, 5)  # 5 edges for 4 bins
                    df[f'{col}_Group'] = pd.cut(
                        df[col], bins=bin_edges, 
                        labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                        include_lowest=True
                    )

            group_col = f'{col}_Group'
            usage_churn = df.groupby(group_col)['Churn'].agg(['mean', 'count']).reset_index()
            usage_churn.columns = ['Usage Level', 'Churn Rate', 'Customer Count']
            usage_churn['Churn Rate'] = usage_churn['Churn Rate'] * 100
            usage_churn = usage_churn.sort_values('Churn Rate', ascending=False)
            usage_insights[col] = usage_churn.to_dict('records')

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Usage Level', y='Churn Rate', data=usage_churn)
            plt.title(f'Churn Rate by {col} Level')
            plt.xlabel(f'{col} Level')
            plt.ylabel('Churn Rate (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{col}_churn_analysis.png'.replace(' ', '_')))
            plt.close()
            
            print(f"Completed analysis for {col}")
            
        except Exception as e:
            print(f"Error analyzing {col}: {str(e)}")
            continue

    if 'TotalServices' in df.columns and 'MonthlyCharges' in df.columns:
        df['RevenuePerService'] = df['MonthlyCharges'] / df['TotalServices'].replace(0, 1)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TotalServices', y='MonthlyCharges', hue='Churn', data=df)
        plt.title('Services vs. Monthly Charges by Churn Status')
        plt.savefig(os.path.join(output_dir, 'services_vs_charges.png'))
        plt.close()

        service_efficiency = df.groupby('TotalServices')[['RevenuePerService', 'Churn']].agg({
            'RevenuePerService': 'mean',
            'Churn': ['mean', 'count']
        }).reset_index()
        
        service_efficiency.columns = ['TotalServices', 'Avg Revenue Per Service', 'Churn Rate', 'Customer Count']
        service_efficiency['Churn Rate'] = service_efficiency['Churn Rate'] * 100
        service_efficiency = service_efficiency[service_efficiency['Customer Count'] >= 30]

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Total Services')
        ax1.set_ylabel('Avg Revenue Per Service ($)', color='tab:blue')
        ax1.plot(service_efficiency['TotalServices'], service_efficiency['Avg Revenue Per Service'], 
                color='tab:blue', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Churn Rate (%)', color='tab:red')
        ax2.plot(service_efficiency['TotalServices'], service_efficiency['Churn Rate'], 
                color='tab:red', marker='s')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title('Service Efficiency vs Churn Rate')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, 'service_efficiency_analysis.png'))
        plt.close()

        usage_insights['service_efficiency'] = service_efficiency.to_dict('records')
    
    return usage_insights

def customer_journey_analysis(df, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    journey_insights = {}

    if 'tenure' in df.columns:
        tenure_bins = [0, 12, 24, 36, 48, 60, float('inf')]
        tenure_labels = ['0-12 months', '13-24 months', '25-36 months', 
                        '37-48 months', '49-60 months', '60+ months']
        
        df['TenureGroup'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels)

        tenure_churn = df.groupby('TenureGroup')['Churn'].agg(['mean', 'count']).reset_index()
        tenure_churn.columns = ['Tenure Group', 'Churn Rate', 'Customer Count']
        tenure_churn['Churn Rate'] = tenure_churn['Churn Rate'] * 100

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Tenure Group', y='Churn Rate', data=tenure_churn)
        plt.title('Churn Rate by Tenure Group')
        plt.xlabel('Tenure')
        plt.ylabel('Churn Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tenure_churn_analysis.png'))
        plt.close()

        journey_insights['tenure_churn'] = tenure_churn.to_dict('records')

    if 'Number of Referrals' in df.columns:
        df['Referral_Group'] = pd.cut(
            df['Number of Referrals'], 
            bins=[-1, 0, 1, 2, 3, float('inf')],
            labels=['0', '1', '2', '3', '4+']
        )

        referral_churn = df.groupby('Referral_Group')['Churn'].agg(['mean', 'count']).reset_index()
        referral_churn.columns = ['Referrals', 'Churn Rate', 'Customer Count']
        referral_churn['Churn Rate'] = referral_churn['Churn Rate'] * 100

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Referrals', y='Churn Rate', data=referral_churn)
        plt.title('Churn Rate by Number of Referrals')
        plt.xlabel('Number of Referrals')
        plt.ylabel('Churn Rate (%)')
        plt.savefig(os.path.join(output_dir, 'referral_churn_analysis.png'))
        plt.close()

        journey_insights['referral_churn'] = referral_churn.to_dict('records')

    if 'Contract' in df.columns:
        contract_churn = df.groupby('Contract')['Churn'].agg(['mean', 'count']).reset_index()
        contract_churn.columns = ['Contract Type', 'Churn Rate', 'Customer Count']

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Contract Type', y='Churn Rate', data=contract_churn)
        plt.title('Churn Rate by Contract Type')
        plt.ylabel('Churn Rate (%)')
        plt.savefig(os.path.join(output_dir, 'contract_churn_analysis.png'))
        plt.close()

        journey_insights['contract_churn'] = contract_churn.to_dict('records')
    
    return journey_insights

def overall_business_insights(df, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    business_insights = {}

    total_customers = len(df)
    churn_rate = df['Churn'].mean() * 100
    
    if 'MonthlyCharges' in df.columns:
        monthly_revenue = df['MonthlyCharges'].sum()
        avg_monthly_charge = df['MonthlyCharges'].mean()
        revenue_at_risk = df[df['Churn'] == 1]['MonthlyCharges'].sum() * 12
        business_insights['financial'] = {
            'total_customers': total_customers,
            'monthly_revenue': monthly_revenue,
            'avg_monthly_charge': avg_monthly_charge,
            'annual_revenue_at_risk': revenue_at_risk,
            'churn_rate': churn_rate
        }

    if 'risk_segment' in df.columns:
        segment_analysis = df.groupby('risk_segment').agg({
            'Churn': 'mean',
            'customerID': 'count' if 'customerID' in df.columns else 'size',
            'MonthlyCharges': 'sum' if 'MonthlyCharges' in df.columns else 'count'
        }).reset_index()
        
        segment_analysis.columns = ['Risk Segment', 'Churn Rate', 'Customer Count', 'Monthly Revenue']
        segment_analysis['Churn Rate'] = segment_analysis['Churn Rate'] * 100
        segment_analysis['Segment %'] = segment_analysis['Customer Count'] / total_customers * 100
        
        business_insights['segment_analysis'] = segment_analysis.to_dict('records')

    churn_drivers = pd.DataFrame({
        'Feature': [],
        'Category': [],
        'Segment': [],
        'Churn Rate': [],
        'Customer Count': [],
        'Overall Churn': [],
        'Relative Risk': []
    })

    for feature in ['Contract', 'InternetService', 'PaymentMethod', 'TenureGroup']:
        if feature in df.columns:
            feature_churn = df.groupby(feature).agg({
                'Churn': ['mean', 'count'],
            }).reset_index()
            
            feature_churn.columns = [feature, 'Churn Rate', 'Customer Count']
            feature_churn['Churn Rate'] = feature_churn['Churn Rate'] * 100
            feature_churn['Overall Churn'] = churn_rate
            feature_churn['Relative Risk'] = feature_churn['Churn Rate'] / churn_rate

            feature_churn = feature_churn.rename(columns={feature: 'Segment'})
            feature_churn['Feature'] = feature

            if feature == 'Contract' or feature == 'TenureGroup':
                feature_churn['Category'] = 'Relationship'
            elif feature == 'InternetService':
                feature_churn['Category'] = 'Service'
            elif feature == 'PaymentMethod':
                feature_churn['Category'] = 'Billing'

            churn_drivers = pd.concat([churn_drivers, feature_churn])

    churn_drivers = churn_drivers.sort_values('Relative Risk', ascending=False)
    churn_drivers.to_csv(os.path.join(output_dir, 'churn_risk_factors.csv'), index=False)

    plt.figure(figsize=(12, 8))
    top_drivers = churn_drivers.head(10)
    sns.barplot(x='Relative Risk', y='Segment', hue='Feature', data=top_drivers)
    plt.title('Top 10 Churn Risk Factors')
    plt.xlabel('Relative Risk (vs. Overall Churn Rate)')
    plt.axvline(x=1, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_churn_risk_factors.png'))
    plt.close()

    business_insights['top_risk_factors'] = top_drivers.to_dict('records')
    
    return business_insights

def calculate_targeted_retention_roi(df, segment_definition, intervention_cost_per_customer, estimated_retention_rate):
    target_df = df.copy()

    for key, value in segment_definition.items():
        if key in target_df.columns:
            target_df = target_df[target_df[key] == value]

    num_customers = len(target_df)
    total_cost = num_customers * intervention_cost_per_customer

    if 'MonthlyCharges' in target_df.columns:
        monthly_revenue = target_df['MonthlyCharges'].sum()
        if 'churn_probability' in target_df.columns:
            revenue_at_risk = target_df['MonthlyCharges'] * target_df['churn_probability']
            revenue_at_risk = revenue_at_risk.sum() * 12
        else:
            revenue_at_risk = monthly_revenue * target_df['Churn'].mean() * 12
        
        annual_revenue_saved = revenue_at_risk * estimated_retention_rate
        roi = (annual_revenue_saved - total_cost) / total_cost if total_cost > 0 else 0

        if 'CLV' in target_df.columns:
            avg_clv = target_df['CLV'].mean()
            clv_impact = avg_clv * num_customers * target_df['Churn'].mean() * estimated_retention_rate
            clv_roi = (clv_impact - total_cost) / total_cost if total_cost > 0 else 0
        else:
            avg_clv = monthly_revenue * 12 * 3  # Approximate 3-year CLV
            clv_impact = avg_clv * num_customers * target_df['Churn'].mean() * estimated_retention_rate
            clv_roi = (clv_impact - total_cost) / total_cost if total_cost > 0 else 0
    else:
        monthly_revenue = 0
        annual_revenue_saved = 0
        revenue_at_risk = 0
        roi = -1.0
        avg_clv = 0
        clv_impact = 0
        clv_roi = -1.0
    
    return {
        'segment_definition': segment_definition,
        'customers_targeted': num_customers,
        'segment_churn_rate': target_df['Churn'].mean() * 100,
        'intervention_cost': total_cost,
        'monthly_revenue': monthly_revenue,
        'annual_revenue_at_risk': revenue_at_risk,
        'potential_annual_revenue_saved': annual_revenue_saved,
        'estimated_roi': roi,
        'estimated_roi_percent': roi * 100,
        'avg_customer_clv': avg_clv,
        'potential_clv_impact': clv_impact,
        'clv_roi': clv_roi,
        'clv_roi_percent': clv_roi * 100
    }