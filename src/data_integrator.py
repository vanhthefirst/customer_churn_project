import pandas as pd
import numpy as np
import os

def load_datasets(data_dir='data'):
    original_df = pd.read_csv(os.path.join(data_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))

    customer_churn_df = pd.read_csv(os.path.join(data_dir, 'telecom_customer_churn.csv'))
    zipcode_population_df = pd.read_csv(os.path.join(data_dir, 'telecom_zipcode_population.csv'))
    data_dict_df = pd.read_csv(os.path.join(data_dir, 'telecom_data_dictionary.csv'), encoding='cp1252')
    
    return original_df, customer_churn_df, zipcode_population_df, data_dict_df

def preprocess_datasets(original_df, customer_churn_df, zipcode_population_df):
    original = original_df.copy()
    customer_churn = customer_churn_df.copy()
    zipcode_pop = zipcode_population_df.copy()

    customer_churn.rename(columns={'Customer ID': 'customerID'}, inplace=True)
    customer_churn.rename(columns={
        'Gender': 'gender',
        'Married': 'marital_status',
        'Number of Dependents': 'dependents_count',
        'Phone Service': 'phone_service',
        'Internet Service': 'internet_service',
        'Contract': 'contract_type',
        'Paperless Billing': 'paperless_billing',
        'Payment Method': 'payment_method',
        'Monthly Charge': 'monthly_charges',
        'Total Charges': 'total_charges',
        'Tenure in Months': 'tenure_months'
    }, inplace=True)

    for col in ['phone_service', 'paperless_billing']:
        if col in customer_churn.columns:
            customer_churn[col] = customer_churn[col].map({'Yes': 1, 'No': 0})
    
    return original, customer_churn, zipcode_pop

def create_primary_dataset(original_df, customer_churn_df):
    common_customers = set(original_df['customerID']).intersection(set(customer_churn_df['customerID']))
    print(f"Number of common customers: {len(common_customers)}")
    
    # If the datasets have common customers, perform inner join
    if len(common_customers) > 0:
        common_cols = set(original_df.columns).intersection(set(customer_churn_df.columns))
        common_cols.remove('customerID')  # Keep customerID for joining

        churn_cols_to_use = [col for col in customer_churn_df.columns if col not in common_cols or col == 'customerID']
        
        # Perform inner join - only keep customers that exist in both datasets
        merged_df = original_df.merge(
            customer_churn_df[churn_cols_to_use],
            on='customerID',
            how='inner'
        )
        
        print(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
    else:
        print("No common customers found between datasets. Using original dataset.")
        return original_df

def enhance_with_geographic_data(merged_df, zipcode_population_df):
    zipcode_population_df.rename(columns={'Zip Code': 'zip_code'}, inplace=True)

    if 'Zip Code' in merged_df.columns:
        merged_df.rename(columns={'Zip Code': 'zip_code'}, inplace=True)

        geo_enhanced_df = merged_df.merge(
            zipcode_population_df,
            on='zip_code',
            how='left'
        )

        if 'Latitude' in geo_enhanced_df.columns and 'Longitude' in geo_enhanced_df.columns:
            geo_enhanced_df['population_density'] = geo_enhanced_df['Population'] / 10
        
        return geo_enhanced_df
    else:
        print("Zip Code not found in primary dataset. Skipping geographic enhancement.")
        return merged_df

def create_churn_details_dataset(customer_churn_df):
    if 'customerID' in customer_churn_df.columns and 'Churn Reason' in customer_churn_df.columns:
        churn_details_cols = ['customerID', 'Customer Status', 'Churn Category', 'Churn Reason']
        churn_details_df = customer_churn_df[churn_details_cols].copy()
        churn_details_df = churn_details_df[churn_details_df['Customer Status'] == 'Churned']
        
        return churn_details_df
    else:
        print("Required columns for churn details not found.")
        return None

def create_usage_metrics_dataset(customer_churn_df):
    if 'customerID' in customer_churn_df.columns:
        usage_cols = ['customerID', 'Avg Monthly Long Distance Charges', 
                      'Avg Monthly GB Download', 'Total Extra Data Charges',
                      'Total Long Distance Charges']

        usage_cols = [col for col in usage_cols if col in customer_churn_df.columns]
        
        if len(usage_cols) > 1:  # Need at least customerID and one metric
            usage_df = customer_churn_df[usage_cols].copy()
            return usage_df
        else:
            print("Insufficient usage metrics found.")
            return None
    else:
        print("Required column 'customerID' not found.")
        return None

def engineer_additional_features(enhanced_df):
    df = enhanced_df.copy()
    
    # 1. Service-related features
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']

    existing_service_cols = [col for col in service_cols if col in df.columns]
    
    if existing_service_cols:
        df['TotalServices'] = df[existing_service_cols].apply(
            lambda row: sum(1 for item in row if item not in ['No', 'No internet service']), axis=1
        )

        if 'TechSupport' in df.columns:
            df['HasTechSupport'] = df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        if 'OnlineSecurity' in df.columns:
            df['HasOnlineSecurity'] = df['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        if 'StreamingTV' in df.columns and 'StreamingMovies' in df.columns:
            df['StreamingServices'] = ((df['StreamingTV'] == 'Yes') & 
                                      (df['StreamingMovies'] == 'Yes')).astype(int)
    
    # 2. Financial features
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        safe_tenure = df['tenure'].replace(0, 1) # Avoid division by zero

        df['CLV'] = df['tenure'] * df['MonthlyCharges']

        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(0)
            df['AvgMonthlySpend'] = df['TotalCharges'] / safe_tenure
    
    # 3. Additional geographic features
    if 'Population' in df.columns and 'zip_code' in df.columns:
        zip_customer_counts = df['zip_code'].value_counts().to_dict()
        df['CustomersInZip'] = df['zip_code'].map(zip_customer_counts)
        df['ZipPenetrationRate'] = df['CustomersInZip'] / df['Population']
    
    # 4. Customer engagement features
    if 'Number of Referrals' in df.columns:
        df['HasReferrals'] = (df['Number of Referrals'] > 0).astype(int)
    
    # 5. Tenure-related features
    if 'tenure' in df.columns:
        df['TenureMonths'] = df['tenure']

        tenure_bins = [0, 12, 24, 36, 48, 60, float('inf')]
        df['TenureGroup'] = pd.cut(df['tenure'], bins=tenure_bins, labels=False)
    
    # 6. Contract risk factor
    if 'Contract' in df.columns:
        contract_col = 'Contract'
    elif 'contract_type' in df.columns:
        contract_col = 'contract_type'
    else:
        contract_col = None
        
    if contract_col:
        unique_contracts = df[contract_col].unique()

        if 'Month-to-month' in unique_contracts:
            df['ContractRiskFactor'] = df[contract_col].map({
                'Month-to-month': 2, 
                'One year': 1, 
                'Two year': 0
            })
        elif 'Monthly' in unique_contracts:
            df['ContractRiskFactor'] = df[contract_col].map({
                'Monthly': 2,
                'One Year': 1,
                'Two Year': 0
            })

    if df.isnull().any().any():
        print("Handling NaN values introduced during feature engineering...")

        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = df[col].fillna(0)

        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    return df

def save_datasets(primary_df, geo_df=None, churn_details_df=None, usage_df=None, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)

    primary_df.to_csv(os.path.join(output_dir, 'telco_churn_primary.csv'), index=False)
    print(f"Primary dataset saved with shape: {primary_df.shape}")

    if geo_df is not None:
        geo_df.to_csv(os.path.join(output_dir, 'telco_churn_geo_enhanced.csv'), index=False)
        print(f"Geo-enhanced dataset saved with shape: {geo_df.shape}")
    
    if churn_details_df is not None:
        churn_details_df.to_csv(os.path.join(output_dir, 'telco_churn_reasons.csv'), index=False)
        print(f"Churn details dataset saved with shape: {churn_details_df.shape}")
    
    if usage_df is not None:
        usage_df.to_csv(os.path.join(output_dir, 'telco_usage_metrics.csv'), index=False)
        print(f"Usage metrics dataset saved with shape: {usage_df.shape}")