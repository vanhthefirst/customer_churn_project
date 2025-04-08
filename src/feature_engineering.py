"""
Feature engineering module for customer churn prediction.
"""
import pandas as pd
import numpy as np

def create_service_features(df):
    df['TotalServices'] = df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
        lambda row: sum(1 for item in row if item not in ['No', 'No internet service']), axis=1
    )

    df['HasTechSupport'] = df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['HasOnlineSecurity'] = df['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['StreamingServices'] = ((df['StreamingTV'] == 'Yes') & 
                              (df['StreamingMovies'] == 'Yes')).astype(int)
    
    return df

def create_financial_features(df):
    safe_tenure = df['tenure'].replace(0, 1)

    df['CLV'] = df['tenure'] * df['MonthlyCharges']
    df['AvgMonthlySpend'] = df['TotalCharges'] / safe_tenure
    df['CLV'] = df['CLV'].fillna(0)
    df['AvgMonthlySpend'] = df['AvgMonthlySpend'].fillna(0)
    
    return df

def create_categorical_features(df):

    df['TenureMonths'] = df['tenure']
    tenure_bins = [0, 12, 24, 36, 48, 60, np.inf]
    df['TenureGroup'] = pd.cut(df['tenure'], bins=tenure_bins, labels=False)

    df['ContractRiskFactor'] = df['Contract'].map({
        'Month-to-month': 2, 
        'One year': 1, 
        'Two year': 0
    })
    
    return df

def prepare_modeling_data(df):
    categorical_cols_to_encode = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                             'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                             'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'TenureGroup']
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    if 'customerID' in X.columns:
        X = X.drop('customerID', axis=1)

    X_encoded = pd.get_dummies(X, columns=categorical_cols_to_encode, drop_first=True)
    
    print(f"Final feature set shape: {X_encoded.shape}")
    return X_encoded, y

def engineer_features(df):
    df = create_service_features(df)
    df = create_financial_features(df)
    df = create_categorical_features(df)

    if df.isnull().any().any():
        print("Handling NaN values introduced during feature engineering...")

        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = df[col].fillna(0)

        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    return df