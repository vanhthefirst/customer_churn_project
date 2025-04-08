"""
Data preprocessing module for customer churn prediction.
"""
import pandas as pd
import numpy as np
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df

def handle_missing_values(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[(df['TotalCharges'].isnull()) & (df['tenure'] == 0), 'TotalCharges'] = 0

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def encode_categorical_variables(df):
    binary_vars = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for var in binary_vars:
        df[var] = df[var].map({'Yes': 1, 'No': 0})
    
    return df

def preprocess_data(file_path):
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    print(f"Missing values after preprocessing:\n{df.isnull().sum()}")
    return df