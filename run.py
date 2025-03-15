"""
Customer Churn Prediction & Retention Strategy
Run script for model training and dashboard startup
"""
import os
import argparse
import pandas as pd
import pickle
import subprocess
import sys

def train_model():
    """Train and save the churn prediction model"""
    from src.data_processor import preprocess_data
    from src.feature_engineering import engineer_features
    from src.model_trainer import (
        prepare_modeling_data, split_data, 
        train_random_forest, save_model, save_column_info
    )
    
    print("Loading and preprocessing data...")
    df = preprocess_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print("Engineering features...")
    df = engineer_features(df)
    
    # Save engineered data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/telco_churn_engineered.csv', index=False)
    print("Engineered data saved to data/telco_churn_engineered.csv")
    
    print("Preparing data for modeling...")
    X, y = prepare_modeling_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Training Random Forest model...")
    model, _ = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and column information
    save_model(model, 'models/churn_model.pkl')
    save_column_info(X_train, 'models/X_train_columns.csv')
    
    print("Model training complete!")
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        print("Starting Streamlit dashboard...")
        subprocess.run(["streamlit", "run", "app/app.py"])
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it with 'pip install streamlit'")
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Customer Churn Prediction & Retention Strategy')
    parser.add_argument('--train', action='store_true', help='Train and save model')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    
    args = parser.parse_args()
    
    if not (args.train or args.dashboard):
        parser.print_help()
    
    if args.train:
        train_success = train_model()
        if not train_success:
            sys.exit(1)
    
    if args.dashboard:
        dashboard_success = launch_dashboard()
        if not dashboard_success:
            sys.exit(1)