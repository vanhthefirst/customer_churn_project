"""
Customer Churn Prediction & Retention Strategy
Run script for model training and dashboard startup
"""
import os
import argparse
import pandas as pd
import subprocess
import sys
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

def train_model():
    from src.data_processor import preprocess_data
    from src.feature_engineering import engineer_features
    from src.model_trainer import (
        prepare_modeling_data, split_data, save_model, save_column_info, 
        train_logistic_regression, train_random_forest, train_xgboost,
    )
    
    print("Loading and preprocessing data...")
    df = preprocess_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print("Engineering features...")
    df = engineer_features(df)
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/telco_churn_engineered.csv', index=False)
    print("Engineered data saved to data/processed/telco_churn_engineered.csv")
    
    print("Preparing data for modeling...")
    X, y = prepare_modeling_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training Logistic Regression model...")
    lr_model, lr_auc = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    print("Training Random Forest model...")
    rf_model, rf_auc = train_random_forest(X_train, y_train, X_test, y_test)

    print("Training XGBoost model...")
    xgb_model, xgb_auc = train_xgboost(X_train, y_train, X_test, y_test)

    models = {
        'Logistic Regression': (lr_model, lr_auc),
        'Random Forest': (rf_model, rf_auc),
        'XGBoost': (xgb_model, xgb_auc)
    }

    best_model_name = max(models, key=lambda k: models[k][1])
    final_model, best_auc = models[best_model_name]

    comparison = " vs ".join([f"{name}: {auc:.4f}" for name, (_, auc) in models.items()])

    print(f"{best_model_name} model selected - AUC: {comparison}")

    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel Performance Summary:")
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    os.makedirs('models', exist_ok=True)
    save_model(final_model, 'models/churn_model.pkl')
    save_column_info(X_train, 'models/X_train_columns.csv')
    
    print("Model training complete!")
    return True

def run_business_analysis():
    from src.business_analysis import (
        geographic_churn_analysis, service_usage_analysis,
        customer_journey_analysis, overall_business_insights
    )
    
    print("Running enhanced business analysis...")
    data_path = 'data/processed/telco_churn_primary.csv'

    if not os.path.exists(data_path):
        print(f"Error: Enhanced dataset not found at {data_path}")
        print("Please run '--integrate-data' first to create the enhanced dataset")
        return False

    os.makedirs('results', exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded enhanced dataset with shape: {df.shape}")

    if df['Churn'].dtype == 'object':
        print("Converting Churn column to numeric...")
        if df['Churn'].iloc[0] in ['Yes', 'No']:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        else:
            try:
                df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce').fillna(0).astype(int)
            except:
                print("Warning: Could not convert Churn column properly")

    print("\n" + "="*50)
    print("GEOGRAPHIC ANALYSIS")
    print("="*50)
    try:
        geo_insights = geographic_churn_analysis(df)
        
        if geo_insights.get('high_churn_cities'):
            print("\nTop 3 Cities with Highest Churn:")
            for i, city in enumerate(geo_insights['high_churn_cities'][:3]):
                print(f"  {i+1}. {city['City']}: {city['Churn Rate']:.2f}% (n={city['Customer Count']})")
    except Exception as e:
        print(f"Error in geographic analysis: {str(e)}")
        geo_insights = {}
    
    print("\n" + "="*50)
    print("SERVICE USAGE ANALYSIS")
    print("="*50)
    usage_insights = service_usage_analysis(df)
    
    if usage_insights.get('TotalServices'):
        print("\nChurn by Service Level:")
        for level in usage_insights['TotalServices']:
            print(f"  {level['Usage Level']} usage: {level['Churn Rate']:.2f}% churn (n={level['Customer Count']})")
    
    print("\n" + "="*50)
    print("CUSTOMER JOURNEY ANALYSIS")
    print("="*50)
    journey_insights = customer_journey_analysis(df)
    
    if journey_insights.get('tenure_churn'):
        print("\nChurn by Tenure:")
        for group in journey_insights['tenure_churn']:
            print(f"  {group['Tenure Group']}: {group['Churn Rate']:.2f}% churn (n={group['Customer Count']})")
    
    if journey_insights.get('contract_churn'):
        print("\nChurn by Contract Type:")
        for contract in journey_insights['contract_churn']:
            print(f"  {contract['Contract Type']}: {contract['Churn Rate']:.2f}% churn")
    
    print("\n" + "="*50)
    print("OVERALL BUSINESS INSIGHTS")
    print("="*50)
    business_insights = overall_business_insights(df)
    
    if business_insights.get('financial'):
        fin = business_insights['financial']
        print(f"\nTotal Customers: {fin['total_customers']:,}")
        print(f"Monthly Revenue: ${fin['monthly_revenue']:,.2f}")
        print(f"Average Monthly Charge: ${fin['avg_monthly_charge']:.2f}")
        print(f"Overall Churn Rate: {fin['churn_rate']:.2f}%")
        print(f"Annual Revenue at Risk: ${fin['annual_revenue_at_risk']:,.2f}")
    
    if business_insights.get('top_risk_factors'):
        print("\nTop 3 Churn Risk Factors:")
        for i, factor in enumerate(business_insights['top_risk_factors'][:3]):
            print(f"  {i+1}. {factor['Feature']}: {factor['Segment']} ({factor['Relative Risk']:.2f}x risk)")
    
    print("\nBusiness analysis completed successfully!")
    print("Detailed results saved to the 'results' directory")
    return True

def launch_dashboard():
    try:
        print("Starting Streamlit dashboard...")
        print("Press Ctrl+C to stop the dashboard")

        process = subprocess.Popen(["streamlit", "run", "app/app.py", "--server.port", "2408"])
        process.wait()
        return True
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it with 'pip install streamlit'")
        return False
    except KeyboardInterrupt:
        print("\nStreamlit dashboard stopped by user.")
        
        if 'process' in locals() and process:
            print("Shutting down streamlit process...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Forcing process to shut down...")
                process.kill()
            except Exception as e:
                print(f"Error while shutting down: {e}")
        
        return True
    except Exception as e:
        print(f"Error running Streamlit dashboard: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Customer Churn Prediction & Retention Strategy')
    parser.add_argument('--integrate-data', action='store_true', help='Integrate new datasets')
    parser.add_argument('--analyze', action='store_true', help='Run enhanced business analysis')
    parser.add_argument('--train', action='store_true', help='Train and save model')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    
    args = parser.parse_args()
    
    if not (args.integrate_data or args.analyze or args.train or args.dashboard):
        parser.print_help()
    
    if args.integrate_data:
        from src.data_integrator import load_datasets, preprocess_datasets, create_primary_dataset, enhance_with_geographic_data, engineer_additional_features
        print("Integrating datasets...")
        original_df, customer_churn_df, zipcode_population_df, data_dict_df = load_datasets('data')
        original, customer_churn, zipcode_pop = preprocess_datasets(original_df, customer_churn_df, zipcode_population_df)
        enhanced_df = create_primary_dataset(original, customer_churn)
        enhanced_df = enhance_with_geographic_data(enhanced_df, zipcode_pop)
        enhanced_df = engineer_additional_features(enhanced_df)
        os.makedirs('data/processed', exist_ok=True)
        enhanced_df.to_csv('data/processed/telco_churn_primary.csv', index=False)
        print("Enhanced dataset saved to data/processed/telco_churn_primary.csv")

    if args.analyze:
        analysis_success = run_business_analysis()
        if not analysis_success:
            sys.exit(1)

    if args.train:
        train_success = train_model()
        if not train_success:
            sys.exit(1)
    
    if args.dashboard:
        dashboard_success = launch_dashboard()
        if not dashboard_success:
            sys.exit(1)