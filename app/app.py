"""
Main Streamlit application for Customer Churn Prediction and Analysis.
"""
import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import preprocess_data
from src.feature_engineering import engineer_features
from src.model_trainer import load_model
from src.business_analysis import add_churn_probability, overall_business_insights
from pages.overview import show_overview
from pages.customer_prediction import show_customer_prediction
from pages.segment_analysis import show_segment_analysis
from pages.retention_strategies import show_retention_strategies
from pages.ab_testing import show_ab_testing

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def file_exists(filepath):
    return os.path.isfile(filepath)

def find_data_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

    possible_data_paths = [
        os.path.join(project_root, 'data', 'processed', 'telco_churn_primary.csv'),
        os.path.join(project_root, 'data', 'processed', 'telco_churn_engineered.csv'),
        os.path.join(current_dir, '..', 'data', 'processed', 'telco_churn_primary.csv'),
        os.path.join(current_dir, '..', 'data', 'processed', 'telco_churn_engineered.csv'),
        os.path.join(project_root, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        os.path.join(current_dir, '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
    ]

    data_path = next((path for path in possible_data_paths if file_exists(path)), None)
    return data_path, possible_data_paths

def find_model_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    model_dir = os.path.join(project_root, 'models')
    model_path = os.path.join(model_dir, 'churn_model.pkl')
    columns_path = os.path.join(model_dir, 'X_train_columns.csv')
    
    return model_path, columns_path

def get_revenue_metrics(df):
    insights = overall_business_insights(df)

    if 'financial' in insights:
        fin = insights['financial']
        metrics = {
            'total_customers': fin['total_customers'],
            'monthly_revenue': fin['monthly_revenue'],
            'churn_rate': fin['churn_rate'],
            'revenue_at_risk': fin['annual_revenue_at_risk'],
            'revenue_at_risk_percentage': (fin['annual_revenue_at_risk'] / (fin['monthly_revenue'] * 12) * 100) if fin['monthly_revenue'] > 0 else 0
        }
    else:
        total_customers = len(df)
        monthly_revenue = df['MonthlyCharges'].sum() if 'MonthlyCharges' in df.columns else 0
        churn_rate = df['Churn'].mean() * 100
        revenue_at_risk = df[df['Churn'] == 1]['MonthlyCharges'].sum() * 12 if 'MonthlyCharges' in df.columns else 0
        revenue_at_risk_percentage = (revenue_at_risk / (monthly_revenue * 12) * 100) if monthly_revenue > 0 else 0
        
        metrics = {
            'total_customers': total_customers,
            'monthly_revenue': monthly_revenue,
            'churn_rate': churn_rate,
            'revenue_at_risk': revenue_at_risk,
            'revenue_at_risk_percentage': revenue_at_risk_percentage
        }
    
    return metrics

@st.cache_data
def load_and_prepare_data():
    data_path, _ = find_data_file()
    model_path, columns_path = find_model_files()

    if not data_path or not file_exists(data_path):
        return None, None, None, None, "data_missing"
    
    if not file_exists(model_path) or not file_exists(columns_path):
        return None, None, None, None, "model_missing"
    
    try:
        is_integrated_dataset = 'telco_churn_primary.csv' in data_path or 'telco_churn_engineered.csv' in data_path
        
        if is_integrated_dataset:
            df = pd.read_csv(data_path)
            if df['Churn'].dtype == 'object':
                df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        else:
            df = preprocess_data(data_path)
            df = engineer_features(df)

        model = load_model(model_path)
        X_train_columns = pd.read_csv(columns_path).columns.tolist()
        df_with_proba = add_churn_probability(df, model, X_train_columns)
        metrics = get_revenue_metrics(df_with_proba)
        
        return df_with_proba, model, metrics, X_train_columns, "success"
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return None, None, None, None, (str(e), error_trace)

def main():
    st.title("Customer Churn Prediction & Retention Analysis")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Overview", "Customer Prediction", "Segment Analysis", "Retention Strategies", "A/B Testing for Retention Strategies"]
    )

    with st.spinner("Loading data and model..."):
        data_path, possible_paths = find_data_file()
        model_path, columns_path = find_model_files()
        
        df, model, metrics, X_train_columns, status = load_and_prepare_data()

    if status == "data_missing":
        st.error("📋 Could not find the customer data CSV file. Please check the following paths:")
        for path in possible_paths:
            st.error(f"- {path}")
        st.error("Make sure the data file exists in one of these locations.")
        return
        
    elif status == "model_missing":
        st.error("❗ Model files not found. You need to train the model first.")
        st.info("📝 Run the following command from the project root directory to train the model:")
        st.code("python run.py --train")
        st.info("Once training is complete, refresh this page.")
        return
        
    elif isinstance(status, tuple):
        error_msg, error_trace = status
        st.error(f"Error loading data: {error_msg}")
        with st.expander("See detailed error trace"):
            st.code(error_trace)
        st.error("Please make sure the model has been trained correctly and the data is valid.")
        return

    if df is None:
        st.error("Could not load data and model. Please check the error messages above.")
        return

    if page == "Overview":
        show_overview(df, metrics)
    elif page == "Customer Prediction":
        show_customer_prediction(model, X_train_columns)
    elif page == "Segment Analysis":
        show_segment_analysis(df)
    elif page == "Retention Strategies":
        show_retention_strategies(df)
    elif page == "A/B Testing for Retention Strategies":
        show_ab_testing(df, model, X_train_columns)

if __name__ == "__main__":
    main()