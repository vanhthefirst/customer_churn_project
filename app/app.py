"""
Main Streamlit application for Customer Churn Prediction and Analysis.
"""
import streamlit as st
import sys
import os
import pandas as pd
import copy

# Add the parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import local modules
from src.data_processor import preprocess_data
from src.feature_engineering import engineer_features
from src.model_trainer import load_model
from src.business_analysis import add_churn_probability, get_revenue_metrics

# Import page modules
from pages.overview import show_overview
from pages.customer_prediction import show_customer_prediction
from pages.segment_analysis import show_segment_analysis
from pages.retention_strategies import show_retention_strategies
from pages.ab_testing import show_ab_testing

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to check if file exists
def file_exists(filepath):
    return os.path.isfile(filepath)

# Find data file without displaying messages
def find_data_file():
    # Get the absolute path to the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Data paths - try different relative paths
    possible_data_paths = [
        os.path.join(project_root, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        os.path.join(project_root, 'data', 'WA_FnUseC_TelcoCustomerChurn.csv'),
        os.path.join(current_dir, '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        os.path.join(current_dir, '..', 'data', 'WA_FnUseC_TelcoCustomerChurn.csv')
    ]
    
    # Find the first valid data path
    data_path = next((path for path in possible_data_paths if file_exists(path)), None)
    return data_path, possible_data_paths

# Find model files without displaying messages
def find_model_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    model_dir = os.path.join(project_root, 'models')
    model_path = os.path.join(model_dir, 'churn_model.pkl')
    columns_path = os.path.join(model_dir, 'X_train_columns.csv')
    
    return model_path, columns_path

# Cache data loading to improve performance - allow mutations
@st.cache(allow_output_mutation=True)
def load_and_prepare_data():
    """Load, preprocess data and add predictions without Streamlit UI calls"""
    # Find data and model files
    data_path, _ = find_data_file()
    model_path, columns_path = find_model_files()
    
    # Check if files exist
    if not data_path or not file_exists(data_path):
        return None, None, None, None, "data_missing"
    
    if not file_exists(model_path) or not file_exists(columns_path):
        return None, None, None, None, "model_missing"
    
    try:
        # Load and preprocess data (no UI calls here)
        df = preprocess_data(data_path)
        
        # Engineer features
        df = engineer_features(df)
        
        # Load model and column information
        model = load_model(model_path)
        X_train_columns = pd.read_csv(columns_path).columns.tolist()
        
        # Add predictions
        df_with_proba = add_churn_probability(df, model, X_train_columns)
        
        # Get metrics
        metrics = get_revenue_metrics(df_with_proba)
        
        return df_with_proba, model, metrics, X_train_columns, "success"
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return None, None, None, None, (str(e), error_trace)

# Main application
def main():
    """Main application function"""
    # App title and description
    st.title("Customer Churn Prediction & Retention Analysis")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Overview", "Customer Prediction", "Segment Analysis", "Retention Strategies", "A/B Testing for Retention Strategies"]
    )
    
    # Load data (cached for performance)
    with st.spinner("Loading data and model..."):
        data_path, possible_paths = find_data_file()
        model_path, columns_path = find_model_files()
        
        df, model, metrics, X_train_columns, status = load_and_prepare_data()
    
    # Handle different loading statuses
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
        
    elif isinstance(status, tuple):  # Exception occurred
        error_msg, error_trace = status
        st.error(f"Error loading data: {error_msg}")
        with st.expander("See detailed error trace"):
            st.code(error_trace)
        st.error("Please make sure the model has been trained correctly and the data is valid.")
        return
    
    # If data loading failed
    if df is None:
        st.error("Could not load data and model. Please check the error messages above.")
        return
    
    # Show selected page
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