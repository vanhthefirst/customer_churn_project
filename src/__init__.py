"""
Customer Churn Prediction & Retention Strategy Analysis
A data-driven approach to customer retention

This package contains modules for data processing, feature engineering,
model training, and business analysis for customer churn prediction.
"""

from .data_processor import preprocess_data, load_data, handle_missing_values
from .feature_engineering import engineer_features, create_service_features, create_financial_features
from .model_trainer import (
    prepare_modeling_data,
    train_random_forest,
    train_logistic_regression,
    split_data,
    save_model,
    load_model
)
from .business_analysis import (
    add_churn_probability, 
    get_segment_insights,
    calculate_roi,
    get_revenue_metrics
)

__all__ = [
    'preprocess_data',
    'load_data',
    'handle_missing_values',
    'engineer_features',
    'create_service_features', 
    'create_financial_features',
    'prepare_modeling_data',
    'train_random_forest',
    'train_logistic_regression',
    'split_data',
    'save_model',
    'load_model',
    'add_churn_probability',
    'get_segment_insights',
    'calculate_roi',
    'get_revenue_metrics'
]