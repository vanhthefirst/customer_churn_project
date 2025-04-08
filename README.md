# Customer Churn Prediction & Retention Strategy Analysis

A comprehensive data science solution to address customer churn in business services, combining predictive modeling with actionable business strategies.

## Project Overview

This project develops a machine learning model to predict customer churn and provides data-driven recommendations for customer retention. The solution includes:

1. A machine learning model for churn prediction
2. Automated feature importance analysis to identify key churn factors
3. Customer segmentation by risk level
4. Revenue impact analysis
5. Targeted retention strategies with ROI calculations
6. An interactive dashboard for visualization and monitoring

## Project Structure

```
customer_churn_project/
├── data/
│   ├── telco_churn_engineered.csv
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   └── business_analysis.py
├── app/
│   ├── app.py
│   └── pages/
│       ├── __init__.py
│       ├── overview.py
│       ├── customer_prediction.py
│       ├── segment_analysis.py
|       ├── retention_strategies.py
│       └── ab_testing.py
├── models/
│   ├── churn_model.pkl
│   └── X_train_columns.csv
├── requirements.txt
├── run.py
├── setup.py
└── README.md
```

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model

```
python run.py --train
```

This will:
- Load and preprocess the customer data
- Engineer relevant features
- Train a machine learning model
- Save the model and necessary artifacts

### 2. Launch the Dashboard

```
python run.py --dashboard
```

This will start the Streamlit dashboard for interactive analysis.

## Dashboard Features

The interactive dashboard includes:

- **Overview**: Key metrics and visualizations of churn distribution
- **Customer Prediction**: Predict churn risk for individual customers
- **Segment Analysis**: Analyze customer segments and their characteristics
- **Retention Strategies**: Evaluate and compare retention interventions
- **A/B Testing for Retention Strategies**: Test the impact of retention interventions on reducing the churn rate

## Model Performance

The model achieves:
```
AUC (Area Under ROC Curve): 0.8267  Accuracy: 0.7935  Precision: 0.6397  Recall: 0.5080  F1 Score: 0.5663
```

## Key Findings

1. Month-to-month contracts are the strongest predictors of churn
2. Customers with Fiber optic internet without security services show high churn rates
3. Customer tenure is strongly negatively correlated with churn
4. Electronic check payment method correlates with higher churn
5. New customers with high monthly charges are especially vulnerable

## Recommended Strategies

Based on ROI analysis, the most effective retention strategies are:

1. Contract upgrade incentives for high-risk customers
2. Service bundle upgrades for Fiber customers without protection
3. Loyalty rewards program for medium-risk customers
4. Targeted support for high-value customers

## License
Dataset's copyright from IBM's kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data