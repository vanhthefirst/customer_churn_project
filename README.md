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
├── app/
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── ab_testing.py
│   │   ├── customer_prediction.py
│   │   ├── overview.py
│   │   ├── retention_strategies.py
│   │   └── segment_analysis.py
│   └── app.py
├── data/
│   ├── processed/
│   │   ├── telco_churn_engineered.csv
│   │   └── telco_churn_primary.csv
│   ├── marketing_campaign.csv
│   ├── telecom_customer_churn.csv
│   ├── telecom_data_dictionary.csv
│   ├── telecom_zipcode_population.csv
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.pkl
│   └── X_train_columns.csv
├── results/
│   ├── .png files
│   ├── churn_by_city.csv
│   └── churn_risk_factors.csv
├── src/
│   ├── __init__.py
│   ├── business_analysis.py
│   ├── data_integrator.py
│   ├── data_processor.py
│   ├── feature_engineering.py
│   └── model_trainer.py
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Integrate datasets

```
python run.py --integrate-data
```

This will integrate the dataset of the basic model with new datasets to train an enhanced model.

To view the data information, run the following command after calling `--integrate-data`:

```
python run.py --analyze
```

This will provide insights from the datasets, including Geography Analysis, Service Usage Analysis, Customer Journey Analysis, and Overall Business Insights.

### 2. Train the Model

```
python run.py --train
```

This will:
- Load and preprocess the customer data
- Engineer relevant features
- Train a machine learning model
- Save the model and necessary artifacts

### 3. Launch the Dashboard

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

Data collected from `WA_Fn-UseC_-Telco-Customer-Churn.csv`: 7043 customers and 21 features.
The basic model is trained by Random Forest Algorithm, achiving:
- AUC (Area Under ROC Curve): 0.8267
- Accuracy:                   0.7935
- Precision:                  0.6397
- Recall:                     0.5080
- F1-Score:                   0.5663

After integrating new datasets `telecom_customer_churn.csv`, `telecom_zipcode_population.csv`, and `telecom_data_dictionary.csv`, the model is trained from 7043 customers and 39 features.
The enhanced model use Logistic Regression and Random Forest Algorithm and choose the better result. Surprisingly, Logistic Regression achieves 0.8457 whereas Random Forest only reaches 0.8332 at AUC metrics.
Below is the best value obtains from training Logistic Regression:
- AUC (Area Under ROC Curve): 0.8457 
- Accuracy:                   0.7452
- Precision:                  0.5127
- Recall:                     0.8075
- F1-Score:                   0.6272


Notably, the model prioritizes recall over precision, meaning it's better at finding potential churners but may flag some loyal customers as risks.

For a customer churn model, this balance is often appropriate since missing a potential churner (false negative) is typically more costly than incorrectly flagging a loyal customer (false positive).

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
The integrated datasets' copyright was collected at: https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics