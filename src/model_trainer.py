"""
Model training module for customer churn prediction.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import pickle

def load_enhanced_data(data_path='data/processed/telco_churn_primary.csv'):
    print(f"Loading enhanced data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    if 'Churn' not in df.columns:
        if 'Customer Status' in df.columns:
            print("Converting 'Customer Status' to binary 'Churn' target")
            df['Churn'] = df['Customer Status'].map({'Churned': 1, 'Stayed': 0})
        else:
            raise ValueError("Target variable 'Churn' not found in dataset!")
    
    return df

def prepare_modeling_data(df):
    X = df.drop(['Churn', 'customerID'] if 'customerID' in df.columns else ['Churn'], axis=1)
    y = df['Churn']

    id_columns = [col for col in X.columns if col.lower().endswith('id')]
    X = X.drop(id_columns, axis=1, errors='ignore')

    if X.isnull().any().any():
        print("Warning: NaN values found in features. Imputing with appropriate values.")

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            X[col] = X[col].fillna(X[col].mean())

        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")

    cat_cols = X.select_dtypes(include=['object']).columns

    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    if X_encoded.isnull().any().any():
        print("Warning: NaN values still present after encoding. Filling with 0.")
        X_encoded = X_encoded.fillna(0)
    
    print(f"Final feature set shape: {X_encoded.shape}")
    return X_encoded, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train, X_test, y_test):
    lr_model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced',
        C=0.8  # Slightly increased regularisation
    )
    lr_model.fit(X_train, y_train)

    train_pred = lr_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    print(f"Training AUC: {train_auc:.4f}")
    
    test_pred = lr_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    print(f"Test AUC: {test_auc:.4f}")

    y_pred = (test_pred > 0.5).astype(int)
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return lr_model, test_auc

def train_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    train_pred = rf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    print(f"Training AUC: {train_auc:.4f}")
    
    test_pred = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    print(f"Test AUC: {test_auc:.4f}")

    y_pred = (test_pred > 0.5).astype(int)
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred))

    return rf, test_auc

def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    train_pred = xgb_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    print(f"Training AUC: {train_auc:.4f}")

    test_pred = xgb_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    print(f"Test AUC: {test_auc:.4f}")

    y_pred = (test_pred > 0.5).astype(int)
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred))

    return xgb_model, test_auc

def plot_feature_importance(model, X_train, n_features=15):
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(n_features))
    plt.title(f'Top {n_features} Features for Churn Prediction')
    plt.tight_layout()
    
    return plt, feature_importances

def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully to {file_path}")

def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def save_column_info(X_train, file_path):
    pd.DataFrame(columns=X_train.columns).to_csv(file_path, index=False)
    print(f"Column information saved to {file_path}")