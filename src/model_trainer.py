"""
Model training module for customer churn prediction.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def prepare_modeling_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn'] 

    if 'customerID' in X.columns:
        X = X.drop('customerID', axis=1)

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
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    lr_pred = lr_model.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_pred)
    print(f"Logistic Regression AUC: {lr_auc:.4f}")
    
    return lr_model, lr_auc

def train_random_forest(X_train, y_train, X_test, y_test, param_grid=None):
    if param_grid is None:
        param_grid = {
            'n_estimators': [100],
            'max_depth': [10],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    rf_pred_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)

    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)

    print(f"Random Forest Metrics:")
    print(f"  - AUC: {rf_auc:.4f}")
    print(f"  - Accuracy: {rf_accuracy:.4f}")
    print(f"  - Precision: {rf_precision:.4f}")
    print(f"  - Recall: {rf_recall:.4f}")
    print(f"  - F1-Score: {rf_f1:.4f}")

    return rf, rf_auc

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