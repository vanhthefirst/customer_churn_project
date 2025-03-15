"""
Customer prediction page for the Churn Prediction dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

def show_customer_prediction(model, X_train_columns):
    """Display customer prediction form and results"""
    st.header("Customer Churn Prediction")
    st.markdown("""
    Use this form to predict churn probability for individual customers. 
    Enter customer characteristics and click 'Predict' to see the results.
    """)
    
    # Create columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.radio("Gender", ["Male", "Female"])
        senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
        partner = st.radio("Has Partner", ["No", "Yes"])
        dependents = st.radio("Has Dependents", ["No", "Yes"])
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        contract = st.selectbox("Contract Type", 
                              ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])
        
    with col2:
        st.subheader("Services")
        phone_service = st.radio("Phone Service", ["No", "Yes"])
        
        if phone_service == "Yes":
            multiple_lines = st.radio("Multiple Lines", ["No", "Yes"])
        else:
            multiple_lines = "No phone service"
        
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        if internet_service != "No":
            online_security = st.radio("Online Security", ["No", "Yes"])
            online_backup = st.radio("Online Backup", ["No", "Yes"])
            device_protection = st.radio("Device Protection", ["No", "Yes"])
            tech_support = st.radio("Tech Support", ["No", "Yes"])
            streaming_tv = st.radio("Streaming TV", ["No", "Yes"])
            streaming_movies = st.radio("Streaming Movies", ["No", "Yes"])
        else:
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"
        
        st.subheader("Billing")
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5)
        total_charges = monthly_charges * tenure
        st.info(f"Total Charges: ${total_charges:.2f}")
    
    # Create predict button
    predict_button = st.button("Predict Churn Probability", type="primary")
    
    # Prediction section
    if predict_button:
        # Create input data dictionary
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': 1 if partner == "Yes" else 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == "Yes" else 0,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])
        
        # Feature engineering
        # Total services
        input_df['TotalServices'] = input_df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                               'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
            lambda row: sum(1 for item in row if item not in ['No', 'No internet service']), axis=1
        )
        
        # Service flags
        input_df['HasTechSupport'] = input_df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
        input_df['HasOnlineSecurity'] = input_df['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0)
        input_df['StreamingServices'] = ((input_df['StreamingTV'] == 'Yes') & 
                                    (input_df['StreamingMovies'] == 'Yes')).astype(int)
        
        # Financial metrics
        input_df['CLV'] = input_df['tenure'] * input_df['MonthlyCharges']
        input_df['AvgMonthlySpend'] = input_df['TotalCharges'] / input_df['tenure'].replace(0, 1)
        
        # Categorical features (using numeric TenureGroup)
        tenure_bins = [0, 12, 24, 36, 48, 60, np.inf]
        input_df['TenureGroup'] = pd.cut(input_df['tenure'], bins=tenure_bins, labels=False)
        
        input_df['ContractRiskFactor'] = input_df['Contract'].map({
            'Month-to-month': 2, 
            'One year': 1, 
            'Two year': 0
        })
        
        # Prepare for model prediction - one-hot encoding
        cat_cols = input_df.select_dtypes(include=['object']).columns
        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
        
        # Create a copy of the columns list to avoid mutation warnings
        columns_for_model = X_train_columns.copy()
        
        # Align with training columns
        model_input = pd.DataFrame(index=[0], columns=columns_for_model).fillna(0)
        
        # Copy values from input_encoded to model_input for columns that exist in both
        for col in input_encoded.columns:
            if col in model_input.columns:
                model_input[col] = input_encoded[col].values
        
        # Make prediction
        try:
            churn_probability = model.predict_proba(model_input)[0, 1]
            
            # Display prediction results
            st.header("Prediction Results")
            
            # Determine risk category
            if churn_probability < 0.25:
                risk_category = "Low Risk"
                color = "green"
            elif churn_probability < 0.5:
                risk_category = "Medium-Low Risk"
                color = "blue"
            elif churn_probability < 0.75:
                risk_category = "Medium-High Risk"
                color = "orange"
            else:
                risk_category = "High Risk"
                color = "red"
            
            # Create nice visualization for prediction
            col3, col4 = st.columns([1, 1])
            
            with col3:
                # Create gauge chart
                fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})
                
                # Gauge settings
                theta = np.linspace(0, 180, 100) * np.pi / 180
                r = [1] * 100
                
                # Plot background
                ax.bar(np.pi/2, 1, width=np.pi, bottom=0, color='lightgray', alpha=0.5)
                
                # Fill gauge based on probability
                width = np.pi * churn_probability
                ax.bar(np.pi/2, 1, width=width, bottom=0, color=color, alpha=0.8)
                
                # Remove unnecessary elements
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines.clear()
                
                # Add text labels
                ax.text(np.pi/2, 0.5, f"{churn_probability:.1%}", ha='center', va='center', fontsize=24)
                ax.text(np.pi/2, 0.2, "Churn Probability", ha='center', va='center', fontsize=12)
                
                # Add risk category
                ax.text(np.pi/2, -0.2, risk_category, ha='center', va='center', 
                      fontsize=14, fontweight='bold', color=color)
                
                st.pyplot(fig)
            
            with col4:
                # Display key risk factors
                st.subheader("Key Risk Factors")
                
                risk_factors = []
                
                if contract == "Month-to-month":
                    risk_factors.append("Month-to-month contract")
                
                if tenure < 12:
                    risk_factors.append("Customer tenure less than 1 year")
                    
                if internet_service == "Fiber optic" and (online_security == "No" or tech_support == "No"):
                    risk_factors.append("Fiber optic without security or tech support")
                    
                if payment_method == "Electronic check":
                    risk_factors.append("Payment by electronic check")
                    
                if input_df['TotalServices'].values[0] <= 1 and internet_service != "No":
                    risk_factors.append("Low service adoption")
                
                # Display risk factors or safe factors
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"⚠️ **{factor}**")
                else:
                    st.markdown("✅ No major risk factors identified")
                
                # Retention recommendations
                st.subheader("Retention Recommendations")
                
                if churn_probability > 0.5:
                    if contract == "Month-to-month":
                        st.markdown("🔹 Offer contract upgrade incentives")
                    
                    if internet_service == "Fiber optic" and online_security == "No":
                        st.markdown("🔹 Promote online security services")
                    
                    if tech_support == "No":
                        st.markdown("🔹 Offer complimentary tech support")
                        
                    if input_df['TotalServices'].values[0] <= 2:
                        st.markdown("🔹 Bundle additional services at discount")
                else:
                    st.markdown("🔹 Regular satisfaction checks")
                    st.markdown("🔹 Loyalty rewards program")
                    
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check that all input features match the expected format.")