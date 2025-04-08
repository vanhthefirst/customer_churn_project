"""
Retention strategies page for the Churn Prediction dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.business_analysis import calculate_targeted_retention_roi

def calculate_roi(df, cost_per_customer, estimated_retention_rate):
    segment_definition = {}

    return calculate_targeted_retention_roi(df, segment_definition, cost_per_customer, estimated_retention_rate)

def show_retention_strategies(df):
    st.header("Retention Strategy Recommendations")
    st.markdown("""
    This dashboard evaluates potential retention strategies and calculates their expected ROI.
    Compare different interventions to identify the most cost-effective approaches.
    """)

    retention_strategies = {
        'High Risk': [
            {
                'name': 'Contract Upgrade with Bonus',
                'description': 'Offer significant discount (20%) for 12-month contract commitment',
                'cost_per_customer': 100,  # Approximate cost
                'estimated_retention_rate': 0.35,  # 35% effectiveness
                'target_segment': 'High Risk customers on month-to-month contracts'
            },
            {
                'name': 'Service Bundle Upgrade',
                'description': 'Free security & tech support upgrades for 6 months',
                'cost_per_customer': 120,  # $20/month for 6 months
                'estimated_retention_rate': 0.30,  # 30% effectiveness
                'target_segment': 'High Risk customers with fiber service but no protection'
            },
            {
                'name': 'Premium Customer Service',
                'description': 'Dedicated support rep, priority service, courtesy check-ins',
                'cost_per_customer': 40,  # $40 per customer for dedicated support
                'estimated_retention_rate': 0.25,  # 25% effectiveness
                'target_segment': 'High Risk customers with high monthly charges'
            }
        ],
        'Medium-High Risk': [
            {
                'name': 'Loyalty Rewards Program',
                'description': 'Points for loyalty, redeemable for bill credits or service upgrades',
                'cost_per_customer': 30,  # $30 per customer in rewards
                'estimated_retention_rate': 0.25,  # 25% effectiveness
                'target_segment': 'Medium-High Risk customers with 12+ months tenure'
            }
        ],
        'Medium-Low Risk': [
            {
                'name': 'Service Review & Optimization',
                'description': 'Personalized service review to identify cost-saving opportunities',
                'cost_per_customer': 10,  # $10 per customer for review
                'estimated_retention_rate': 0.15,  # 15% effectiveness
                'target_segment': 'Medium-Low Risk customers with high bills'
            }
        ],
        'Low Risk': [
            {
                'name': 'Appreciation Communication',
                'description': 'Personalized thank you and recognition of loyalty',
                'cost_per_customer': 2,  # $2 per customer
                'estimated_retention_rate': 0.05,  # 5% effectiveness
                'target_segment': 'Low Risk customers with long tenure'
            }
        ]
    }

    risk_segment = st.selectbox(
        "Select risk segment to analyze", 
        ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"],
        index=0
    )

    st.subheader(f"Retention Strategies for {risk_segment} Segment")

    segment_customers = df[df['risk_segment'] == risk_segment]
    num_customers = len(segment_customers)
    monthly_revenue = segment_customers['MonthlyCharges'].sum()
    annual_revenue = monthly_revenue * 12

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Customers in Segment", f"{num_customers:,}")
    with col2:
        st.metric("Monthly Revenue", f"${monthly_revenue:,.2f}")
    with col3:
        st.metric("Annual Revenue", f"${annual_revenue:,.2f}")

    st.subheader("Available Strategies")
    
    if risk_segment in retention_strategies:
        strategies = retention_strategies[risk_segment]

        for i, strategy in enumerate(strategies):
            with st.expander(f"Strategy {i+1}: {strategy['name']}", expanded=True if i == 0 else False):
                st.markdown(f"**Description**: {strategy['description']}")
                st.markdown(f"**Target**: {strategy['target_segment']}")

                if "month-to-month" in strategy['target_segment'].lower():
                    target_df = segment_customers[segment_customers['Contract'] == 'Month-to-month']
                elif "fiber" in strategy['target_segment'].lower():
                    target_df = segment_customers[(segment_customers['InternetService'] == 'Fiber optic')]
                elif "high" in strategy['target_segment'].lower() and "charge" in strategy['target_segment'].lower():
                    charge_threshold = df['MonthlyCharges'].median()
                    target_df = segment_customers[segment_customers['MonthlyCharges'] > charge_threshold]
                elif "12+" in strategy['target_segment'].lower() or "tenure" in strategy['target_segment'].lower():
                    target_df = segment_customers[segment_customers['tenure'] >= 12]
                else:
                    target_df = segment_customers

                roi_data = calculate_roi(
                    target_df, 
                    strategy['cost_per_customer'],
                    strategy['estimated_retention_rate']
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Customers Targeted", f"{roi_data['customers_targeted']:,}")
                    st.metric("Cost per Customer", f"${strategy['cost_per_customer']:.2f}")
                
                with col2:
                    st.metric("Total Cost", f"${roi_data['intervention_cost']:,.2f}")
                    st.metric("Est. Effectiveness", f"{strategy['estimated_retention_rate']*100:.1f}%")
                
                with col3:
                    st.metric("Revenue Saved", f"${roi_data['potential_annual_revenue_saved']:,.2f}")
                    roi_pct = roi_data['estimated_roi_percent'] if 'estimated_roi_percent' in roi_data else roi_data.get('estimated_roi', 0) * 100
                    st.metric("ROI", f"{roi_pct:.1f}%", delta="positive" if roi_pct > 0 else "negative")

                fig, ax = plt.subplots(figsize=(10, 5))
                labels = ['Cost', 'Revenue Saved']
                values = [roi_data['intervention_cost'], roi_data['potential_annual_revenue_saved']]
                colors = ['#ff9999', '#66b3ff']
                
                ax.bar(labels, values, color=colors)
                ax.set_title('Cost vs. Revenue Saved')
                ax.set_ylabel('Amount ($)')

                for i, v in enumerate(values):
                    ax.text(i, v/2, f"${v:,.0f}", ha='center', fontweight='bold', color='white')
                
                st.pyplot(fig)
    else:
        st.info("No strategies defined for this segment.")

    st.subheader("Strategy Comparison")

    comparison_data = []
    
    for segment, strategies in retention_strategies.items():
        segment_customers = df[df['risk_segment'] == segment]
        
        for strategy in strategies:
            if "month-to-month" in strategy['target_segment'].lower():
                target_df = segment_customers[segment_customers['Contract'] == 'Month-to-month']
            elif "fiber" in strategy['target_segment'].lower():
                target_df = segment_customers[(segment_customers['InternetService'] == 'Fiber optic')]
            elif "high" in strategy['target_segment'].lower() and "charge" in strategy['target_segment'].lower():
                charge_threshold = df['MonthlyCharges'].median()
                target_df = segment_customers[segment_customers['MonthlyCharges'] > charge_threshold]
            elif "12+" in strategy['target_segment'].lower() or "tenure" in strategy['target_segment'].lower():
                target_df = segment_customers[segment_customers['tenure'] >= 12]
            else:
                target_df = segment_customers

            if len(target_df) > 0:
                roi_data = calculate_roi(
                    target_df, 
                    strategy['cost_per_customer'],
                    strategy['estimated_retention_rate']
                )
                
                roi_pct = roi_data['estimated_roi_percent'] if 'estimated_roi_percent' in roi_data else roi_data.get('estimated_roi', 0) * 100
                
                comparison_data.append({
                    'Risk Segment': segment,
                    'Strategy': strategy['name'],
                    'Target Customers': roi_data['customers_targeted'],
                    'Total Cost': roi_data['intervention_cost'],
                    'Revenue Saved': roi_data['potential_annual_revenue_saved'],
                    'ROI %': roi_pct
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROI %', ascending=False)

        st.dataframe(comparison_df)

        fig, ax = plt.subplots(figsize=(10, 5))
        top_strategies = comparison_df.head(5)
        strategy_labels = [f"{row['Strategy']} ({row['Risk Segment']})" for _, row in top_strategies.iterrows()]
        roi_values = top_strategies['ROI %']
        
        ax.barh(strategy_labels, roi_values)
        ax.set_title('Top 5 Strategies by ROI')
        ax.set_xlabel('ROI %')

        for i, v in enumerate(roi_values):
            ax.text(v + 5, i, f"{v:.1f}%", va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No comparison data available.")

    st.subheader("Implementation Recommendations")
    
    st.markdown("""
    ### Implementation Phases:
    
    1. **Phase 1 (Month 1-2)**: 
       - Launch highest ROI strategies first
       - Focus on high-risk segment to prevent immediate churn
       
    2. **Phase 2 (Month 3-4)**:
       - Implement medium-risk strategies
       - Review effectiveness of Phase 1 interventions
       
    3. **Phase 3 (Month 5-6)**:
       - Deploy remaining strategies
       - Comprehensive evaluation of program effectiveness
    
    ### Monitoring KPIs:
    - Reduction in churn rate by segment
    - Strategy conversion rates
    - ROI by intervention
    - Customer satisfaction scores
    """)