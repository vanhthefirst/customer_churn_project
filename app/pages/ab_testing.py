"""
A/B Testing page for the Churn Prediction dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import copy
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.business_analysis import calculate_targeted_retention_roi

def calculate_roi(df, cost_per_customer, estimated_retention_rate):
    segment_definition = {}

    return calculate_targeted_retention_roi(df, segment_definition, cost_per_customer, estimated_retention_rate)

def show_ab_testing(df, model=None, X_train_columns=None):
    df = copy.deepcopy(df)
    
    # Step 1: Define test parameters
    st.subheader("1. Define Test Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_name = st.text_input("Test Name", "Retention Strategy A/B Test", key="test_name")
        test_duration = st.slider("Test Duration (days)", 30, 180, 90, step=30, key="test_duration")
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95, key="confidence_level")
    
    with col2:
        segment_type = st.selectbox(
            "Select Segment Type for Testing",
            ["All Customers", "Risk Level", "Contract Type", "Internet Service", "Tenure Group", "Payment Method"],
            key="segment_type"
        )

        segment_column_map = {
            "Risk Level": "risk_segment",
            "Contract Type": "Contract",
            "Internet Service": "InternetService",
            "Tenure Group": "TenureGroup",
            "Payment Method": "PaymentMethod"
        }
        
        if segment_type != "All Customers":
            segment_column = segment_column_map[segment_type]

            if segment_column == "TenureGroup" and df[segment_column].dtype in [np.int64, np.float64]:
                tenure_map = {
                    0: '0-1 year',
                    1: '1-2 years', 
                    2: '2-3 years',
                    3: '3-4 years',
                    4: '4-5 years',
                    5: '5+ years'
                }
                
                unique_values = sorted(df[segment_column].unique())
                segment_values = [tenure_map.get(val, f"Group {val}") for val in unique_values]
                
                selected_label = st.selectbox(
                    f"Select {segment_type} for Testing",
                    segment_values,
                    key="segment_value"
                )
                
                # Map back to numeric
                reverse_map = {v: k for k, v in tenure_map.items()}
                selected_value = reverse_map.get(selected_label, 0)

                segment_df = df[df[segment_column] == selected_value]
            else:
                segment_values = sorted(df[segment_column].unique().tolist())

                if segment_column == "risk_segment":
                    segment_values = ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"]
                
                selected_segment = st.selectbox(
                    f"Select {segment_type} for Testing",
                    segment_values,
                    key="segment_value"
                )

                segment_df = df[df[segment_column] == selected_segment]
        else:
            segment_df = df

    st.metric("Testing Segment Size", f"{len(segment_df):,} customers")
    
    # Step 2: Define control and test groups
    st.subheader("2. Define Control and Test Groups")

    test_size_pct = st.slider("Test Group Size (% of segment)", 10, 50, 50, key="test_size")
    total_customers = len(segment_df)
    test_size = int(total_customers * (test_size_pct / 100))
    control_size = total_customers - test_size
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Control Group Size", f"{control_size:,} customers")
    with col4:
        st.metric("Test Group Size", f"{test_size:,} customers")
    
    # Step 3: Select strategy for test group
    st.subheader("3. Select Strategy for Test Group")

    strategies = [
        {
            'name': 'Contract Upgrade Incentive',
            'description': 'Offer 20% discount for 12-month contract commitment',
            'cost_per_customer': 100,
            'base_retention_rate': 0.35,
        },
        {
            'name': 'Service Bundle Upgrade',
            'description': 'Free security & tech support upgrades for 6 months',
            'cost_per_customer': 120,
            'base_retention_rate': 0.30,
        },
        {
            'name': 'Premium Customer Service',
            'description': 'Dedicated support rep and priority service',
            'cost_per_customer': 40,
            'base_retention_rate': 0.25,
        },
        {
            'name': 'Loyalty Rewards Program',
            'description': 'Points for loyalty, redeemable for bill credits',
            'cost_per_customer': 30,
            'base_retention_rate': 0.25,
        },
        {
            'name': 'Service Review & Optimization',
            'description': 'Personalized service review to identify cost-saving opportunities',
            'cost_per_customer': 10,
            'base_retention_rate': 0.15,
        },
        {
            'name': 'Appreciation Communication',
            'description': 'Personalized thank you and recognition of loyalty',
            'cost_per_customer': 2,
            'base_retention_rate': 0.05,
        },
        {
            'name': 'Custom Strategy',
            'description': 'Define your own strategy parameters',
            'cost_per_customer': 50,
            'base_retention_rate': 0.20,
        }
    ]

    strategy_names = [s['name'] for s in strategies]
    selected_strategy_name = st.selectbox("Select Strategy", strategy_names, key="strategy")

    selected_strategy = next((s for s in strategies if s['name'] == selected_strategy_name), None)

    if selected_strategy_name == "Custom Strategy":
        custom_description = st.text_input("Strategy Description", "Custom retention strategy", key="custom_desc")
        custom_cost = st.number_input("Cost per Customer ($)", 1.0, 500.0, 50.0, step=1.0, key="custom_cost")
        custom_rate = st.slider("Expected Retention Rate (%)", 1, 50, 20, key="custom_rate") / 100
        
        selected_strategy['description'] = custom_description
        selected_strategy['cost_per_customer'] = custom_cost
        selected_strategy['base_retention_rate'] = custom_rate

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Strategy", selected_strategy['name'])
    with col6:
        st.metric("Cost per Customer", f"${selected_strategy['cost_per_customer']:.2f}")
    with col7:
        st.metric("Base Retention Rate", f"{selected_strategy['base_retention_rate']*100:.1f}%")
    
    st.markdown(f"**Description**: {selected_strategy['description']}")

    effectiveness_adjustment = st.slider(
        "Effectiveness Adjustment (%)", 
        -50, 50, 0, 
        help="Adjust expected effectiveness up or down from the base rate"
    )

    adjusted_retention_rate = selected_strategy['base_retention_rate'] * (1 + effectiveness_adjustment/100)
    adjusted_retention_rate = min(max(adjusted_retention_rate, 0), 1)  # Clamp between 0 and 1
    
    st.metric(
        "Adjusted Retention Rate", 
        f"{adjusted_retention_rate*100:.1f}%", 
        f"{effectiveness_adjustment:+d}% from base"
    )
    
    # Step 4: Simulate A/B test
    st.subheader("4. Simulate A/B Test")
    
    # Add seed control for reproducible results
    use_seed = st.checkbox("Use fixed random seed (for consistent results)", value=True)
    if use_seed:
        random_seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
    else:
        random_seed = None
    
    run_simulation = st.button("Run Simulation", type="primary")
    
    if run_simulation:
        st.markdown("---")
        st.subheader("Simulation Results")

        if use_seed:
            np.random.seed(random_seed)

        segment_df_sorted = segment_df.sort_values('churn_probability')
        
        # Use stratified sampling for balanced test/control groups 
        # by taking every nth row for test group based on desired proportion
        n = int(100 / test_size_pct)
        segment_df_sorted['group'] = 'control'
        test_indices = list(range(0, len(segment_df_sorted), n))
        segment_df_sorted.iloc[test_indices, segment_df_sorted.columns.get_loc('group')] = 'test'

        current_test_pct = (segment_df_sorted['group'] == 'test').mean() * 100
        if current_test_pct < test_size_pct:
            control_indices = segment_df_sorted[segment_df_sorted['group'] == 'control'].index
            additional_needed = int(len(segment_df_sorted) * (test_size_pct/100 - current_test_pct/100))
            additional_indices = np.random.choice(control_indices, size=min(additional_needed, len(control_indices)), replace=False)
            segment_df_sorted.loc[additional_indices, 'group'] = 'test'

        segment_df = segment_df_sorted.copy()
        
        test_group = segment_df[segment_df['group'] == 'test']
        control_group = segment_df[segment_df['group'] == 'control']

        actual_test_size = len(test_group)
        actual_control_size = len(control_group)

        baseline_churn_rate = segment_df['Churn'].mean() * 100
        baseline_monthly_revenue = segment_df['MonthlyCharges'].mean()
        baseline_clv = segment_df['CLV'].mean() if 'CLV' in segment_df.columns else baseline_monthly_revenue * 12

        avg_churn_prob = segment_df['churn_probability'].mean()
        avg_monthly_charges = segment_df['MonthlyCharges'].mean()

        test_group = segment_df[segment_df['group'] == 'test']
        control_group = segment_df[segment_df['group'] == 'control']

        test_avg_churn = test_group['churn_probability'].mean()
        control_avg_churn = control_group['churn_probability'].mean()
        test_avg_charges = test_group['MonthlyCharges'].mean()
        control_avg_charges = control_group['MonthlyCharges'].mean()

        group_balance = {
            'test_size': len(test_group),
            'control_size': len(control_group),
            'test_avg_churn_prob': test_avg_churn,
            'control_avg_churn_prob': control_avg_churn,
            'test_avg_monthly_charges': test_avg_charges,
            'control_avg_monthly_charges': control_avg_charges,
            'churn_prob_diff_pct': (test_avg_churn - control_avg_churn) / control_avg_churn * 100 if control_avg_churn > 0 else 0,
            'charges_diff_pct': (test_avg_charges - control_avg_charges) / control_avg_charges * 100 if control_avg_charges > 0 else 0
        }
        
        original_churn_probs = test_group['churn_probability'].values
        new_churn_probs = original_churn_probs * (1 - adjusted_retention_rate)

        control_churns = (control_group['churn_probability'] > 0.5).sum()
        control_churn_rate = control_churns / actual_control_size * 100

        test_churns_baseline = (test_group['churn_probability'] > 0.5).sum()
        test_churn_rate_baseline = test_churns_baseline / actual_test_size * 100

        test_churns_treated = (new_churn_probs > 0.5).sum()
        test_churn_rate_treated = test_churns_treated / actual_test_size * 100

        absolute_lift = test_churn_rate_baseline - test_churn_rate_treated
        relative_lift = absolute_lift / test_churn_rate_baseline * 100 if test_churn_rate_baseline > 0 else 0

        monthly_revenue_test = test_group['MonthlyCharges'].sum()
        monthly_revenue_saved = monthly_revenue_test * (adjusted_retention_rate) * (test_churn_rate_baseline / 100)
        annual_revenue_saved = monthly_revenue_saved * 12

        total_cost = actual_test_size * selected_strategy['cost_per_customer']

        roi = (annual_revenue_saved - total_cost) / total_cost * 100 if total_cost > 0 else 0

        try:
            z_stat, p_value = stats.proportions_ztest(
                [test_churns_treated, control_churns], 
                [actual_test_size, actual_control_size]
            )
            is_significant = p_value < (1 - confidence_level/100)
        except:
            # Fallback if scipy is not available or calculation fails
            p_value = 0.5  # Neutral p-value
            is_significant = absolute_lift > (baseline_churn_rate * 0.1)  # Simple heuristic

        col8, col9, col10 = st.columns(3)
        
        with col8:
            st.metric("Baseline Churn Rate", f"{baseline_churn_rate:.2f}%")
            st.metric("Control Group Churn Rate", f"{control_churn_rate:.2f}%")
        
        with col9:
            st.metric("Test Group Baseline Churn", f"{test_churn_rate_baseline:.2f}%")
            st.metric("Test Group Treated Churn", f"{test_churn_rate_treated:.2f}%", 
                     f"{-absolute_lift:.2f}%" if absolute_lift > 0 else f"{absolute_lift:.2f}%",
                     delta_color="inverse" if absolute_lift > 0 else "normal")
        
        with col10:
            st.metric("Relative Improvement", f"{relative_lift:.1f}%" if relative_lift > 0 else f"{-relative_lift:.1f}%")
            st.metric("Statistical Significance", 
                     "Yes" if is_significant else "No", 
                     f"p-value: {p_value:.4f}" if 'p_value' in locals() else "")

        st.subheader("Group Balance Information")
        
        col_balance1, col_balance2 = st.columns(2)
        
        with col_balance1:
            st.metric("Test Group Avg Churn Prob", f"{group_balance['test_avg_churn_prob']:.2%}")
            st.metric("Control Group Avg Churn Prob", f"{group_balance['control_avg_churn_prob']:.2%}")
            st.metric("Difference", f"{group_balance['churn_prob_diff_pct']:.1f}%")
        
        with col_balance2:
            st.metric("Test Group Avg Monthly Charge", f"${group_balance['test_avg_monthly_charges']:.2f}")
            st.metric("Control Group Avg Monthly Charge", f"${group_balance['control_avg_monthly_charges']:.2f}")
            st.metric("Difference", f"{group_balance['charges_diff_pct']:.1f}%")

        st.subheader("Financial Impact")
        
        col11, col12, col13 = st.columns(3)
        
        with col11:
            st.metric("Total Strategy Cost", f"${total_cost:,.2f}")
        
        with col12:
            st.metric("Annual Revenue Saved", f"${annual_revenue_saved:,.2f}")
        
        with col13:
            st.metric("ROI", f"{roi:.1f}%")

        st.subheader("Visualizations")
        
        col14, col15 = st.columns(2)
        
        with col14:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            
            groups = ['Control Group', 'Test Group (Before)', 'Test Group (After)']
            churn_rates = [control_churn_rate, test_churn_rate_baseline, test_churn_rate_treated]
            colors = ['#dddddd', '#ffb3b3', '#b3e6b3']
 
            max_rate = max(churn_rates) if max(churn_rates) > 0 else 0.5
            min_rate = 0

            display_rates = [max(rate, 0.01) for rate in churn_rates]
            
            bars = ax1.bar(groups, display_rates, color=colors)
            
            ax1.set_title('Churn Rate Comparison')
            ax1.set_ylabel('Churn Rate (%)')
            ax1.set_ylim(min_rate, max_rate * 1.2)

            for i, bar in enumerate(bars):
                height = churn_rates[i]
                ax1.text(bar.get_x() + bar.get_width()/2., 
                         display_rates[i] + max_rate * 0.05,
                         f'{height:.2f}%', ha='center', va='bottom')
            
            st.pyplot(fig1)
        
        with col15:
            # Cost vs Revenue Saved
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            
            labels = ['Strategy Cost', 'Revenue Saved']
            values = [total_cost, annual_revenue_saved]
            colors = ['#ff9999', '#99ff99']
            
            if annual_revenue_saved < 1:
                revenue_note = "(No measurable revenue impact)"
            else:
                revenue_note = ""
            
            bars = ax2.bar(labels, values, color=colors)
            
            ax2.set_title('Cost vs. Revenue Impact')
            ax2.set_ylabel('Amount ($)')

            for i, bar in enumerate(bars):
                height = values[i]
                if i == 1 and height < 1:
                    label_text = f"${height:.2f}\n{revenue_note}"
                else:
                    label_text = f"${height:,.0f}"

                if height < 1:
                    y_pos = max(values) * 0.05
                    va = 'bottom'
                else:
                    y_pos = height * 0.5
                    va = 'center'
                
                ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                         label_text, ha='center', va=va, 
                         color='black', fontweight='bold')

            if annual_revenue_saved < total_cost * 0.05:
                ax2.set_ylim(0, total_cost * 1.2)
            
            st.pyplot(fig2)

        st.subheader("Projected Cumulative Impact")

        months = range(1, 13)

        costs = [total_cost] + [0] * (len(months) - 1)
        cumulative_costs = np.cumsum(costs)

        monthly_savings = [monthly_revenue_saved] * len(months)
        cumulative_savings = np.cumsum(monthly_savings)
        net_impact = cumulative_savings - cumulative_costs

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(months, cumulative_costs, 'r--', label='Cumulative Cost')
        ax3.plot(months, cumulative_savings, 'g-', label='Cumulative Revenue Saved')
        ax3.plot(months, net_impact, 'b-', label='Net Impact')

        if max(cumulative_savings) < 1:
            ax3.annotate('No measurable revenue impact', 
                         xy=(6, cumulative_costs[5]/2),
                         ha='center', 
                         fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7))

        if any(net_impact > 0):
            breakeven_month = next((i+1 for i, impact in enumerate(net_impact) if impact > 0), None)
            if breakeven_month:
                ax3.axvline(x=breakeven_month, color='gray', linestyle='--', alpha=0.7)
                ax3.text(breakeven_month + 0.1, max(net_impact)/2, f'Break-even: Month {breakeven_month}', 
                        rotation=90, verticalalignment='center')
        
        ax3.set_title('Projected Financial Impact Over Time')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Amount ($)')
        ax3.set_xticks(months)

        if max(cumulative_savings) < max(cumulative_costs) * 0.05:
            y_max = max(cumulative_costs) * 1.2
            y_min = min(net_impact) * 1.2 if min(net_impact) < 0 else -max(cumulative_costs) * 0.1
            ax3.set_ylim(y_min, y_max)
        
        ax3.legend()
        
        st.pyplot(fig3)

        st.subheader("Sensitivity Analysis")

        fig_sens, ax_sens = plt.subplots(figsize=(10, 6))
        
        retention_rates = np.linspace(0.1, 0.6, 6)
        roi_values = []
        
        for rate in retention_rates:
            # Same calculation as main ROI but with different rates
            test_monthly_revenue = test_group['MonthlyCharges'].sum()
            test_monthly_saved = test_monthly_revenue * rate * (test_churn_rate_baseline / 100)
            test_annual_saved = test_monthly_saved * 12
            test_roi = (test_annual_saved - total_cost) / total_cost * 100 if total_cost > 0 else 0
            roi_values.append(test_roi)

        if test_churn_rate_baseline < 0.01:
            ax_sens.text(0.5, 0.5, 
                        "Insufficient baseline churn to generate positive ROI.\nThe test group has minimal churn risk.", 
                        ha='center', va='center', 
                        transform=ax_sens.transAxes,
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                        fontsize=12)

            ax_sens.plot(retention_rates * 100, roi_values, 'bo-')
            
        else:
            ax_sens.plot(retention_rates * 100, roi_values, 'bo-')

            if min(roi_values) < 0 and max(roi_values) > 0:
                for i in range(len(roi_values)-1):
                    if (roi_values[i] < 0 and roi_values[i+1] > 0) or (roi_values[i] > 0 and roi_values[i+1] < 0):
                        x1, y1 = retention_rates[i] * 100, roi_values[i]
                        x2, y2 = retention_rates[i+1] * 100, roi_values[i+1]
                        breakeven_x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
                        ax_sens.axvline(x=breakeven_x, color='g', linestyle='--', alpha=0.7)
                        ax_sens.text(breakeven_x + 1, 20, f'Break-even: {breakeven_x:.1f}%', 
                                    color='g', fontweight='bold')
                        break
        
        ax_sens.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax_sens.set_xlabel('Retention Rate (%)')
        ax_sens.set_ylabel('ROI (%)')
        ax_sens.set_title('ROI Sensitivity to Retention Rate')
        ax_sens.grid(True, alpha=0.3)
        
        # Set reasonable y-limits that won't break with extreme negative values
        if min(roi_values) < -100:
            ax_sens.set_ylim(-100, max(50, max(roi_values) * 1.1))
        
        st.pyplot(fig_sens)

        st.subheader("Strategy Recommendation")
        
        if roi > 50:
            recommendation = "Strong Recommendation"
            recommendation_text = "This strategy shows excellent potential with a high ROI and significant churn reduction."
            recommendation_color = "#009900"  # Dark green
        elif roi > 20:
            recommendation = "Recommended"
            recommendation_text = "This strategy shows good potential with a positive ROI and meaningful churn reduction."
            recommendation_color = "#00cc00"  # Green
        elif roi > 0:
            recommendation = "Conditionally Recommended"
            recommendation_text = "This strategy has a positive ROI, but consider optimizing to improve returns."
            recommendation_color = "#cccc00"  # Yellow
        else:
            recommendation = "Not Recommended"
            recommendation_text = "This strategy does not show a positive ROI. Consider alternative approaches."
            recommendation_color = "#cc0000"  # Red
        
        st.markdown(f"<h3 style='color: {recommendation_color}'>{recommendation}</h3>", unsafe_allow_html=True)
        st.markdown(recommendation_text)

        if use_seed:
            st.success("✅ Results are consistent due to fixed random seed and balanced sampling.")
        else:
            st.info("ℹ️ Enable fixed random seed for consistent results between simulation runs.")

        if not is_significant:
            st.warning("⚠️ Results are not statistically significant. Consider increasing sample size or test duration.")
        
        if relative_lift < 10:
            st.info("ℹ️ The relative improvement is relatively small. Consider testing more impactful strategies.")