import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference.predict_freight_cost import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# Session State Initialization
if "freight_history" not in st.session_state:
    st.session_state.freight_history = []

if "invoice_history" not in st.session_state:
    st.session_state.invoice_history = []

# Page configuration
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="💼",
    layout="wide"
)


# Header section
st.markdown("""
# 💼 Vendor Invoice Intelligence Portal
Machine Learning for Freight Cost Forecasting and Invoice Risk Assessment
            
This machine learning application helps finance teams:
- **Forecast freight costs accurately**
- **Detect risk or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
""")

st.divider()

# Sidebar
st.sidebar.title("🔍 Model Selection")
selected_model = st.sidebar.radio(
    "",
    [
        "🚚 Freight Cost Prediction",
        "📄 Invoice Risk Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**Business Impact**

✓ Improved freight cost forecasting

✓ Reduced invoice risk and anomalies

✓ Faster invoice approval process

✓ Better vendor spend visibility
""")

# Freight Cost Prediction
if selected_model == "🚚 Freight Cost Prediction":
    st.subheader("🚚 Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Estimate freight cost for vendor invoices to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):
        col1, col2 = st.columns(2)
        with col1:
            quantity = st.number_input(
                "📊 Quantity",
                min_value=1,
                value=1200
            )

        with col2:
            dollars = st.number_input(
                "💰 Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )

        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:
        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }

        if dollars > 50000:
            st.warning(
                "⚠️ Invoice amount exceeds the maximum value seen during training."
            )

        if quantity > 5000:
            st.warning(
                "⚠️ Quantity exceeds the maximum value seen during training."
            )

        prediction = predict_freight_cost(input_data)['Predicted_Freight']

        st.session_state.freight_history.append({
            "Quantity": quantity,
            "Invoice Dollars": dollars,
            "Predicted Freight": prediction[0]
        })

        st.success("🚚 Freight Cost Estimate Generated Successfully")
        
        col1, col2, col3 = st.columns(3)

        col1.metric("Quantity", f"{quantity:,}")

        col2.metric(
            "Invoice Dollars",
            f"${dollars:,.2f}"
        )

        col3.metric(
        label="Estimated Freight Cost",
        value=f"${prediction[0]:,.2f}"
        )
        
        # Load model comparison metrics
        with open("models/freight_metrics.json", "r") as f:
            metrics = json.load(f)
            
        comparison_df = pd.DataFrame(metrics).T.reset_index()

        comparison_df.columns = [
            "Model",
            "MAE",
            "RMSE",
            "R²",
            "CV MAE"
        ]
        
        st.subheader("📊 Model Comparison")

        st.dataframe(
            comparison_df,
            use_container_width=True
        )
        
        best_model = comparison_df.loc[
            comparison_df["MAE"].idxmin(),
            "Model"
        ]

        st.success(f"Best Model: {best_model}")

        

        if prediction[0] < 50:
            st.success(
                "🟢 Low Freight Cost\n\nRecommendation: Freight cost appears within normal range."
            )

        elif prediction[0] < 150:
            st.warning(
                "🟡 Medium Freight Cost\n\nRecommendation: Review freight cost before final approval."
            )

        else:
            st.error(
                "🔴 High Freight Cost\n\nRecommendation: High freight cost detected. Consider vendor review."
            )

        st.subheader("🔍 Prediction Explanation")

        reasons = []

        # Rule 1: Dollar impact 
        if dollars > 10000:
            reasons.append("🔴 High invoice value significantly increases freight cost.")

        elif dollars > 5000:
            reasons.append("🟡 Moderate invoice value contributes to freight cost.")

        else:
            reasons.append("🟢 Low invoice value keeps freight cost stable.")

        # Rule 2: Quantity impact
        if quantity > 2000:
            reasons.append("🔴 Large shipment quantity increases logistics cost.")

        elif quantity > 1000:
            reasons.append("🟡 Moderate shipment size affects freight moderately.")

        else:
            reasons.append("🟢 Small shipment size keeps freight low.")

        # Display reasoning
        for r in reasons:
            st.write("• " + r)

        st.info(
            """
            This prediction is based on historical vendor invoice patterns
            learned from the training dataset.
            """
        )
        
    if st.session_state.freight_history:

        history_df = pd.DataFrame(
            st.session_state.freight_history
        )
        
        total_predictions = len(history_df)
        avg_freight = history_df["Predicted Freight"].mean()
        max_freight = history_df["Predicted Freight"].max()
        min_freight = history_df["Predicted Freight"].min()
        
        st.subheader("📊 Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", total_predictions)
        col2.metric("Avg Freight", f"${avg_freight:,.2f}")
        col3.metric("Max Freight", f"${max_freight:,.2f}")
        col4.metric("Min Freight", f"${min_freight:,.2f}")
        
        # Trend Visualization
        if len(history_df) > 1:

            st.subheader("📈 Freight Prediction Trends")

            st.line_chart(
                data=history_df,
                x="Invoice Dollars",
                y="Predicted Freight"
            )
            
        st.subheader("📜 Prediction History")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        # Download History
        st.download_button(
            label="📥 Download Prediction History",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        # Delete Single Prediction
        row_to_delete = st.selectbox(
            "Select prediction to delete",
            history_df.index
        )

        if st.button("🗑️ Delete Selected Prediction"):
            st.session_state.freight_history.pop(row_to_delete)
            st.rerun()

        
# Invoice Flag Prediction
else:
    st.subheader("📄 Invoice Risk Prediction")

    st.markdown("""
    **Objective**
    Assess invoice risk and identify transactions that may require manual review before payment.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=100
            )
            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1000.0
            )
            days_po_to_invoice = st.number_input(
                "Days PO to Invoice",
                min_value=0,
                value=20
            )

        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=20000.0
            )
            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=80
            )
        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=17000.0
            )
            total_brands = st.number_input(
                "Total Brands",
                min_value=1,
                value=8
            )

        submit_flag = st.form_submit_button("Evaluate Invoice Risk")

    if submit_flag:
        input_data = {
        "invoice_quantity": [invoice_quantity],
        "invoice_dollars": [invoice_dollars],
        "Freight": [freight],
        "total_brands": [total_brands],
        "total_item_quantity": [total_item_quantity],
        "total_item_dollars": [total_item_dollars],
        "days_po_to_invoice": [days_po_to_invoice]
        }

        result = predict_invoice_flag(input_data)

        flag_prediction = result["Predicted_Flag"].iloc[0]

        risk_probability = result["Risk_Probability"].iloc[0]
        risk_level = result["Risk_Level"].iloc[0]

        risk_drivers = []

        freight_pct = freight / max(invoice_dollars, 1)

        amount_difference_pct = (
            abs(invoice_dollars - total_item_dollars)
            / max(total_item_dollars, 1)
        )

        quantity_difference_pct = (
            abs(invoice_quantity - total_item_quantity)
            / max(total_item_quantity, 1)
        )

        if amount_difference_pct > 0.10:
            risk_drivers.append(
                (
                    "Invoice Amount Mismatch",
                    amount_difference_pct * 100
                )
            )

        if quantity_difference_pct > 0.25:
            risk_drivers.append(
                (
                    "Quantity Difference",
                    quantity_difference_pct * 100
                )
            )

        if freight_pct > 0.15:
            risk_drivers.append(
                (
                    "High Freight Cost",
                    freight_pct * 100
                )
            )

        if days_po_to_invoice > 15:
            risk_drivers.append(
                (
                    "PO to Invoice Delay",
                    days_po_to_invoice
                )
            )

        if total_brands > 10:
            risk_drivers.append(
                (
                    "High Brand Count",
                    total_brands
                )
            )
        risk_drivers = sorted(
            risk_drivers,
            key=lambda x: x[1],
            reverse=True
        )

        st.session_state.invoice_history.append(
            {
                "Invoice Dollars": invoice_dollars,
                "Freight Cost": freight,
                "Risk Probability": risk_probability,
                "Risk Level": risk_level,
                "Manual Approval": "Yes" if flag_prediction == 1 else "No"
            }
        )

        st.markdown(
            f"**Predicted Risk Probability:** {risk_probability:.2f}%"
        )
        
        st.subheader("📌 Final Decision")

        if risk_probability >= 70:
            st.error(
                "🚨 High Risk Invoice — Manual Approval Required"
            )

        elif risk_probability >= 40:
            st.warning(
                "⚠ Medium Risk Invoice — Review Recommended"
            )

        else:
            st.success(
                "✅ Low Risk Invoice — Approved for Automatic Processing"
            )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_probability,
                title={"text": "Invoice Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 70], "color": "gold"},
                        {"range": [70, 100], "color": "lightcoral"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": risk_probability
                    }
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("💡 Recommendation")

        if risk_probability >= 70:

            st.markdown("""
                - Perform manual invoice review
                - Validate freight charges
                - Verify purchase order details
                - Confirm quantity reconciliation
                """)

        elif risk_probability >= 40:

            st.markdown("""
                - Review invoice before payment
                - Check for quantity variances
                - Verify vendor charges
                """)

        else:

            st.markdown("""
                - Invoice appears normal
                - Eligible for streamlined approval
                - No significant anomalies detected
                """)

        st.subheader("📈 Top Risk Drivers")

        if risk_drivers:

            st.markdown(
                    "<br>".join(
                        [
                            f"✓ {driver} (+{score:.1f})"
                            for driver, score in risk_drivers
                        ]
                    ),
                    unsafe_allow_html=True
            )

        else:

            st.success(
                "No significant risk drivers detected."
            )

        if st.session_state.invoice_history:

            st.subheader("📜 Invoice Prediction History")

            invoice_history_df = pd.DataFrame(
                st.session_state.invoice_history
            )

            st.dataframe(
                invoice_history_df,
                use_container_width=True
            )

            st.download_button(
                label="📥 Download Invoice History",
                data=invoice_history_df.to_csv(index=False),
                file_name="invoice_prediction_history.csv",
                mime="text/csv"
            )

            row_to_delete = st.selectbox(
                "Select invoice prediction to delete",
                invoice_history_df.index,
                key="invoice_delete"
            )

            if st.button("🗑️ Delete Selected Invoice Prediction"):

                st.session_state.invoice_history.pop(
                    row_to_delete
                )

                st.rerun()

