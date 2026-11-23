import streamlit as st
import pandas as pd
from dashboard.utils.ui import apply_custom_ui

apply_custom_ui()

from dashboard.utils.model_utils import get_monitoring_data

from src.alert_management import (
    generate_alerts,
    get_alerts,
    update_alert_status,
    link_alert_to_case
)

from src.case_management import create_case


st.set_page_config(
    page_title="Fraud Alert Center",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Fraud Alert Center")
st.caption("Centralized monitoring and prioritization of suspicious transactions")


# ============================================================
# LOAD DATA & GENERATE ALERTS
# ============================================================

df = get_monitoring_data()

new_alerts = generate_alerts(
    df,
    threshold=0.60
)

alerts_df = get_alerts()


if new_alerts > 0:
    st.toast(f"{new_alerts} new alerts generated")


# ============================================================
# KPI OVERVIEW
# ============================================================

total_alerts = len(alerts_df)

new_count = len(
    alerts_df[alerts_df["status"] == "New"]
)

acknowledged_count = len(
    alerts_df[alerts_df["status"] == "Acknowledged"]
)

investigating_count = len(
    alerts_df[alerts_df["status"] == "Investigating"]
)

critical_count = len(
    alerts_df[alerts_df["priority"] == "Critical"]
)

high_count = len(
    alerts_df[alerts_df["priority"] == "High"]
)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Alerts", total_alerts)
col2.metric("New Alerts", new_count)
col3.metric("Critical", critical_count)
col4.metric("High Priority", high_count)
col5.metric("Investigating", investigating_count)


# ============================================================
# FILTERS
# ============================================================

st.divider()

st.subheader("Alert Queue")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:

    selected_priorities = st.multiselect(
        "Priority",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium"]
    )

with filter_col2:

    selected_statuses = st.multiselect(
        "Status",
        ["New", "Acknowledged", "Investigating", "Resolved"],
        default=["New", "Acknowledged", "Investigating"]
    )

with filter_col3:

    min_probability = st.slider(
        "Minimum Fraud Probability",
        min_value=0.60,
        max_value=1.00,
        value=0.60,
        step=0.05
    )


filtered_alerts = alerts_df.copy()

if selected_priorities:

    filtered_alerts = filtered_alerts[
        filtered_alerts["priority"].isin(selected_priorities)
    ]

if selected_statuses:

    filtered_alerts = filtered_alerts[
        filtered_alerts["status"].isin(selected_statuses)
    ]

filtered_alerts = filtered_alerts[
    filtered_alerts["fraud_probability"] >= min_probability
]


# ============================================================
# ALERT TABLE
# ============================================================

if filtered_alerts.empty:

    st.info("No alerts match the selected filters.")

else:

    display_df = filtered_alerts.copy()

    display_df["fraud_probability"] = (
        display_df["fraud_probability"] * 100
    ).round(2)

    display_df = display_df.rename(columns={
        "alert_id": "Alert ID",
        "transaction_id": "Transaction ID",
        "account_id": "Account",
        "amount": "Amount",
        "fraud_probability": "Fraud Probability (%)",
        "priority": "Priority",
        "status": "Status",
        "created_at": "Created At",
        "case_id": "Linked Case"
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ALERT DETAILS & ACTIONS
# ============================================================

st.divider()

st.subheader("Alert Investigation")

if not filtered_alerts.empty:

    selected_alert_id = st.selectbox(
        "Select Alert",
        filtered_alerts["alert_id"].tolist(),
        key="selected_alert"
    )

    alert = filtered_alerts[
        filtered_alerts["alert_id"] == selected_alert_id
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transaction",
        alert["transaction_id"]
    )

    col2.metric(
        "Account",
        alert["account_id"]
    )

    col3.metric(
        "Amount",
        f"{alert['amount']:,.2f}"
    )

    col4.metric(
        "Fraud Probability",
        f"{alert['fraud_probability']:.2%}"
    )

    st.write(f"**Priority:** {alert['priority']}")
    st.write(f"**Current Status:** {alert['status']}")
    st.write(f"**Created:** {alert['created_at']}")

    if alert["case_id"]:
        st.success(f"Linked Case: {alert['case_id']}")


    # --------------------------------------------------------
    # ACKNOWLEDGE ALERT
    # --------------------------------------------------------

    if alert["status"] == "New":

        if st.button(
            "✅ Acknowledge Alert",
            key="acknowledge_alert"
        ):

            update_alert_status(
                selected_alert_id,
                "Acknowledged"
            )

            st.success("Alert acknowledged.")
            st.rerun()


    # --------------------------------------------------------
    # RESOLVE ALERT
    # --------------------------------------------------------

    if alert["status"] != "Resolved":

        if st.button(
            "✔ Mark as Resolved",
            key="resolve_alert"
        ):

            update_alert_status(
                selected_alert_id,
                "Resolved"
            )

            st.success("Alert resolved.")
            st.rerun()


    # --------------------------------------------------------
    # CREATE CASE
    # --------------------------------------------------------

    if not alert["case_id"]:

        st.markdown("### Create Investigation Case")

        case_notes = st.text_area(
            "Investigation Notes",
            placeholder="Initial observations regarding this alert...",
            key="alert_case_notes"
        )

        if st.button(
            "📁 Convert Alert to Case",
            key="convert_alert_to_case"
        ):

            probability = float(alert["fraud_probability"])

            risk_score = probability * 100

            if probability >= 0.95:
                severity = "Critical"
            elif probability >= 0.80:
                severity = "High"
            elif probability >= 0.60:
                severity = "Medium"
            else:
                severity = "Low"

            case_id = create_case(
                account_id=alert["account_id"],
                risk_score=risk_score,
                severity=severity,
                notes=(
                    f"Created from Alert ID: {selected_alert_id}\n"
                    f"Transaction ID: {alert['transaction_id']}\n"
                    f"Fraud Probability: {probability:.4f}\n"
                    f"{case_notes}"
                )
            )

            link_alert_to_case(
                selected_alert_id,
                case_id
            )

            st.success(
                f"Alert converted to Case: {case_id}"
            )

            st.rerun()


    # --------------------------------------------------------
    # LINKED CASE
    # --------------------------------------------------------

    else:

        st.info(
            f"This alert has already been converted to Case "
            f"{alert['case_id']}."
        )

else:

    st.info("Select an alert to investigate.")