# dashboard/pages/1_Transaction_Monitoring.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.utils.ui import apply_custom_ui

apply_custom_ui()

from dashboard.utils.model_utils import get_monitoring_data


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Transaction Monitoring",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return get_monitoring_data()


df = load_data()


# =========================================================
# PAGE HEADER
# =========================================================

st.title("🔎 Transaction Monitoring")

st.markdown(
    """
    Real-time transaction monitoring powered by the trained
    LightGBM fraud detection model.
    """
)

st.caption(
    "Synthetic transaction dataset | Model-generated risk scores | "
    "Decision threshold: 95%"
)


# =========================================================
# FILTERS
# =========================================================

st.sidebar.header("Filters")

filtered_df = df.copy()


# Date filter
if "timestamp" in filtered_df.columns:

    filtered_df["timestamp"] = pd.to_datetime(
        filtered_df["timestamp"],
        errors="coerce"
    )

    valid_dates = filtered_df["timestamp"].dropna()

    if not valid_dates.empty:

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        selected_dates = st.sidebar.date_input(
            "Transaction Date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

            start_date, end_date = selected_dates

            filtered_df = filtered_df[
                filtered_df["timestamp"].dt.date.between(
                    start_date,
                    end_date
                )
            ]


# Amount filter
if "amount" in filtered_df.columns:

    min_amount = float(filtered_df["amount"].min())
    max_amount = float(filtered_df["amount"].max())

    amount_range = st.sidebar.slider(
        "Transaction Amount",
        min_value=min_amount,
        max_value=max_amount,
        value=(min_amount, max_amount)
    )

    filtered_df = filtered_df[
        filtered_df["amount"].between(
            amount_range[0],
            amount_range[1]
        )
    ]


# Risk level filter
risk_options = ["Low", "Medium", "High", "Critical"]

selected_risks = st.sidebar.multiselect(
    "Risk Level",
    options=risk_options,
    default=risk_options
)

filtered_df = filtered_df[
    filtered_df["risk_level"].astype(str).isin(selected_risks)
]


# Flagged filter
flagged_filter = st.sidebar.selectbox(
    "Model Status",
    ["All Transactions", "Flagged Only", "Not Flagged Only"]
)

if flagged_filter == "Flagged Only":

    filtered_df = filtered_df[
        filtered_df["model_prediction"] == 1
    ]

elif flagged_filter == "Not Flagged Only":

    filtered_df = filtered_df[
        filtered_df["model_prediction"] == 0
    ]


# Transaction type filter
if "transaction_type" in filtered_df.columns:

    transaction_types = filtered_df[
        "transaction_type"
    ].dropna().unique().tolist()

    selected_types = st.sidebar.multiselect(
        "Transaction Type",
        options=transaction_types,
        default=transaction_types
    )

    filtered_df = filtered_df[
        filtered_df["transaction_type"].isin(selected_types)
    ]


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_transactions = len(filtered_df)

flagged_transactions = int(
    filtered_df["model_prediction"].sum()
)

flagged_rate = (
    flagged_transactions / total_transactions * 100
    if total_transactions > 0
    else 0
)

critical_transactions = int(
    (filtered_df["risk_level"] == "Critical").sum()
)

high_risk_transactions = int(
    (filtered_df["risk_level"] == "High").sum()
)

total_transaction_volume = filtered_df["amount"].sum()

flagged_amount = filtered_df.loc[
    filtered_df["model_prediction"] == 1,
    "amount"
].sum()

average_risk_score = (
    filtered_df["risk_score"].mean()
    if total_transactions > 0
    else 0
)


# =========================================================
# KPI CARDS
# =========================================================

st.subheader("Monitoring Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Flagged Transactions",
    f"{flagged_transactions:,}",
    f"{flagged_rate:.2f}% of filtered transactions"
)

col3.metric(
    "Critical Alerts",
    f"{critical_transactions:,}"
)

col4.metric(
    "High Risk",
    f"{high_risk_transactions:,}"
)

col5.metric(
    "Flagged Amount",
    f"{flagged_amount:,.2f}"
)


st.divider()


# =========================================================
# RISK DISTRIBUTION
# =========================================================

st.subheader("Risk Distribution")

risk_order = ["Low", "Medium", "High", "Critical"]

risk_counts = (
    filtered_df["risk_level"]
    .astype(str)
    .value_counts()
    .reindex(risk_order, fill_value=0)
    .reset_index()
)

risk_counts.columns = ["Risk Level", "Transactions"]


col1, col2 = st.columns(2)


with col1:

    fig_risk = px.bar(
        risk_counts,
        x="Risk Level",
        y="Transactions",
        color="Risk Level",
        category_orders={
            "Risk Level": risk_order
        },
        title="Transaction Count by Risk Level"
    )

    fig_risk.update_layout(
        showlegend=False,
        height=400
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )


with col2:

    fig_pie = px.pie(
        risk_counts,
        names="Risk Level",
        values="Transactions",
        title="Risk Distribution (%)",
        hole=0.45
    )

    fig_pie.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


st.divider()


# =========================================================
# TRANSACTION TREND
# =========================================================

st.subheader("Transaction Trends")

if "timestamp" in filtered_df.columns:

    trend_df = filtered_df.dropna(
        subset=["timestamp"]
    ).copy()

    if not trend_df.empty:

        trend_df["date"] = (
            trend_df["timestamp"].dt.date
        )

        daily_transactions = (
            trend_df.groupby("date")
            .agg(
                total_transactions=("transaction_id", "count"),
                flagged_transactions=("model_prediction", "sum"),
                transaction_volume=("amount", "sum")
            )
            .reset_index()
        )

        fig_trend = go.Figure()

        fig_trend.add_trace(
            go.Scatter(
                x=daily_transactions["date"],
                y=daily_transactions["total_transactions"],
                name="Total Transactions",
                mode="lines"
            )
        )

        fig_trend.add_trace(
            go.Scatter(
                x=daily_transactions["date"],
                y=daily_transactions["flagged_transactions"],
                name="Flagged Transactions",
                mode="lines"
            )
        )

        fig_trend.update_layout(
            title="Daily Transaction Activity",
            xaxis_title="Date",
            yaxis_title="Transaction Count",
            height=400,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )

        # Volume chart
        fig_volume = px.area(
            daily_transactions,
            x="date",
            y="transaction_volume",
            title="Daily Transaction Volume"
        )

        fig_volume.update_layout(
            height=350
        )

        st.plotly_chart(
            fig_volume,
            use_container_width=True
        )


st.divider()


# =========================================================
# RISK ANALYSIS
# =========================================================

st.subheader("Risk Analysis")

col1, col2 = st.columns(2)


with col1:

    fig_amount_risk = px.scatter(
        filtered_df,
        x="amount",
        y="fraud_probability",
        color="risk_level",
        hover_data=[
            "transaction_id",
            "risk_score",
            "priority"
        ],
        title="Transaction Amount vs Fraud Probability"
    )

    fig_amount_risk.update_layout(
        height=450,
        xaxis_title="Transaction Amount",
        yaxis_title="Fraud Probability"
    )

    st.plotly_chart(
        fig_amount_risk,
        use_container_width=True
    )


with col2:

    fig_hist = px.histogram(
        filtered_df,
        x="amount",
        color="risk_level",
        nbins=50,
        title="Transaction Amount Distribution"
    )

    fig_hist.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )


# =========================================================
# TRANSACTION TYPE ANALYSIS
# =========================================================

if "transaction_type" in filtered_df.columns:

    st.subheader("Transaction Type Risk Analysis")

    type_analysis = (
        filtered_df.groupby("transaction_type", observed=False)
        .agg(
            total_transactions=("transaction_id", "count"),
            flagged_transactions=("model_prediction", "sum"),
            average_amount=("amount", "mean"),
            average_risk=("risk_score", "mean")
        )
        .reset_index()
    )

    type_analysis["flagged_rate"] = (
        type_analysis["flagged_transactions"]
        / type_analysis["total_transactions"]
        * 100
    )

    fig_type = px.bar(
        type_analysis,
        x="transaction_type",
        y="flagged_rate",
        color="average_risk",
        title="Flagged Transaction Rate by Transaction Type",
        labels={
            "flagged_rate": "Flagged Rate (%)",
            "transaction_type": "Transaction Type"
        }
    )

    st.plotly_chart(
        fig_type,
        use_container_width=True
    )

    st.dataframe(
        type_analysis,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# =========================================================
# INVESTIGATION QUEUE
# =========================================================

st.subheader("🚨 Investigation Queue")

st.caption(
    "Transactions are ranked using model risk score and transaction amount."
)


queue_df = filtered_df[
    filtered_df["model_prediction"] == 1
].copy()


queue_df = queue_df.sort_values(
    by=["priority_score", "fraud_probability"],
    ascending=False
)


display_columns = [
    "transaction_id",
    "timestamp",
    "amount",
    "fraud_probability",
    "risk_score",
    "risk_level",
    "priority",
    "priority_score",
    "model_status"
]


display_columns = [
    column
    for column in display_columns
    if column in queue_df.columns
]


if not queue_df.empty:

    queue_display = queue_df[display_columns].copy()

    queue_display["fraud_probability"] = (
        queue_display["fraud_probability"] * 100
    )

    queue_display = queue_display.rename(
        columns={
            "fraud_probability": "Fraud Probability (%)",
            "risk_score": "Risk Score",
            "priority_score": "Priority Score",
            "transaction_id": "Transaction ID",
            "timestamp": "Timestamp",
            "amount": "Amount",
            "risk_level": "Risk Level",
            "priority": "Priority",
            "model_status": "Model Status"
        }
    )

    st.dataframe(
        queue_display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No flagged transactions found for the selected filters."
    )


st.divider()


# =========================================================
# TRANSACTION INVESTIGATION
# =========================================================

st.subheader("Transaction Investigation")

if not filtered_df.empty:

    transaction_options = filtered_df[
        "transaction_id"
    ].astype(str).tolist()

    selected_transaction = st.selectbox(
        "Select Transaction ID",
        transaction_options
    )

    transaction = filtered_df[
        filtered_df["transaction_id"].astype(str)
        == selected_transaction
    ].iloc[0]


    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Amount",
        f"{transaction['amount']:,.2f}"
    )

    col2.metric(
        "Fraud Probability",
        f"{transaction['fraud_probability']:.2%}"
    )

    col3.metric(
        "Risk Score",
        f"{transaction['risk_score']:.2f}/100"
    )

    col4.metric(
        "Priority",
        str(transaction["priority"])
    )


    st.markdown("### Transaction Details")

    detail_columns = [
        "transaction_id",
        "timestamp",
        "amount",
        "risk_level",
        "risk_score",
        "fraud_probability",
        "model_prediction",
        "priority",
        "priority_score",
        "transaction_type",
        "sender_id",
        "receiver_id",
        "is_fraud",
        "fraud_scenario"
    ]

    detail_columns = [
        column
        for column in detail_columns
        if column in transaction.index
    ]

    detail_df = pd.DataFrame(
        {
            "Feature": detail_columns,
            "Value": [
                transaction[column]
                for column in detail_columns
            ]
        }
    )

    st.dataframe(
        detail_df,
        use_container_width=True,
        hide_index=True
    )


    # Feature details
    st.markdown("### Model Features")

    feature_columns = [
        feature
        for feature in [
            "amount_log",
            "hour",
            "day_of_week",
            "is_weekend",
            "transaction_type_code",
            "tx_count_1h",
            "tx_count_24h",
            "amount_sum_1h",
            "amount_sum_24h",
            "previous_avg_amount",
            "previous_std_amount",
            "has_transaction_history",
            "amount_vs_avg",
            "amount_zscore",
            "sender_prior_tx_count",
            "receiver_prior_tx_count"
        ]
        if feature in transaction.index
    ]

    feature_detail = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Value": [
                transaction[feature]
                for feature in feature_columns
            ]
        }
    )

    st.dataframe(
        feature_detail,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# EXPORT
# =========================================================

st.divider()

st.subheader("Export")

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Transactions CSV",
    data=csv_data,
    file_name="transaction_monitoring_export.csv",
    mime="text/csv"
)

queue_csv = queue_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Investigation Queue CSV",
    data=queue_csv,
    file_name="fraud_investigation_queue.csv",
    mime="text/csv"
)

