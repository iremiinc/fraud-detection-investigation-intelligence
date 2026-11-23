import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Fraud Intelligence",
    page_icon="🛡️",
    layout="wide"
)

from dashboard.utils.ui import apply_custom_ui

apply_custom_ui()

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "all_transactions.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    return df

df = load_data()



# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Fraud Detection & Investigation Dashboard")

st.markdown(
    """
    Monitor transaction activity, identify suspicious behavior,
    and investigate fraud patterns across the transaction ecosystem.
    """
)

st.markdown("---")

# --------------------------------------------------
# FILTERS
# --------------------------------------------------

st.subheader("Filters")

col1, col2 = st.columns(2)

with col1:
    scenarios = sorted(df["fraud_scenario"].dropna().unique())

    selected_scenarios = st.multiselect(
        "Fraud Scenario",
        scenarios,
        default=scenarios
    )

with col2:
    if df["timestamp"].notna().any():
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()

        selected_dates = st.date_input(
            "Transaction Date Range",
            value=(min_date, max_date)
        )
    else:
        selected_dates = None

filtered_df = df[
    df["fraud_scenario"].isin(selected_scenarios)
].copy()

if selected_dates and len(selected_dates) == 2:

    start_date, end_date = selected_dates

    filtered_df = filtered_df[
        filtered_df["timestamp"].dt.date.between(
            start_date,
            end_date
        )
    ]

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------

total_transactions = len(filtered_df)

fraud_transactions = filtered_df["is_fraud"].sum()

fraud_rate = (
    fraud_transactions / total_transactions * 100
    if total_transactions > 0
    else 0
)

total_accounts = pd.concat([
    filtered_df["sender_id"],
    filtered_df["receiver_id"]
]).nunique()

total_amount = filtered_df["amount"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraud Transactions",
    f"{int(fraud_transactions):,}"
)

col3.metric(
    "Fraud Rate",
    f"{fraud_rate:.2f}%"
)

col4.metric(
    "Accounts Involved",
    f"{total_accounts:,}"
)

st.markdown("---")

# --------------------------------------------------
# TRANSACTION VOLUME & FRAUD
# --------------------------------------------------

st.subheader("Transaction Overview")

col1, col2 = st.columns(2)

with col1:

    transaction_type_counts = (
        filtered_df["fraud_scenario"]
        .value_counts()
        .reset_index()
    )

    transaction_type_counts.columns = [
        "Scenario",
        "Count"
    ]

    fig_scenario = px.bar(
        transaction_type_counts,
        x="Scenario",
        y="Count",
        title="Transactions by Scenario"
    )

    fig_scenario.update_layout(
        xaxis_title="Scenario",
        yaxis_title="Transaction Count"
    )

    st.plotly_chart(
        fig_scenario,
        width="stretch"
    )

with col2:

    fraud_distribution = (
        filtered_df["is_fraud"]
        .value_counts()
        .reset_index()
    )

    fraud_distribution.columns = [
        "Label",
        "Count"
    ]

    fraud_distribution["Label"] = (
        fraud_distribution["Label"]
        .map({
            0: "Legitimate",
            1: "Fraud"
        })
    )

    fig_fraud = px.pie(
        fraud_distribution,
        names="Label",
        values="Count",
        title="Fraud vs Legitimate Transactions",
        hole=0.45
    )

    st.plotly_chart(
        fig_fraud,
        width="stretch"
    )

# --------------------------------------------------
# TRANSACTION AMOUNT ANALYSIS
# --------------------------------------------------

st.markdown("---")

st.subheader("Transaction Amount Analysis")

col1, col2 = st.columns(2)

with col1:

    fig_amount = px.histogram(
        filtered_df,
        x="amount",
        color="is_fraud",
        nbins=50,
        title="Transaction Amount Distribution",
        labels={
            "amount": "Transaction Amount",
            "is_fraud": "Fraud Label"
        }
    )

    st.plotly_chart(
        fig_amount,
        width="stretch"
    )

with col2:

    scenario_amount = (
        filtered_df.groupby("fraud_scenario")["amount"]
        .sum()
        .reset_index()
        .sort_values("amount", ascending=False)
    )

    fig_scenario_amount = px.bar(
        scenario_amount,
        x="fraud_scenario",
        y="amount",
        title="Transaction Volume by Scenario"
    )

    st.plotly_chart(
        fig_scenario_amount,
        width="stretch"
    )

# --------------------------------------------------
# RECENT TRANSACTIONS
# --------------------------------------------------

st.markdown("---")

st.subheader("Recent Transactions")

display_columns = [
    "transaction_id",
    "timestamp",
    "sender_id",
    "receiver_id",
    "amount",
    "fraud_scenario",
    "is_fraud"
]

display_columns = [
    col for col in display_columns
    if col in filtered_df.columns
]

recent_transactions = (
    filtered_df.sort_values(
        "timestamp",
        ascending=False
    )
    .head(20)
)

st.dataframe(
    recent_transactions[display_columns],
    width="stretch",
    hide_index=True
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Fraud Intelligence | Synthetic Transaction Monitoring Platform"
)