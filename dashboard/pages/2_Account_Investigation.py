
import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.utils.ui import apply_custom_ui

apply_custom_ui()

from dashboard.utils.model_utils import get_monitoring_data


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Account Investigation",
    page_icon="👤",
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
# HEADER
# =========================================================

st.title("👤 Account Investigation")

st.markdown(
    "Investigate account-level transaction activity, "
    "suspicious behavior, and model-detected fraud risk."
)

st.caption(
    "Account intelligence based on transaction history "
    "and LightGBM risk predictions."
)


# =========================================================
# ACCOUNT AGGREGATION
# =========================================================

account_records = []

account_columns = ["sender_id", "receiver_id"]

for account_column in account_columns:

    if account_column not in df.columns:
        continue

    temp = df.copy()

    temp["account_id"] = temp[account_column]

    temp = temp.dropna(subset=["account_id"])

    account_stats = (
        temp.groupby("account_id")
        .agg(
            total_transactions=("transaction_id", "count"),
            total_amount=("amount", "sum"),
            avg_transaction_amount=("amount", "mean"),
            flagged_transactions=("model_prediction", "sum"),
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            critical_transactions=(
                "risk_level",
                lambda x: (x == "Critical").sum()
            ),
            fraud_transactions=("is_fraud", "sum")
        )
        .reset_index()
    )

    account_stats["account_role"] = (
        "Sender" if account_column == "sender_id"
        else "Receiver"
    )

    account_records.append(account_stats)


if account_records:

    accounts = pd.concat(
        account_records,
        ignore_index=True
    )

else:

    st.error("Sender and receiver account columns were not found.")
    st.stop()


# =========================================================
# ACCOUNT METRICS
# =========================================================

accounts["flagged_rate"] = (
    accounts["flagged_transactions"]
    / accounts["total_transactions"]
    * 100
)

accounts["fraud_rate"] = (
    accounts["fraud_transactions"]
    / accounts["total_transactions"]
    * 100
)


# Account risk classification
accounts["account_risk"] = pd.cut(
    accounts["avg_risk_score"],
    bins=[-0.01, 40, 70, 95, 100.01],
    labels=["Low", "Medium", "High", "Critical"]
)


# Account investigation priority
accounts["investigation_priority_score"] = (
    accounts["avg_risk_score"] * 0.50
    + accounts["flagged_rate"] * 0.30
    + accounts["fraud_rate"] * 0.20
)


accounts["investigation_priority"] = pd.cut(
    accounts["investigation_priority_score"],
    bins=[-0.01, 30, 55, 80, 100.01],
    labels=[
        "P4 - Low",
        "P3 - Medium",
        "P2 - High",
        "P1 - Urgent"
    ]
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Account Filters")

role_options = accounts["account_role"].unique().tolist()

selected_roles = st.sidebar.multiselect(
    "Account Role",
    options=role_options,
    default=role_options
)

filtered_accounts = accounts[
    accounts["account_role"].isin(selected_roles)
].copy()


risk_options = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

selected_account_risks = st.sidebar.multiselect(
    "Account Risk Level",
    options=risk_options,
    default=risk_options
)

filtered_accounts = filtered_accounts[
    filtered_accounts["account_risk"]
    .astype(str)
    .isin(selected_account_risks)
]


min_transactions = st.sidebar.slider(
    "Minimum Transaction Count",
    min_value=1,
    max_value=int(accounts["total_transactions"].max()),
    value=1
)

filtered_accounts = filtered_accounts[
    filtered_accounts["total_transactions"]
    >= min_transactions
]


# =========================================================
# OVERVIEW
# =========================================================

st.subheader("Account Intelligence Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accounts Analyzed",
    f"{filtered_accounts['account_id'].nunique():,}"
)

col2.metric(
    "High / Critical Accounts",
    f"{filtered_accounts['account_risk'].isin(['High', 'Critical']).sum():,}"
)

col3.metric(
    "Flagged Transactions",
    f"{filtered_accounts['flagged_transactions'].sum():,}"
)

col4.metric(
    "Fraud-Linked Transactions",
    f"{filtered_accounts['fraud_transactions'].sum():,}"
)


st.divider()


# =========================================================
# ACCOUNT RISK DISTRIBUTION
# =========================================================

st.subheader("Account Risk Distribution")

risk_distribution = (
    filtered_accounts["account_risk"]
    .astype(str)
    .value_counts()
    .reindex(risk_options, fill_value=0)
    .reset_index()
)

risk_distribution.columns = [
    "Risk Level",
    "Accounts"
]


col1, col2 = st.columns(2)


with col1:

    fig_risk = px.bar(
        risk_distribution,
        x="Risk Level",
        y="Accounts",
        color="Risk Level",
        title="Accounts by Risk Level"
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
        risk_distribution,
        names="Risk Level",
        values="Accounts",
        hole=0.45,
        title="Account Risk Composition"
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
# HIGH RISK ACCOUNT TABLE
# =========================================================

st.subheader("🚨 High-Risk Accounts")

high_risk_accounts = filtered_accounts.sort_values(
    by="investigation_priority_score",
    ascending=False
)


high_risk_display = high_risk_accounts[
    [
        "account_id",
        "account_role",
        "total_transactions",
        "total_amount",
        "avg_transaction_amount",
        "flagged_transactions",
        "flagged_rate",
        "fraud_transactions",
        "fraud_rate",
        "avg_risk_score",
        "account_risk",
        "investigation_priority"
    ]
].copy()


high_risk_display = high_risk_display.rename(
    columns={
        "account_id": "Account ID",
        "account_role": "Role",
        "total_transactions": "Transactions",
        "total_amount": "Total Amount",
        "avg_transaction_amount": "Avg Amount",
        "flagged_transactions": "Flagged",
        "flagged_rate": "Flagged Rate (%)",
        "fraud_transactions": "Ground Truth Fraud",
        "fraud_rate": "Ground Truth Fraud Rate (%)",
        "avg_risk_score": "Avg Risk Score",
        "account_risk": "Risk Level",
        "investigation_priority": "Priority"
    }
)


st.dataframe(
    high_risk_display,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# ACCOUNT SELECTION
# =========================================================

st.subheader("🔍 Investigate Individual Account")

account_options = (
    filtered_accounts["account_id"]
    .astype(str)
    .sort_values()
    .unique()
    .tolist()
)


if not account_options:

    st.warning("No accounts match the selected filters.")

else:

    selected_account = st.selectbox(
        "Select Account",
        account_options
    )


    # =====================================================
    # SELECTED ACCOUNT TRANSACTIONS
    # =====================================================

    account_transactions = df[
        (
            df["sender_id"].astype(str)
            == selected_account
        )
        |
        (
            df["receiver_id"].astype(str)
            == selected_account
        )
    ].copy()


    account_transactions = account_transactions.sort_values(
        "timestamp"
    )


    # Account role
    account_transactions["account_role"] = "Receiver"

    account_transactions.loc[
        account_transactions["sender_id"].astype(str)
        == selected_account,
        "account_role"
    ] = "Sender"


    # =====================================================
    # ACCOUNT KPI
    # =====================================================

    total_tx = len(account_transactions)

    total_sent = account_transactions.loc[
        account_transactions["sender_id"].astype(str)
        == selected_account,
        "amount"
    ].sum()

    total_received = account_transactions.loc[
        account_transactions["receiver_id"].astype(str)
        == selected_account,
        "amount"
    ].sum()

    account_flagged = int(
        account_transactions["model_prediction"].sum()
    )

    account_fraud = int(
        account_transactions["is_fraud"].sum()
    )

    avg_risk = account_transactions[
        "risk_score"
    ].mean()

    max_risk = account_transactions[
        "risk_score"
    ].max()


    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Transactions",
        f"{total_tx:,}"
    )

    col2.metric(
        "Total Sent",
        f"{total_sent:,.2f}"
    )

    col3.metric(
        "Total Received",
        f"{total_received:,.2f}"
    )

    col4.metric(
        "Flagged",
        f"{account_flagged:,}"
    )

    col5.metric(
        "Average Risk Score",
        f"{avg_risk:.2f}"
    )


    # =====================================================
    # ACCOUNT RISK PROFILE
    # =====================================================

    st.markdown("### Account Risk Profile")

    selected_account_stats = accounts[
        accounts["account_id"].astype(str)
        == selected_account
    ]

    st.dataframe(
        selected_account_stats,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # TRANSACTION HISTORY
    # =====================================================

    st.markdown("### Transaction History")

    transaction_columns = [
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
        "fraud_probability",
        "risk_score",
        "risk_level",
        "model_status",
        "is_fraud",
        "fraud_scenario"
    ]

    transaction_columns = [
        column
        for column in transaction_columns
        if column in account_transactions.columns
    ]


    history = account_transactions[
        transaction_columns
    ].copy()


    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # ACCOUNT TRANSACTION TREND
    # =====================================================

    st.markdown("### Transaction Activity Over Time")

    if "timestamp" in account_transactions.columns:

        daily_activity = (
            account_transactions
            .dropna(subset=["timestamp"])
            .groupby(
                account_transactions["timestamp"].dt.date
            )
            .agg(
                transaction_count=("transaction_id", "count"),
                transaction_amount=("amount", "sum"),
                flagged_count=("model_prediction", "sum")
            )
            .reset_index()
        )

        daily_activity.columns = [
            "Date",
            "Transaction Count",
            "Transaction Amount",
            "Flagged Transactions"
        ]


        fig_activity = px.line(
            daily_activity,
            x="Date",
            y=[
                "Transaction Count",
                "Flagged Transactions"
            ],
            title="Account Activity"
        )

        st.plotly_chart(
            fig_activity,
            use_container_width=True
        )


        fig_amount = px.bar(
            daily_activity,
            x="Date",
            y="Transaction Amount",
            title="Daily Transaction Amount"
        )

        st.plotly_chart(
            fig_amount,
            use_container_width=True
        )


    # =====================================================
    # SENDER / RECEIVER ANALYSIS
    # =====================================================

    st.markdown("### Counterparty Analysis")

    outgoing = account_transactions[
        account_transactions["sender_id"].astype(str)
        == selected_account
    ]

    incoming = account_transactions[
        account_transactions["receiver_id"].astype(str)
        == selected_account
    ]


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("#### Top Receivers")

        top_receivers = (
            outgoing.groupby("receiver_id")
            .agg(
                transactions=("transaction_id", "count"),
                total_amount=("amount", "sum"),
                flagged=("model_prediction", "sum"),
                avg_risk=("risk_score", "mean")
            )
            .sort_values(
                "total_amount",
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        st.dataframe(
            top_receivers,
            use_container_width=True,
            hide_index=True
        )


    with col2:

        st.markdown("#### Top Senders")

        top_senders = (
            incoming.groupby("sender_id")
            .agg(
                transactions=("transaction_id", "count"),
                total_amount=("amount", "sum"),
                flagged=("model_prediction", "sum"),
                avg_risk=("risk_score", "mean")
            )
            .sort_values(
                "total_amount",
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        st.dataframe(
            top_senders,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # RISK DISTRIBUTION FOR SELECTED ACCOUNT
    # =====================================================

    st.markdown("### Transaction Risk Distribution")

    selected_risk_distribution = (
        account_transactions["risk_level"]
        .astype(str)
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"],
            fill_value=0
        )
        .reset_index()
    )

    selected_risk_distribution.columns = [
        "Risk Level",
        "Transactions"
    ]


    fig_selected_risk = px.bar(
        selected_risk_distribution,
        x="Risk Level",
        y="Transactions",
        color="Risk Level",
        title=f"Risk Distribution: {selected_account}"
    )

    st.plotly_chart(
        fig_selected_risk,
        use_container_width=True
    )


    # =====================================================
    # SUSPICIOUS TRANSACTIONS
    # =====================================================

    st.markdown("### 🚨 Suspicious Transactions")

    suspicious_transactions = account_transactions[
        account_transactions["model_prediction"] == 1
    ].sort_values(
        "fraud_probability",
        ascending=False
    )


    if not suspicious_transactions.empty:

        st.dataframe(
            suspicious_transactions[
                transaction_columns
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No model-flagged transactions for this account."
        )


