# dashboard/pages/3_Transaction_Network.py

import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
from dashboard.utils.ui import apply_custom_ui

apply_custom_ui()

from pyvis.network import Network

from dashboard.utils.model_utils import get_monitoring_data


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Transaction Network",
    page_icon="🕸️",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return get_monitoring_data()


df = load_data().copy()


# =========================================================
# HEADER
# =========================================================

st.title("🕸️ Transaction Network")

st.markdown(
    """
    Explore financial relationships between accounts through an
    interactive transaction network. Identify suspicious connections,
    high-risk accounts and unusual transaction patterns.
    """
)

st.caption(
    "Account = Node | Transaction = Edge | "
    "Node risk = Account-level model risk | Edge = Transaction"
)


# =========================================================
# DATA PREPARATION
# =========================================================

required_columns = [
    "transaction_id",
    "sender_id",
    "receiver_id",
    "amount",
    "fraud_probability",
    "risk_score",
    "risk_level",
    "model_prediction"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing required columns: {missing_columns}"
    )

    st.stop()


df["sender_id"] = df["sender_id"].astype(str)
df["receiver_id"] = df["receiver_id"].astype(str)


if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Network Filters")


# Risk filter
risk_options = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

selected_risks = st.sidebar.multiselect(
    "Transaction Risk Level",
    options=risk_options,
    default=["High", "Critical"]
)


filtered_df = df[
    df["risk_level"].astype(str).isin(selected_risks)
].copy()


# Flagged only
flagged_only = st.sidebar.checkbox(
    "Show Model-Flagged Transactions Only",
    value=False
)

if flagged_only:

    filtered_df = filtered_df[
        filtered_df["model_prediction"] == 1
    ]


# Amount filter
if not filtered_df.empty:

    amount_min = float(
        filtered_df["amount"].min()
    )

    amount_max = float(
        filtered_df["amount"].max()
    )

    if amount_min < amount_max:

        amount_range = st.sidebar.slider(
            "Transaction Amount",
            min_value=amount_min,
            max_value=amount_max,
            value=(amount_min, amount_max)
        )

        filtered_df = filtered_df[
            filtered_df["amount"].between(
                amount_range[0],
                amount_range[1]
            )
        ]


# Date filter
if "timestamp" in filtered_df.columns:

    valid_dates = filtered_df["timestamp"].dropna()

    if not valid_dates.empty:

        start_date = valid_dates.min().date()
        end_date = valid_dates.max().date()

        selected_dates = st.sidebar.date_input(
            "Date Range",
            value=(start_date, end_date),
            min_value=start_date,
            max_value=end_date
        )

        if (
            isinstance(selected_dates, tuple)
            and len(selected_dates) == 2
        ):

            date_start, date_end = selected_dates

            filtered_df = filtered_df[
                filtered_df["timestamp"].dt.date.between(
                    date_start,
                    date_end
                )
            ]


# Graph limit
max_edges = st.sidebar.slider(
    "Maximum Edges in Graph",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100
)


# =========================================================
# NETWORK PREPARATION
# =========================================================

if filtered_df.empty:

    st.warning(
        "No transactions match the selected filters."
    )

    st.stop()


# Sort by risk score
network_transactions = filtered_df.sort_values(
    by="fraud_probability",
    ascending=False
).head(max_edges).copy()


# Create graph
G = nx.DiGraph()


# =========================================================
# NODE ATTRIBUTES
# =========================================================

all_accounts = pd.concat(
    [
        network_transactions["sender_id"],
        network_transactions["receiver_id"]
    ]
).unique()


account_stats = []

for account in all_accounts:

    outgoing = network_transactions[
        network_transactions["sender_id"] == account
    ]

    incoming = network_transactions[
        network_transactions["receiver_id"] == account
    ]

    involved = pd.concat(
        [outgoing, incoming]
    )

    account_stats.append(
        {
            "account_id": account,

            "outgoing_transactions": len(outgoing),

            "incoming_transactions": len(incoming),

            "total_transactions": len(involved),

            "total_sent": outgoing["amount"].sum(),

            "total_received": incoming["amount"].sum(),

            "total_volume": involved["amount"].sum(),

            "flagged_transactions": involved[
                "model_prediction"
            ].sum(),

            "average_risk": involved[
                "risk_score"
            ].mean(),

            "maximum_risk": involved[
                "risk_score"
            ].max(),

            "fraud_probability": involved[
                "fraud_probability"
            ].mean()
        }
    )


account_stats = pd.DataFrame(account_stats)


# =========================================================
# NODE RISK LEVEL
# =========================================================

account_stats["account_risk"] = pd.cut(
    account_stats["average_risk"],
    bins=[-0.01, 40, 70, 95, 100.01],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
)


# =========================================================
# ADD NODES
# =========================================================

for _, account in account_stats.iterrows():

    account_id = account["account_id"]

    account_risk = str(
        account["account_risk"]
    )

    risk_colors = {
        "Low": "#2ECC71",
        "Medium": "#F1C40F",
        "High": "#E67E22",
        "Critical": "#E74C3C"
    }

    node_color = risk_colors.get(
        account_risk,
        "#95A5A6"
    )


    node_size = (
        15
        + np.sqrt(
            account["total_transactions"]
        ) * 4
    )


    tooltip = f"""
    Account: {account_id}

    Risk Level: {account_risk}

    Avg Risk Score: {account['average_risk']:.2f}

    Transactions: {account['total_transactions']}

    Outgoing: {account['outgoing_transactions']}

    Incoming: {account['incoming_transactions']}

    Total Volume: {account['total_volume']:,.2f}

    Flagged Transactions: {account['flagged_transactions']}
    """


    G.add_node(
        account_id,
        label=account_id,
        title=tooltip,
        color=node_color,
        size=node_size
    )


# =========================================================
# ADD EDGES
# =========================================================

for _, tx in network_transactions.iterrows():

    sender = tx["sender_id"]
    receiver = tx["receiver_id"]

    amount = tx["amount"]

    risk = tx["risk_score"]

    probability = tx["fraud_probability"]


    if risk >= 95:

        edge_color = "#E74C3C"

    elif risk >= 70:

        edge_color = "#E67E22"

    elif risk >= 40:

        edge_color = "#F1C40F"

    else:

        edge_color = "#95A5A6"


    edge_width = (
        1 + np.log1p(amount) / 3
    )


    edge_tooltip = f"""
    Transaction ID: {tx['transaction_id']}

    Amount: {amount:,.2f}

    Fraud Probability: {probability:.2%}

    Risk Score: {risk:.2f}

    Risk Level: {tx['risk_level']}
    """


    G.add_edge(
        sender,
        receiver,
        title=edge_tooltip,
        color=edge_color,
        width=edge_width,
        arrows="to"
    )


# =========================================================
# NETWORK KPI CALCULATIONS
# =========================================================

total_nodes = G.number_of_nodes()

total_edges = G.number_of_edges()
total_transactions_in_network = len(network_transactions)

flagged_network_transactions = int(
    network_transactions["model_prediction"].sum()
)

network_volume = network_transactions[
    "amount"
].sum()

suspicious_accounts = int(
    (
        account_stats["account_risk"].isin(
            ["High", "Critical"]
        )
    ).sum()
)

critical_accounts = int(
    (
        account_stats["account_risk"] == "Critical"
    ).sum()
)


# =========================================================
# NETWORK METRICS
# =========================================================

undirected_graph = G.to_undirected()

connected_components = (
    nx.number_connected_components(
        undirected_graph
    )
    if total_nodes > 0
    else 0
)

average_degree = (
    sum(dict(G.degree()).values()) / total_nodes
    if total_nodes > 0
    else 0
)

density = nx.density(
    undirected_graph
) if total_nodes > 1 else 0


# =========================================================
# KPI DISPLAY
# =========================================================

st.subheader("Network Overview")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Accounts / Nodes", f"{total_nodes:,}")
col2.metric("Unique Connections", f"{total_edges:,}")
col3.metric("Transactions", f"{total_transactions_in_network:,}")
col4.metric("Flagged Transactions", f"{flagged_network_transactions:,}")
col5.metric("Suspicious Accounts", f"{suspicious_accounts:,}")
col6.metric("Network Volume", f"{network_volume:,.2f}")


# =========================================================
# SECONDARY NETWORK METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Connected Components",
    f"{connected_components:,}"
)

col2.metric(
    "Average Degree",
    f"{average_degree:.2f}"
)

col3.metric(
    "Network Density",
    f"{density:.5f}"
)

col4.metric(
    "Critical Accounts",
    f"{critical_accounts:,}"
)


st.divider()


# =========================================================
# NETWORK GRAPH
# =========================================================

st.subheader("Interactive Transaction Network")


# Legend
st.markdown(
    """
    **Node Colors**

    🟢 Low Risk &nbsp;&nbsp;
    🟡 Medium Risk &nbsp;&nbsp;
    🟠 High Risk &nbsp;&nbsp;
    🔴 Critical Risk
    """
)


# Create PyVis graph
net = Network(
    height="750px",
    width="100%",
    directed=True,
    bgcolor="#0E1117",
    font_color="white",
    notebook=False
)


net.from_nx(G)


# Physics
net.set_options(
    """
    {
      "nodes": {
        "font": {
          "size": 12,
          "color": "white"
        },
        "borderWidth": 1
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        }
      },
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "minVelocity": 0.75,
        "stabilization": {
          "iterations": 150
        }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """
)


# Generate HTML
graph_html = net.generate_html()


components.html(
    graph_html,
    height=780,
    scrolling=True
)


st.caption(
    "Drag nodes to explore relationships. Hover over nodes or edges "
    "to inspect details. Use zoom and navigation controls."
)


st.divider()


# =========================================================
# NETWORK RISK ANALYSIS
# =========================================================

st.subheader("Network Risk Analysis")


col1, col2 = st.columns(2)


# Risk distribution
with col1:

    node_risk_distribution = (
        account_stats["account_risk"]
        .astype(str)
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"],
            fill_value=0
        )
        .reset_index()
    )

    node_risk_distribution.columns = [
        "Risk Level",
        "Accounts"
    ]


    fig_node_risk = px.bar(
        node_risk_distribution,
        x="Risk Level",
        y="Accounts",
        color="Risk Level",
        title="Account Risk Distribution"
    )

    fig_node_risk.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_node_risk,
        use_container_width=True
    )


# Edge risk distribution
with col2:

    edge_risk_distribution = (
        network_transactions["risk_level"]
        .astype(str)
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"],
            fill_value=0
        )
        .reset_index()
    )

    edge_risk_distribution.columns = [
        "Risk Level",
        "Transactions"
    ]


    fig_edge_risk = px.bar(
        edge_risk_distribution,
        x="Risk Level",
        y="Transactions",
        color="Risk Level",
        title="Transaction Risk Distribution"
    )

    fig_edge_risk.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_edge_risk,
        use_container_width=True
    )


st.divider()


# =========================================================
# TOP CONNECTED ACCOUNTS
# =========================================================

st.subheader("Top Connected Accounts")


degree_df = pd.DataFrame(
    G.degree(),
    columns=[
        "Account",
        "Degree"
    ]
)


degree_df = degree_df.merge(
    account_stats,
    left_on="Account",
    right_on="account_id",
    how="left"
)


degree_df = degree_df.sort_values(
    "Degree",
    ascending=False
).head(20)


fig_degree = px.bar(
    degree_df,
    x="Account",
    y="Degree",
    color="account_risk",
    title="Top 20 Accounts by Network Degree",
    hover_data=[
        "total_transactions",
        "total_volume",
        "average_risk",
        "flagged_transactions"
    ]
)


st.plotly_chart(
    fig_degree,
    use_container_width=True
)


st.dataframe(
    degree_df[
        [
            "Account",
            "Degree",
            "total_transactions",
            "total_volume",
            "flagged_transactions",
            "average_risk",
            "account_risk"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# SUSPICIOUS ACCOUNTS
# =========================================================

st.subheader("🚨 Suspicious Accounts Ranking")


suspicious_account_table = account_stats.sort_values(
    by=[
        "average_risk",
        "flagged_transactions",
        "total_volume"
    ],
    ascending=False
)


suspicious_account_table = suspicious_account_table[
    [
        "account_id",
        "total_transactions",
        "outgoing_transactions",
        "incoming_transactions",
        "total_volume",
        "flagged_transactions",
        "average_risk",
        "maximum_risk",
        "account_risk"
    ]
]


st.dataframe(
    suspicious_account_table.head(50),
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# ACCOUNT SELECTION
# =========================================================

st.subheader("Account Network Investigation")


account_list = sorted(
    account_stats["account_id"].unique().tolist()
)


selected_account = st.selectbox(
    "Select an account",
    account_list
)


selected_account_data = account_stats[
    account_stats["account_id"] == selected_account
].iloc[0]


neighbors = list(
    G.neighbors(selected_account)
)


incoming_neighbors = list(
    G.predecessors(selected_account)
)


outgoing_neighbors = list(
    G.successors(selected_account)
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Connections",
    G.degree(selected_account)
)

col2.metric(
    "Outgoing Connections",
    len(outgoing_neighbors)
)

col3.metric(
    "Incoming Connections",
    len(incoming_neighbors)
)

col4.metric(
    "Average Risk",
    f"{selected_account_data['average_risk']:.2f}"
)


# Account profile
st.markdown("### Account Profile")


st.dataframe(
    pd.DataFrame(
        {
            "Metric": [
                "Account ID",
                "Account Risk",
                "Total Transactions",
                "Outgoing Transactions",
                "Incoming Transactions",
                "Total Sent",
                "Total Received",
                "Total Transaction Volume",
                "Flagged Transactions",
                "Average Risk Score",
                "Maximum Risk Score"
            ],

            "Value": [
                selected_account,
                selected_account_data["account_risk"],
                selected_account_data["total_transactions"],
                selected_account_data["outgoing_transactions"],
                selected_account_data["incoming_transactions"],
                selected_account_data["total_sent"],
                selected_account_data["total_received"],
                selected_account_data["total_volume"],
                selected_account_data["flagged_transactions"],
                selected_account_data["average_risk"],
                selected_account_data["maximum_risk"]
            ]
        }
    ),
    use_container_width=True,
    hide_index=True
)


# Connected accounts
st.markdown("### Connected Accounts")


neighbor_df = account_stats[
    account_stats["account_id"].isin(neighbors)
].sort_values(
    "average_risk",
    ascending=False
)


st.dataframe(
    neighbor_df[
        [
            "account_id",
            "total_transactions",
            "total_volume",
            "flagged_transactions",
            "average_risk",
            "account_risk"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# SELECTED ACCOUNT TRANSACTIONS
# =========================================================

st.markdown("### Transactions Involving Selected Account")


selected_transactions = network_transactions[
    (
        network_transactions["sender_id"]
        == selected_account
    )
    |
    (
        network_transactions["receiver_id"]
        == selected_account
    )
].copy()


selected_transactions = selected_transactions.sort_values(
    "fraud_probability",
    ascending=False
)


st.dataframe(
    selected_transactions[
        [
            "transaction_id",
            "timestamp",
            "sender_id",
            "receiver_id",
            "amount",
            "fraud_probability",
            "risk_score",
            "risk_level",
            "model_status"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# EXPORT
# =========================================================

st.divider()

st.subheader("Export Network Data")


network_export = network_transactions.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "Download Network Transactions",
    data=network_export,
    file_name="transaction_network.csv",
    mime="text/csv"
)


account_export = account_stats.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "Download Account Network Metrics",
    data=account_export,
    file_name="account_network_metrics.csv",
    mime="text/csv"
)


# =========================================================
# FOOTNOTE
# =========================================================

st.caption(
    "Network visualization is generated from the selected transaction subset. "
    "Account risk is aggregated from model-generated transaction scores. "
    "Ground-truth labels are not used to determine node colors."
)