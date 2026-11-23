
import pandas as pd
import numpy as np
import networkx as nx


# =========================================================
# NETWORK INTELLIGENCE
# =========================================================

def build_network_intelligence_context(
    df,
    risk_threshold=0.95,
    top_n=15
):
    """
    Build transaction-network intelligence context
    using transaction and model prediction data.

    Ground-truth fraud labels are intentionally excluded
    from AI context to prevent label leakage.
    """

    df = df.copy()


    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------

    required_columns = [
        "sender_id",
        "receiver_id",
        "amount"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


    # -----------------------------------------------------
    # Clean data
    # -----------------------------------------------------

    df = df.dropna(
        subset=[
            "sender_id",
            "receiver_id",
            "amount"
        ]
    )


    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )


    df = df.dropna(
        subset=["amount"]
    )


    # -----------------------------------------------------
    # Risk information
    # -----------------------------------------------------

    if "fraud_probability" in df.columns:

        df["fraud_probability"] = pd.to_numeric(
            df["fraud_probability"],
            errors="coerce"
        )

    else:

        df["fraud_probability"] = np.nan


    if "model_prediction" in df.columns:

        df["model_prediction"] = pd.to_numeric(
            df["model_prediction"],
            errors="coerce"
        )

    else:

        df["model_prediction"] = np.nan


    df["high_risk"] = (
        df["fraud_probability"] >= risk_threshold
    )


    # -----------------------------------------------------
    # Build directed transaction graph
    # -----------------------------------------------------

    G = nx.DiGraph()


    for _, row in df.iterrows():

        sender = str(row["sender_id"])
        receiver = str(row["receiver_id"])

        amount = float(row["amount"])

        probability = row["fraud_probability"]


        if pd.isna(probability):

            probability = 0.0


        G.add_edge(

            sender,

            receiver,

            amount=amount,

            fraud_probability=float(probability),

            high_risk=bool(
                probability >= risk_threshold
            )

        )


    # -----------------------------------------------------
    # Network-level metrics
    # -----------------------------------------------------

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()


    if total_nodes > 0:

        density = nx.density(G)

        weak_components = (
            nx.number_weakly_connected_components(G)
        )

        largest_component = max(
            (
                len(component)
                for component in nx.weakly_connected_components(G)
            ),
            default=0
        )

    else:

        density = 0
        weak_components = 0
        largest_component = 0


    total_transactions = len(df)


    high_risk_transactions = int(
        df["high_risk"].sum()
    )


    high_risk_rate = (

        high_risk_transactions / total_transactions

        if total_transactions > 0

        else 0

    )


    # -----------------------------------------------------
    # Node-level activity
    # -----------------------------------------------------

    outgoing = (

        df.groupby("sender_id")

        .agg(

            outgoing_transactions=(
                "transaction_id",
                "count"
            )
            if "transaction_id" in df.columns
            else (
                "amount",
                "count"
            ),

            outgoing_amount=(
                "amount",
                "sum"
            ),

            avg_outgoing_amount=(
                "amount",
                "mean"
            ),

            outgoing_high_risk_count=(
                "high_risk",
                "sum"
            ),

            outgoing_avg_fraud_probability=(
                "fraud_probability",
                "mean"
            )

        )

    )


    incoming = (

        df.groupby("receiver_id")

        .agg(

            incoming_transactions=(
                "transaction_id",
                "count"
            )
            if "transaction_id" in df.columns
            else (
                "amount",
                "count"
            ),

            incoming_amount=(
                "amount",
                "sum"
            ),

            avg_incoming_amount=(
                "amount",
                "mean"
            ),

            incoming_high_risk_count=(
                "high_risk",
                "sum"
            ),

            incoming_avg_fraud_probability=(
                "fraud_probability",
                "mean"
            )

        )

    )


    # -----------------------------------------------------
    # Combine account metrics
    # -----------------------------------------------------

    node_metrics = (

        outgoing.join(
            incoming,
            how="outer"
        )

        .fillna(0)

    )


    node_metrics.index = (
        node_metrics.index.astype(str)
    )


    # -----------------------------------------------------
    # Graph degree
    # -----------------------------------------------------

    node_metrics["in_degree"] = (

        pd.Series(
            dict(G.in_degree()),
            dtype=float
        )

    )


    node_metrics["out_degree"] = (

        pd.Series(
            dict(G.out_degree()),
            dtype=float
        )

    )


    node_metrics["total_degree"] = (

        node_metrics["in_degree"]

        + node_metrics["out_degree"]

    )


    node_metrics = node_metrics.fillna(0)


    # -----------------------------------------------------
    # Risk concentration
    # -----------------------------------------------------

    node_metrics["total_high_risk_transactions"] = (

        node_metrics["outgoing_high_risk_count"]

        + node_metrics["incoming_high_risk_count"]

    )


    node_metrics["total_transaction_amount"] = (

        node_metrics["outgoing_amount"]

        + node_metrics["incoming_amount"]

    )


    # -----------------------------------------------------
    # Suspicious node ranking
    # -----------------------------------------------------

    suspicious_nodes = (

        node_metrics.sort_values(

            [

                "total_high_risk_transactions",

                "total_degree",

                "total_transaction_amount"

            ],

            ascending=False

        )

        .head(top_n)

    )


    # -----------------------------------------------------
    # Suspicious edges
    # -----------------------------------------------------

    suspicious_edges = (

        df[df["high_risk"]]

        .sort_values(
            "fraud_probability",
            ascending=False
        )

        .head(top_n)

    )


    # -----------------------------------------------------
    # Network statistics
    # -----------------------------------------------------

    total_amount = df["amount"].sum()


    high_risk_amount = (

        df.loc[
            df["high_risk"],
            "amount"
        ]

        .sum()

    )


    # -----------------------------------------------------
    # Build AI context
    # -----------------------------------------------------

    context_parts = []


    context_parts.append(
        "TRANSACTION NETWORK OVERVIEW"
    )


    context_parts.append(
        f"Total Nodes: {total_nodes}"
    )


    context_parts.append(
        f"Total Directed Edges: {total_edges}"
    )


    context_parts.append(
        f"Network Density: {density:.6f}"
    )


    context_parts.append(
        f"Weakly Connected Components: {weak_components}"
    )


    context_parts.append(
        f"Largest Connected Component Size: {largest_component}"
    )


    context_parts.append(
        f"Total Transactions: {total_transactions}"
    )


    context_parts.append(
        f"High-Risk Transactions: {high_risk_transactions}"
    )


    context_parts.append(
        f"High-Risk Transaction Rate: {high_risk_rate:.2%}"
    )


    context_parts.append(
        f"Total Transaction Amount: {total_amount:,.2f}"
    )


    context_parts.append(
        f"High-Risk Transaction Amount: {high_risk_amount:,.2f}"
    )


    # -----------------------------------------------------
    # Suspicious Accounts
    # -----------------------------------------------------

    context_parts.append(
        "\nTOP ACCOUNTS BY NETWORK RISK AND ACTIVITY"
    )


    for account, row in suspicious_nodes.iterrows():

        context_parts.append(

            f"""
Account: {account}
Incoming Transactions: {row['incoming_transactions']:.0f}
Outgoing Transactions: {row['outgoing_transactions']:.0f}
Incoming Amount: {row['incoming_amount']:.2f}
Outgoing Amount: {row['outgoing_amount']:.2f}
Incoming High-Risk Transactions: {row['incoming_high_risk_count']:.0f}
Outgoing High-Risk Transactions: {row['outgoing_high_risk_count']:.0f}
Total High-Risk Transactions: {row['total_high_risk_transactions']:.0f}
In-Degree: {row['in_degree']:.0f}
Out-Degree: {row['out_degree']:.0f}
Total Degree: {row['total_degree']:.0f}
"""

        )


    # -----------------------------------------------------
    # High-risk transaction connections
    # -----------------------------------------------------

    context_parts.append(
        "\nTOP HIGH-RISK TRANSACTION CONNECTIONS"
    )


    for _, row in suspicious_edges.iterrows():

        transaction_id = (

            row.get(
                "transaction_id",
                "Unknown"
            )

        )


        context_parts.append(

            f"""
Transaction ID: {transaction_id}
Sender: {row['sender_id']}
Receiver: {row['receiver_id']}
Amount: {row['amount']:.2f}
Fraud Probability: {row['fraud_probability']:.4f}
"""

        )


    # -----------------------------------------------------
    # Network interpretation notes
    # -----------------------------------------------------

    context_parts.append(

        """
NETWORK ANALYSIS NOTES:

- A high-degree account has many transaction connections.
- High incoming amounts may indicate a collection account.
- High outgoing amounts may indicate a distribution account.
- Multiple high-risk connections may indicate a suspicious network pattern.
- Network connectivity alone does not prove fraud.
- Model predictions are risk indicators, not confirmed fraud labels.
- Ground-truth fraud labels were not included in this context.
"""

    )


    return {

        "graph": G,

        "node_metrics": node_metrics,

        "suspicious_nodes": suspicious_nodes,

        "suspicious_edges": suspicious_edges,

        "context": "\n".join(context_parts),

        "statistics": {

            "total_nodes": total_nodes,

            "total_edges": total_edges,

            "density": density,

            "weak_components": weak_components,

            "largest_component": largest_component,

            "total_transactions": total_transactions,

            "high_risk_transactions": high_risk_transactions,

            "high_risk_rate": high_risk_rate,

            "total_amount": total_amount,

            "high_risk_amount": high_risk_amount

        }

    }
