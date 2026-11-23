
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from dashboard.utils.ui import apply_custom_ui

apply_custom_ui()

from src.case_management import (
    get_case_events,
    log_event
)
from dashboard.utils.model_utils import get_monitoring_data

from src.ai.network_intelligence import (
    build_network_intelligence_context
)

from src.ai.fraud_analyst import (
    ask_network_intelligence
)
from src.case_management import (
    create_case,
    get_all_cases,
    update_case_status,
    update_case_notes,
    delete_case
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Network Intelligence",
    page_icon="🕸️",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🕸️ AI Network Intelligence")

st.markdown(
    """
    Analyze transaction relationships, suspicious account connections,
    risk concentration and potential money-flow patterns using
    graph analytics and AI-powered investigation.
    """
)

st.markdown("---")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return get_monitoring_data()


try:

    df = load_data()


except Exception as e:

    st.error(
        f"Could not load transaction data: {e}"
    )

    st.stop()


if df.empty:

    st.warning(
        "No transaction data available."
    )

    st.stop()


# =========================================================
# BUILD NETWORK INTELLIGENCE
# =========================================================

try:

    network_result = (
        build_network_intelligence_context(
            df,
            risk_threshold=0.95,
            top_n=20
        )
    )


except Exception as e:

    st.error(
        f"Network analysis failed: {e}"
    )

    st.stop()


G = network_result["graph"]

node_metrics = network_result["node_metrics"]

suspicious_nodes = network_result["suspicious_nodes"]

suspicious_edges = network_result["suspicious_edges"]

network_context = network_result["context"]

stats = network_result["statistics"]


# =========================================================
# NETWORK OVERVIEW
# =========================================================

st.subheader("Network Overview")


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Accounts",
    f"{stats['total_nodes']:,}"
)


col2.metric(
    "Unique Connections",
    f"{stats['total_edges']:,}"
)


col3.metric(
    "High-Risk Transactions",
    f"{stats['high_risk_transactions']:,}"
)


col4.metric(
    "High-Risk Rate",
    f"{stats['high_risk_rate']:.2%}"
)


col1, col2, col3 = st.columns(3)


col1.metric(
    "Total Transaction Volume",
    f"{stats['total_amount']:,.2f}"
)


col2.metric(
    "High-Risk Transaction Volume",
    f"{stats['high_risk_amount']:,.2f}"
)


col3.metric(
    "Connected Components",
    f"{stats['weak_components']:,}"
)


st.markdown("---")


# =========================================================
# INTERACTIVE TRANSACTION NETWORK
# =========================================================

st.subheader("Interactive Transaction Network")

st.caption(
    "Visualizing high-risk transaction flows. "
    "Node size represents network activity, while node color "
    "indicates involvement in high-risk transactions."
)


# ---------------------------------------------------------
# GRAPH SETTINGS
# ---------------------------------------------------------

graph_col1, graph_col2 = st.columns(2)

with graph_col1:

    graph_risk_threshold = st.slider(
        "Fraud Probability Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.95,
        step=0.01,
        key="network_graph_threshold"
    )


with graph_col2:

    max_graph_transactions = st.slider(
        "Maximum Transactions to Display",
        min_value=20,
        max_value=150,
        value=75,
        step=5,
        key="network_graph_transaction_limit"
    )


# ---------------------------------------------------------
# PREPARE GRAPH DATA
# ---------------------------------------------------------

graph_df = df.copy()


if "fraud_probability" not in graph_df.columns:

    st.warning(
        "Fraud probability data is unavailable."
    )

else:

    graph_df["fraud_probability"] = pd.to_numeric(
        graph_df["fraud_probability"],
        errors="coerce"
    )


    graph_df = graph_df.dropna(
        subset=[
            "sender_id",
            "receiver_id",
            "fraud_probability"
        ]
    )


    # Select high-risk transactions
    high_risk_graph_df = graph_df[
        graph_df["fraud_probability"]
        >= graph_risk_threshold
    ].copy()


    # Sort by fraud probability
    high_risk_graph_df = high_risk_graph_df.sort_values(
        "fraud_probability",
        ascending=False
    )


    # Limit graph complexity
    high_risk_graph_df = high_risk_graph_df.head(
        max_graph_transactions
    )


    if high_risk_graph_df.empty:

        st.info(
            "No transactions found above the selected threshold."
        )


    else:

        # -------------------------------------------------
        # BUILD GRAPH
        # -------------------------------------------------

        graph = nx.DiGraph()


        for _, row in high_risk_graph_df.iterrows():

            sender = str(row["sender_id"])

            receiver = str(row["receiver_id"])


            graph.add_node(sender)

            graph.add_node(receiver)


            graph.add_edge(
                sender,
                receiver,
                amount=float(row["amount"]),
                fraud_probability=float(
                    row["fraud_probability"]
                ),
                transaction_id=str(
                    row.get(
                        "transaction_id",
                        "Unknown"
                    )
                )
            )


        # -------------------------------------------------
        # GRAPH LAYOUT
        # -------------------------------------------------

        positions = nx.spring_layout(
            graph,
            seed=42,
            k=0.8,
            iterations=50
        )

# ============================================================
# ACCOUNT-LEVEL NETWORK INVESTIGATION
# ============================================================

st.divider()
st.header("🔎 Account-Level Network Investigation")

account_options = sorted(
    set(df["sender_id"].astype(str).unique()) |
    set(df["receiver_id"].astype(str).unique())
)

selected_account = st.selectbox(
    "Select an account to investigate",
    account_options,
    key="network_selected_account"
)

account_df = df.copy()

account_df["sender_id"] = account_df["sender_id"].astype(str)
account_df["receiver_id"] = account_df["receiver_id"].astype(str)

account_transactions = account_df[
    (account_df["sender_id"] == selected_account) |
    (account_df["receiver_id"] == selected_account)
].copy()

incoming = account_transactions[
    account_transactions["receiver_id"] == selected_account
]

outgoing = account_transactions[
    account_transactions["sender_id"] == selected_account
]

if "fraud_probability" in account_transactions.columns:
    account_transactions["fraud_probability"] = pd.to_numeric(
        account_transactions["fraud_probability"],
        errors="coerce"
    ).fillna(0)

    high_risk_transactions = account_transactions[
        account_transactions["fraud_probability"] >= 0.95
    ]
else:
    high_risk_transactions = pd.DataFrame()

# Account metrics
total_transactions = len(account_transactions)
incoming_count = len(incoming)
outgoing_count = len(outgoing)

incoming_amount = incoming["amount"].sum() if not incoming.empty else 0
outgoing_amount = outgoing["amount"].sum() if not outgoing.empty else 0

high_risk_count = len(high_risk_transactions)

total_amount = (
    account_transactions["amount"].sum()
    if not account_transactions.empty
    else 0
)

# Connected accounts
connected_accounts = set()

connected_accounts.update(
    account_transactions.loc[
        account_transactions["sender_id"] == selected_account,
        "receiver_id"
    ].tolist()
)

connected_accounts.update(
    account_transactions.loc[
        account_transactions["receiver_id"] == selected_account,
        "sender_id"
    ].tolist()
)

connected_accounts.discard(selected_account)

# KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", total_transactions)
col2.metric("Connected Accounts", len(connected_accounts))
col3.metric("High-Risk Transactions", high_risk_count)
col4.metric("Total Transaction Volume", f"{total_amount:,.2f}")

col5, col6, col7 = st.columns(3)

col5.metric("Incoming Transactions", incoming_count)
col6.metric("Incoming Volume", f"{incoming_amount:,.2f}")
col7.metric("Outgoing Volume", f"{outgoing_amount:,.2f}")

# Transaction direction analysis
st.subheader("Transaction Flow")

flow_col1, flow_col2 = st.columns(2)

with flow_col1:
    st.markdown("### 📥 Incoming Transactions")

    st.write(f"**Transaction Count:** {incoming_count}")
    st.write(f"**Total Amount:** {incoming_amount:,.2f}")

    if not incoming.empty:
        st.dataframe(
            incoming.sort_values(
                "fraud_probability",
                ascending=False
            ) if "fraud_probability" in incoming.columns else incoming,
            use_container_width=True,
            hide_index=True
        )

with flow_col2:
    st.markdown("### 📤 Outgoing Transactions")

    st.write(f"**Transaction Count:** {outgoing_count}")
    st.write(f"**Total Amount:** {outgoing_amount:,.2f}")

    if not outgoing.empty:
        st.dataframe(
            outgoing.sort_values(
                "fraud_probability",
                ascending=False
            ) if "fraud_probability" in outgoing.columns else outgoing,
            use_container_width=True,
            hide_index=True
        )

# Connected account analysis
st.subheader("Connected Accounts")

if connected_accounts:

    connection_data = []

    for connected_account in connected_accounts:

        related_transactions = account_transactions[
            (
                account_transactions["sender_id"] == connected_account
            ) |
            (
                account_transactions["receiver_id"] == connected_account
            )
        ]

        connection_data.append({
            "Account": connected_account,
            "Transaction Count": len(related_transactions),
            "Total Amount": related_transactions["amount"].sum(),
            "High-Risk Transactions": (
                len(
                    related_transactions[
                        related_transactions["fraud_probability"] >= 0.95
                    ]
                )
                if "fraud_probability" in related_transactions.columns
                else 0
            )
        })

    connections_df = pd.DataFrame(connection_data)

    connections_df = connections_df.sort_values(
        "High-Risk Transactions",
        ascending=False
    )

    st.dataframe(
        connections_df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No connected accounts found.")

# Account-level risk transaction list
st.subheader("🚨 High-Risk Activity")

if not high_risk_transactions.empty:

    st.dataframe(
        high_risk_transactions.sort_values(
            "fraud_probability",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No transactions above the 0.95 risk threshold.")

# ============================================================
# ACCOUNT AI INVESTIGATION
# ============================================================

st.divider()

st.subheader("🤖 AI Account Investigation")

account_context = f"""
ACCOUNT INVESTIGATION

Selected Account: {selected_account}

TOTAL ACTIVITY
- Total transactions: {total_transactions}
- Total transaction volume: {total_amount:.2f}
- Connected accounts: {len(connected_accounts)}

INCOMING ACTIVITY
- Incoming transactions: {incoming_count}
- Incoming volume: {incoming_amount:.2f}

OUTGOING ACTIVITY
- Outgoing transactions: {outgoing_count}
- Outgoing volume: {outgoing_amount:.2f}

RISK
- Transactions with fraud probability >= 0.95: {high_risk_count}

CONNECTED ACCOUNTS
{connections_df.to_string(index=False) if connected_accounts else "No connected accounts"}

HIGH-RISK TRANSACTIONS
{high_risk_transactions.head(20).to_string(index=False) if not high_risk_transactions.empty else "No high-risk transactions"}
"""

account_question = st.text_area(
    "Ask about this account",
    placeholder="Why is this account suspicious? Does it appear to be a money mule? Analyze its transaction flow...",
    key="account_ai_question"
)

if st.button(
    "Analyze Selected Account",
    key="analyze_selected_account"
):

    if account_question.strip():

        with st.spinner("AI is investigating this account..."):

            try:
                response = ask_network_intelligence(
                    account_question,
                    account_context
                )

                st.markdown("### AI Investigation Report")
                st.markdown(response)

            except Exception as e:
                st.error(f"AI analysis failed: {e}")

    else:
        st.warning("Please enter a question first.")
        # -------------------------------------------------
        # NODE METRICS
        # -------------------------------------------------

        node_risk_counts = {}


        for node in graph.nodes():

            node_risk_counts[node] = 0


        for _, row in high_risk_graph_df.iterrows():

            sender = str(row["sender_id"])

            receiver = str(row["receiver_id"])


            node_risk_counts[sender] += 1

            node_risk_counts[receiver] += 1


        node_degrees = dict(
            graph.degree()
        )


        # -------------------------------------------------
        # EDGE TRACE
        # -------------------------------------------------

        edge_x = []

        edge_y = []

        edge_hover = []


        for source, target, data in graph.edges(data=True):

            x0, y0 = positions[source]

            x1, y1 = positions[target]


            edge_x.extend(
                [x0, x1, None]
            )


            edge_y.extend(
                [y0, y1, None]
            )


            edge_hover.append(

                f"Transaction: {data['transaction_id']}<br>"
                f"Amount: {data['amount']:,.2f}<br>"
                f"Fraud Probability: "
                f"{data['fraud_probability']:.2%}"

            )


        edge_trace = go.Scatter(

            x=edge_x,

            y=edge_y,

            mode="lines",

            line=dict(
                width=1.2,
                color="rgba(255,80,80,0.35)"
            ),

            hoverinfo="none",

            showlegend=False

        )


        # -------------------------------------------------
        # NODE TRACE
        # -------------------------------------------------

        node_x = []

        node_y = []

        node_text = []

        node_sizes = []

        node_colors = []

        node_hover = []


        for node in graph.nodes():

            x, y = positions[node]


            node_x.append(x)

            node_y.append(y)


            degree = node_degrees[node]

            risk_count = node_risk_counts[node]


            # Node size
            node_sizes.append(
                15 + degree * 5
            )


            # Node color
            node_colors.append(
                risk_count
            )


            node_text.append(
                node
            )


            node_hover.append(

                f"Account: {node}<br>"
                f"Network Degree: {degree}<br>"
                f"High-Risk Connections: {risk_count}"

            )


        node_trace = go.Scatter(

            x=node_x,

            y=node_y,

            mode="markers+text",

            text=node_text,

            textposition="top center",

            textfont=dict(
                size=9
            ),

            marker=dict(

                size=node_sizes,

                color=node_colors,

                colorscale="Reds",

                colorbar=dict(
                    title="High-Risk<br>Connections"
                ),

                line=dict(
                    width=1,
                    color="white"
                ),

                opacity=0.9

            ),

            hovertext=node_hover,

            hoverinfo="text",

            showlegend=False

        )


        # -------------------------------------------------
        # FIGURE
        # -------------------------------------------------

        fig = go.Figure(

            data=[
                edge_trace,
                node_trace
            ]

        )


        fig.update_layout(

            height=700,

            title="High-Risk Transaction Network",

            showlegend=False,

            hovermode="closest",

            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            ),

            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),

            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),

            template="plotly_dark"

        )


        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -------------------------------------------------
        # GRAPH SUMMARY
        # -------------------------------------------------

        graph_metric_col1, graph_metric_col2, graph_metric_col3 = (
            st.columns(3)
        )


        graph_metric_col1.metric(
            "Graph Accounts",
            graph.number_of_nodes()
        )


        graph_metric_col2.metric(
            "Graph Connections",
            graph.number_of_edges()
        )


        graph_metric_col3.metric(
            "Displayed Transactions",
            len(high_risk_graph_df)
        )


        with st.expander(
            "View transactions included in the graph"
        ):

            graph_display_columns = [

                "transaction_id",

                "sender_id",

                "receiver_id",

                "amount",

                "fraud_probability"

            ]


            graph_available_columns = [

                col

                for col in graph_display_columns

                if col in high_risk_graph_df.columns

            ]


            st.dataframe(

                high_risk_graph_df[
                    graph_available_columns
                ],

                use_container_width=True,

                hide_index=True

            )


# =========================================================
# TOP SUSPICIOUS ACCOUNTS
# =========================================================

st.subheader("Accounts of Interest")

st.caption(
    "Accounts ranked using high-risk transaction involvement, "
    "network degree and transaction volume."
)


if suspicious_nodes.empty:

    st.info(
        "No suspicious accounts identified."
    )


else:

    display_node_metrics = suspicious_nodes.reset_index()

    display_node_metrics = display_node_metrics.rename(
        columns={
            "index": "Account ID"
        }
    )


    preferred_columns = [

        "Account ID",

        "incoming_transactions",

        "outgoing_transactions",

        "incoming_amount",

        "outgoing_amount",

        "incoming_high_risk_count",

        "outgoing_high_risk_count",

        "total_high_risk_transactions",

        "in_degree",

        "out_degree",

        "total_degree"

    ]


    available_columns = [

        col

        for col in preferred_columns

        if col in display_node_metrics.columns

    ]


    st.dataframe(

        display_node_metrics[available_columns],

        use_container_width=True,

        hide_index=True

    )


st.markdown("---")


# =========================================================
# HIGH-RISK TRANSACTION CONNECTIONS
# =========================================================

st.subheader("High-Risk Transaction Connections")


if suspicious_edges.empty:

    st.info(
        "No high-risk transaction connections found."
    )


else:

    edge_columns = [

        "transaction_id",

        "sender_id",

        "receiver_id",

        "amount",

        "fraud_probability",

        "model_prediction"

    ]


    available_edge_columns = [

        col

        for col in edge_columns

        if col in suspicious_edges.columns

    ]


    st.dataframe(

        suspicious_edges[available_edge_columns],

        use_container_width=True,

        hide_index=True

    )


st.markdown("---")

# ============================================================
# AUTOMATED FRAUD INVESTIGATION REPORT
# ============================================================

st.subheader("📋 Automated Investigation Report")

if st.button(
    "Generate Investigation Report",
    key="generate_investigation_report"
):

    investigation_question = """
    Generate a structured fraud investigation report for the selected account.

    Analyze the available evidence and cover the following sections:

    1. Executive Summary
    2. Risk Assessment
    3. Suspicious Behavioral Patterns
    4. Incoming and Outgoing Transaction Analysis
    5. Network Relationships
    6. Potential Money Mule or Funnel Account Indicators
    7. Evidence Supporting Suspicion
    8. Recommended Investigation Actions
    9. Confidence Level and Limitations

    Important:
    - Do not claim fraud is confirmed.
    - Distinguish observed evidence from hypotheses.
    - Do not invent missing information.
    - Explain why the account deserves investigation.
    - Base the report exclusively on the supplied data.
    """

    with st.spinner("Generating investigation report..."):

        try:

            report = ask_network_intelligence(
                investigation_question,
                account_context
            )

            st.session_state["investigation_report"] = report
            st.session_state["investigation_report_account"] = selected_account


        except Exception as e:

            st.error(f"Report generation failed: {e}")

# Add report generation event to an existing case
existing_cases = get_all_cases()

account_cases = existing_cases[
    existing_cases["account_id"] == selected_account
]

if not account_cases.empty:

    latest_case = account_cases.iloc[0]

    log_event(
        latest_case["case_id"],
        "AI_REPORT_GENERATED",
        "AI-generated investigation report created for the selected account."
    )

# Display generated report
if (
    "investigation_report" in st.session_state
    and st.session_state.get("investigation_report_account") == selected_account
):

    report = st.session_state["investigation_report"]

    st.markdown("---")
    st.markdown(report)

    st.download_button(
        label="📥 Download Investigation Report",
        data=f"""
FRAUD INVESTIGATION REPORT

Account: {selected_account}

{report}
""",
        file_name=f"investigation_report_{selected_account}.md",
        mime="text/markdown",
        key="download_investigation_report"
    )

# =========================================================
# AI NETWORK INVESTIGATION
# =========================================================

st.subheader("🧠 AI Network Analyst")


st.caption(
    "The AI analyzes network structure, account activity "
    "and model-predicted risk using the available data."
)


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

quick_questions = [

    "Identify the most suspicious accounts in this network and explain why.",

    "Are there potential collection or distribution accounts in this network?",

    "Analyze the concentration of high-risk transactions between accounts.",

    "What suspicious money-flow patterns can be identified?",

    "Which accounts should investigators prioritize and why?",

    "Does this network provide enough evidence to suggest coordinated fraud? "

]


selected_prompt = st.selectbox(

    "Suggested investigation questions",

    [
        "Select a question..."
    ] + quick_questions

)


# =========================================================
# CHAT HISTORY
# =========================================================

if "network_analyst_messages" not in st.session_state:

    st.session_state.network_analyst_messages = []


# =========================================================
# DISPLAY HISTORY
# =========================================================

for message in st.session_state.network_analyst_messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input(
    "Ask the AI Network Analyst..."
)


quick_prompt_clicked = st.button(
    "Analyze Selected Question",
    key="analyze_network_prompt"
)


# =========================================================
# DETERMINE QUESTION
# =========================================================

question_to_send = None


if user_input:

    question_to_send = user_input


elif quick_prompt_clicked:

    if selected_prompt != "Select a question...":

        question_to_send = selected_prompt

    else:

        st.warning(
            "Please select a question first."
        )


# =========================================================
# SEND QUESTION
# =========================================================

if question_to_send:


    # Save user message

    st.session_state.network_analyst_messages.append(

        {

            "role": "user",

            "content": question_to_send

        }

    )


    with st.chat_message("user"):

        st.markdown(
            question_to_send
        )


    # Previous conversation

    previous_messages = (

        st.session_state.network_analyst_messages[:-1]

    )


    conversation_context = ""


    for message in previous_messages:

        conversation_context += (

            f"\n{message['role'].upper()}: "

            f"{message['content']}\n"

        )


    complete_context = f"""

NETWORK DATA:

{network_context}


PREVIOUS CONVERSATION:

{conversation_context}

"""


    # AI response

    with st.chat_message("assistant"):

        with st.spinner(
            "AI Network Analyst is analyzing the network..."
        ):

            try:

                response = ask_network_intelligence(

                    question=question_to_send,

                    context=complete_context

                )


                st.markdown(
                    response
                )


                st.session_state.network_analyst_messages.append(

                    {

                        "role": "assistant",

                        "content": response

                    }

                )


            except Exception as e:

                st.error(
                    f"AI network analysis failed: {e}"
                )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🕸️ Network Intelligence"
    )


    st.markdown(

        """
        **Capabilities**

        - Account relationship analysis
        - High-risk connection detection
        - Network risk concentration
        - Incoming/outgoing flow analysis
        - Suspicious account identification
        - Investigation prioritization
        """

    )


    st.markdown("---")


    st.markdown(

        """
        **Important**

        Network relationships and model risk
        predictions are investigative indicators.
        They do not independently confirm fraud
        or coordinated criminal activity.
        """

    )


    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.network_analyst_messages = []

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")


st.caption(
    "AI Network Intelligence | NetworkX | OpenRouter | "
    "Fraud Investigation Intelligence Platform"
)

# ============================================================
# CASE MANAGEMENT
# ============================================================

st.divider()

st.header("📁 Fraud Case Management")

# Determine account risk score
if "fraud_probability" in account_transactions.columns:

    account_risk_score = (
        account_transactions["fraud_probability"].max() * 100
        if not account_transactions.empty
        else 0
    )

else:
    account_risk_score = 0


if account_risk_score >= 95:
    severity = "Critical"
elif account_risk_score >= 80:
    severity = "High"
elif account_risk_score >= 60:
    severity = "Medium"
else:
    severity = "Low"


st.subheader("Create Investigation Case")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Account", selected_account)
col2.metric("Risk Score", f"{account_risk_score:.2f}/100")
col3.metric("Severity", severity)

initial_notes = st.text_area(
    "Initial Investigation Notes",
    placeholder="Describe the reason for opening this case...",
    key="new_case_notes"
)

if st.button("➕ Create Case", key="create_fraud_case"):

    case_id = create_case(
        account_id=selected_account,
        risk_score=account_risk_score,
        severity=severity,
        notes=initial_notes
    )

    st.success(f"Case successfully created: {case_id}")

    st.rerun()


# ============================================================
# EXISTING CASES
# ============================================================

st.subheader("Investigation Cases")

cases_df = get_all_cases()

if not cases_df.empty:

    # Status summary
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Cases",
        len(cases_df)
    )

    col2.metric(
        "Open",
        len(cases_df[cases_df["status"] == "Open"])
    )

    col3.metric(
        "Investigating",
        len(cases_df[cases_df["status"] == "Investigating"])
    )

    col4.metric(
        "Escalated",
        len(cases_df[cases_df["status"] == "Escalated"])
    )

    st.dataframe(
        cases_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # CASE MANAGEMENT ACTIONS
    # ========================================================

    st.subheader("Manage Existing Case")

    selected_case = st.selectbox(
        "Select Case",
        cases_df["case_id"].tolist(),
        key="manage_selected_case"
    )

    case_row = cases_df[
        cases_df["case_id"] == selected_case
    ].iloc[0]

    st.write(f"**Account:** {case_row['account_id']}")
    st.write(f"**Severity:** {case_row['severity']}")
    st.write(f"**Risk Score:** {case_row['risk_score']}")

    # Update status
    new_status = st.selectbox(
        "Update Status",
        [
            "Open",
            "Investigating",
            "Escalated",
            "Closed"
        ],
        index=[
            "Open",
            "Investigating",
            "Escalated",
            "Closed"
        ].index(case_row["status"]),
        key="case_status_update"
    )

    if st.button("Update Case Status"):

        update_case_status(
            selected_case,
            new_status
        )

        st.success("Case status updated.")
        st.rerun()

    # Update notes
    updated_notes = st.text_area(
        "Analyst Notes",
        value=case_row["analyst_notes"],
        key="case_notes_update"
    )

    if st.button("Save Analyst Notes"):

        update_case_notes(
            selected_case,
            updated_notes
        )

        st.success("Analyst notes saved.")
        st.rerun()

    # Delete case
    with st.expander("Delete Case"):

        st.warning("This action permanently deletes the selected case.")

        if st.button(
            "Delete Selected Case",
            type="primary"
        ):

            delete_case(selected_case)

            st.success("Case deleted.")
            st.rerun()

else:

    st.info("No investigation cases have been created yet.")

# ============================================================
# INVESTIGATION TIMELINE
# ============================================================

st.divider()

st.header("🕒 Investigation Timeline")

cases_df = get_all_cases()

if not cases_df.empty:

    timeline_case = st.selectbox(
        "Select case for timeline",
        cases_df["case_id"].tolist(),
        key="timeline_selected_case"
    )

    selected_case_info = cases_df[
        cases_df["case_id"] == timeline_case
    ].iloc[0]

    st.write(
        f"Account: **{selected_case_info['account_id']}**"
    )

    events_df = get_case_events(timeline_case)

    if not events_df.empty:

        for _, event in events_df.iterrows():

            with st.container(border=True):

                col1, col2 = st.columns([1, 4])

                col1.markdown(
                    f"**{event['event_type']}**"
                )

                col2.caption(
                    event["created_at"]
                )

                st.write(
                    event["event_description"]
                )

    else:

        st.info("No events recorded for this case.")

else:

    st.info("Create a case first to view its investigation timeline.")