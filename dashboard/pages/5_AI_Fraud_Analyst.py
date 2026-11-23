
import streamlit as st
import pandas as pd
import numpy as np
from dashboard.utils.ui import apply_custom_ui

apply_custom_ui()

from dashboard.utils.model_utils import get_monitoring_data
from src.ai.context_builder import build_transaction_context
from src.ai.fraud_analyst import ask_fraud_analyst
from src.ai.context_builder import (
    build_transaction_context,
    build_shap_context
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Fraud Analyst",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Fraud Analyst")

st.markdown(
    """
    Investigate suspicious transactions using AI-powered
    analysis grounded in transaction data and machine learning
    predictions.
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
    st.error(f"Could not load monitoring data: {e}")
    st.stop()


if df.empty:
    st.warning("No transaction data available.")
    st.stop()


# =========================================================
# TRANSACTION SELECTION
# =========================================================

st.subheader("Select Transaction")

sort_column = (
    "fraud_probability"
    if "fraud_probability" in df.columns
    else "risk_score"
)

if sort_column in df.columns:

    sorted_df = df.sort_values(
        sort_column,
        ascending=False
    )

else:

    sorted_df = df.copy()


# Make sure transaction ID column exists
if "transaction_id" in sorted_df.columns:

    transaction_id_column = "transaction_id"

else:

    sorted_df = sorted_df.reset_index()

    transaction_id_column = "index"


transaction_options = (
    sorted_df[transaction_id_column]
    .astype(str)
    .tolist()
)


selected_transaction_id = st.selectbox(
    "Choose a transaction to investigate",
    transaction_options
)


selected_transaction = sorted_df[
    sorted_df[transaction_id_column].astype(str)
    == selected_transaction_id
].iloc[0]


# =========================================================
# TRANSACTION OVERVIEW
# =========================================================

st.markdown("### Transaction Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    amount = selected_transaction.get(
        "amount",
        0
    )

    st.metric(
        "Transaction Amount",
        f"{amount:,.2f}"
        if pd.notna(amount)
        else "N/A"
    )


with col2:

    probability = selected_transaction.get(
        "fraud_probability",
        None
    )

    if probability is not None and pd.notna(probability):

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    else:

        st.metric(
            "Fraud Probability",
            "N/A"
        )


with col3:

    risk_score = selected_transaction.get(
        "risk_score",
        None
    )

    st.metric(
        "Risk Score",
        f"{risk_score:.1f}"
        if risk_score is not None
        and pd.notna(risk_score)
        else "N/A"
    )


with col4:

    risk_level = selected_transaction.get(
        "risk_level",
        "Unknown"
    )

    st.metric(
        "Risk Level",
        str(risk_level)
    )


# =========================================================
# TRANSACTION DETAILS
# =========================================================

with st.expander(
    "View Transaction Details",
    expanded=True
):

    display_fields = [
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
        "transaction_type",
        "fraud_probability",
        "risk_score",
        "risk_level",
        "model_prediction",
        "priority",
        "priority_score"
    ]

    available_fields = [
        col for col in display_fields
        if col in selected_transaction.index
    ]

    details = pd.DataFrame({
        "Field": available_fields,
        "Value": [
            selected_transaction[col]
            for col in available_fields
        ]
    })

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# BUILD AI CONTEXT
# =========================================================


# =========================================================
# BUILD AI CONTEXT
# =========================================================

try:

    # Build basic transaction context
    transaction_context = build_transaction_context(
        selected_transaction
    )


    # Calculate SHAP explanation
    shap_context = build_shap_context(
        selected_transaction
    )


    # Rebuild context with SHAP information
    transaction_context = build_transaction_context(
        selected_transaction,
        shap_context=shap_context
    )


except Exception as e:

    st.warning(
        f"SHAP explanation could not be generated: {e}"
    )

    # Continue with basic transaction context
    transaction_context = build_transaction_context(
        selected_transaction
    )



# =========================================================
# AI INVESTIGATION
# =========================================================

st.markdown("---")

st.subheader("🧠 AI Investigation")

st.caption(
    "The AI analyst uses the selected transaction's actual "
    "data and model outputs to provide an investigation assessment."
)


# =========================================================
# QUICK PROMPTS
# =========================================================

quick_questions = [
    "Analyze this transaction and explain why it is suspicious.",
    "Identify the key fraud indicators in this transaction.",
    "What investigation actions should an analyst take?",
    "Explain this transaction in simple terms.",
    "What additional information would help investigate this transaction?",
    "This transaction has a high fraud probability. Does this necessarily mean it is fraudulent? Explain your assessment based on the available evidence."
]


selected_prompt = st.selectbox(
    "Suggested questions",
    ["Select a suggested question..."] + quick_questions
)


# =========================================================
# CHAT HISTORY
# =========================================================

if "fraud_analyst_messages" not in st.session_state:

    st.session_state.fraud_analyst_messages = []


# Reset chat when transaction changes

if "previous_transaction_id" not in st.session_state:

    st.session_state.previous_transaction_id = (
        selected_transaction_id
    )


if (
    st.session_state.previous_transaction_id
    != selected_transaction_id
):

    st.session_state.fraud_analyst_messages = []

    st.session_state.previous_transaction_id = (
        selected_transaction_id
    )


# =========================================================
# DISPLAY PREVIOUS MESSAGES
# =========================================================

for message in st.session_state.fraud_analyst_messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask the AI Fraud Analyst about this transaction..."
)


# =========================================================
# QUICK PROMPT BUTTON
# =========================================================

quick_prompt_clicked = st.button(
    "Analyze Selected Question",
    key="analyze_quick_prompt"
)


# =========================================================
# DETERMINE QUESTION
# =========================================================

question_to_send = None


# Manual input has priority
if user_input:

    question_to_send = user_input


# Quick prompt is used only when its button is clicked
elif quick_prompt_clicked:

    if selected_prompt != "Select a suggested question...":

        question_to_send = selected_prompt

    else:

        st.warning(
            "Please select a suggested question first."
        )


# =========================================================
# SEND QUESTION
# =========================================================

if question_to_send:

    # -----------------------------------------------------
    # Save and display user question
    # -----------------------------------------------------

    st.session_state.fraud_analyst_messages.append(
        {
            "role": "user",
            "content": question_to_send
        }
    )

    with st.chat_message("user"):

        st.markdown(question_to_send)


    # -----------------------------------------------------
    # Prepare conversation history
    # -----------------------------------------------------

    previous_messages = (
        st.session_state.fraud_analyst_messages[:-1]
    )

    conversation_context = ""

    for message in previous_messages:

        conversation_context += (
            f"\n{message['role'].upper()}: "
            f"{message['content']}\n"
        )


    # -----------------------------------------------------
    # Complete context
    # -----------------------------------------------------

    complete_context = f"""
CURRENT TRANSACTION CONTEXT:

{transaction_context}

PREVIOUS CONVERSATION:

{conversation_context}
"""


    # -----------------------------------------------------
    # AI RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "AI Fraud Analyst is investigating..."
        ):

            try:

                response = ask_fraud_analyst(
                    question=question_to_send,
                    context=complete_context
                )

                st.markdown(response)

                st.session_state.fraud_analyst_messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

            except Exception as e:

                st.error(
                    f"AI analysis failed: {e}"
                )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🤖 AI Fraud Analyst")

    st.markdown(
        """
        **Capabilities**

        - Transaction risk analysis
        - Fraud indicator identification
        - Investigation recommendations
        - Behavioral analysis
        - Model prediction interpretation
        """
    )

    st.markdown("---")

    st.markdown(
        """
        **Important**

        AI-generated analysis is decision support.
        A model prediction does not independently
        confirm fraudulent activity.
        """
    )

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.fraud_analyst_messages = []

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Fraud Analyst | OpenRouter | "
    "LightGBM-powered Fraud Detection Platform"
)
