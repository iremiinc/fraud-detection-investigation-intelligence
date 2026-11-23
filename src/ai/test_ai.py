from dashboard.utils.model_utils import get_monitoring_data

from src.ai.context_builder import build_transaction_context
from src.ai.fraud_analyst import ask_fraud_analyst


# =========================================================
# LOAD REAL DASHBOARD DATA
# =========================================================

df = get_monitoring_data()

print("Dataset shape:", df.shape)


# =========================================================
# SELECT A REAL HIGH-RISK TRANSACTION
# =========================================================

high_risk_transactions = df.sort_values(
    "fraud_probability",
    ascending=False
)

transaction = high_risk_transactions.iloc[0]


print("\nSelected transaction:")
print(transaction)


# =========================================================
# BUILD CONTEXT
# =========================================================

context = build_transaction_context(transaction)


print("\nGenerated context:")
print(context)


# =========================================================
# ASK AI ANALYST
# =========================================================

question = """
Analyze this transaction.

Explain why it is considered high risk,
identify its main suspicious indicators,
and recommend investigation actions.
"""

response = ask_fraud_analyst(
    question=question,
    context=context
)


# =========================================================
# OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("AI FRAUD ANALYST")
print("=" * 60)

print(response)