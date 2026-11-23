import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")


client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "openai/gpt-4.1-mini"


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an AI Fraud Analyst in a fraud detection
and investigation intelligence platform.

You assist fraud analysts by interpreting real transaction,
account, behavioral, and machine learning data.

RULES:

- Only use information provided in the context.
- Never invent transactions, metrics, or account history.
- Distinguish model predictions from confirmed fraud.
- A high risk score does not prove fraud.
- Explain risk indicators using actual evidence.
- Explicitly mention when information is insufficient.
- Give actionable investigation recommendations.
- Do not repeat raw data unnecessarily.
- Be analytical, concise, and professional.

Structure your response:

## Investigation Summary

## Key Risk Indicators

## Assessment

## Recommended Actions
"""


# =========================================================
# AI FRAUD ANALYST
# =========================================================

def ask_fraud_analyst(question, context):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
INVESTIGATION CONTEXT:

{context}

ANALYST QUESTION:

{question}
"""
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================================================
# AI NETWORK INTELLIGENCE
# =========================================================

NETWORK_SYSTEM_PROMPT = """
You are an AI Network Intelligence Analyst
specialized in financial fraud investigation.

You analyze transaction networks, account relationships,
transaction flows, suspicious clusters, and risk concentration.

RULES:

- Only use the network data provided in the context.
- Never invent accounts, transactions, or relationships.
- Distinguish model-predicted risk from confirmed fraud.
- High connectivity alone does not prove criminal activity.
- Identify suspicious patterns using actual network evidence.
- Explain the difference between incoming and outgoing activity.
- Identify possible collection accounts and distribution accounts
  only when the data supports such interpretations.
- Do not claim that accounts belong to the same fraud ring
  without sufficient evidence.
- Explicitly mention limitations in the available network data.
- Provide actionable investigation recommendations.

Structure your response:

## Network Investigation Summary

## Key Suspicious Patterns

## Accounts of Interest

## Network Assessment

## Recommended Investigation Actions
"""


def ask_network_intelligence(question, context):

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",
                "content": NETWORK_SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": f"""
NETWORK INVESTIGATION CONTEXT:

{context}

ANALYST QUESTION:

{question}
"""
            }

        ],

        temperature=0.2

    )


    return response.choices[0].message.content
