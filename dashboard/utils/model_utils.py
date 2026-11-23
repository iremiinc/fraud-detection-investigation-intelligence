# dashboard/utils/model_utils.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "all_transactions.csv"
MODEL_DATA_PATH = BASE_DIR / "data" / "processed" / "model_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "lightgbm_model.pkl"


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_NAME = "LightGBM"
DECISION_THRESHOLD = 0.95

FEATURES = [
    "amount",
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
    "receiver_prior_tx_count",
]


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# =========================================================
# LOAD TRANSACTION DATA
# =========================================================

@st.cache_data
def load_transaction_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Transaction data not found: {DATA_PATH}"
        )

    if not MODEL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Model dataset not found: {MODEL_DATA_PATH}"
        )

    # Original transactions
    transactions = pd.read_csv(DATA_PATH)

    # Model features
    model_data = pd.read_csv(MODEL_DATA_PATH)

    # Timestamp
    if "timestamp" in transactions.columns:
        transactions["timestamp"] = pd.to_datetime(
            transactions["timestamp"],
            errors="coerce"
        )

    # Required columns
    required_columns = ["transaction_id"] + FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in model_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    # Keep only transaction ID and model features
    model_data = model_data[required_columns].copy()

    # Prevent duplicate transaction IDs
    model_data = model_data.drop_duplicates(
        subset="transaction_id"
    )

    # Normalize transaction IDs before merging
    transactions["transaction_id"] = (
        transactions["transaction_id"].astype(str)
    )

    model_data["transaction_id"] = (
        model_data["transaction_id"].astype(str)
    )

    # Remove overlapping feature columns
    overlapping_features = [
        column
        for column in FEATURES
        if column in transactions.columns
    ]

    transactions = transactions.drop(
        columns=overlapping_features
    )

    # Merge original transaction information
    # with engineered model features
    df = transactions.merge(
        model_data,
        on="transaction_id",
        how="left",
        validate="one_to_one"
    )

    return df


# =========================================================
# MODEL PREDICTIONS
# =========================================================

@st.cache_data
def generate_predictions():

    df = load_transaction_data()
    model = load_model()

    # Feature matrix
    X = df[FEATURES].copy()

    # LightGBM supports NaN values
    X_array = X.to_numpy(dtype=np.float64)

    # Fraud probability
    probabilities = model.predict_proba(X_array)[:, 1]

    df["fraud_probability"] = probabilities

    # Risk score: 0-100
    df["risk_score"] = df["fraud_probability"] * 100

    # Model prediction
    df["model_prediction"] = (
        df["fraud_probability"] >= DECISION_THRESHOLD
    ).astype(int)

    # =====================================================
    # RISK LEVEL
    # =====================================================

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, 40, 70, 95, 100.01],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True
    )

    # =====================================================
    # PRIORITY SCORE
    # =====================================================

    # Amount percentile
    amount_percentile = (
        df["amount"]
        .rank(pct=True)
        .fillna(0)
        * 100
    )

    # Risk + transaction amount
    df["priority_score"] = (
        df["risk_score"] * 0.70
        + amount_percentile * 0.30
    )

    df["priority"] = pd.cut(
        df["priority_score"],
        bins=[-0.01, 30, 55, 80, 100.01],
        labels=["P4 - Low", "P3 - Medium", "P2 - High", "P1 - Urgent"],
        include_lowest=True
    )

    # =====================================================
    # MODEL STATUS
    # =====================================================

    df["model_status"] = np.where(
        df["model_prediction"] == 1,
        "Flagged",
        "Not Flagged"
    )

    # =====================================================
    # GROUND TRUTH STATUS
    # =====================================================

    if "is_fraud" in df.columns:

        df["ground_truth_status"] = np.where(
            df["is_fraud"] == 1,
            "Fraud",
            "Legitimate"
        )

    return df


# =========================================================
# DASHBOARD DATA
# =========================================================

def get_monitoring_data():

    return generate_predictions()