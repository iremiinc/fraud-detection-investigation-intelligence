
import pandas as pd
import numpy as np
import joblib
import shap

from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "lightgbm_model.pkl"


# =========================================================
# MODEL FEATURES
# =========================================================

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
    "receiver_prior_tx_count"
]


# =========================================================
# SHAP EXPLANATION
# =========================================================

def build_shap_context(transaction):

    """
    Calculate SHAP contributions for a selected transaction
    and return a human-readable explanation context.
    """

    if isinstance(transaction, pd.Series):

        transaction = transaction.to_dict()


    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = joblib.load(MODEL_PATH)


    # -----------------------------------------------------
    # Prepare transaction features
    # -----------------------------------------------------

    feature_values = {}

    for feature in FEATURES:

        feature_values[feature] = transaction.get(
            feature,
            0
        )


    X_transaction = pd.DataFrame(
        [feature_values],
        columns=FEATURES
    )


    # Convert to NumPy to avoid feature-name mismatch
    X_numpy = X_transaction.to_numpy()


    # -----------------------------------------------------
    # SHAP Explainer
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_numpy
    )


    # -----------------------------------------------------
    # Handle binary classification outputs
    # -----------------------------------------------------

    if isinstance(shap_values, list):

        shap_values = shap_values[1]


    shap_values = np.array(shap_values)


    if shap_values.ndim == 3:

        shap_values = shap_values[:, :, 1]


    shap_values = shap_values.reshape(-1)


    # -----------------------------------------------------
    # Build SHAP DataFrame
    # -----------------------------------------------------

    shap_df = pd.DataFrame({

        "Feature": FEATURES,

        "Feature Value": [
            feature_values[feature]
            for feature in FEATURES
        ],

        "SHAP Contribution": shap_values,

        "Absolute Contribution": np.abs(
            shap_values
        )

    })


    shap_df = shap_df.sort_values(
        "Absolute Contribution",
        ascending=False
    )


    # -----------------------------------------------------
    # Positive / Negative Contributions
    # -----------------------------------------------------

    shap_df["Impact"] = np.where(
        shap_df["SHAP Contribution"] > 0,
        "Increases fraud risk",
        "Decreases fraud risk"
    )


    # -----------------------------------------------------
    # Format Context
    # -----------------------------------------------------

    context_parts = []

    context_parts.append(
        "MODEL EXPLANATION USING SHAP"
    )


    context_parts.append(
        "\nTOP FEATURES INFLUENCING THE MODEL:"
    )


    for _, row in shap_df.head(10).iterrows():

        context_parts.append(

            f"""
Feature: {row['Feature']}
Feature Value: {row['Feature Value']}
SHAP Contribution: {row['SHAP Contribution']:.6f}
Impact: {row['Impact']}
"""

        )


    # -----------------------------------------------------
    # Overall interpretation
    # -----------------------------------------------------

    positive_features = shap_df[
        shap_df["SHAP Contribution"] > 0
    ].sort_values(
        "SHAP Contribution",
        ascending=False
    )


    negative_features = shap_df[
        shap_df["SHAP Contribution"] < 0
    ].sort_values(
        "SHAP Contribution",
        ascending=True
    )


    context_parts.append(
        "\nFEATURES INCREASING FRAUD RISK:"
    )


    for _, row in positive_features.head(5).iterrows():

        context_parts.append(

            f"{row['Feature']}: "
            f"value={row['Feature Value']}, "
            f"SHAP={row['SHAP Contribution']:.6f}"

        )


    context_parts.append(
        "\nFEATURES DECREASING FRAUD RISK:"
    )


    for _, row in negative_features.head(5).iterrows():

        context_parts.append(

            f"{row['Feature']}: "
            f"value={row['Feature Value']}, "
            f"SHAP={row['SHAP Contribution']:.6f}"

        )


    return "\n".join(context_parts)


# =========================================================
# TRANSACTION CONTEXT
# =========================================================

def build_transaction_context(
    transaction,
    account_context=None,
    shap_context=None
):

    if isinstance(transaction, pd.Series):

        transaction = transaction.to_dict()


    context_parts = []


    # -----------------------------------------------------
    # Transaction Information
    # -----------------------------------------------------

    context_parts.append(
        "TRANSACTION INFORMATION"
    )


    transaction_fields = [

        "transaction_id",
        "timestamp",
        "amount",
        "transaction_type",
        "sender_id",
        "receiver_id",
        "hour",
        "day_of_week",
        "is_weekend"

    ]


    for field in transaction_fields:

        if field in transaction:

            context_parts.append(
                f"{field}: {transaction[field]}"
            )


    # -----------------------------------------------------
    # Model Assessment
    # -----------------------------------------------------

    context_parts.append(
        "\nMODEL ASSESSMENT"
    )


    model_fields = [

        "fraud_probability",
        "risk_score",
        "model_prediction",
        "risk_level",
        "priority",
        "priority_score"

    ]


    for field in model_fields:

        if field in transaction:

            context_parts.append(
                f"{field}: {transaction[field]}"
            )


    # -----------------------------------------------------
    # Transaction Behavior
    # -----------------------------------------------------

    context_parts.append(
        "\nTRANSACTION BEHAVIOR"
    )


    behavioral_fields = [

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


    for field in behavioral_fields:

        if field in transaction:

            context_parts.append(
                f"{field}: {transaction[field]}"
            )


    # -----------------------------------------------------
    # Account Context
    # -----------------------------------------------------

    if account_context:

        context_parts.append(
            "\nACCOUNT INVESTIGATION CONTEXT"
        )


        for key, value in account_context.items():

            context_parts.append(
                f"{key}: {value}"
            )


    # -----------------------------------------------------
    # SHAP Explanation
    # -----------------------------------------------------

    if shap_context is not None:

        context_parts.append(
            "\nMODEL EXPLANATION (SHAP)"
        )


        if isinstance(shap_context, str):

            context_parts.append(
                shap_context
            )


        elif isinstance(shap_context, pd.DataFrame):

            shap_context = shap_context.sort_values(
                "Absolute Contribution",
                ascending=False
            )


            for _, row in shap_context.head(10).iterrows():

                context_parts.append(

                    f"{row['Feature']}: "
                    f"value={row['Feature Value']}, "
                    f"contribution={row['SHAP Contribution']:.4f}"

                )


    return "\n".join(context_parts)
