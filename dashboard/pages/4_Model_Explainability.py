import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from dashboard.utils.ui import apply_custom_ui

apply_custom_ui()

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Model Explainability",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Model Explainability")
st.caption(
    "Technical evaluation and interpretation of the trained "
    "fraud detection model."
)

# =========================================================
# PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "lightgbm_model.pkl"
X_TEST_PATH = ROOT_DIR / "models" / "X_test.pkl"
Y_TEST_PATH = ROOT_DIR / "models" / "y_test.pkl"
X_TRAIN_PATH = ROOT_DIR / "models" / "X_train.pkl"
Y_TRAIN_PATH = ROOT_DIR / "models" / "y_train.pkl"

# =========================================================
# LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    X_test = joblib.load(X_TEST_PATH)
    y_test = joblib.load(Y_TEST_PATH)
    X_train = joblib.load(X_TRAIN_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)

    return model, X_train, y_train, X_test, y_test


model, X_train, y_train, X_test, y_test = load_model_and_data()

# =========================================================
# PREPARE DATA
# =========================================================

if isinstance(X_test, pd.DataFrame):
    feature_names = X_test.columns.tolist()
    X_test_np = X_test.to_numpy()
else:
    X_test_np = np.asarray(X_test)
    feature_names = [
        f"Feature {i}"
        for i in range(X_test_np.shape[1])
    ]

if isinstance(X_train, pd.DataFrame):
    X_train_np = X_train.to_numpy()
else:
    X_train_np = np.asarray(X_train)

y_test_np = np.asarray(y_test).reshape(-1)
y_train_np = np.asarray(y_train).reshape(-1)

# Model predictions
fraud_probabilities = model.predict_proba(X_test_np)[:, 1]

# Default threshold
DEFAULT_THRESHOLD = 0.95

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Model Configuration")

threshold = st.sidebar.slider(
    "Fraud Classification Threshold",
    min_value=0.01,
    max_value=0.99,
    value=DEFAULT_THRESHOLD,
    step=0.01
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Current Threshold",
    f"{threshold:.2f}"
)

st.sidebar.caption(
    "Transactions with fraud probability above the threshold "
    "are classified as fraud."
)

# =========================================================
# CALCULATE METRICS
# =========================================================

predictions = (
    fraud_probabilities >= threshold
).astype(int)

accuracy = accuracy_score(y_test_np, predictions)

precision = precision_score(
    y_test_np,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test_np,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_np,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_np,
    fraud_probabilities
)

pr_auc = average_precision_score(
    y_test_np,
    fraud_probabilities
)

tn, fp, fn, tp = confusion_matrix(
    y_test_np,
    predictions
).ravel()

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.header("1. Model Performance")

st.caption(
    "Evaluation metrics are calculated using the existing held-out test dataset."
)

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

col2.metric(
    "Precision",
    f"{precision:.4f}"
)

col3.metric(
    "Recall",
    f"{recall:.4f}"
)

col4.metric(
    "F1 Score",
    f"{f1:.4f}"
)

col5.metric(
    "ROC-AUC",
    f"{roc_auc:.4f}"
)

col6.metric(
    "PR-AUC",
    f"{pr_auc:.4f}"
)

st.markdown("---")

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.header("2. Confusion Matrix")

cm = np.array([
    [tn, fp],
    [fn, tp]
])

cm_labels = [
    ["True Negative", "False Positive"],
    ["False Negative", "True Positive"]
]

fig_cm = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Predicted Legitimate", "Predicted Fraud"],
        y=["Actual Legitimate", "Actual Fraud"],
        text=[
            [
                f"TN: {tn:,}",
                f"FP: {fp:,}"
            ],
            [
                f"FN: {fn:,}",
                f"TP: {tp:,}"
            ]
        ],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True
    )
)

fig_cm.update_layout(
    title="Confusion Matrix",
    height=450,
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label"
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)

cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)

cm_col1.metric("True Negatives", f"{tn:,}")
cm_col2.metric("False Positives", f"{fp:,}")
cm_col3.metric("False Negatives", f"{fn:,}")
cm_col4.metric("True Positives", f"{tp:,}")

with st.expander("Understanding the Confusion Matrix"):

    st.markdown("""
    - **True Negative (TN):** Legitimate transactions correctly classified as legitimate.
    - **False Positive (FP):** Legitimate transactions incorrectly flagged as fraud.
    - **False Negative (FN):** Fraudulent transactions missed by the model.
    - **True Positive (TP):** Fraudulent transactions correctly detected.

    In fraud detection, False Negatives are particularly important because
    they represent fraudulent transactions that were not detected.
    """)

st.markdown("---")

# =========================================================
# GLOBAL FEATURE IMPORTANCE
# =========================================================

st.header("3. Global Feature Importance")

importance_values = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

feature_importance_df = (
    feature_importance_df
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

top_n = st.slider(
    "Number of Features to Display",
    min_value=5,
    max_value=len(feature_importance_df),
    value=min(15, len(feature_importance_df))
)

top_features = feature_importance_df.head(top_n)

fig_importance = px.bar(
    top_features.sort_values("Importance"),
    x="Importance",
    y="Feature",
    orientation="h",
    title=f"Top {top_n} Important Features",
    text="Importance"
)

fig_importance.update_layout(
    height=max(450, top_n * 30),
    xaxis_title="Feature Importance",
    yaxis_title="Feature",
    showlegend=False
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

st.dataframe(
    feature_importance_df,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

st.header("4. Threshold Analysis")

st.write(
    "Changing the classification threshold affects the trade-off "
    "between detecting fraud and generating false alerts."
)

thresholds = np.linspace(0.01, 0.99, 100)

threshold_results = []

for t in thresholds:

    threshold_predictions = (
        fraud_probabilities >= t
    ).astype(int)

    threshold_results.append({
        "Threshold": t,
        "Precision": precision_score(
            y_test_np,
            threshold_predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test_np,
            threshold_predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_test_np,
            threshold_predictions,
            zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

fig_threshold = go.Figure()

fig_threshold.add_trace(
    go.Scatter(
        x=threshold_df["Threshold"],
        y=threshold_df["Precision"],
        mode="lines",
        name="Precision"
    )
)

fig_threshold.add_trace(
    go.Scatter(
        x=threshold_df["Threshold"],
        y=threshold_df["Recall"],
        mode="lines",
        name="Recall"
    )
)

fig_threshold.add_trace(
    go.Scatter(
        x=threshold_df["Threshold"],
        y=threshold_df["F1"],
        mode="lines",
        name="F1 Score"
    )
)

fig_threshold.add_vline(
    x=threshold,
    line_dash="dash",
    annotation_text=f"Current Threshold: {threshold:.2f}"
)

fig_threshold.update_layout(
    title="Precision, Recall and F1 by Threshold",
    xaxis_title="Classification Threshold",
    yaxis_title="Metric Value",
    yaxis_range=[0, 1],
    height=500
)

st.plotly_chart(
    fig_threshold,
    use_container_width=True
)

# Threshold impact

threshold_comparison = []

for t in thresholds:

    p = (
        fraud_probabilities >= t
    ).astype(int)

    tn_t, fp_t, fn_t, tp_t = confusion_matrix(
        y_test_np,
        p
    ).ravel()

    threshold_comparison.append({
        "Threshold": t,
        "False Positives": fp_t,
        "False Negatives": fn_t,
        "True Positives": tp_t,
        "True Negatives": tn_t
    })

threshold_comparison_df = pd.DataFrame(
    threshold_comparison
)

fig_errors = go.Figure()

fig_errors.add_trace(
    go.Scatter(
        x=threshold_comparison_df["Threshold"],
        y=threshold_comparison_df["False Positives"],
        mode="lines",
        name="False Positives"
    )
)

fig_errors.add_trace(
    go.Scatter(
        x=threshold_comparison_df["Threshold"],
        y=threshold_comparison_df["False Negatives"],
        mode="lines",
        name="False Negatives"
    )
)

fig_errors.add_vline(
    x=threshold,
    line_dash="dash",
    annotation_text=f"Current: {threshold:.2f}"
)

fig_errors.update_layout(
    title="False Positives vs False Negatives",
    xaxis_title="Classification Threshold",
    yaxis_title="Number of Transactions",
    height=450
)

st.plotly_chart(
    fig_errors,
    use_container_width=True
)

st.markdown("---")

# =========================================================
# LOCAL TRANSACTION EXPLANATION
# =========================================================

st.header("5. Transaction-Level Explanation")

st.write(
    "Select a transaction from the test dataset to examine "
    "its prediction and the features contributing to the decision."
)

# Select transaction from test dataset

selected_index = st.selectbox(
    "Select Test Transaction",
    options=list(range(len(X_test_np))),
    format_func=lambda x: f"Test Transaction #{x}"
)

selected_probability = fraud_probabilities[selected_index]
selected_prediction = int(
    selected_probability >= threshold
)
selected_actual = int(
    y_test_np[selected_index]
)

# Transaction metrics

col1, col2, col3 = st.columns(3)

col1.metric(
    "Fraud Probability",
    f"{selected_probability:.2%}"
)

col2.metric(
    "Model Prediction",
    "FRAUD" if selected_prediction == 1 else "LEGITIMATE"
)

col3.metric(
    "Actual Label",
    "FRAUD" if selected_actual == 1 else "LEGITIMATE"
)

# Prediction status

if selected_prediction == selected_actual:

    if selected_actual == 1:
        st.success(
            "The model correctly identified this fraudulent transaction."
        )
    else:
        st.success(
            "The model correctly classified this legitimate transaction."
        )

else:

    if selected_actual == 1:
        st.error(
            "False Negative: This fraudulent transaction was missed by the model."
        )
    else:
        st.warning(
            "False Positive: This legitimate transaction was flagged by the model."
        )

# Feature values

selected_features = pd.DataFrame({
    "Feature": feature_names,
    "Value": X_test_np[selected_index]
})

st.subheader("Transaction Feature Values")

st.dataframe(
    selected_features,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# SHAP EXPLANATION
# =========================================================

st.write("X_test shape:", X_test.shape)
st.write("Model type:", type(model))
st.write("SHAP section reached")

import shap
import matplotlib.pyplot as plt

st.subheader("🔎 SHAP Explanation")

try:

    # -----------------------------------------------------
    # Select a real transaction from the test dataset
    # -----------------------------------------------------

    transaction_idx = st.selectbox(
        "Select Transaction",
        range(len(X_test)),
        index=0
    )

    # Convert to NumPy if necessary
    X_test_np = (
        X_test.values
        if isinstance(X_test, pd.DataFrame)
        else np.asarray(X_test)
    )

    transaction = X_test_np[
        transaction_idx:transaction_idx + 1
    ]

    # -----------------------------------------------------
    # Feature names
    # -----------------------------------------------------

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

    feature_names = (
        FEATURES
        if len(FEATURES) == transaction.shape[1]
        else [f"Feature {i}" for i in range(transaction.shape[1])]
    )

    # -----------------------------------------------------
    # SHAP Explainer
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(transaction)

    # Binary classification handling
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    # Handle SHAP output dimensions
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_values = shap_values.reshape(
        transaction.shape[0],
        transaction.shape[1]
    )

    transaction_shap_values = shap_values[0]

    # -----------------------------------------------------
    # Expected Value
    # -----------------------------------------------------

    expected_value = explainer.expected_value

    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value.flatten()

        if len(expected_value) > 1:
            expected_value = expected_value[1]
        else:
            expected_value = expected_value[0]

    # -----------------------------------------------------
    # Waterfall Plot
    # -----------------------------------------------------

    explanation = shap.Explanation(
        values=transaction_shap_values,
        base_values=float(expected_value),
        data=transaction[0],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        explanation,
        max_display=10,
        show=False
    )

    st.pyplot(plt.gcf())

    plt.close("all")

    # -----------------------------------------------------
    # SHAP CONTRIBUTION TABLE
    # -----------------------------------------------------

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Feature Value": transaction[0],
        "SHAP Contribution": transaction_shap_values
    })

    shap_df["Absolute Contribution"] = (
        shap_df["SHAP Contribution"].abs()
    )

    shap_df = (
        shap_df
        .sort_values(
            "Absolute Contribution",
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.subheader("Feature Contributions")

    st.dataframe(
        shap_df[
            [
                "Feature",
                "Feature Value",
                "SHAP Contribution"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # Fraud Increasing / Decreasing Factors
    # -----------------------------------------------------

    positive_features = shap_df[
        shap_df["SHAP Contribution"] > 0
    ].head(5)

    negative_features = shap_df[
        shap_df["SHAP Contribution"] < 0
    ].head(5)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🔴 Fraud-Increasing Factors")

        if not positive_features.empty:

            for _, row in positive_features.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"(Contribution: "
                    f"{row['SHAP Contribution']:.4f})"
                )

        else:

            st.write("No positive SHAP contributions.")

    with col2:

        st.markdown("### 🟢 Fraud-Decreasing Factors")

        if not negative_features.empty:

            for _, row in negative_features.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"(Contribution: "
                    f"{row['SHAP Contribution']:.4f})"
                )

        else:

            st.write("No negative SHAP contributions.")

except Exception as e:

    st.error(
        f"SHAP explanation could not be generated: {e}"
    )

    st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Model Explainability | LightGBM Fraud Detection Model | "
    "Evaluation based on the existing held-out test dataset"
)