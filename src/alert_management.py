import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd

DB_PATH = Path("data/processed/fraud_cases.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_alerts_db():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fraud_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE,
            account_id TEXT,
            amount REAL,
            fraud_probability REAL,
            priority TEXT,
            status TEXT DEFAULT 'New',
            created_at TEXT,
            case_id TEXT
        )
    """)

    conn.commit()
    conn.close()


def generate_alerts(df, threshold=0.60):

    initialize_alerts_db()

    if "fraud_probability" not in df.columns:
        return 0

    risk_df = df.copy()

    risk_df["fraud_probability"] = pd.to_numeric(
        risk_df["fraud_probability"],
        errors="coerce"
    )

    risk_df = risk_df[
        risk_df["fraud_probability"] >= threshold
    ]

    conn = get_connection()

    created_count = 0

    for _, row in risk_df.iterrows():

        transaction_id = str(row.get("transaction_id", ""))

        if not transaction_id:
            continue

        probability = float(row["fraud_probability"])

        if probability >= 0.95:
            priority = "Critical"
        elif probability >= 0.80:
            priority = "High"
        elif probability >= 0.60:
            priority = "Medium"
        else:
            priority = "Low"

        account_id = str(row.get("sender_id", "Unknown"))

        amount = float(row.get("amount", 0))

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor = conn.execute("""
            INSERT OR IGNORE INTO fraud_alerts (
                transaction_id,
                account_id,
                amount,
                fraud_probability,
                priority,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction_id,
            account_id,
            amount,
            probability,
            priority,
            "New",
            now
        ))

        if cursor.rowcount > 0:
            created_count += 1

    conn.commit()
    conn.close()

    return created_count


def get_alerts():

    initialize_alerts_db()

    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT *
        FROM fraud_alerts
        ORDER BY fraud_probability DESC
    """, conn)

    conn.close()

    return df


def update_alert_status(alert_id, status):

    initialize_alerts_db()

    conn = get_connection()

    conn.execute("""
        UPDATE fraud_alerts
        SET status = ?
        WHERE alert_id = ?
    """, (status, alert_id))

    conn.commit()
    conn.close()


def link_alert_to_case(alert_id, case_id):

    initialize_alerts_db()

    conn = get_connection()

    conn.execute("""
        UPDATE fraud_alerts
        SET status = 'Investigating',
            case_id = ?
        WHERE alert_id = ?
    """, (case_id, alert_id))

    conn.commit()
    conn.close()