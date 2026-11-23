import sqlite3
from pathlib import Path
from datetime import datetime
import uuid
import pandas as pd

DB_PATH = Path("data/processed/fraud_cases.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            risk_score REAL,
            severity TEXT,
            status TEXT DEFAULT 'Open',
            analyst_notes TEXT DEFAULT '',
            created_at TEXT,
            updated_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS case_events (
            event_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_description TEXT,
            created_at TEXT,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        )
    """)

    conn.commit()
    conn.close()


def log_event(case_id, event_type, description):

    initialize_database()

    conn = get_connection()

    event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute("""
        INSERT INTO case_events (
            event_id,
            case_id,
            event_type,
            event_description,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        event_id,
        case_id,
        event_type,
        description,
        now
    ))

    conn.commit()
    conn.close()


def create_case(account_id, risk_score, severity, notes=""):

    initialize_database()

    conn = get_connection()

    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute("""
        INSERT INTO cases (
            case_id,
            account_id,
            risk_score,
            severity,
            status,
            analyst_notes,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        case_id,
        account_id,
        risk_score,
        severity,
        "Open",
        notes,
        now,
        now
    ))

    conn.commit()
    conn.close()

    log_event(
        case_id,
        "CASE_CREATED",
        f"Case created for account {account_id}. Severity: {severity}. Risk score: {risk_score:.2f}."
    )

    if notes.strip():

        log_event(
            case_id,
            "NOTE_ADDED",
            "Initial investigation notes added during case creation."
        )

    return case_id


def get_all_cases():

    initialize_database()

    conn = get_connection()

    rows = conn.execute("""
        SELECT *
        FROM cases
        ORDER BY created_at DESC
    """).fetchall()

    columns = [
        "case_id",
        "account_id",
        "risk_score",
        "severity",
        "status",
        "analyst_notes",
        "created_at",
        "updated_at"
    ]

    result = pd.DataFrame(rows, columns=columns)

    conn.close()

    return result


def get_case_events(case_id):

    initialize_database()

    conn = get_connection()

    rows = conn.execute("""
        SELECT *
        FROM case_events
        WHERE case_id = ?
        ORDER BY created_at DESC
    """, (case_id,)).fetchall()

    columns = [
        "event_id",
        "case_id",
        "event_type",
        "event_description",
        "created_at"
    ]

    result = pd.DataFrame(rows, columns=columns)

    conn.close()

    return result


def update_case_status(case_id, status):

    initialize_database()

    conn = get_connection()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    old_status = conn.execute("""
        SELECT status
        FROM cases
        WHERE case_id = ?
    """, (case_id,)).fetchone()

    conn.execute("""
        UPDATE cases
        SET status = ?, updated_at = ?
        WHERE case_id = ?
    """, (status, now, case_id))

    conn.commit()
    conn.close()

    previous_status = old_status[0] if old_status else "Unknown"

    if previous_status != status:

        log_event(
            case_id,
            "STATUS_CHANGED",
            f"Case status changed from {previous_status} to {status}."
        )


def update_case_notes(case_id, notes):

    initialize_database()

    conn = get_connection()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute("""
        UPDATE cases
        SET analyst_notes = ?, updated_at = ?
        WHERE case_id = ?
    """, (notes, now, case_id))

    conn.commit()
    conn.close()

    log_event(
        case_id,
        "NOTE_UPDATED",
        "Analyst notes updated."
    )


def delete_case(case_id):

    initialize_database()

    conn = get_connection()

    conn.execute("""
        DELETE FROM case_events
        WHERE case_id = ?
    """, (case_id,))

    conn.execute("""
        DELETE FROM cases
        WHERE case_id = ?
    """, (case_id,))

    conn.commit()
    conn.close()