
import pandas as pd
import numpy as np


def generate_normal_transactions(
    customers: pd.DataFrame,
    n_transactions: int = 98_500,
    simulation_days: int = 30,
    start_date: str = "2026-01-01",
    random_state: int = 42
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    # Initial account balances
    balances = customers.set_index(
        "account_id"
    )["initial_balance"].to_dict()

    account_ids = customers["account_id"].to_numpy()

    # Customer information lookup
    customer_info = customers.set_index("account_id")

    # Generate timestamps
    start = pd.Timestamp(start_date)

    timestamps = start + pd.to_timedelta(
        rng.uniform(
            0,
            simulation_days * 24 * 60 * 60,
            size=n_transactions
        ),
        unit="s"
    )

    timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)

    transactions = []

    transaction_types = ["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN"]

    for i in range(n_transactions):

        # Select sender
        sender_id = rng.choice(account_ids)

        sender_info = customer_info.loc[sender_id]

        sender_balance = balances[sender_id]

        # Select receiver
        receiver_id = rng.choice(account_ids)

        while receiver_id == sender_id:
            receiver_id = rng.choice(account_ids)

        # Transaction type
        transaction_type = rng.choice(
            transaction_types,
            p=[0.30, 0.35, 0.20, 0.15]
        )

        # Generate amount
        multiplier = {
            "STANDARD": 1.0,
            "PREMIUM": 2.5,
            "BUSINESS": 5.0
        }[sender_info["account_type"]]

        amount = rng.lognormal(
            mean=3.5,
            sigma=1.0
        ) * multiplier

        amount = min(amount, 100_000)

        # Ensure sufficient balance
        if transaction_type != "CASH_IN":

            if sender_balance < 1:
                continue

            amount = min(amount, sender_balance)

            if amount <= 0:
                continue

        # Balance update
        sender_balance_before = balances[sender_id]

        if transaction_type == "CASH_IN":
            balances[sender_id] += amount

        else:
            balances[sender_id] -= amount
            balances[receiver_id] += amount

        sender_balance_after = balances[sender_id]

        # Save transaction
        transactions.append({
            "transaction_id": f"TXN_{i+1:07d}",
            "timestamp": timestamps.iloc[i],
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "transaction_type": transaction_type,
            "amount": round(amount, 2),
            "sender_balance_before": round(sender_balance_before, 2),
            "sender_balance_after": round(sender_balance_after, 2),
            "device_id": sender_info["registered_device"],
            "location": sender_info["home_location"],
            "is_fraud": 0
        })

    return pd.DataFrame(transactions)
