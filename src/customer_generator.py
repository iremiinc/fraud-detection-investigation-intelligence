
import pandas as pd
import numpy as np


def generate_customers(
    n_accounts: int = 10_000,
    random_state: int = 42
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    account_ids = [
        f"ACC_{i:06d}"
        for i in range(1, n_accounts + 1)
    ]

    customer_ids = [
        f"CUST_{i:06d}"
        for i in range(1, n_accounts + 1)
    ]

    account_types = rng.choice(
        ["STANDARD", "PREMIUM", "BUSINESS"],
        size=n_accounts,
        p=[0.70, 0.20, 0.10]
    )

    # Account balances
    balances = np.zeros(n_accounts)

    for i, account_type in enumerate(account_types):

        if account_type == "STANDARD":
            balances[i] = rng.lognormal(
                mean=8.5,
                sigma=1.0
            )

        elif account_type == "PREMIUM":
            balances[i] = rng.lognormal(
                mean=10.0,
                sigma=0.8
            )

        else:
            balances[i] = rng.lognormal(
                mean=11.0,
                sigma=1.0
            )

    risk_profiles = rng.choice(
        ["LOW", "MEDIUM", "HIGH"],
        size=n_accounts,
        p=[0.65, 0.30, 0.05]
    )

    locations = rng.choice(
        [
            "Istanbul",
            "Ankara",
            "Izmir",
            "Bursa",
            "Antalya",
            "Adana",
            "Konya",
            "Gaziantep"
        ],
        size=n_accounts
    )

    device_ids = [
        f"DEV_{i:06d}"
        for i in range(1, n_accounts + 1)
    ]

    customers = pd.DataFrame({
        "account_id": account_ids,
        "customer_id": customer_ids,
        "account_type": account_types,
        "initial_balance": balances.round(2),
        "risk_profile": risk_profiles,
        "home_location": locations,
        "registered_device": device_ids
    })

    return customers
