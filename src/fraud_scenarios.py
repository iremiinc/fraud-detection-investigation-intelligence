
import pandas as pd
import numpy as np


def generate_rapid_transactions(
    customers: pd.DataFrame,
    n_transactions: int = 300,
    simulation_days: int = 30,
    start_date: str = "2026-01-01",
    random_state: int = 42
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    # 30 accounts, 10 transactions per account
    n_accounts = 30
    transactions_per_account = n_transactions // n_accounts

    if n_transactions % n_accounts != 0:
        raise ValueError(
            "n_transactions must be divisible by 30."
        )

    # Select accounts with sufficient initial balances
    eligible = customers[
        customers["initial_balance"] >= 10_000
    ].copy()

    if len(eligible) < n_accounts:
        raise ValueError("Not enough eligible accounts.")

    selected = eligible.sample(
        n=n_accounts,
        random_state=random_state
    ).reset_index(drop=True)

    account_info = customers.set_index("account_id")

    receivers = customers["account_id"].to_numpy()

    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=simulation_days)

    transactions = []

    for account in selected.itertuples(index=False):

        sender_id = account.account_id
        balance = float(account.initial_balance)

        # Each account starts a suspicious burst
        burst_start = start + pd.Timedelta(
            seconds=int(
                rng.integers(
                    0,
                    simulation_days * 24 * 60 * 60
                )
            )
        )

        for j in range(transactions_per_account):

            receiver_id = rng.choice(receivers)

            while receiver_id == sender_id:
                receiver_id = rng.choice(receivers)

            timestamp = (
                burst_start
                + pd.Timedelta(seconds=int(rng.integers(0, 300)))
            )

            amount = min(
                float(rng.uniform(100, 500)),
                balance
            )

            if amount <= 0:
                continue

            balance_before = balance
            balance -= amount

            transactions.append({
                "transaction_id": f"RAPID_{sender_id}_{j+1:03d}",
                "timestamp": timestamp,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "transaction_type": "TRANSFER",
                "amount": round(amount, 2),
                "sender_balance_before": round(balance_before, 2),
                "sender_balance_after": round(balance, 2),
                "device_id": account.registered_device,
                "location": account.home_location,
                "is_fraud": 1,
                "fraud_scenario": "RAPID_TRANSACTIONS"
            })

    return pd.DataFrame(transactions)

def generate_unusual_amount_transactions(
    customers: pd.DataFrame,
    n_transactions: int = 300,
    simulation_days: int = 30,
    start_date: str = "2026-01-01",
    random_state: int = 43
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    selected = customers.sample(
        n=n_transactions,
        random_state=random_state
    ).reset_index(drop=True)

    account_info = customers.set_index("account_id")
    all_accounts = customers["account_id"].to_numpy()

    start = pd.Timestamp(start_date)

    transactions = []

    for i, account in enumerate(selected.itertuples(index=False)):

        sender_id = account.account_id

        # Generate an unusually high amount
        normal_amount = rng.lognormal(
            mean=3.5,
            sigma=1.0
        )

        amount = max(
            normal_amount * 15,
            10_000
        )

        # Keep the amount within available balance
        amount = min(
            amount,
            float(account.initial_balance)
        )

        if amount <= 0:
            continue

        receiver_id = rng.choice(all_accounts)

        while receiver_id == sender_id:
            receiver_id = rng.choice(all_accounts)

        timestamp = start + pd.Timedelta(
            seconds=int(
                rng.integers(
                    0,
                    simulation_days * 24 * 60 * 60
                )
            )
        )

        transactions.append({
            "transaction_id": f"UNUSUAL_{i+1:05d}",
            "timestamp": timestamp,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "transaction_type": "TRANSFER",
            "amount": round(amount, 2),
            "sender_balance_before": round(
                float(account.initial_balance), 2
            ),
            "sender_balance_after": round(
                float(account.initial_balance) - amount, 2
            ),
            "device_id": account.registered_device,
            "location": account.home_location,
            "is_fraud": 1,
            "fraud_scenario": "UNUSUAL_AMOUNT"
        })

    return pd.DataFrame(transactions)

def generate_account_takeover_transactions(
    customers: pd.DataFrame,
    n_transactions: int = 300,
    simulation_days: int = 30,
    start_date: str = "2026-01-01",
    random_state: int = 44
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    # Her hesap için 5 işlem oluşturacağız.
    transactions_per_account = 5
    n_accounts = n_transactions // transactions_per_account

    if n_transactions % transactions_per_account != 0:
        raise ValueError("n_transactions must be divisible by 5.")

    eligible = customers[
        customers["initial_balance"] >= 5_000
    ].copy()

    selected = eligible.sample(
        n=n_accounts,
        random_state=random_state
    )

    all_accounts = customers["account_id"].to_numpy()
    all_locations = customers["home_location"].unique()

    start = pd.Timestamp(start_date)
    transactions = []

    for account in selected.itertuples(index=False):

        sender_id = account.account_id
        balance = float(account.initial_balance)

        # Saldırganın kullandığı yeni cihaz
        new_device = f"UNKNOWN_{rng.integers(100000, 999999)}"

        # Hesabın kayıtlı konumundan farklı bir konum
        possible_locations = [
            loc for loc in all_locations
            if loc != account.home_location
        ]
        attack_location = rng.choice(possible_locations)

        # Beş işlem kısa bir zaman aralığında gerçekleşiyor.
        attack_start = start + pd.Timedelta(
            seconds=int(rng.integers(
                0, simulation_days * 24 * 60 * 60
            ))
        )

        for j in range(transactions_per_account):

            receiver_id = rng.choice(all_accounts)

            while receiver_id == sender_id:
                receiver_id = rng.choice(all_accounts)

            amount = min(
                float(rng.uniform(500, 2_000)),
                balance
            )

            if amount <= 0:
                continue

            balance_before = balance
            balance -= amount

            timestamp = attack_start + pd.Timedelta(
                seconds=int(rng.integers(0, 600))
            )

            transactions.append({
                "transaction_id": f"ATO_{sender_id}_{j+1:03d}",
                "timestamp": timestamp,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "transaction_type": "TRANSFER",
                "amount": round(amount, 2),
                "sender_balance_before": round(balance_before, 2),
                "sender_balance_after": round(balance, 2),
                "device_id": new_device,
                "location": attack_location,
                "is_fraud": 1,
                "fraud_scenario": "ACCOUNT_TAKEOVER"
            })

    return pd.DataFrame(transactions)

def generate_money_laundering_transactions(
    customers: pd.DataFrame,
    n_transactions: int = 300,
    simulation_days: int = 30,
    start_date: str = "2026-01-01",
    random_state: int = 45
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    transactions_per_chain = 10
    n_chains = n_transactions // transactions_per_chain

    if n_transactions % transactions_per_chain != 0:
        raise ValueError("n_transactions must be divisible by 10.")

    # Her zincir için 11 farklı hesap gerekir.
    required_accounts = n_chains * (transactions_per_chain + 1)

    if len(customers) < required_accounts:
        raise ValueError("Not enough customer accounts.")

    # Başlangıç hesapları yeterli bakiyeye sahip olmalı.
    eligible = customers[
        customers["initial_balance"] >= 5_000
    ].copy()

    if len(eligible) < n_chains:
        raise ValueError("Not enough eligible starting accounts.")

    selected = eligible.sample(
        n=n_chains,
        random_state=random_state
    )

    # Zincirlerde kullanılacak hesapları birbirinden ayır.
    used_ids = set(selected["account_id"])
    remaining = customers[
        ~customers["account_id"].isin(used_ids)
    ]

    receivers = remaining.sample(
        n=n_chains * transactions_per_chain,
        random_state=random_state
    )

    receiver_ids = receivers["account_id"].to_numpy()

    balances = customers.set_index(
        "account_id"
    )["initial_balance"].to_dict()

    customer_info = customers.set_index("account_id")

    start = pd.Timestamp(start_date)
    transactions = []

    receiver_index = 0

    for chain_number, source in enumerate(
        selected.itertuples(index=False), start=1
    ):

        # Zincirin başlangıç hesabı
        sender_id = source.account_id

        # Zincirin 10 alıcı hesabı
        chain_receivers = receiver_ids[
            receiver_index:receiver_index + transactions_per_chain
        ]

        receiver_index += transactions_per_chain

        chain_start = start + pd.Timedelta(
            seconds=int(rng.integers(
                0, simulation_days * 24 * 60 * 60
            ))
        )

        for step, receiver_id in enumerate(
            chain_receivers, start=1
        ):

            sender_balance = balances[sender_id]

            # İlk transfer zincire para sokar.
            if step == 1:
                amount = float(rng.uniform(1_000, 2_000))

            # Sonraki hesaplar gelen paranın büyük kısmını aktarır.
            else:
                amount = incoming_amount * float(
                    rng.uniform(0.85, 0.98)
                )

            amount = min(amount, sender_balance)

            if amount <= 0:
                break

            receiver_balance_before = balances[receiver_id]

            sender_balance_before = balances[sender_id]
            balances[sender_id] -= amount
            balances[receiver_id] += amount

            incoming_amount = amount

            sender_info = customer_info.loc[sender_id]

            timestamp = chain_start + pd.Timedelta(
                seconds=int(rng.integers(
                    (step - 1) * 30,
                    (step - 1) * 30 + 20
                ))
            )

            transactions.append({
                "transaction_id": f"ML_{chain_number:03d}_{step:02d}",
                "timestamp": timestamp,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "transaction_type": "TRANSFER",
                "amount": round(amount, 2),
                "sender_balance_before": round(sender_balance_before, 2),
                "sender_balance_after": round(balances[sender_id], 2),
                "device_id": sender_info["registered_device"],
                "location": sender_info["home_location"],
                "is_fraud": 1,
                "fraud_scenario": "MONEY_LAUNDERING"
            })

            # Bir sonraki işlemde alıcı, gönderen olur.
            sender_id = receiver_id

    return pd.DataFrame(transactions)

def generate_coordinated_fraud_transactions(
    customers: pd.DataFrame,
    n_transactions: int = 300,
    simulation_days: int = 30,
    start_date: str = "2026-01-01",
    random_state: int = 46
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    senders_per_group = 10
    n_groups = n_transactions // senders_per_group

    if n_transactions % senders_per_group != 0:
        raise ValueError("n_transactions must be divisible by 10.")

    # Gönderici hesapların yeterli bakiyesi olmalı.
    eligible_senders = customers[
        customers["initial_balance"] >= 2_000
    ].copy()

    if len(eligible_senders) < n_transactions:
        raise ValueError("Not enough eligible sender accounts.")

    # 300 farklı gönderici hesap seç.
    selected_senders = eligible_senders.sample(
        n=n_transactions,
        random_state=random_state
    ).reset_index(drop=True)

    # 30 farklı ortak alıcı hesap seç.
    sender_ids = set(selected_senders["account_id"])

    eligible_receivers = customers[
        ~customers["account_id"].isin(sender_ids)
    ]

    selected_receivers = eligible_receivers.sample(
        n=n_groups,
        random_state=random_state + 1
    ).reset_index(drop=True)

    start = pd.Timestamp(start_date)
    transactions = []

    for group_number in range(n_groups):

        receiver = selected_receivers.iloc[group_number]
        receiver_id = receiver["account_id"]

        # Her grup için ortak bir zaman aralığı.
        group_start = start + pd.Timedelta(
            seconds=int(rng.integers(
                0, simulation_days * 24 * 60 * 60
            ))
        )

        group_senders = selected_senders.iloc[
            group_number * senders_per_group:
            (group_number + 1) * senders_per_group
        ]

        for sender in group_senders.itertuples(index=False):

            sender_id = sender.account_id
            sender_balance = float(sender.initial_balance)

            amount = float(rng.uniform(300, 1_000))
            amount = min(amount, sender_balance)

            if amount <= 0:
                continue

            timestamp = group_start + pd.Timedelta(
                seconds=int(rng.integers(0, 15 * 60))
            )

            transactions.append({
                "transaction_id": (
                    f"COORD_{group_number + 1:03d}_"
                    f"{sender_id}"
                ),
                "timestamp": timestamp,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "transaction_type": "TRANSFER",
                "amount": round(amount, 2),
                "sender_balance_before": round(
                    sender_balance, 2
                ),
                "sender_balance_after": round(
                    sender_balance - amount, 2
                ),
                "device_id": sender.registered_device,
                "location": sender.home_location,
                "is_fraud": 1,
                "fraud_scenario": "COORDINATED_FRAUD"
            })

    return pd.DataFrame(transactions)
