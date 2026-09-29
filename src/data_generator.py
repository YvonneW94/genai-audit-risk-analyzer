from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
OUTPUT_PATH = Path("data/synthetic_transactions.csv")


def generate_transactions(number_of_rows: int = 1000) -> pd.DataFrame:
    """Generate synthetic procurement transactions with seeded audit risks."""

    rng = np.random.default_rng(RANDOM_SEED)

    vendors = [
        "Northstar Consulting",
        "Blue Harbor Technology",
        "Summit Office Supply",
        "Greenfield Logistics",
        "Redwood Marketing",
        "Metro Data Services",
        "Pacific Equipment",
        "Brightline Solutions",
    ]

    departments = [
        "Finance",
        "Marketing",
        "Operations",
        "Engineering",
        "Human Resources",
    ]

    approvers = [
        "A. Chen",
        "J. Smith",
        "M. Garcia",
        "S. Patel",
        "L. Johnson",
    ]

    descriptions = [
        "Professional services",
        "Software subscription",
        "Office equipment",
        "Data services",
        "Marketing services",
        "Logistics support",
        "Technical consulting",
    ]

    business_dates = pd.bdate_range(
    start="2026-01-01",
    end="2026-08-31",
)

    df = pd.DataFrame(
        {
            "transaction_id": [
                f"TXN-{number:05d}" for number in range(1, number_of_rows + 1)
            ],
            "vendor_id": [
                f"V-{number:03d}"
                for number in rng.integers(1, len(vendors) + 1, size=number_of_rows)
            ],
            "vendor_name": rng.choice(vendors, size=number_of_rows),
            "invoice_number": [
                f"INV-{number:06d}"
                for number in rng.integers(100000, 999999, size=number_of_rows)
            ],
            "transaction_date": pd.to_datetime(
                rng.choice(business_dates, size=number_of_rows)
            ),
            "amount": np.round(
                rng.lognormal(mean=7.4, sigma=0.65, size=number_of_rows),
                2,
            ),
            "approver": rng.choice(approvers, size=number_of_rows),
            "department": rng.choice(departments, size=number_of_rows),
            "payment_method": rng.choice(
                ["ACH", "Wire", "Corporate Card"],
                size=number_of_rows,
                p=[0.65, 0.20, 0.15],
            ),
            "description": rng.choice(descriptions, size=number_of_rows),
            "expected_risk_type": "normal",
        }
    )

    # Risk 1: missing approvals
    missing_approval_rows = list(range(10, 20))
    df.loc[missing_approval_rows, "approver"] = ""
    df.loc[
        missing_approval_rows,
        "expected_risk_type",
    ] = "missing_approval"

    # Risk 2: duplicate payments
    for source_row, duplicate_row in zip(range(30, 40), range(40, 50)):
        duplicate_fields = [
            "vendor_id",
            "vendor_name",
            "invoice_number",
            "amount",
            "department",
            "payment_method",
            "description",
        ]

        df.loc[duplicate_row, duplicate_fields] = df.loc[
            source_row,
            duplicate_fields,
        ].values

        df.loc[
            [source_row, duplicate_row],
            "expected_risk_type",
        ] = "duplicate_payment"

    # Risk 3: unusually large transactions
    amount_outlier_rows = list(range(60, 70))
    df.loc[amount_outlier_rows, "amount"] = rng.uniform(
        50000,
        100000,
        size=len(amount_outlier_rows),
    ).round(2)

    df.loc[
        amount_outlier_rows,
        "expected_risk_type",
    ] = "amount_outlier"

    # Risk 4: seeded weekend transactions
    weekend_rows = list(range(80, 90))
    weekend_dates = pd.date_range(
        start="2026-03-07",
        periods=len(weekend_rows),
        freq="7D",
    )

    df.loc[weekend_rows, "transaction_date"] = weekend_dates
    df.loc[
        weekend_rows,
        "expected_risk_type",
    ] = "weekend_transaction"

    # Risk 5: split transactions below a $10,000 approval threshold
    split_rows = [100, 101, 102]

    df.loc[split_rows, "vendor_id"] = "V-999"
    df.loc[split_rows, "vendor_name"] = "Vertex Advisory"
    df.loc[split_rows, "transaction_date"] = pd.Timestamp("2026-05-12")
    df.loc[split_rows, "amount"] = [9800.00, 9700.00, 9600.00]
    df.loc[split_rows, "department"] = "Finance"
    df.loc[split_rows, "approver"] = "A. Chen"
    df.loc[
        split_rows,
        "expected_risk_type",
    ] = "split_transaction"

    df["transaction_date"] = df["transaction_date"].dt.strftime("%Y-%m-%d")

    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    transactions = generate_transactions()
    transactions.to_csv(OUTPUT_PATH, index=False)

    print(f"Created {len(transactions):,} synthetic transactions.")
    print(f"Saved file to: {OUTPUT_PATH}")
    print()
    print("Risk distribution:")
    print(transactions["expected_risk_type"].value_counts())


if __name__ == "__main__":
    main()

