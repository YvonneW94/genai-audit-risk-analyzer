import pandas as pd

from src.risk_rules import (
    detect_duplicate_payments,
    detect_missing_approvals,
    detect_weekend_transactions,
)


def make_test_data():
    return pd.DataFrame(
        {
            "transaction_id": ["TXN-1", "TXN-2", "TXN-3"],
            "vendor_id": ["V-1", "V-1", "V-2"],
            "vendor_name": ["Vendor A", "Vendor A", "Vendor B"],
            "invoice_number": ["INV-100", "INV-100", "INV-200"],
            "transaction_date": pd.to_datetime(
                ["2026-03-02", "2026-03-02", "2026-03-07"]
            ),
            "amount": [5000.00, 5000.00, 1200.00],
            "approver": ["A. Chen", "A. Chen", ""],
            "department": ["Finance", "Finance", "Operations"],
        }
    )


def test_duplicate_payment_detection():
    data = make_test_data()

    result = detect_duplicate_payments(data)

    assert len(result) == 2
    assert set(result["transaction_id"]) == {"TXN-1", "TXN-2"}
    assert set(result["detected_risk_type"]) == {
        "duplicate_payment"
    }


def test_missing_approval_detection():
    data = make_test_data()

    result = detect_missing_approvals(data)

    assert len(result) == 1
    assert result.iloc[0]["transaction_id"] == "TXN-3"
    assert result.iloc[0]["detected_risk_type"] == "missing_approval"


def test_weekend_transaction_detection():
    data = make_test_data()

    result = detect_weekend_transactions(data)

    assert len(result) == 1
    assert result.iloc[0]["transaction_id"] == "TXN-3"
    assert result.iloc[0]["detected_risk_type"] == (
        "weekend_transaction"
    )
