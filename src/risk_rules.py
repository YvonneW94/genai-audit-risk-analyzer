from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/synthetic_transactions.csv")
OUTPUT_PATH = Path("data/flagged_transactions.csv")
APPROVAL_THRESHOLD = 10000


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare transaction data."""

    required_columns = {
        "transaction_id",
        "vendor_id",
        "vendor_name",
        "invoice_number",
        "transaction_date",
        "amount",
        "approver",
        "department",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    prepared = df.copy()
    prepared["transaction_date"] = pd.to_datetime(
        prepared["transaction_date"],
        errors="coerce",
    )
    prepared["amount"] = pd.to_numeric(
        prepared["amount"],
        errors="coerce",
    )

    return prepared


def format_flags(
    rows: pd.DataFrame,
    rule_name: str,
    evidence: str,
) -> pd.DataFrame:
    """Convert flagged rows into a consistent result format."""

    result = rows.copy()
    result["detected_risk_type"] = rule_name
    result["rule_evidence"] = evidence

    return result


def detect_duplicate_payments(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_mask = df.duplicated(
        subset=[
            "vendor_id",
            "invoice_number",
            "amount",
        ],
        keep=False,
    )

    return format_flags(
        df.loc[duplicate_mask],
        "duplicate_payment",
        "Same vendor, invoice number, and amount appeared more than once.",
    )


def detect_missing_approvals(df: pd.DataFrame) -> pd.DataFrame:
    approval_text = df["approver"].fillna("").astype(str).str.strip()
    missing_mask = approval_text.eq("")

    return format_flags(
        df.loc[missing_mask],
        "missing_approval",
        "The transaction does not contain an approver.",
    )


def detect_weekend_transactions(df: pd.DataFrame) -> pd.DataFrame:
    weekend_mask = df["transaction_date"].dt.dayofweek >= 5

    return format_flags(
        df.loc[weekend_mask],
        "weekend_transaction",
        "The transaction was recorded on Saturday or Sunday.",
    )


def detect_split_transactions(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "vendor_id",
        "transaction_date",
        "department",
    ]

    group_summary = (
        df.groupby(group_columns, dropna=False)
        .agg(
            group_count=("transaction_id", "count"),
            group_total=("amount", "sum"),
            maximum_payment=("amount", "max"),
        )
        .reset_index()
    )

    suspicious_groups = group_summary.loc[
        (group_summary["group_count"] >= 2)
        & (group_summary["group_total"] >= APPROVAL_THRESHOLD)
        & (group_summary["maximum_payment"] < APPROVAL_THRESHOLD)
    ]

    flagged = df.merge(
        suspicious_groups[group_columns + ["group_total"]],
        on=group_columns,
        how="inner",
    )

    if flagged.empty:
        return format_flags(
            flagged,
            "split_transaction",
            "No suspicious split transactions detected.",
        )

    flagged["detected_risk_type"] = "split_transaction"
    flagged["rule_evidence"] = flagged["group_total"].apply(
        lambda total: (
            f"Same-day payments totaled ${total:,.2f}, "
            f"while each payment remained below "
            f"the ${APPROVAL_THRESHOLD:,.2f} approval threshold."
        )
    )

    return flagged.drop(columns=["group_total"])


def detect_amount_outliers(df: pd.DataFrame) -> pd.DataFrame:
    q1 = df["amount"].quantile(0.25)
    q3 = df["amount"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = df["amount"] > upper_bound

    flagged = format_flags(
        df.loc[outlier_mask],
        "amount_outlier",
        "Transaction amount exceeded the dataset IQR upper bound.",
    )

    flagged["outlier_upper_bound"] = round(upper_bound, 2)

    return flagged


def run_all_rules(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_data(df)

    results = [
        detect_duplicate_payments(prepared),
        detect_missing_approvals(prepared),
        detect_weekend_transactions(prepared),
        detect_split_transactions(prepared),
        detect_amount_outliers(prepared),
    ]

    nonempty_results = [
        result for result in results if not result.empty
    ]

    if not nonempty_results:
        return pd.DataFrame()

    return pd.concat(
        nonempty_results,
        ignore_index=True,
        sort=False,
    )


def main() -> None:
    transactions = pd.read_csv(DATA_PATH)
    flagged = run_all_rules(transactions)

    flagged.to_csv(OUTPUT_PATH, index=False)

    print(f"Analyzed {len(transactions):,} transactions.")
    print(f"Generated {len(flagged):,} risk flags.")
    print(f"Saved results to: {OUTPUT_PATH}")
    print()
    print("Detected risk distribution:")
    print(flagged["detected_risk_type"].value_counts())


if __name__ == "__main__":
    main()

