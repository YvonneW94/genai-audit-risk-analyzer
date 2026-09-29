import json
import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-6-luna")
API_KEY = os.getenv("OPENAI_API_KEY")

RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_type": {
            "type": "string",
            "enum": [
                "duplicate_payment",
                "missing_approval",
                "weekend_transaction",
                "split_transaction",
                "amount_outlier",
                "other",
            ],
        },
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "explanation": {
            "type": "string",
        },
        "recommended_audit_procedure": {
            "type": "string",
        },
    },
    "required": [
        "risk_type",
        "risk_level",
        "explanation",
        "recommended_audit_procedure",
    ],
    "additionalProperties": False,
}


def get_client() -> OpenAI:
    if not API_KEY:
        raise ValueError(
            "OPENAI_API_KEY was not found. Check the local .env file."
        )

    return OpenAI(api_key=API_KEY)


def enrich_risk(transaction: dict[str, Any]) -> dict[str, str]:
    """Generate a structured audit-risk assessment for one flagged row."""

    client = get_client()

    transaction_text = json.dumps(
        transaction,
        indent=2,
        default=str,
    )

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=(
            "You assist an internal audit data science team. "
            "Assess only the transaction and deterministic rule evidence "
            "provided by the user. Do not invent amounts, dates, policies, "
            "or supporting facts. The flag is a review indicator, not a "
            "confirmed finding. Keep the explanation and recommended "
            "procedure concise."
        ),
        input=(
            "Review this flagged transaction and return a structured "
            f"risk assessment:\n\n{transaction_text}"
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "risk_assessment",
                "schema": RISK_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=800,
        store=False,
    )

    return json.loads(response.output_text)


def main() -> None:
    flagged = pd.read_csv("data/flagged_transactions.csv")

    if flagged.empty:
        raise ValueError("No flagged transactions were found.")

    test_transaction = flagged.iloc[0].to_dict()

    print(f"Model: {MODEL_NAME}")
    print(
        "Testing transaction:",
        test_transaction["transaction_id"],
    )
    print("Calling the OpenAI API...")

    assessment = enrich_risk(test_transaction)

    print()
    print("Structured GenAI assessment:")
    print(
        json.dumps(
            assessment,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
