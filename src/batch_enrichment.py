import time
from pathlib import Path

import pandas as pd

from src.llm_enrichment import enrich_risk


INPUT_PATH = Path("data/flagged_transactions.csv")
OUTPUT_PATH = Path("data/genai_risk_assessments.csv")
SAMPLE_SIZE = 10


def main() -> None:
    all_flagged = pd.read_csv(INPUT_PATH)

    flagged = (
        all_flagged.groupby(
            "detected_risk_type",
            group_keys=False,
        )
        .head(2)
        .reset_index(drop=True)
     )
    results = []

    for index, row in flagged.iterrows():
        transaction = row.to_dict()

        print(
            f"Analyzing {index + 1}/{len(flagged)}: "
            f"{transaction['transaction_id']}"
        )

        try:
            assessment = enrich_risk(transaction)

            results.append(
                {
                    **transaction,
                    "genai_risk_type": assessment["risk_type"],
                    "genai_risk_level": assessment["risk_level"],
                    "genai_explanation": assessment["explanation"],
                    "genai_recommended_procedure": assessment[
                        "recommended_audit_procedure"
                    ],
                    "genai_status": "success",
                }
            )

        except Exception as error:
            results.append(
                {
                    **transaction,
                    "genai_risk_type": "",
                    "genai_risk_level": "",
                    "genai_explanation": "",
                    "genai_recommended_procedure": "",
                    "genai_status": f"error: {type(error).__name__}",
                }
            )

            print(f"Error: {error}")

        time.sleep(0.5)

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_PATH, index=False)

    successful = output["genai_status"].eq("success").sum()

    print()
    print(f"Completed {len(output)} assessments.")
    print(f"Successful assessments: {successful}")
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
