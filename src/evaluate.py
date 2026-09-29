from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/genai_risk_assessments.csv")
OUTPUT_PATH = Path("data/evaluation_summary.csv")


def main() -> None:
    results = pd.read_csv(INPUT_PATH)

    results["structured_output_success"] = (
        results["genai_status"] == "success"
    )

    results["classification_match"] = (
        results["detected_risk_type"]
        == results["genai_risk_type"]
    )

    total = len(results)
    successful = int(results["structured_output_success"].sum())
    matched = int(results["classification_match"].sum())

    success_rate = successful / total if total else 0
    agreement_rate = matched / total if total else 0

    summary = pd.DataFrame(
        {
            "metric": [
                "evaluation_sample_size",
                "structured_output_success_rate",
                "rule_classification_agreement_rate",
            ],
            "value": [
                total,
                success_rate,
                agreement_rate,
            ],
        }
    )

    summary.to_csv(OUTPUT_PATH, index=False)

    print("GenAI Evaluation Summary")
    print("-------------------------")
    print(f"Sample size: {total}")
    print(f"Structured-output success: {success_rate:.1%}")
    print(f"Rule classification agreement: {agreement_rate:.1%}")
    print(f"Saved summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

