from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.risk_rules import run_all_rules


DEFAULT_DATA_PATH = Path("data/synthetic_transactions.csv")
GENAI_RESULTS_PATH = Path("data/genai_risk_assessments.csv")

st.set_page_config(
    page_title="GenAI Audit Risk Analyzer",
    page_icon="🔎",
    layout="wide",
)

st.title("GenAI Audit Risk Analyzer")

st.write(
    """
    Upload procurement transaction data or use the included synthetic
    dataset to identify potential control exceptions and audit risks.
    """
)

st.info(
    "This demonstration uses synthetic data. Risk flags require "
    "professional review and should not be treated as final audit findings."
)


uploaded_file = st.file_uploader(
    "Upload a transaction CSV",
    type=["csv"],
)


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA_PATH)


if uploaded_file is not None:
    transactions = pd.read_csv(uploaded_file)
    data_source = "Uploaded dataset"
else:
    transactions = load_default_data()
    data_source = "Included synthetic dataset"


st.subheader("Data preview")
st.caption(data_source)
st.dataframe(
    transactions.head(20),
    use_container_width=True,
)


if st.button("Run Audit Analysis", type="primary"):
    try:
        flagged = run_all_rules(transactions)

        unique_flagged = flagged.drop_duplicates(
            subset=["transaction_id"]
        )

        total_transactions = len(transactions)
        flagged_transactions = unique_flagged["transaction_id"].nunique()
        flag_rate = (
            flagged_transactions / total_transactions
            if total_transactions
            else 0
        )
        flagged_amount = unique_flagged["amount"].sum()

        st.subheader("Audit overview")

        column_1, column_2, column_3, column_4 = st.columns(4)

        column_1.metric(
            "Total Transactions",
            f"{total_transactions:,}",
        )
        column_2.metric(
            "Flagged Transactions",
            f"{flagged_transactions:,}",
        )
        column_3.metric(
            "Flag Rate",
            f"{flag_rate:.1%}",
        )
        column_4.metric(
            "Flagged Amount",
            f"${flagged_amount:,.0f}",
        )

        risk_counts = (
            flagged["detected_risk_type"]
            .value_counts()
            .rename_axis("risk_type")
            .reset_index(name="flag_count")
        )

        chart_1, chart_2 = st.columns(2)

        with chart_1:
            st.subheader("Risk flags by type")

            risk_chart = px.bar(
                risk_counts,
                x="risk_type",
                y="flag_count",
                color="risk_type",
                labels={
                    "risk_type": "Risk Type",
                    "flag_count": "Number of Flags",
                },
            )

            risk_chart.update_layout(
                showlegend=False,
                xaxis_tickangle=-25,
            )

            st.plotly_chart(
                risk_chart,
                use_container_width=True,
            )

        with chart_2:
            st.subheader("Risk flags by department")

            department_counts = (
                flagged.groupby("department")
                .size()
                .reset_index(name="flag_count")
                .sort_values("flag_count", ascending=False)
            )

            department_chart = px.bar(
                department_counts,
                x="department",
                y="flag_count",
                color="department",
                labels={
                    "department": "Department",
                    "flag_count": "Number of Flags",
                },
            )

            department_chart.update_layout(
                showlegend=False,
                xaxis_tickangle=-25,
            )

            st.plotly_chart(
                department_chart,
                use_container_width=True,
            )

        st.subheader("Flagged transactions")

        display_columns = [
            "transaction_id",
            "transaction_date",
            "vendor_name",
            "department",
            "amount",
            "detected_risk_type",
            "rule_evidence",
        ]

        st.dataframe(
            flagged[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        csv_output = flagged.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Flagged Transactions",
            data=csv_output,
            file_name="flagged_transactions.csv",
            mime="text/csv",
        )

        st.subheader("GenAI risk assessments")

        if GENAI_RESULTS_PATH.exists():
            genai_results = pd.read_csv(GENAI_RESULTS_PATH)

            successful_results = genai_results.loc[
                genai_results["genai_status"] == "success"
            ].copy()

            successful_results["classification_match"] = (
                successful_results["detected_risk_type"]
                == successful_results["genai_risk_type"]
            )

            success_rate = (
                len(successful_results) / len(genai_results)
                if len(genai_results)
                else 0
            )

            agreement_rate = (
                successful_results["classification_match"].mean()
                if len(successful_results)
                else 0
            )

            genai_column_1, genai_column_2, genai_column_3 = (
                st.columns(3)
            )

            genai_column_1.metric(
                "GenAI Assessments",
                f"{len(genai_results):,}",
            )
            genai_column_2.metric(
                "Structured Output Success",
                f"{success_rate:.1%}",
            )
            genai_column_3.metric(
                "Rule Classification Agreement",
                f"{agreement_rate:.1%}",
            )

            genai_display_columns = [
                "transaction_id",
                "detected_risk_type",
                "genai_risk_type",
                "genai_risk_level",
                "genai_explanation",
                "genai_recommended_procedure",
            ]

            st.dataframe(
                genai_results[genai_display_columns],
                use_container_width=True,
                hide_index=True,
            )

            genai_csv = genai_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download GenAI Assessments",
                data=genai_csv,
                file_name="genai_risk_assessments.csv",
                mime="text/csv",
            )

            st.caption(
                "GenAI outputs are decision-support signals and "
                "require human audit review."
            )

        else:
            st.warning(
                "No saved GenAI assessments were found. Run "
                "`python -m src.batch_enrichment` first."
            )
    except Exception as error:
        st.error(f"Analysis failed: {error}")
