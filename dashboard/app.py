import pandas as pd
import streamlit as st

from prediction import (
    predict_classification,
    predict_cluster,
)
from preprocessor import create_age_group


st.set_page_config(
    page_title="Bank Prototype",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 Bank Prototype")
st.write("Enter customer and transaction information to identify the customer segment.")


LOCATIONS = [
    "San Diego",
    "Houston",
    "Mesa",
    "Raleigh",
    "Oklahoma City",
    "Seattle",
    "Indianapolis",
    "Detroit",
    "Nashville",
    "Albuquerque",
    "Memphis",
    "Louisville",
    "Denver",
    "Austin",
    "Columbus",
    "Los Angeles",
    "Las Vegas",
    "Milwaukee",
    "Miami",
    "Baltimore",
    "San Francisco",
    "San Antonio",
    "Philadelphia",
    "Charlotte",
    "Tucson",
    "Kansas City",
    "Virginia Beach",
    "Omaha",
    "Dallas",
    "Atlanta",
    "Boston",
    "Jacksonville",
    "Fort Worth",
    "Colorado Springs",
    "Sacramento",
    "Fresno",
    "Portland",
    "Washington",
    "Chicago",
    "New York",
    "Phoenix",
    "San Jose",
    "El Paso",
]

TRANSACTION_TYPES = ["Debit", "Credit"]
CHANNELS = ["ATM", "Online", "Branch"]
OCCUPATIONS = ["Doctor", "Student", "Retired", "Engineer"]


def transaction_form(form_key):
    with st.form(form_key):
        left_column, right_column = st.columns(2)

        with left_column:
            transaction_amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0,
                key=f"{form_key}_amount",
            )

            transaction_type = st.selectbox(
                "Transaction Type",
                TRANSACTION_TYPES,
                key=f"{form_key}_type",
            )

            location = st.selectbox(
                "Location",
                LOCATIONS,
                key=f"{form_key}_location",
            )

            channel = st.selectbox(
                "Channel",
                CHANNELS,
                key=f"{form_key}_channel",
            )

        with right_column:
            customer_age = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=80,
                value=30,
                step=1,
                key=f"{form_key}_age",
            )

            customer_occupation = st.selectbox(
                "Customer Occupation",
                OCCUPATIONS,
                key=f"{form_key}_occupation",
            )

            transaction_duration = st.number_input(
                "Transaction Duration",
                min_value=0.0,
                value=100.0,
                key=f"{form_key}_duration",
            )

            login_attempts = st.number_input(
                "Login Attempts",
                min_value=1,
                value=1,
                step=1,
                key=f"{form_key}_login",
            )

            account_balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                value=5000.0,
                key=f"{form_key}_balance",
            )

        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True,
        )

    age_group = create_age_group(customer_age)

    data = pd.DataFrame(
        [
            {
                "TransactionAmount": transaction_amount,
                "CustomerAge": customer_age,
                "TransactionDuration": transaction_duration,
                "LoginAttempts": login_attempts,
                "AccountBalance": account_balance,
                "TransactionType": transaction_type,
                "Location": location,
                "Channel": channel,
                "CustomerOccupation": customer_occupation,
                "AgeGroupBin": age_group,
            }
        ]
    )

    return data, submitted


tab_clustering, tab_classification = st.tabs(
    [
        "📊 Clustering",
        "🎯 Classification",
    ]
)


with tab_clustering:
    st.subheader("Clustering")

    cluster_data, cluster_submitted = transaction_form("clustering_form")

    if cluster_submitted:
        try:
            result = predict_cluster(cluster_data)

            st.success(f"Prediction result: Cluster {result['prediction']}")

            with st.expander("View input data"):
                st.dataframe(
                    cluster_data,
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


with tab_classification:
    st.subheader("Classification")

    classification_data, classification_submitted = transaction_form(
        "classification_form"
    )

    if classification_submitted:
        try:
            result = predict_classification(classification_data)

            st.success(f"Prediction result: Class {result['prediction']}")

            with st.expander("View input data"):
                st.dataframe(
                    classification_data,
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
