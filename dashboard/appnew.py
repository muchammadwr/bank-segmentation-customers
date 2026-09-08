import joblib
import pandas as pd
import streamlit as st
from preprocessor import preprocess_clustering


# ==========================================
# CONFIGURATION PAGE
# ==========================================


st.set_page_config(
    page_title="Bank Transaction Segmentation",
    page_icon="🏦",
)

# ==========================================
# LOAD
# ==========================================


st.title("🏦 Bank Transaction Segmentation")
st.write(
    "Enter customer and transaction information to identify the transaction segment."
)


tab_clustering, tab_classification = st.tabs(
    [
        "📊 Clustering",
        "🎯 Classification",
    ]
)


# ==========================================
# CATEGORICAL OPTIONS
# ==========================================
locations = [
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


TransactionType = ["Debit", "Credit"]

Channel = ["ATM", "Online", "Branch"]

CustomerOccupation = ["Doctor", "Student", "Retired", "Engineer"]


# ==========================================
# AGE GROUP FUNCTION
# ==========================================
def create_age_group(age):
    if 18 <= age <= 32:
        return "rendah"

    elif 33 <= age <= 55:
        return "sedang"

    elif 56 <= age <= 80:
        return "tinggi"

    else:
        raise ValueError("CustomerAge harus berada antara 18 dan 80.")


# ==========================================
# CLUSTERING TAB
# ==========================================

with tab_clustering:
    st.subheader("Customer and Transaction Input")
    with st.form("clustering_form"):
        left_column, right_column = st.columns(2)
        with left_column:
            transaction_amount = st.number_input(
                "Transaction Amount", min_value=0.0, value=100.0
            )

            transaction_type = st.selectbox("Transaction Type", options=TransactionType)

            location = st.selectbox("Location", options=locations)

            channel = st.selectbox("Channel", options=Channel)

        with right_column:
            customer_age = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=80,
                value=30,
                step=1,
            )
            customer_occupation = st.selectbox(
                "Customer Occupation", CustomerOccupation
            )

            transaction_duration = st.number_input(
                "Transaction Duration",
                min_value=10.0,
                max_value=300.0,
                value=100.0,
            )

            login_attempts = st.number_input(
                "Login Attempts",
                min_value=1,
                value=1,
                step=1,
            )

            account_balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                value=5000.0,
            )

        predict = st.form_submit_button(
            "Prediction",
            use_container_width=True,
        )

        if predict:
            try:
                age_group = create_age_group(customer_age)
                input_data = pd.DataFrame(
                    [
                        {
                            "TransactionAmount": transaction_amount,
                            "TransactionType": transaction_type,
                            "Location": location,
                            "Channel": channel,
                            "CustomerAge": customer_age,
                            "CustomerOccupation": customer_occupation,
                            "TransactionDuration": transaction_duration,
                            "LoginAttempts": login_attempts,
                            "AccountBalance": account_balance,
                            "AgeGroupBin": age_group,
                        }
                    ],
                )
                st.success("Input data was successfully created.")
                st.dataframe(input_data)
                processed_data = preprocess_clustering(input_data)
                st.success("Data was successfully preprocessed.")

                with st.expander("View Raw Data"):
                    st.dataframe(
                        input_data,
                        hide_index=True,
                        use_container_width=True,
                    )

                with st.expander("View Preprocessed Data"):
                    st.dataframe(
                        processed_data,
                        hide_index=True,
                        use_container_width=True,
                    )

            except ValueError as error:
                st.error(str(error))


with tab_classification:
    st.subheader("Customer and Transaction Input")
    st.info(
        "Classification form will be added after the clustering form works correctly."
    )
