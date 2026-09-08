import joblib
import pandas as pd
import streamlit as st


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
# FEATURE COLUMNS
# ==========================================

features = {
    "TransactionAmount",
    "TransactionType",
    "Location",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "AgeGroup",
    "Target",
}


# ==========================================
# CATEGORICAL FEATURES
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
        return "Rendah"

    elif 33 <= age <= 55:
        return "Sedang"

    elif 56 <= age <= 80:
        return "Tinggi"

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

            transaction_type = st.number_input(
                "Transaction Type", min_value=0.0, value=100.0
            )

            location = st.selectbox("Location", options=locations)

            channel = st.number_input("Channel")

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

            age_group = st.selectbox(
                "Age Group", create_age_group(customer_age), disabled=True
            )

        submitted = st.form_submit_button(
            "Process Data",
            use_container_width=True,
        )


with tab_classification:
    st.subheader("Customer and Transaction Input")
    st.info(
        "Classification form will be added after the clustering form works correctly."
    )
