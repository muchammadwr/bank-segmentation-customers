import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation Prototype",
    page_icon="👥",
    layout="wide"
)

FEATURE_COLUMNS = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance"
]

CLUSTER_PROFILES = {
    0: {
        "name": (
            "Nasabah dengan Durasi Transaksi "
            "Sedikit Lebih Lama"
        ),
        "description": (
            "Nasabah memiliki usia, durasi transaksi, "
            "dan saldo rekening yang sedikit lebih tinggi "
            "dibandingkan rata-rata. Nilai transaksinya "
            "sedikit lebih rendah."
        )
    },
    1: {
        "name": (
            "Nasabah dengan Nilai Transaksi "
            "Sedikit Lebih Tinggi"
        ),
        "description": (
            "Nasabah memiliki nilai transaksi sedikit "
            "lebih tinggi, tetapi usia, durasi transaksi, "
            "dan saldo rekeningnya sedikit lebih rendah "
            "dibandingkan rata-rata."
        )
    }
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(
        "models/scaler_model.joblib"
    )

    pca = joblib.load(
        "models/PCA_model_clustering.h5"
    )

    classifier = joblib.load(
        "models/tuning_classification.h5"
    )

    return scaler, pca, classifier


scaler, pca, classifier = load_artifacts()


def predict_cluster(input_df):
    ordered_df = input_df.reindex(
        columns=FEATURE_COLUMNS
    )

    scaled_data = scaler.transform(
        ordered_df
    )

    # Gunakan jika classifier dilatih
    # menggunakan hasil PCA
    pca_data = pca.transform(
        scaled_data
    )

    cluster = int(
        classifier.predict(pca_data)[0]
    )

    confidence = None

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(
            pca_data
        )[0]

        confidence = float(
            probabilities.max()
        )

    return cluster, confidence


st.title("👥 Customer Segmentation Prototype")

st.caption(
    "Prototype for classifying customers based on "
    "their transaction characteristics."
)

prediction_tab, profile_tab, about_tab = st.tabs([
    "Segment Prediction",
    "Cluster Profiles",
    "About Model"
])


with prediction_tab:
    st.subheader("Customer Transaction Information")

    col1, col2 = st.columns(2)

    with col1:
        transaction_amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30
        )

        transaction_duration = st.number_input(
            "Transaction Duration",
            min_value=0.0,
            value=60.0
        )

    with col2:
        login_attempts = st.number_input(
            "Login Attempts",
            min_value=0,
            value=1
        )

        account_balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=5000.0
        )

    if st.button(
        "Predict Customer Segment",
        type="primary",
        use_container_width=True
    ):
        input_df = pd.DataFrame([{
            "TransactionAmount": transaction_amount,
            "CustomerAge": customer_age,
            "TransactionDuration": transaction_duration,
            "LoginAttempts": login_attempts,
            "AccountBalance": account_balance
        }])

        try:
            cluster, confidence = predict_cluster(
                input_df
            )

            profile = CLUSTER_PROFILES[cluster]

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Cluster",
                f"Cluster {cluster}"
            )

            if confidence is not None:
                col2.metric(
                    "Prediction Confidence",
                    f"{confidence:.2%}"
                )

            st.success(
                f"Customer belongs to Cluster {cluster}"
            )

            st.markdown(
                f"### {profile['name']}"
            )

            st.write(
                profile["description"]
            )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )


with profile_tab:
    col1, col2 = st.columns(2)

    with col1:
        st.info("Cluster 0")
        st.subheader(
            CLUSTER_PROFILES[0]["name"]
        )
        st.write(
            CLUSTER_PROFILES[0]["description"]
        )

    with col2:
        st.info("Cluster 1")
        st.subheader(
            CLUSTER_PROFILES[1]["name"]
        )
        st.write(
            CLUSTER_PROFILES[1]["description"]
        )


with about_tab:
    st.subheader("About This Prototype")

    st.write("""
    This prototype classifies customers into transaction
    segments generated from an unsupervised clustering
    process. The cluster labels are then learned by a
    classification model to support predictions for new
    customer transactions.
    """)

    st.warning(
        "The differences between the clusters are "
        "relatively small. Predictions should therefore "
        "be treated as exploratory segmentation results."
    )