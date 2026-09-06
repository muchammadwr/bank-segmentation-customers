import joblib
import pandas as pd
import streamlit as st

from preprocess import MODEL_DIR, load_preprocessors
from prediction import (
    predict_cluster,
    predict_classification,
)


# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Segmentasi Nasabah",
    page_icon="🏦",
)


# ==========================================
# LOAD ARTEFAK
# ==========================================
@st.cache_resource
def load_shared_resources():
    return load_preprocessors()


@st.cache_resource
def load_cluster_model():
    return joblib.load(MODEL_DIR / "model_clustering.h5")


@st.cache_resource
def load_classifier():
    model = joblib.load(MODEL_DIR / "decision_tree_model.h5")

    schema = joblib.load(MODEL_DIR / "classification_schema.joblib")

    return model, schema


st.title("🏦 Segmentasi Nasabah")
st.write("Masukkan data transaksi dalam satuan asli. Kelompok usia dihitung otomatis.")

try:
    encoders, scalers, age_config = load_shared_resources()

except FileNotFoundError as error:
    st.error(f"Artefak preprocessing belum tersedia: {error.filename}")
    st.stop()


# ==========================================
# FORM INPUT
# ==========================================
def transaction_form(prefix, categories):
    with st.form(key=f"{prefix}_form"):
        left, right = st.columns(2)

        with left:
            transaction_amount = st.number_input(
                "Jumlah transaksi",
                min_value=0.0,
                value=500.0,
                key=f"{prefix}_amount",
            )

            transaction_type = st.selectbox(
                "Jenis transaksi",
                options=categories["TransactionType"],
                key=f"{prefix}_type",
            )

            location = st.selectbox(
                "Lokasi",
                options=categories["Location"],
                key=f"{prefix}_location",
            )

            channel = st.selectbox(
                "Channel",
                options=categories["Channel"],
                key=f"{prefix}_channel",
            )

        with right:
            customer_age = st.number_input(
                "Usia nasabah",
                min_value=18,
                max_value=80,
                value=30,
                step=1,
                key=f"{prefix}_age",
            )

            customer_occupation = st.selectbox(
                "Pekerjaan",
                options=categories["CustomerOccupation"],
                key=f"{prefix}_occupation",
            )

            transaction_duration = st.number_input(
                "Durasi transaksi",
                min_value=0.0,
                value=60.0,
                help="Gunakan satuan yang sama dengan dataset.",
                key=f"{prefix}_duration",
            )

            # Dataset training setelah filtering hanya
            # memiliki LoginAttempts = 1.
            login_attempts = st.number_input(
                "Jumlah percobaan login",
                min_value=1,
                value=1,
                step=1,
                disabled=True,
                help=(
                    "Dibatasi ke 1 karena hanya nilai ini "
                    "yang tersedia pada data training."
                ),
                key=f"{prefix}_login",
            )

            account_balance = st.number_input(
                "Saldo rekening",
                min_value=0.0,
                value=5000.0,
                key=f"{prefix}_balance",
            )

        submitted = st.form_submit_button(
            "Prediksi",
            use_container_width=True,
        )

    data = pd.DataFrame(
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
            }
        ]
    )

    return data, submitted


# ==========================================
# TAMPILKAN HASIL TERSIMPAN
# ==========================================
def display_result(state_key, target_column):
    if state_key not in st.session_state:
        return

    result = st.session_state[state_key]
    label = int(result[target_column].iloc[0])

    st.success(f"Hasil prediksi: Cluster {label}")

    st.dataframe(
        result,
        hide_index=True,
        use_container_width=True,
    )


# ==========================================
# TABS
# ==========================================
tab_cluster, tab_classification = st.tabs(
    [
        "📊 Clustering",
        "🎯 Classification",
    ]
)


with tab_cluster:
    st.subheader("Clustering Nasabah")
    st.write("Menentukan cluster menggunakan KMeans tanpa PCA.")

    try:
        cluster_model = load_cluster_model()

    except FileNotFoundError as error:
        st.error(f"Model belum tersedia: {error.filename}")

    else:
        cluster_categories = {
            column: encoder.classes_.tolist() for column, encoder in encoders.items()
        }

        cluster_data, cluster_submitted = transaction_form(
            prefix="clustering",
            categories=cluster_categories,
        )

        if cluster_submitted:
            st.session_state.pop("cluster_result", None)

            try:
                result = predict_cluster(
                    data=cluster_data,
                    model=cluster_model,
                    encoders=encoders,
                    scalers=scalers,
                    age_config=age_config,
                )

                st.session_state["cluster_result"] = result

            except (ValueError, KeyError) as error:
                st.error(f"Prediksi gagal: {error}")

        display_result(
            state_key="cluster_result",
            target_column="Cluster",
        )


with tab_classification:
    st.subheader("Classification Nasabah")
    st.write("Memprediksi label cluster menggunakan Decision Tree.")

    try:
        classifier, schema = load_classifier()

    except FileNotFoundError as error:
        st.error(f"Model atau schema belum tersedia: {error.filename}")

    else:
        classification_data, classification_submitted = transaction_form(
            prefix="classification",
            categories=schema["categories"],
        )

        if classification_submitted:
            st.session_state.pop("classification_result", None)

            try:
                result = predict_classification(
                    data=classification_data,
                    model=classifier,
                    schema=schema,
                    scalers=scalers,
                    age_config=age_config,
                )

                st.session_state["classification_result"] = result

            except (ValueError, KeyError) as error:
                st.error(f"Prediksi gagal: {error}")

        display_result(
            state_key="classification_result",
            target_column="Target",
        )

        st.caption(
            "Target berasal dari label clustering. Hasil ini bukan penilaian fraud."
        )
