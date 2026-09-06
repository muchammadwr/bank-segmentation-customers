import pandas as pd
import streamlit as st


# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Segmentasi Nasabah",
    page_icon="🏦",
)

st.title("🏦 Segmentasi Nasabah")
st.write("Analisis data transaksi menggunakan model clustering dan classification.")


# ==========================================
# FORM INPUT YANG DIGUNAKAN KEDUA TAB
# ==========================================
def transaction_form(prefix):
    # Pilihan kategori ini hanya contoh.
    # Nantinya gunakan seluruh kategori dari training.
    with st.form(key=f"{prefix}_form"):
        col1, col2 = st.columns(2)

        with col1:
            transaction_amount = st.number_input(
                "Jumlah transaksi",
                min_value=0.0,
                value=500.0,
                key=f"{prefix}_amount",
            )

            transaction_type = st.selectbox(
                "Jenis transaksi",
                ["Debit", "Credit"],
                key=f"{prefix}_type",
            )

            location = st.selectbox(
                "Lokasi",
                ["San Diego", "Houston", "Mesa"],
                key=f"{prefix}_location",
            )

            channel = st.selectbox(
                "Channel",
                ["ATM", "Online", "Branch"],
                key=f"{prefix}_channel",
            )

        with col2:
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
                ["Doctor", "Engineer", "Retired", "Student"],
                key=f"{prefix}_occupation",
            )

            transaction_duration = st.number_input(
                "Durasi transaksi (ikuti satuan dataset)",
                min_value=0.0,
                value=60.0,
                key=f"{prefix}_duration",
            )

            login_attempts = st.number_input(
                "Jumlah percobaan login",
                min_value=1,
                value=1,
                step=1,
                key=f"{prefix}_login",
            )

            account_balance = st.number_input(
                "Saldo rekening",
                min_value=0.0,
                value=5000.0,
                key=f"{prefix}_balance",
            )

        submitted = st.form_submit_button(
            "Proses data",
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
# DUA TAB DALAM SATU HALAMAN
# ==========================================
tab_clustering, tab_classification = st.tabs(
    [
        "📊 Clustering",
        "🎯 Classification",
    ]
)


with tab_clustering:
    st.subheader("Clustering Nasabah")
    st.write("Tentukan kelompok transaksi menggunakan KMeans.")

    cluster_data, cluster_submitted = transaction_form(prefix="clustering")

    if cluster_submitted:
        st.session_state["clustering_input"] = cluster_data

    if "clustering_input" in st.session_state:
        st.write("**Data yang dikirim:**")
        st.dataframe(
            st.session_state["clustering_input"],
            hide_index=True,
            use_container_width=True,
        )

        st.info("Form sudah berfungsi. Prediksi KMeans belum dihubungkan.")


with tab_classification:
    st.subheader("Classification Nasabah")
    st.write(
        "Prediksi Target menggunakan model classification "
        "yang mempelajari label cluster."
    )

    classification_data, classification_submitted = transaction_form(
        prefix="classification"
    )

    if classification_submitted:
        st.session_state["classification_input"] = classification_data

    if "classification_input" in st.session_state:
        st.write("**Data yang dikirim:**")
        st.dataframe(
            st.session_state["classification_input"],
            hide_index=True,
            use_container_width=True,
        )

        st.info("Form sudah berfungsi. Prediksi classification belum dihubungkan.")
