from pathlib import Path

import joblib
import pandas as pd

from preprocessor import preprocess_clustering

# ==========================================
# PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ==========================================
# LOAD MODELS
# ==========================================


clustering_model = joblib.load(MODEL_DIR / "model_clustering.h5")

# ==========================================
# CLUSTER INFORMATION
# ==========================================

CLUSTER_INFORMATION = {
    0: {
        "label": "Average Stable Customer",
        "description": (
            "Nasabah dengan usia, durasi transaksi, dan saldo sedikit "
            "lebih tinggi, tetapi nilai transaksinya sedikit lebih rendah. "
            "Karakteristiknya masih mendekati rata-rata populasi."
        ),
    },
    1: {
        "label": "Active Transaction Customer",
        "description": (
            "Nasabah dengan usia, durasi transaksi, dan saldo sedikit "
            "lebih rendah, tetapi nilai transaksinya sedikit lebih tinggi."
        ),
    },
}


# ==========================================
# CLUSTER PREDICTION
# ==========================================


def predict_cluster(data: pd.DataFrame) -> dict:
    """
    Melakukan preprocessing, PCA, dan prediksi cluster.

    Args:
        data:
            DataFrame berisi data mentah dari Streamlit.

    Returns:
        Dictionary berisi hasil preprocessing, PCA,
        nomor cluster, label, dan deskripsi cluster.
    """

    processed_data = preprocess_clustering(data)
    prediction = clustering_model.predict(processed_data)
    cluster_number = int(prediction[0])
    cluster_info = CLUSTER_INFORMATION.get(
        cluster_number,
        {
            "label": f"Cluster {cluster_number}",
            "description": "Deskripsi cluster belum tersedia.",
        },
    )

    return {
        "cluster": cluster_number,
        "label": cluster_info["label"],
        "description": cluster_info["description"],
        "processed_data": processed_data,
    }
