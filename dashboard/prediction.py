from pathlib import Path

import joblib
import pandas as pd

from preprocessor import (
    preprocess_clustering,
    preprocess_classification,
)


# ==========================================
# PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ==========================================
# LOAD MODELS
# ==========================================

clustering_model = joblib.load(MODEL_DIR / "model_clustering.h5")

classification_model = joblib.load(MODEL_DIR / "tuning_classification.h5")


# ==========================================
# RESULT INFORMATION
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


CLASSIFICATION_INFORMATION = {
    0: {
        "label": "Average Stable Customer",
        "description": (
            "Model classification memprediksi input sebagai segmen pelanggan Cluster 0."
        ),
    },
    1: {
        "label": "Active Transaction Customer",
        "description": (
            "Model classification memprediksi input sebagai segmen pelanggan Cluster 1."
        ),
    },
}


# ==========================================
# CLUSTERING PREDICTION
# ==========================================


def predict_cluster(data: pd.DataFrame) -> dict:
    processed_data = preprocess_clustering(data)

    prediction = clustering_model.predict(processed_data)

    cluster_number = int(prediction[0])

    information = CLUSTER_INFORMATION.get(
        cluster_number,
        {
            "label": f"Cluster {cluster_number}",
            "description": "Deskripsi cluster belum tersedia.",
        },
    )

    return {
        "cluster": cluster_number,
        "label": information["label"],
        "description": information["description"],
        "processed_data": processed_data,
    }


# ==========================================
# CLASSIFICATION PREDICTION
# ==========================================


def predict_classification(data: pd.DataFrame) -> dict:
    processed_data = preprocess_classification(data)

    prediction = classification_model.predict(processed_data)

    target = int(prediction[0])

    information = CLASSIFICATION_INFORMATION.get(
        target,
        {
            "label": f"Class {target}",
            "description": "Deskripsi kelas belum tersedia.",
        },
    )

    return {
        "target": target,
        "label": information["label"],
        "description": information["description"],
        "processed_data": processed_data,
    }
