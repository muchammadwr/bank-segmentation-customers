from pathlib import Path

import joblib


# ==========================================
# PATH
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ==========================================
# LOAD MODEL
# ==========================================
cluster_model = joblib.load(MODEL_DIR / "model_clustering.h5")


# ==========================================
# CLUSTER PREDICTION
# ==========================================
def predict_cluster(data):
    """
    Memprediksi cluster dari data yang sudah
    melalui encoding dan scaling.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame hasil preprocess_clustering().

    Returns
    -------
    int
        Label cluster untuk baris pertama.
    """
    prediction = cluster_model.predict(data)

    cluster_label = int(prediction[0])

    return cluster_label
