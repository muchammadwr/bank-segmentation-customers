from pathlib import Path

import joblib
import pandas as pd

from preprocessor import preprocess_clustering

# ==========================================
# LOAD MODELS
# ==========================================

clustering_bundle = joblib.load("clustering_bundle.joblib")

classification_pipeline = joblib.load("classification_pipeline.joblib")


# ==========================================
# CLUSTERING PREDICTION
# ==========================================


def predict_cluster(data: pd.DataFrame) -> dict:
    prepared_data = preprocess_clustering(
        data=data,
        bundle=clustering_bundle,
    )

    prediction = clustering_bundle["pipeline"].predict(prepared_data)

    cluster = int(prediction[0])

    return {
        "prediction": cluster,
        "prepared_data": prepared_data,
    }


# ==========================================
# CLASSIFICATION PREDICTION
# ==========================================


def predict_classification(data: pd.DataFrame) -> dict:
    feature_columns = classification_pipeline.feature_names_in_.tolist()

    prepared_data = data.loc[:, feature_columns].copy()

    prediction = classification_pipeline.predict(prepared_data)

    target = int(prediction[0])

    return {
        "prediction": target,
        "prepared_data": prepared_data,
    }
