from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
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
]

NUMERICAL_COLUMNS = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
]

CATEGORICAL_COLUMNS = [
    "TransactionType",
    "Location",
    "Channel",
    "CustomerOccupation",
    "AgeGroup",
]


def load_preprocessors():
    encoders = {
        column: joblib.load(MODEL_DIR / f"encoder_{column}.joblib")
        for column in CATEGORICAL_COLUMNS
    }

    scalers = {
        column: joblib.load(MODEL_DIR / f"scaler_{column}.joblib")
        for column in NUMERICAL_COLUMNS
    }

    age_config = joblib.load(MODEL_DIR / "age_group_bins.joblib")

    return encoders, scalers, age_config


def validate_categories(series, categories):
    unknown = ~series.isin(categories)

    if unknown.any():
        values = series.loc[unknown].unique().tolist()

        raise ValueError(f"Kategori tidak dikenal pada {series.name}: {values}")


def add_age_group(data, age_scaler, age_config):
    result = data.copy()

    # Batas bin training berasal dari CustomerAge
    # yang sudah distandardisasi.
    scaled_age = age_scaler.transform(result[["CustomerAge"]]).ravel()

    bins = np.asarray(
        age_config["bins"],
        dtype=float,
    ).copy()

    # Kebijakan input baru: usia di luar rentang training
    # masuk kelompok terluar; batas internal tidak berubah.
    bins[0] = -np.inf
    bins[-1] = np.inf

    result["AgeGroup"] = pd.cut(
        scaled_age,
        bins=bins,
        labels=age_config["labels"],
        include_lowest=True,
    )

    # CustomerAge pada result tetap dalam satuan asli.
    return result


def preprocess_clustering(data, encoders, scalers):
    result = data.loc[:, FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_COLUMNS:
        encoder = encoders[column]

        validate_categories(
            result[column],
            encoder.classes_,
        )

        result[column] = encoder.transform(result[column])

    for column in NUMERICAL_COLUMNS:
        result[column] = scalers[column].transform(result[[column]]).ravel()

    return result.loc[:, FEATURE_COLUMNS]


def preprocess_classification(data, schema):
    result = data.loc[:, FEATURE_COLUMNS].copy()

    for column, categories in schema["categories"].items():
        validate_categories(
            result[column],
            categories,
        )

        # Seluruh kategori training harus tersedia
        # agar drop_first memakai kategori acuan yang sama.
        result[column] = pd.Categorical(
            result[column],
            categories=categories,
        )

    result = pd.get_dummies(
        result,
        columns=list(schema["categories"]),
        drop_first=True,
    )

    return result.reindex(
        columns=schema["feature_columns"],
        fill_value=0,
    )
