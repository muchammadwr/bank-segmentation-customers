from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==========================================
# PATH
# ==========================================


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ==========================================
# FEATURES TRAINING
# ==========================================


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

# ==========================================
# LOAD ENCODER
# ==========================================


encoder_TransactionType = joblib.load(MODEL_DIR / "encoder_TransactionType.joblib")
encoder_Location = joblib.load(MODEL_DIR / "encoder_Location.joblib")
encoder_Channel = joblib.load(MODEL_DIR / "encoder_Channel.joblib")
encoder_CustomerOccupation = joblib.load(
    MODEL_DIR / "encoder_CustomerOccupation.joblib"
)
encoder_AgeGroup = joblib.load(MODEL_DIR / "encoder_AgeGroup.joblib")
age_group_config = joblib.load(MODEL_DIR / "age_group_bins.joblib")


# ==========================================
# LOAD SCALER
# ==========================================


scaler_TransactionAmount = joblib.load(MODEL_DIR / "scaler_TransactionAmount.joblib")
scaler_CustomerAge = joblib.load(MODEL_DIR / "scaler_CustomerAge.joblib")
scaler_TransactionDuration = joblib.load(
    MODEL_DIR / "scaler_TransactionDuration.joblib"
)
scaler_LoginAttempts = joblib.load(MODEL_DIR / "scaler_LoginAttempts.joblib")
scaler_AccountBalance = joblib.load(MODEL_DIR / "scaler_AccountBalance.joblib")


def load_preprocessors():

    encoders = {
        "TransactionType": encoder_TransactionType,
        "Location": encoder_Location,
        "Channel": encoder_Channel,
        "CustomerOccupation": encoder_CustomerOccupation,
        "AgeGroup": encoder_AgeGroup,
    }

    scalers = {
        "TransactionAmount": scaler_TransactionAmount,
        "CustomerAge": scaler_CustomerAge,
        "TransactionDuration": scaler_TransactionDuration,
        "LoginAttempts": scaler_LoginAttempts,
        "AccountBalance": scaler_AccountBalance,
    }

    return encoders, scalers, age_group_config


# ==========================================
# CREATE AGE GROUP
# ==========================================
def add_age_group(data, age_scaler, age_config):
    data = data.copy()
    scaled_age = age_scaler.transform(data[["CustomerAge"]]).ravel()
    bins = np.asarray(
        age_config["bins"],
        dtype=float,
    ).copy()
    bins[0] = -np.inf
    bins[-1] = np.inf
    data["AgeGroup"] = pd.cut(
        scaled_age,
        bins=bins,
        labels=age_config["labels"],
        include_lowest=True,
    )
    return data


# ==========================================
# CATEGORICAL VALIDATION
# ==========================================


def validate_categories(series, categories):
    unknown = ~series.isin(categories)
    if unknown.any():
        values = series.loc[unknown].unique().tolist()
        raise ValueError(f"Kategori tidak dikenal pada {series.name}: {values}")


# ==========================================
# PREPROCESSING CLUSTERING
# ==========================================


def preprocess_clustering(data, encoders, scalers):
    data = data.loc[:, FEATURE_COLUMNS].copy()
    df = pd.DataFrame(index=data.index)
    for column, encoder in encoders.items():
        validate_categories(
            data[column],
            encoder.classes_,
        )
    # Scaler
    df["TransactionAmount"] = (
        scalers["TransactionAmount"].transform(data[["TransactionAmount"]]).ravel()
    )
    df["CustomerAge"] = scalers["CustomerAge"].transform(data[["CustomerAge"]]).ravel()
    df["TransactionDuration"] = (
        scalers["TransactionDuration"].transform(data[["TransactionDuration"]]).ravel()
    )
    df["LoginAttempts"] = (
        scalers["LoginAttempts"].transform(data[["LoginAttempts"]]).ravel()
    )
    df["AccountBalance"] = (
        scalers["AccountBalance"].transform(data[["AccountBalance"]]).ravel()
    )
    # Encoding
    df["TransactionType"] = encoders["TransactionType"].transform(
        data["TransactionType"]
    )
    df["Location"] = encoders["Location"].transform(data["Location"])
    df["Channel"] = encoders["Channel"].transform(data["Channel"])
    df["CustomerOccupation"] = encoders["CustomerOccupation"].transform(
        data["CustomerOccupation"]
    )
    df["AgeGroup"] = encoders["AgeGroup"].transform(data["AgeGroup"])
    return df.loc[:, FEATURE_COLUMNS]


# ==========================================
# PREPROCESSING CLASSIFICATION
# ==========================================


def preprocess_classification(data, schema):
    data = data.loc[:, FEATURE_COLUMNS].copy()
    for column, categories in schema["categories"].items():
        validate_categories(
            data[column],
            categories,
        )
        data[column] = pd.Categorical(
            data[column],
            categories=categories,
        )
    df = pd.get_dummies(
        data,
        columns=list(schema["categories"]),
        drop_first=True,
    )
    return df.reindex(
        columns=schema["feature_columns"],
        fill_value=0,
    )
