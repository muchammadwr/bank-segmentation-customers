from pathlib import Path

import joblib
import pandas as pd


# ==========================================
# PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ==========================================
# LOAD CLUSTERING PREPROCESSORS
# ==========================================

encoder_transaction_type = joblib.load(MODEL_DIR / "encoder_TransactionType.joblib")
encoder_location = joblib.load(MODEL_DIR / "encoder_Location.joblib")
encoder_channel = joblib.load(MODEL_DIR / "encoder_Channel.joblib")
encoder_customer_occupation = joblib.load(
    MODEL_DIR / "encoder_CustomerOccupation.joblib"
)
encoder_age_group = joblib.load(MODEL_DIR / "encoder_AgeGroupBin.joblib")
scaler_numerical = joblib.load(MODEL_DIR / "scaler_numerical_cols.joblib")

decision_tree_model = joblib.load(MODEL_DIR / "decision_tree_model.h5")
tuning_classification = joblib.load(MODEL_DIR / "tuning_classification.h5")
explore_random_forest_classifier_classification = joblib.load(
    MODEL_DIR / "explore_random_forest_classifier_classification.h5"
)


# ==========================================
# LOAD CLASSIFICATION PREPROCESSING METADATA
# ==========================================

# classification_preprocessing = joblib.load(
#     MODEL_DIR / "classification_preprocessing.joblib"
# )


# ==========================================
# FEATURES
# ==========================================

FEATURES = [
    "TransactionAmount",
    "TransactionType",
    "Location",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "AgeGroupBin",
]

CATEGORICAL_COLUMNS = [
    "TransactionType",
    "Location",
    "Channel",
    "CustomerOccupation",
    "AgeGroupBin",
]


NUMERICAL_FEATURES = scaler_numerical.feature_names_in_.tolist()


# ==========================================
# ENCODERS
# ==========================================

ENCODERS = {
    "TransactionType": encoder_transaction_type,
    "Location": encoder_location,
    "Channel": encoder_channel,
    "CustomerOccupation": encoder_customer_occupation,
    "AgeGroupBin": encoder_age_group,
}


# ==========================================
# PREPROCESSING
# ==========================================


def preprocess_clustering(data):
    result = data.copy()
    result = result.loc[:, FEATURES]

    for column in CATEGORICAL_COLUMNS:
        encoder = ENCODERS[column]
        result[column] = encoder.transform(result[column].astype(str))

    result[NUMERICAL_FEATURES] = scaler_numerical.transform(result[NUMERICAL_FEATURES])
    return result.loc[:, FEATURES]


