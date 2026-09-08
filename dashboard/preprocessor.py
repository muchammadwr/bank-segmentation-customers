from pathlib import Path

import joblib

# ==========================================
# PATH
# ==========================================


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ==========================================
# FEATURE COLUMNS
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
    "AgeGroupBin",
]


CATEGORICAL_COLUMNS = [
    "TransactionType",
    "Location",
    "Channel",
    "CustomerOccupation",
    "AgeGroupBin",
]


# ==========================================
# LOAD PREPROCESSING OBJECTS
# ==========================================
encoder_transaction_type = joblib.load(MODEL_DIR / "encoder_TransactionType.joblib")

encoder_location = joblib.load(MODEL_DIR / "encoder_Location.joblib")

encoder_channel = joblib.load(MODEL_DIR / "encoder_Channel.joblib")

encoder_customer_occupation = joblib.load(
    MODEL_DIR / "encoder_CustomerOccupation.joblib"
)

encoder_age_group = joblib.load(MODEL_DIR / "encoder_AgeGroupBin.joblib")

scaler_numerical = joblib.load(MODEL_DIR / "scaler_numerical_cols.joblib")

NUMERICAL_COLUMNS = scaler_numerical.feature_names_in_.tolist()
ENCODERS = {
    "TransactionType": encoder_transaction_type,
    "Location": encoder_location,
    "Channel": encoder_channel,
    "CustomerOccupation": encoder_customer_occupation,
    "AgeGroupBin": encoder_age_group,
}


# ==========================================
# VALIDATION
# ==========================================


def validate_input_columns(data):
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Kolom input belum lengkap: {missing_columns}")


def validate_category(data, column, encoder):
    unknown_values = data.loc[
        ~data[column].isin(encoder.classes_),
        column,
    ].unique()

    if len(unknown_values) > 0:
        raise ValueError(
            f"Kategori tidak dikenali pada {column}: {unknown_values.tolist()}"
        )


# ==========================================
# CLUSTERING PREPROCESSING
# ==========================================
def preprocess_clustering(data):
    """
    Mengubah data mentah dari Streamlit menjadi
    data numerik sesuai preprocessing training.
    """
    result = data.copy()
    validate_input_columns(result)
    result = result.loc[:, FEATURE_COLUMNS]
    for column in CATEGORICAL_COLUMNS:
        encoder = ENCODERS[column]

        validate_category(
            result,
            column,
            encoder,
        )
        result[column] = encoder.transform(result[column].astype(str))

    result[NUMERICAL_COLUMNS] = scaler_numerical.transform(result[NUMERICAL_COLUMNS])

    return result.loc[:, FEATURE_COLUMNS]
