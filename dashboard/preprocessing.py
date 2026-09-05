import joblib
import pandas as pd


# ================================================
# LOAD MODELS
# ================================================

# Encoder
encoder_AgeGroup = joblib.load("models/encoder_AgeGroup.joblib")
encoder_Channel = joblib.load("models/encoder_Channel.joblib")
encoder_CustomerOccupation = joblib.load("models/encoder_CustomerOccupation.joblib")
encoder_Location = joblib.load("models/encoder_Location.joblib")
encoder_TransactionType = joblib.load("models/encoder_TransactionType.joblib")

# Scaler
scaler_TransactionAmount = joblib.load("models/scaler_TransactionAmount.joblib")
scaler_AccountBalance = joblib.load("models/scaler_AccountBalance.joblib")
scaler_CustomerAge = joblib.load("models/scaler_CustomerAge.joblib")
scaler_TransactionDuration = joblib.load("models/scaler_TransactionDuration.joblib")
scaler_LoginAttempts = joblib.load("models/scaler_LoginAttempts.joblib")


# ================================================
# PREPROCESSING
# ================================================
def data_preprocessing(data):
    """PPreprocessing data

    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction

    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
