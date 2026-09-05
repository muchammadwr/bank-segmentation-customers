import joblib
import pandas as pd


# ================================================
# LOAD MODELS
# ================================================

# Encoder
encoder_TransactionType = joblib.load("models/encoder_TransactionType.joblib")
encoder_Location = joblib.load("models/encoder_Location.joblib")
encoder_Channel = joblib.load("models/encoder_Channel.joblib")
encoder_CustomerOccupation = joblib.load("models/encoder_CustomerOccupation.joblib")
encoder_AgeGroup = joblib.load("models/encoder_AgeGroup.joblib")

# Scaler
scaler_TransactionAmount = joblib.load("models/scaler_TransactionAmount.joblib")
scaler_CustomerAge = joblib.load("models/scaler_CustomerAge.joblib")
scaler_AccountBalance = joblib.load("models/scaler_AccountBalance.joblib")
scaler_TransactionDuration = joblib.load("models/scaler_TransactionDuration.joblib")
scaler_LoginAttempts = joblib.load("models/scaler_LoginAttempts.joblib")
scaler_AccountBalance = joblib.load("models/scaler_AccountBalance.joblib")


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
    df = pd.DataFrame()
    # Encoder
    df["TransactionType"] = encoder_TransactionType.transform(
        data["TransactionType"][0]
    )
    df["Location"] = encoder_Location.transform(data["Location"][0])
    df["Channel"] = encoder_Channel.transform(data["Channel"])
    df["CustomerOccupation"] = encoder_CustomerOccupation(
        data["encoder_CustomerOccupation"][0]
    )
    df["AgeGroup"] = encoder_AgeGroup.transform(data["AgeGroup"])

    # Scaling
    df["TransactionAmount"] = scaler_TransactionAmount.transform(
        data["scaler_TransactionAmount"]
    )
    df["CustomerAge"] = scaler_CustomerAge.transform(data["scaler_CustomerAge"])
    df["AccountBalance"] = scaler_AccountBalance.transform(
        data["scaler_AccountBalance"]
    )
    df["TransactionDuration"] = scaler_TransactionDuration.transform(
        data["scaler_TransactionDuration"]
    )
    df["LoginAttempts"] = scaler_LoginAttempts.transform(data["scaler_LoginAttempts"])
    df["AccountBalance"] = scaler_AccountBalance.transform(
        data["scaler_AccountBalance"]
    )
