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


# ==========================================
# LOAD ENCODER
# ==========================================


encoder_TransactionType = joblib.load("/models/encoder_TransactionType.joblib")
encoder_TransactionType = joblib.load(MODEL_DIR / "encoder_TransactionType.joblib")
encoder_Location = joblib.load(MODEL_DIR / "encoder_Location.joblib")
encoder_Channel = joblib.load(MODEL_DIR / "encoder_Channel.joblib")
encoder_CustomerOccupation = joblib.load(
    MODEL_DIR / "encoder_CustomerOccupation.joblib"
)
encoder_AgeGroup = joblib.load(MODEL_DIR / "encoder_AgeGroupBin.joblib")


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


# ==========================================
# LOAD SCALER
# ==========================================

def preprocessor():
    