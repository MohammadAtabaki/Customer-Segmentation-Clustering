import joblib
import pandas as pd
import os

MODELS_DIR = "app/utils/models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
DATA_PATH = os.path.join(MODELS_DIR, "rfm_clv.csv")

def save_model(model, scaler, rfm_df):
    """Save trained model, scaler, and processed dataset."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    rfm_df.to_csv(DATA_PATH, index=False)

def load_model():
    """Load pre-trained model, scaler, and RFM dataset if they exist."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(DATA_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        rfm_df = pd.read_csv(DATA_PATH)
        return model, scaler, rfm_df
    return None, None, None
