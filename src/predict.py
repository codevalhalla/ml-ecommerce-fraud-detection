# predict.py

from typing import Dict, Any
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

# =========================
# Load saved model pipeline
# =========================
MODEL_PATH = "../models/fraud_detection_xgb_pipeline.bin"
with open(MODEL_PATH, "rb") as f_in:
    model_data = pickle.load(f_in)

pipeline = model_data["pipeline"]
threshold = model_data["threshold"]

print(f"Model loaded successfully with threshold = {threshold}")

# =========================
# FastAPI App Initialization
# =========================
app = FastAPI(title="Fraud Detection API", version="1.0")

# =========================
# Pydantic Input Schema
# =========================
class TransactionInput(BaseModel):
    transaction_id: int
    user_id: int
    account_age_days: float
    total_transactions_user: int
    avg_amount_user: float
    amount: float
    country: str
    bin_country: str
    channel: str
    merchant_category: str
    promo_used: int
    avs_match: int
    cvv_result: int
    three_ds_flag: int
    transaction_time: str
    shipping_distance_km: float

# =========================
# Prediction Function
# =========================
def predict_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([transaction])
    prob = pipeline.predict_proba(df)[0][1]
    pred = "Fraud" if prob >= threshold else "Legitimate"
    return {
        "fraud_probability": round(float(prob), 4),
        "prediction": pred
    }

# =========================
# API Endpoints
# =========================
@app.get("/")
def home():
    return {"message": "Welcome to Fraud Detection API"}

@app.post("/predict")
def predict_api(transaction: TransactionInput):
    result = predict_transaction(transaction.dict())
    return result

# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
