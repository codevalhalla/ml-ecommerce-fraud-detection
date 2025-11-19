# predict.py

import pickle
import pandas as pd
import numpy as np

# Load trained pipeline and threshold
model_path = "../models/fraud_detection_xgb_pipeline.bin"

with open(model_path, "rb") as f_in:
    model_data = pickle.load(f_in)

pipeline = model_data["pipeline"]
threshold = model_data["threshold"]

print(f"Loaded model pipeline with threshold = {threshold}")


def preprocess_input(data):
    """
    Ensures input is returned as a DataFrame.
    Supports both single dictionary and list of dictionaries.
    """

    if isinstance(data, dict):  # Single transaction
        return pd.DataFrame([data])
    elif isinstance(data, list):  # List of transactions
        return pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a dict or list of dicts.")


def predict_transaction(transaction):
    """
    Returns fraud probability and prediction (Fraud / Legitimate).
    """

    # Convert to DataFrame
    df = preprocess_input(transaction)

    # Predict probabilities
    y_proba = pipeline.predict_proba(df)[:, 1]

    # Apply decision threshold
    y_pred = (y_proba >= threshold).astype(int)

    response = []
    for prob, pred in zip(y_proba, y_pred):
        response.append({
            "fraud_probability": round(prob, 4),
            "prediction": "Fraud" if pred == 1 else "Legitimate"
        })
    return response


# ===========================
# Example Test (Optional)
# ===========================
if __name__ == "__main__":
    sample_transaction = {
        "transaction_id": 999999,
        "user_id": 1111,
        "account_age_days": 120,
        "total_transactions_user": 55,
        "avg_amount_user": 140.75,
        "amount": 950.25,
        "country": "US",
        "bin_country": "IN",
        "channel": "web",
        "merchant_category": "electronics",
        "promo_used": 0,
        "avs_match": 1,
        "cvv_result": 1,
        "three_ds_flag": 1,
        "transaction_time": "2024-01-18T22:15:00Z",
        "shipping_distance_km": 800.0
    }

    print("\nRunning prediction on sample transaction...\n")
    result = predict_transaction(sample_transaction)
    print(result)
    
    transactions = [
    {
        "transaction_id": 101,
        "user_id": 5,
        "account_age_days": 150,
        "total_transactions_user": 22,
        "avg_amount_user": 110.0,
        "amount": 120.0,
        "country": "IN",
        "bin_country": "IN",
        "channel": "app",
        "merchant_category": "fashion",
        "promo_used": 0,
        "avs_match": 1,
        "cvv_result": 1,
        "three_ds_flag": 1,
        "transaction_time": "2024-01-08T12:15:00Z",
        "shipping_distance_km": 12.0
    },
    {
        "transaction_id": 102,
        "user_id": 6,
        "account_age_days": 30,
        "total_transactions_user": 10,
        "avg_amount_user": 120.0,
        "amount": 950.0,
        "country": "US",
        "bin_country": "IN",
        "channel": "web",
        "merchant_category": "electronics",
        "promo_used": 1,
        "avs_match": 0,
        "cvv_result": 0,
        "three_ds_flag": 0,
        "transaction_time": "2024-01-19T02:20:00Z",
        "shipping_distance_km": 1500.0
    }
]

result = predict_transaction(transactions)
print(result)

