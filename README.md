# 🚨 E-Commerce Fraud Detection using Machine Learning  
### 🔍 Predicting fraudulent transactions with XGBoost & FastAPI | Dockerized & Deployed on Render

## Problem Description

E-commerce platforms process millions of transactions every day. Even a small number of fraudulent transactions can lead to major financial losses, chargebacks, operational overhead, and customer dissatisfaction.

Fraud detection is a challenging problem due to:

- Highly imbalanced datasets (fraudulent cases are extremely rare)  
- Constantly evolving fraud patterns  
- The requirement for near real-time decision-making  
- The need to avoid false positives, which can negatively impact genuine users  

The objective of this project is to develop a machine learning–based fraud detection system that predicts whether a transaction is fraudulent using historical data and engineered features.

The project includes:

- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training and comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost)  
- Hyperparameter optimization and threshold tuning  

The final solution is built using the XGBoost model, optimized with a custom decision threshold of `0.80`, which balances fraud detection coverage and false positives.

The model is deployed as a REST API using FastAPI, containerized using Docker, and hosted on Render for real-time inference.

---

### How the Solution Is Used

When the e-commerce platform processes a transaction, it sends the transaction details to the API. The model returns a fraud probability score and a final prediction.

Example output:

```json
{
  "fraud_probability": 0.9875,
  "prediction": "Fraud"
}
```

Based on the model response:

| Prediction | Recommended Action                            |
|------------|-----------------------------------------------|
| Fraud      | Block transaction or send for manual review   |
| Legitimate | Approve transaction                           |

---

## Summary

This project delivers an end-to-end production-ready fraud detection system, including:

- Data loading, preparation, and exploratory analysis  
- Feature engineering and selection  
- Model training, evaluation, and tuning  
- Model packaging using FastAPI  
- Docker-based containerization  
- Deployment on Render with live API access  

The system is suitable for real-world integration into transaction processing pipelines to minimize fraud risk and enhance payment security.




