# E-Commerce Fraud Detection using Machine Learning  
### Predicting fraudulent transactions with XGBoost & FastAPI | Dockerized & Deployed on Render

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

## Exploratory Data Analysis (EDA)

### 1. Dataset Overview

- **Total transactions:** 299,695  
- **Fraudulent transactions:** 6,612 (~2.2%)  
- **Legitimate transactions:** 293,083 (~97.8%)  
- **Unique users:** 6,000  
- **Transactions per user:** approximately 40–60  
- **Missing values:** none (all 17 columns are complete)

**Data source (Kaggle):**  
E-Commerce Fraud Detection Dataset  
https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset  

This is a synthetic but realistic dataset designed to simulate real-world fraud patterns such as:
- Multiple transactions per user
- Cross-country activity (`country` vs `bin_country`)
- Time-based behavior (night vs day)
- Natural class imbalance (~2% fraud)

---

### 2. Feature Overview

| Type              | Features                                                                                                                                              |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Identifier        | `transaction_id`, `user_id`                                                                                                                           |
| Numerical         | `account_age_days`, `total_transactions_user`, `avg_amount_user`, `amount`, `shipping_distance_km`                                                   |
| Categorical       | `country`, `bin_country`, `channel`, `merchant_category`                                                                                              |
| Binary flags      | `promo_used`, `avs_match`, `cvv_result`, `three_ds_flag`                                                                                              |
| Time              | `transaction_time` (later transformed into `hour`, `day_of_week`, `is_night`)                                                                        |
| Target            | `is_fraud` (0 = legitimate, 1 = fraud)                                                                                                                |

From `df.info()`:

- 17 columns (3 float, 9 int, 5 object)
- No missing values in any column

From `df.describe()` (highlights):

- `amount`:  
  - min = 1.00, max = 16,994.74  
  - mean ≈ 177.17  
- `shipping_distance_km`:  
  - min = 0.00, max ≈ 3,748.56  
  - mean ≈ 357.05  
- `is_fraud`:  
  - mean ≈ 0.022 → about 2.2% of rows are fraudulent

---

### 3. Fraud Distribution (Class Imbalance)

The target variable `is_fraud` is extremely imbalanced.

- Legitimate (0): 293,083
- Fraudulent (1): 6,612

This is visualized using:

```text
images/fraud_distribution.png
```
![Fraud vs Non-Fraud Transaction Counts](images/fraud_distribution.png)


