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

# Exploratory Data Analysis (EDA)

## 1. Dataset Overview

- **Total transactions:** `299,695`
- **Fraudulent transactions:** `6,612` (~2.2%)
- **Legitimate transactions:** `293,083` (~97.8%)
- **Unique users:** `6,000`
- **Transactions per user:** ~40–60
- **Missing values:** `None`

**Data source:** Kaggle — E-Commerce Fraud Detection Dataset  
https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset

This is a synthetic but realistic dataset simulating real-world transaction behavior, including:
- Multiple transactions per user
- Cross-country dynamics (`country` vs `bin_country`)
- Time-based fraud patterns
- Natural class imbalance (~2% fraud)

---

## 2. Feature Overview

| Feature Type | Features |
|--------------|----------|
| **Identifier** | `transaction_id`, `user_id` |
| **Numerical** | `account_age_days`, `total_transactions_user`, `avg_amount_user`, `amount`, `shipping_distance_km` |
| **Categorical** | `country`, `bin_country`, `channel`, `merchant_category` |
| **Binary** | `promo_used`, `avs_match`, `cvv_result`, `three_ds_flag` |
| **Time** | `transaction_time` (later transformed) |
| **Target** | `is_fraud` (0 = legitimate, 1 = fraud) |

All 17 columns contain complete values — `0` missing entries.

---

## 3. Fraud Distribution (Class Imbalance)

The target variable is heavily imbalanced:
- **Legitimate:** `293,083`
- **Fraud:** `6,612`

**Image in repository at:** `images/fraud_distribution.png`

**Display in Markdown:**
```markdown
![Fraud vs Non-Fraud Transaction Counts](images/fraud_distribution.png)
```

**Key insight:**
- Fraudulent transactions represent only ~2.2% of the dataset
- Accuracy alone is misleading → we focused on **ROC-AUC, Precision, Recall, and F1-score**

---

## 4. Correlation of Numeric Features

**Image in repository:** `images/correlation_matrix.png`

**Display in Markdown:**
```markdown
![Correlation Matrix of Numeric Features](images/correlation_matrix.png)
```

**Important correlations with `is_fraud`:**
- `shipping_distance_km` → 0.27
- `amount` → 0.20
- `account_age_days` → -0.12

**Insight:** Fraudulent transactions tend to have higher shipping distances and amounts, and often come from newer accounts.

---

## 5. Mutual Information (Categorical/Binary Features)

| Feature | MI Score |
|---------|----------|
| `avs_match` | 0.0169 |
| `cvv_result` | 0.0149 |
| `three_ds_flag` | 0.0103 |
| `channel` | 0.0048 |
| `promo_used` | 0.0019 |
| `country` | 0.0002 |
| `bin_country` | 0.0001 |
| `merchant_category` | 0.0000 |

**Insight:**
- Security-related checks (`avs_match`, `cvv_result`, `three_ds_flag`) are strong indicators of fraud
- Country-based features have low standalone influence but improve performance when encoded effectively

---

## 6. Key Takeaways from EDA

1. **Severe class imbalance (~2.2% fraud)** → Metrics like accuracy are incorrect for model evaluation, use ROC-AUC, Precision, Recall, and F1-score.
2. **Transaction amount and shipping distance** are important fraud indicators.
3. **Newer accounts** (`low account_age_days`) are more likely to commit fraud.
4. **Security flag features** (`avs_match`, `cvv_result`) highly correlate with fraud.
5. **No missing data** → focus was on feature engineering rather than cleaning.

---

## How EDA Influenced Modeling

- Used **tree-based models** (Random Forest, XGBoost) that handle non-linear relationships well.
- Performed **custom feature engineering:**
  - `amount_per_avg_ratio`
  - `cross_country_flag`
  - `country_freq` / `bin_country_freq`
  - `hour`, `day_of_week`, `is_night`
- **Final model selected:** XGBoost
- Applied **threshold tuning at 0.80** to optimize fraud detection while minimizing false positives.
