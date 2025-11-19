# train.py

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

from features import FeatureEngineering 


#  Load dataset
df = pd.read_csv("../data/transactions.csv")

y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])


# Final tuned XGBoost model
final_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    subsample=1.0,
    scale_pos_weight=25,
    n_estimators=500,
    min_child_weight=40,
    max_depth=5,
    learning_rate=0.01,
    colsample_bytree=0.7,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


# 3Build pipeline: FeatureEngineering → DictVectorizer → XGB
pipeline = Pipeline(
    steps=[
        ("featureengineering", FeatureEngineering()),
        ("vectorizer", DictVectorizer(sparse=False)),
        ("model", final_model),
    ]
)


# Train on full dataset
print("\nTraining model on full dataset...")
pipeline.fit(X, y)
print(" Model trained successfully!")


# Store best decision threshold (from your notebook: 0.80)
best_threshold = 0.80


# Save pipeline + threshold
model_path = "../models/fraud_detection_xgb_pipeline.bin"

with open(model_path, "wb") as f_out:
    pickle.dump({"pipeline": pipeline, "threshold": best_threshold}, f_out)

print(f"\n Model saved to: {model_path}")
print(" Saved object contains full pipeline (FE + DictVectorizer + XGB) and threshold.")
