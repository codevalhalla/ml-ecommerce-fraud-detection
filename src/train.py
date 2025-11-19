# train.py

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

# =====================================================
# Custom Feature Engineering Transformer
# =====================================================
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """Calculate frequency only on training data."""
        self.country_freq = X['country'].value_counts(normalize=True)
        self.bin_country_freq = X['bin_country'].value_counts(normalize=True)
        self.default_freq = self.country_freq.mean()  # For unseen values
        return self

    def transform(self, X):
        X = X.copy()

        # Drop identifier columns
        X = X.drop(['transaction_id', 'user_id'], axis=1)

        # Feature: ratio of transaction amount to customer avg amount
        X['amount_per_avg_ratio'] = X['amount'] / X['avg_amount_user']

        # Feature: cross-country prediction
        X['cross_country_flag'] = (X['bin_country'] != X['country']).astype(int)

        # Frequency encoding
        X['country_freq'] = X['country'].map(self.country_freq).fillna(self.default_freq)
        X['bin_country_freq'] = X['bin_country'].map(self.bin_country_freq).fillna(self.default_freq)

        # Drop original country fields
        X = X.drop(['country', 'bin_country'], axis=1)

        # Time-based features
        X['transaction_time'] = pd.to_datetime(X['transaction_time'], utc=True)
        X['hour'] = X['transaction_time'].dt.hour
        X['day_of_week'] = X['transaction_time'].dt.dayofweek
        X['is_night'] = ((X['hour'] >= 0) & (X['hour'] <= 6)).astype(int)

        # Drop timestamp
        X = X.drop(['transaction_time'], axis=1)

        return X


# =====================================================
# Load dataset
# =====================================================
df = pd.read_csv("../data/transactions.csv")
y = df['is_fraud']
X = df.drop(columns=['is_fraud'])

# =====================================================
# Final Model Configuration (Tuned Parameters)
# =====================================================
final_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    subsample=1.0,
    scale_pos_weight=25,
    n_estimators=500,
    min_child_weight=40,
    max_depth=5,
    learning_rate=0.01,
    colsample_bytree=0.7,
    tree_method='hist',
    n_jobs=-1,
    random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ('featureengineering', FeatureEngineering()),
    ('model', final_model)
])

# =====================================================
# Train full model (no data split)
# =====================================================
print("\nTraining model on full dataset...")
pipeline.fit(X, y)
print("Model successfully trained!")

# Decision threshold (from tuning)
best_threshold = 0.80
print(f"Using best decision threshold: {best_threshold}")

# =====================================================
# Save trained pipeline & threshold
# =====================================================
model_path = "../models/fraud_detection_xgb_pipeline.bin"

with open(model_path, "wb") as f_out:
    pickle.dump({"pipeline": pipeline, "threshold": best_threshold}, f_out)

print(f"\nModel saved ➝ {model_path}")
print("Contains full feature engineering and trained model.")
print("Use predict.py to run inference.")

