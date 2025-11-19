# train.py

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

# ======================================
# Custom Feature Engineering Transformer
# ======================================
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.country_freq = X['country'].value_counts(normalize=True)
        self.bin_country_freq = X['bin_country'].value_counts(normalize=True)
        self.default_freq = self.country_freq.mean()
        return self

    def transform(self, X):
        X = X.copy()

        X = X.drop(['transaction_id', 'user_id'], axis=1)

        X['amount_per_avg_ratio'] = X['amount'] / X['avg_amount_user']

        X['cross_country_flag'] = (X['bin_country'] != X['country']).astype(int)

        X['country_freq'] = X['country'].map(self.country_freq).fillna(self.default_freq)
        X['bin_country_freq'] = X['bin_country'].map(self.bin_country_freq).fillna(self.default_freq)

        X = X.drop(['country', 'bin_country'], axis=1)

        X['transaction_time'] = pd.to_datetime(X['transaction_time'], utc=True)
        X['hour'] = X['transaction_time'].dt.hour
        X['day_of_week'] = X['transaction_time'].dt.dayofweek
        X['is_night'] = ((X['hour'] >= 0) & (X['hour'] <= 6)).astype(int)

        X = X.drop(['transaction_time'], axis=1)

        return X.to_dict(orient='records')  # IMPORTANT


# =====================
# Load dataset
# =====================
df = pd.read_csv("data/transactions.csv")
y = df['is_fraud']
X = df.drop(columns=['is_fraud'])

# =====================
# Final Optimized Model
# =====================
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

# =====================
# Build pipeline
# =====================
pipeline = Pipeline([
    ('featureengineering', FeatureEngineering()),
    ('vectorizer', DictVectorizer(sparse=False)),  # Converts dicts → numeric
    ('model', final_model)
])

# =====================
# Train on full data
# =====================
print("\nTraining model on full dataset...")
pipeline.fit(X, y)
print("Model trained successfully!")

best_threshold = 0.80

# =====================
# Save model
# =====================
model_path = "../models/fraud_detection_xgb_pipeline.bin"
with open(model_path, "wb") as f_out:
    pickle.dump({"pipeline": pipeline, "threshold": best_threshold}, f_out)

print(f"\nModel saved to: {model_path}")
print("Use predict.py to run inference.")
