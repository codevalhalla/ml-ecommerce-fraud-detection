# features.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Custom transformer that:
    - Drops ID columns
    - Creates amount_per_avg_ratio
    - Creates cross_country_flag
    - Frequency-encodes country and bin_country
    - Extracts hour, day_of_week, is_night from transaction_time
    - Returns list[dict] suitable for DictVectorizer
    """

    def fit(self, X, y=None):
        # Compute frequency encodings on training data
        self.country_freq = X["country"].value_counts(normalize=True)
        self.bin_country_freq = X["bin_country"].value_counts(normalize=True)

        # Default frequency for unseen countries
        self.default_freq = self.country_freq.mean()

        return self

    def transform(self, X):
        X = X.copy()

        # Drop identifiers
        X = X.drop(["transaction_id", "user_id"], axis=1)

        # Amount ratio
        X["amount_per_avg_ratio"] = X["amount"] / X["avg_amount_user"]

        # Cross-country flag
        X["cross_country_flag"] = (X["bin_country"] != X["country"]).astype(int)

        # Frequency encoding
        X["country_freq"] = X["country"].map(self.country_freq).fillna(self.default_freq)
        X["bin_country_freq"] = X["bin_country"].map(self.bin_country_freq).fillna(
            self.default_freq
        )

        # Drop raw country columns
        X = X.drop(["country", "bin_country"], axis=1)

        # Time-based features
        X["transaction_time"] = pd.to_datetime(X["transaction_time"], utc=True)
        X["hour"] = X["transaction_time"].dt.hour
        X["day_of_week"] = X["transaction_time"].dt.dayofweek
        X["is_night"] = ((X["hour"] >= 0) & (X["hour"] <= 6)).astype(int)

        X = X.drop(["transaction_time"], axis=1)

        # VERY IMPORTANT: DictVectorizer expects list of dicts
        return X.to_dict(orient="records")
