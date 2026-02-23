"""
Quick test to verify model prediction works without errors.
"""

import joblib
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

# Define required custom components
def make_text_array(values):
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    else:
        series = pd.Series(values)
    return series.fillna('').astype(str).values

def comma_tokenizer(text: str) -> List[str]:
    return [token.strip().lower() for token in text.split(',') if token and token.strip()]

class DenseTruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._svd = None
        self.actual_components_ = 0
        self._use_identity = False

    def fit(self, X, y=None):
        n_features = X.shape[1]
        if n_features <= 1:
            self.actual_components_ = n_features
            self._svd = None
            self._use_identity = True
            return self
        self._use_identity = False
        self.actual_components_ = min(self.n_components, n_features - 1)
        self._svd = TruncatedSVD(n_components=self.actual_components_, random_state=self.random_state)
        self._svd.fit(X)
        return self

    def transform(self, X):
        if self._use_identity:
            return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        if self._svd is None or self.actual_components_ == 0:
            return np.zeros((X.shape[0], 0))
        return self._svd.transform(X)

def main():
    artifact_dir = Path(__file__).parent / "artifacts"
    
    # Find latest model and info
    model_files = sorted(artifact_dir.glob("best_anime_rating_model_*.joblib"))
    info_files = sorted(artifact_dir.glob("model_feature_info_*.pkl"))
    
    if not model_files or not info_files:
        print("❌ No model artifacts found!")
        return 1
    
    latest_model = model_files[-1]
    latest_info = info_files[-1]
    
    print(f"Loading model: {latest_model.name}")
    print(f"Loading info: {latest_info.name}")
    
    try:
        model = joblib.load(latest_model)
        with open(latest_info, 'rb') as f:
            feature_info = pickle.load(f)
        
        print("✓ Model and info loaded successfully\n")
        
        # Create test input
        raw_columns = feature_info['raw_feature_names']
        numeric_cols = set(feature_info.get('numeric_features', []))
        categorical_cols = set(feature_info.get('categorical_features', []))
        text_cols = set(feature_info.get('text_features', []))
        
        # Test data
        test_features = {
            'members': 125000,
            'favorites': 5000,
            'scored_by': 110000,
            'popularity': 450,
            'episodes': 24,
            'type': 'TV',
            'source': 'Manga',
            'anime_rating': 'PG-13 - Teens 13 or older',
            'overview': 'A great action anime with amazing characters',
            'genres': 'Action, Adventure, Fantasy',
            'producers': 'Studio A, Studio B',
            'licensors': 'Funimation',
            'studios': 'Bones',
            'genre_count': 3,
            'producer_count': 2,
            'studio_count': 1,
            'licensor_count': 1,
            'overview_char_len': 45,
            'overview_word_len': 7,
            'has_licensor': 1,
            'episodes_missing': 0,
            'log_popularity': np.log1p(450),
            'log_favorites': np.log1p(5000),
            'log_scored_by': np.log1p(110000),
            'log_members': np.log1p(125000),
            'log_episodes': np.log1p(24),
            'members_per_episode': 125000 / 24,
            'favorites_per_episode': 5000 / 24,
            'favorites_per_member': 5000 / 125000,
            'scored_by_per_member': 110000 / 125000,
            'is_long_series': 0
        }
        
        # Build input dataframe
        row = {}
        for column in raw_columns:
            value = test_features.get(column)
            if value is None:
                if column in numeric_cols:
                    value = 0.0
                elif column in categorical_cols:
                    value = "Unknown"
                elif column in text_cols:
                    value = ""
                else:
                    value = 0.0
            row[column] = value
        
        X_test = pd.DataFrame([row])
        
        print("Testing prediction...")
        prediction = model.predict(X_test)[0]
        prediction = max(1.0, min(10.0, float(prediction)))
        
        print(f"✅ Prediction successful: {prediction:.2f}/10.0")
        print("\n🎉 Model is working correctly!")
        return 0
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
