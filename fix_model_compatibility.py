"""
Fix model compatibility by reloading and resaving with current scikit-learn version.
This resolves the '_fill_dtype' attribute error in SimpleImputer.
"""

import joblib
import pickle
from pathlib import Path
from datetime import datetime
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Custom Functions and Classes (needed for unpickling the model)
# ============================================================================

def make_text_array(values):
    """Convert input to text array for TfidfVectorizer."""
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    else:
        series = pd.Series(values)
    return series.fillna('').astype(str).values


def comma_tokenizer(text: str) -> List[str]:
    """Tokenize comma-separated text."""
    return [token.strip().lower() for token in text.split(',') if token and token.strip()]


class DenseTruncatedSVD(BaseEstimator, TransformerMixin):
    """Wrapper for TruncatedSVD that handles edge cases and returns dense arrays."""
    
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
    
    # Find the most recent model and info files
    model_files = sorted(artifact_dir.glob("best_anime_rating_model_*.joblib"))
    info_files = sorted(artifact_dir.glob("model_feature_info_*.pkl"))
    
    if not model_files or not info_files:
        print("Error: No model artifacts found!")
        return 1
    
    latest_model = model_files[-1]
    latest_info = info_files[-1]
    
    print(f"Loading model from: {latest_model.name}")
    print(f"Loading info from: {latest_info.name}")
    
    try:
        # Load the model
        model = joblib.load(latest_model)
        print("✓ Model loaded successfully")
        
        # Load the feature info
        with open(latest_info, 'rb') as f:
            feature_info = pickle.load(f)
        print("✓ Feature info loaded successfully")
        
        # Create new timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save with new timestamp
        new_model_file = artifact_dir / f"best_anime_rating_model_{timestamp}.joblib"
        new_info_file = artifact_dir / f"model_feature_info_{timestamp}.pkl"
        
        # Save model
        joblib.dump(model, new_model_file)
        print(f"✓ Model saved to: {new_model_file.name}")
        
        # Update timestamp in feature_info
        feature_info['timestamp'] = timestamp
        
        # Save feature info
        with open(new_info_file, 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"✓ Feature info saved to: {new_info_file.name}")
        
        print("\n✅ Model compatibility fixed successfully!")
        print(f"\nNew artifacts:")
        print(f"  - {new_model_file.name}")
        print(f"  - {new_info_file.name}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
