"""
Retrain the best model with scikit-learn 1.8.0 to fix compatibility issues.
This recreates the model pipeline from scratch using the current sklearn version.
"""

import warnings
from pathlib import Path
from typing import List
from datetime import datetime

import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ============================================================================
# HELPER FUNCTIONS & CUSTOM TRANSFORMERS
# ============================================================================

def make_text_array(values):
    """Convert series/dataframe to text array for TF-IDF vectorizer."""
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    else:
        series = pd.Series(values)
    return series.fillna('').astype(str).values


def comma_tokenizer(text: str) -> List[str]:
    """Custom tokenizer for comma-separated fields."""
    return [token.strip().lower() for token in text.split(',') if token and token.strip()]


class DenseTruncatedSVD(BaseEstimator, TransformerMixin):
    """Custom SVD transformer that handles edge cases."""
    
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


def count_from_comma_separated(series: pd.Series) -> pd.Series:
    """Count items in comma-separated string."""
    return series.fillna('').astype(str).apply(
        lambda text: sum(1 for token in text.split(',') if token.strip())
    )


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    renamed = df.copy()
    renamed.columns = (
        renamed.columns
        .str.strip()
        .str.lower()
        .str.replace('[^0-9a-zA-Z]+', '_', regex=True)
        .str.strip('_')
    )
    return renamed


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert columns to numeric."""
    updated = df.copy()
    for col in columns:
        if col in updated.columns:
            updated[col] = (
                updated[col]
                .replace({'Unknown': np.nan, 'unknown': np.nan, 'N/A': np.nan, 'UNKNOWN': np.nan})
                .astype(str)
            )
            updated[col] = pd.to_numeric(updated[col], errors='coerce')
    return updated


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print("=" * 70)
    print("RETRAINING BEST MODEL WITH SCIKIT-LEARN 1.8.0")
    print("=" * 70)
    
    # Load data
    data_path = Path(__file__).parent / "Animes.csv"
    if not data_path.exists():
        print(f"❌ Error: Dataset not found at {data_path}")
        return 1
    
    print(f"\n📂 Loading data from {data_path.name}...")
    anime_df = pd.read_csv(data_path)
    print(f"✓ Loaded {anime_df.shape[0]} rows and {anime_df.shape[1]} columns")
    
    # Feature engineering
    print("\n🔧 Performing feature engineering...")
    clean_df = standardize_column_names(anime_df).drop_duplicates(subset=['anime_id'])
    numeric_candidates = ['anime_id', 'average_rating', 'episodes', 'rank', 'popularity', 'favorites', 'scored_by', 'members']
    clean_df = coerce_numeric(clean_df, numeric_candidates)
    
    feature_df = clean_df.copy()
    feature_df['genre_count'] = count_from_comma_separated(feature_df.get('genres', pd.Series(dtype=str)))
    feature_df['producer_count'] = count_from_comma_separated(feature_df.get('producers', pd.Series(dtype=str)))
    feature_df['studio_count'] = count_from_comma_separated(feature_df.get('studios', pd.Series(dtype=str)))
    feature_df['licensor_count'] = count_from_comma_separated(feature_df.get('licensors', pd.Series(dtype=str)))
    feature_df['overview_char_len'] = feature_df.get('overview', pd.Series(dtype=str)).fillna('').astype(str).str.len()
    feature_df['overview_word_len'] = feature_df.get('overview', pd.Series(dtype=str)).fillna('').astype(str).str.split().str.len()
    feature_df['has_licensor'] = (feature_df.get('licensors', pd.Series(dtype=str)).fillna('').str.strip() != '').astype(int)
    feature_df['episodes_missing'] = feature_df['episodes'].isna().astype(int)
    
    target_col = 'average_rating'
    leakage_columns = ['image_url', 'rank', 'name']
    feature_df = feature_df.drop(columns=[col for col in leakage_columns if col in feature_df.columns], errors='ignore')
    
    for col in ['popularity', 'favorites', 'scored_by', 'members']:
        if col in feature_df.columns:
            feature_df[f'log_{col}'] = np.log1p(feature_df[col])
    
    feature_df['log_episodes'] = np.log1p(feature_df['episodes'].fillna(0))
    episode_safe = feature_df['episodes'].replace({0: np.nan})
    member_safe = feature_df['members'].replace({0: np.nan}) if 'members' in feature_df else pd.Series(np.nan, index=feature_df.index)
    favorite_safe = feature_df['favorites'].replace({0: np.nan}) if 'favorites' in feature_df else pd.Series(np.nan, index=feature_df.index)
    
    feature_df['members_per_episode'] = (feature_df['members'] / episode_safe).replace([np.inf, -np.inf], np.nan).fillna(0) if 'members' in feature_df else 0
    feature_df['favorites_per_episode'] = (feature_df['favorites'] / episode_safe).replace([np.inf, -np.inf], np.nan).fillna(0) if 'favorites' in feature_df else 0
    feature_df['favorites_per_member'] = (feature_df['favorites'] / member_safe).replace([np.inf, -np.inf], np.nan).fillna(0) if 'favorites' in feature_df else 0
    feature_df['scored_by_per_member'] = (feature_df['scored_by'] / member_safe).replace([np.inf, -np.inf], np.nan).fillna(0) if 'scored_by' in feature_df else 0
    feature_df['is_long_series'] = (feature_df['episodes'].fillna(0) >= 50).astype(int)
    feature_df = feature_df.drop(columns=['anime_id'], errors='ignore')
    
    print(f"✓ Feature engineering complete: {feature_df.shape}")
    
    # Prepare train/test split
    print("\n📊 Preparing train/test split...")
    model_df = feature_df.dropna(subset=[target_col]).reset_index(drop=True)
    y = model_df[target_col]
    X = model_df.drop(columns=[target_col])
    
    categorical_features = [col for col in ['type', 'source', 'anime_rating'] if col in X.columns]
    text_features = [col for col in ['overview', 'genres', 'producers', 'licensors', 'studios'] if col in X.columns]
    numeric_features = [col for col in X.select_dtypes(include=[np.number]).columns if col not in categorical_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"✓ Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical, {len(text_features)} text")
    
    # Build preprocessing pipeline
    print("\n🔨 Building preprocessing pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    text_transformers = []
    if 'overview' in text_features:
        text_transformers.append((
            'overview',
            Pipeline([
                ('to_str', FunctionTransformer(make_text_array, validate=False)),
                ('tfidf', TfidfVectorizer(max_features=800, ngram_range=(1, 2), min_df=5, stop_words='english')),
                ('svd', DenseTruncatedSVD(n_components=40, random_state=42))
            ]),
            'overview'
        ))
    if 'genres' in text_features:
        text_transformers.append((
            'genres',
            Pipeline([
                ('to_str', FunctionTransformer(make_text_array, validate=False)),
                ('tfidf', TfidfVectorizer(max_features=80, tokenizer=comma_tokenizer, lowercase=True)),
                ('svd', DenseTruncatedSVD(n_components=20, random_state=42))
            ]),
            'genres'
        ))
    for col in ['producers', 'licensors', 'studios']:
        if col in text_features:
            text_transformers.append((
                col,
                Pipeline([
                    ('to_str', FunctionTransformer(make_text_array, validate=False)),
                    ('tfidf', TfidfVectorizer(max_features=60, tokenizer=comma_tokenizer, lowercase=True)),
                    ('svd', DenseTruncatedSVD(n_components=20, random_state=42))
                ]),
                col
            ))
    
    full_transformers = []
    if numeric_features:
        full_transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        full_transformers.append(('cat', categorical_transformer, categorical_features))
    full_transformers.extend(text_transformers)
    
    preprocessor = ColumnTransformer(transformers=full_transformers, remainder='drop', sparse_threshold=0.0)
    
    # Build and train the best model (HistGradientBoostingRegressor based on notebook results)
    print("\n🤖 Training HistGradient Boosting Regressor...")
    best_model = Pipeline([
        ('preprocess', preprocessor),
        ('model', HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=None,
            l2_regularization=0.0,
            random_state=42
        ))
    ])
    
    best_model.fit(X_train, y_train)
    print("✓ Model training complete")
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred_test = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    residuals = y_test - y_pred_test
    
    print(f"✓ Test R²: {test_r2:.4f}")
    print(f"✓ Test RMSE: {test_rmse:.4f}")
    print(f"✓ Test MAE: {test_mae:.4f}")
    
    # Permutation importance
    print("\n🔍 Calculating feature importance...")
    perm_result = permutation_importance(best_model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_result.importances_mean,
        'importance_std': perm_result.importances_std
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    print(f"✓ Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance_mean']:.4f})")
    
    # Prepare artifacts
    print("\n💾 Saving model artifacts...")
    artifact_dir = Path(__file__).parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = artifact_dir / f"best_anime_rating_model_{timestamp}.joblib"
    info_filename = artifact_dir / f"model_feature_info_{timestamp}.pkl"
    metrics_filename = artifact_dir / f"model_metrics_{timestamp}.csv"
    
    # Save model
    joblib.dump(best_model, model_filename)
    print(f"✓ Model saved: {model_filename.name}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'metric': ['test_r2', 'rmse', 'mae'],
        'value': [test_r2, test_rmse, test_mae]
    })
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"✓ Metrics saved: {metrics_filename.name}")
    
    # Prepare feature info
    minimal_columns = [
        col for col in ['members', 'favorites', 'episodes', 'genre_count', 'producer_count',
                       'studio_count', 'licensor_count', 'members_per_episode',
                       'favorites_per_episode', 'favorites_per_member', 'scored_by_per_member']
        if col in X_test.columns
    ]
    
    eval_frame = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test,
        'residual': residuals
    })
    if minimal_columns:
        eval_frame[minimal_columns] = X_test[minimal_columns].reset_index(drop=True)
    
    category_options = {
        'type': sorted(X['type'].dropna().unique().tolist()) if 'type' in X.columns else [],
        'source': sorted(X['source'].dropna().unique().tolist()) if 'source' in X.columns else [],
        'anime_rating': sorted(X['anime_rating'].dropna().unique().tolist()) if 'anime_rating' in X.columns else []
    }
    
    metrics_summary = pd.DataFrame({
        'metric': ['Model', 'Test R²', 'RMSE', 'MAE'],
        'value': ['HistGradient Boosting Regressor', round(test_r2, 4), round(test_rmse, 4), round(test_mae, 4)]
    })
    
    feature_info = {
        'model_name': 'HistGradient Boosting Regressor',
        'timestamp': timestamp,
        'raw_feature_names': X.columns.tolist(),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'text_features': text_features,
        'category_options': category_options,
        'data_info': {
            'total_samples': int(len(model_df)),
            'n_features': int(X.shape[1]),
            'target_mean': float(y.mean()),
            'target_std': float(y.std()),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test))
        },
        'model_performance': {
            'test_r2': float(test_r2),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'training_date': datetime.now().strftime("%Y-%m-%d")
        },
        'feature_importance': importance_df.to_dict(orient='records'),
        'top_features': importance_df.head(10).to_dict(orient='records'),
        'evaluation_frame': eval_frame,
        'metrics_summary': metrics_summary.to_dict(orient='records')
    }
    
    with open(info_filename, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"✓ Feature info saved: {info_filename.name}")
    
    print("\n" + "=" * 70)
    print("✅ MODEL RETRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📦 New artifacts:")
    print(f"   • {model_filename.name}")
    print(f"   • {info_filename.name}")
    print(f"   • {metrics_filename.name}")
    print(f"\n🚀 You can now run the Streamlit app with the updated model.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
