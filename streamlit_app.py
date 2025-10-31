import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

# ============================================================================
# HELPER FUNCTIONS FOR MODEL COMPATIBILITY
# These functions are required for unpickling the trained model pipeline
# They must match the functions used during model training in the notebook
# ============================================================================

def make_text_array(values):
    """Convert series/dataframe to text array for TF-IDF vectorizer."""
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    else:
        series = pd.Series(values)
    return series.fillna('').astype(str).values


def comma_tokenizer(text: str) -> List[str]:
    """Custom tokenizer for comma-separated fields (genres, producers, etc)."""
    return [token.strip().lower() for token in text.split(',') if token and token.strip()]


class DenseTruncatedSVD(BaseEstimator, TransformerMixin):
    """Custom SVD transformer that handles edge cases and returns dense arrays."""
    
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


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="🎌 Anime Rating Predictor",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B6B;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        padding: 1rem;
        background-color: #f0fff0;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF5252;
    }
</style>
""", unsafe_allow_html=True)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATTERN = "best_anime_rating_model_*.joblib"
INFO_PATTERN = "model_feature_info_*.pkl"
ANIME_DATA_PATH = Path(__file__).resolve().parent / "Animes.csv"


@st.cache_data
def load_anime_dataset():
    """Load the anime dataset for name-based selection."""
    try:
        if not ANIME_DATA_PATH.exists():
            return None
        df = pd.read_csv(ANIME_DATA_PATH)
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace('[^0-9a-zA-Z]+', '_', regex=True).str.strip('_')
        return df
    except Exception as e:
        st.error(f"Error loading anime dataset: {e}")
        return None


def _latest_artifact(pattern: str) -> Path | None:
    if not ARTIFACT_DIR.exists():
        return None
    candidates = sorted(ARTIFACT_DIR.glob(pattern))
    return candidates[-1] if candidates else None


@st.cache_resource
def load_model_artifacts() -> Tuple[object, Dict, Path] | Tuple[None, None, None]:
    """Load the most recent serialized model pipeline and metadata."""
    try:
        model_path = _latest_artifact(MODEL_PATTERN)
        info_path = _latest_artifact(INFO_PATTERN)

        if not model_path or not info_path:
            st.error(
                "Model artifacts were not found. Generate them from the training notebook before running the app."
            )
            return None, None, None

        model = joblib.load(model_path)
        with open(info_path, "rb") as f:
            feature_info = pickle.load(f)

        return model, feature_info, model_path
    except Exception as exc:  # pragma: no cover - surface to UI
        st.error(f"Error loading model artifacts: {exc}")
        return None, None, None


def _count_tokens(value: str) -> int:
    if not value:
        return 0
    return sum(1 for token in value.split(',') if token.strip())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator and denominator != 0:
        return float(numerator) / float(denominator)
    return 0.0


def _safe_log(value: float) -> float:
    return float(np.log1p(max(value, 0)))


def _default_option(options: list[str], fallback: str) -> str:
    return options[0] if options else fallback


def create_feature_input_form(feature_info: Dict) -> Dict:
    """Collect raw inputs and engineer the feature dictionary expected by the pipeline."""

    st.markdown('<p class="sub-header">🎬 Enter Anime Details</p>', unsafe_allow_html=True)

    category_options = feature_info.get("category_options", {})
    type_options = category_options.get("type", ["TV", "Movie", "OVA", "Special", "ONA", "Music"])
    source_options = category_options.get("source", ["Original", "Manga", "Light novel", "Other"])
    rating_options = category_options.get(
        "anime_rating",
        [
            "G - All Ages",
            "PG - Children",
            "PG-13 - Teens 13 or older",
            "R - 17+ (violence & profanity)",
            "R+ - Mild Nudity",
            "Rx - Hentai",
        ],
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📊 Engagement Signals**")
        members = st.number_input(
            "Members",
            min_value=0,
            max_value=3_000_000,
            value=125_000,
            help="Number of users who track this anime",
        )
        favorites = st.number_input(
            "Favorites",
            min_value=0,
            max_value=400_000,
            value=5_000,
            help="Users who marked this anime as a favourite",
        )
        scored_by = st.number_input(
            "Score Count",
            min_value=0,
            max_value=3_000_000,
            value=110_000,
            help="Number of users who scored this anime",
        )
        popularity = st.number_input(
            "Popularity Rank",
            min_value=1,
            max_value=10_000,
            value=450,
            help="Overall popularity rank (lower is more popular)",
        )

    with col2:
        st.markdown("**📺 Content Profile**")
        episodes = st.number_input(
            "Episodes",
            min_value=1,
            max_value=2_000,
            value=24,
            help="Total number of released episodes",
        )
        anime_type = st.selectbox(
            "Anime Type",
            options=type_options,
            index=type_options.index("TV") if "TV" in type_options else 0,
        )
        source = st.selectbox(
            "Source Material",
            options=source_options,
            index=source_options.index("Manga") if "Manga" in source_options else 0,
        )
        anime_rating = st.selectbox(
            "Content Rating",
            options=rating_options,
            index=rating_options.index("PG-13 - Teens 13 or older") if "PG-13 - Teens 13 or older" in rating_options else 0,
        )

    with col3:
        st.markdown("**🧾 Textual Context**")
        genres = st.text_input(
            "Genres (comma separated)",
            value="Action, Adventure, Fantasy",
            help="Example: Action, Adventure, Fantasy",
        )
        producers = st.text_input(
            "Producers", value="Aniplex, Bandai Visual", help="Comma separated list"
        )
        studios = st.text_input(
            "Studios", value="Bones", help="Comma separated list"
        )
        licensors = st.text_input(
            "Licensors", value="Funimation", help="Comma separated list (optional)"
        )

    overview = st.text_area(
        "Overview / Synopsis",
        value=(
            "In a world where heroes and villains clash, a young protagonist discovers hidden powers "
            "that could change the fate of humanity."
        ),
        help="Add a concise synopsis. This fuels the TF-IDF features.",
    )

    # Engineered signals
    favorites_per_member = _safe_ratio(favorites, members)
    episodes_missing = int(episodes == 0)
    licensor_count = _count_tokens(licensors)

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Log Members", f"{_safe_log(members):.2f}")
        st.metric("Log Favorites", f"{_safe_log(favorites):.2f}")
    with info_col2:
        st.metric("Log Score Count", f"{_safe_log(scored_by):.2f}")
        st.metric("Favorites / Member", f"{favorites_per_member:.3f}")
    with info_col3:
        st.metric("Log Episodes", f"{_safe_log(episodes):.2f}")
        st.metric("Long Series", "Yes" if episodes >= 50 else "No")

    features: Dict[str, object] = {
        "genres": genres.strip(),
        "overview": overview.strip(),
        "type": anime_type,
        "episodes": float(episodes),
        "producers": producers.strip(),
        "licensors": licensors.strip(),
        "studios": studios.strip(),
        "source": source,
        "anime_rating": anime_rating,
        "popularity": float(popularity),
        "favorites": float(favorites),
        "scored_by": float(scored_by),
        "members": float(members),
        "genre_count": float(_count_tokens(genres)),
        "producer_count": float(_count_tokens(producers)),
        "studio_count": float(_count_tokens(studios)),
        "licensor_count": float(licensor_count),
        "overview_char_len": float(len(overview)),
        "overview_word_len": float(len(overview.split())),
        "has_licensor": float(1 if licensor_count > 0 else 0),
        "episodes_missing": float(episodes_missing),
        "log_popularity": _safe_log(popularity),
        "log_favorites": _safe_log(favorites),
        "log_scored_by": _safe_log(scored_by),
        "log_members": _safe_log(members),
        "log_episodes": _safe_log(episodes),
        "members_per_episode": _safe_ratio(members, episodes),
        "favorites_per_episode": _safe_ratio(favorites, episodes),
        "favorites_per_member": favorites_per_member,
        "scored_by_per_member": _safe_ratio(scored_by, members),
        "is_long_series": float(1 if episodes >= 50 else 0),
    }

    return features


def predict_rating(model, feature_info: Dict, features: Dict) -> float | None:
    """Run inference using the trained pipeline."""

    try:
        raw_columns = feature_info.get("raw_feature_names", [])
        numeric_cols = set(feature_info.get("numeric_features", []))
        categorical_cols = set(feature_info.get("categorical_features", []))
        text_cols = set(feature_info.get("text_features", []))

        row = {}
        for column in raw_columns:
            value = features.get(column)
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

        X_pred = pd.DataFrame([row])
        prediction = float(model.predict(X_pred)[0])
        return max(1.0, min(10.0, prediction))
    except Exception as exc:  # pragma: no cover - reported to UI
        st.error(f"Prediction failed: {exc}")
        return None


def create_feature_importance_chart(feature_info: Dict) -> go.Figure:
    """Build a horizontal bar chart for top permutation importances."""

    importance_records = feature_info.get("feature_importance", [])
    if not importance_records:
        return go.Figure()

    importance_df = (
        pd.DataFrame(importance_records)
        .sort_values("importance_mean", ascending=False)
        .head(10)
        .iloc[::-1]
    )

    fig = px.bar(
        importance_df,
        x="importance_mean",
        y="feature",
        orientation="h",
        color="importance_mean",
        color_continuous_scale="Viridis",
        labels={"importance_mean": "Permutation Importance", "feature": "Feature"},
        title="🔍 Top 10 Features",
    )

    fig.update_layout(height=420, yaxis_title="Feature", xaxis_title="Importance (Δ R²)")
    return fig


def create_dashboard_figures(feature_info: Dict) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """Generate three subplot dashboards (six charts total) for deployment."""

    importance_records = feature_info.get("feature_importance", [])
    importance_df = pd.DataFrame(importance_records)

    eval_frame = feature_info.get("evaluation_frame")
    if isinstance(eval_frame, pd.DataFrame):
        eval_df = eval_frame.copy()
    else:
        eval_df = pd.DataFrame(eval_frame)

    if eval_df.empty:
        eval_df = pd.DataFrame({"actual": [], "predicted": [], "residual": []})

    fig_a = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Top Feature Importance", "Predicted vs Actual"),
    )

    if not importance_df.empty:
        top_imp = importance_df.sort_values("importance_mean", ascending=False).head(10).iloc[::-1]
        fig_a.add_trace(
            go.Bar(
                x=top_imp["importance_mean"],
                y=top_imp["feature"],
                orientation="h",
                marker=dict(color=top_imp["importance_mean"], colorscale="Viridis"),
                name="Importance",
            ),
            row=1,
            col=1,
        )
    else:
        fig_a.add_annotation(
            text="No importance data",
            xref="x1",
            yref="y1",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    if not eval_df.empty:
        diag_min = float(min(eval_df["actual"].min(), eval_df["predicted"].min()))
        diag_max = float(max(eval_df["actual"].max(), eval_df["predicted"].max()))
        fig_a.add_trace(
            go.Scatter(
                x=eval_df["actual"],
                y=eval_df["predicted"],
                mode="markers",
                marker=dict(color="#1f77b4", opacity=0.5, size=8),
                name="Predictions",
            ),
            row=1,
            col=2,
        )
        fig_a.add_trace(
            go.Scatter(
                x=[diag_min, diag_max],
                y=[diag_min, diag_max],
                mode="lines",
                line=dict(color="crimson", dash="dash"),
                name="Ideal Fit",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig_a.update_xaxes(title_text="Importance (Δ R²)", row=1, col=1)
    fig_a.update_yaxes(title_text="Feature", row=1, col=1)
    fig_a.update_xaxes(title_text="Actual Rating", row=1, col=2)
    fig_a.update_yaxes(title_text="Predicted Rating", row=1, col=2)
    fig_a.update_layout(height=420, showlegend=False)

    fig_b = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Prediction Distribution", "Residuals vs Actual"),
    )

    if not eval_df.empty:
        fig_b.add_trace(
            go.Histogram(
                x=eval_df["predicted"],
                nbinsx=30,
                marker=dict(color="#4e79a7"),
                name="Predicted",
            ),
            row=1,
            col=1,
        )
        fig_b.add_trace(
            go.Scatter(
                x=eval_df["actual"],
                y=eval_df["residual"],
                mode="markers",
                marker=dict(color="#f28e2b", opacity=0.5, size=8),
                name="Residuals",
            ),
            row=1,
            col=2,
        )
        fig_b.add_hline(y=0, line=dict(color="gray", dash="dash"), row=1, col=2)

    fig_b.update_xaxes(title_text="Predicted Rating", row=1, col=1)
    fig_b.update_yaxes(title_text="Count", row=1, col=1)
    fig_b.update_xaxes(title_text="Actual Rating", row=1, col=2)
    fig_b.update_yaxes(title_text="Residual", row=1, col=2)
    fig_b.update_layout(height=420, showlegend=False)

    fig_c = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Members vs Predicted", "Favorites/Member vs Predicted"),
    )

    if not eval_df.empty:
        if "members" in eval_df.columns:
            fig_c.add_trace(
                go.Scatter(
                    x=eval_df["members"],
                    y=eval_df["predicted"],
                    mode="markers",
                    marker=dict(color="#59a14f", opacity=0.5, size=8),
                    name="Members",
                ),
                row=1,
                col=1,
            )
        if "favorites_per_member" in eval_df.columns:
            fig_c.add_trace(
                go.Scatter(
                    x=eval_df["favorites_per_member"],
                    y=eval_df["predicted"],
                    mode="markers",
                    marker=dict(color="#b07aa1", opacity=0.5, size=8),
                    name="Fav / Member",
                ),
                row=1,
                col=2,
            )

    fig_c.update_xaxes(title_text="Members", row=1, col=1)
    fig_c.update_yaxes(title_text="Predicted Rating", row=1, col=1)
    fig_c.update_xaxes(title_text="Favorites / Member", row=1, col=2)
    fig_c.update_yaxes(title_text="Predicted Rating", row=1, col=2)
    fig_c.update_layout(height=420, showlegend=False)

    return fig_a, fig_b, fig_c

def create_prediction_gauge(prediction: float) -> go.Figure:
    """Create a gauge chart for the prediction"""

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Predicted Rating"},
            gauge={
                "axis": {"range": [1, 10]},
                "bar": {"color": "#FF6B6B"},
                "steps": [
                    {"range": [1, 5], "color": "#f4f4f4"},
                    {"range": [5, 7], "color": "#ffe082"},
                    {"range": [7, 8.5], "color": "#ffcc80"},
                    {"range": [8.5, 10], "color": "#81c784"},
                ],
            },
        )
    )

    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_feature_contribution_chart(features: Dict, feature_info: Dict) -> go.Figure:
    """Create a bar chart showing top feature values that influence prediction."""
    
    # Get top contributing features
    importance_data = feature_info.get("feature_importance", [])
    if not importance_data:
        return None
    
    # Get top 6 features by importance
    top_features = sorted(importance_data, key=lambda x: x['importance_mean'], reverse=True)[:6]
    
    # Extract feature names and their current values
    feature_names = []
    feature_values = []
    colors = []
    
    for feat in top_features:
        feat_name = feat['feature']
        if feat_name in features:
            value = features[feat_name]
            # Format value for display
            if isinstance(value, (int, float)):
                if feat_name.startswith('log_'):
                    display_val = f"{value:.2f}"
                elif value > 1000:
                    display_val = f"{value:,.0f}"
                else:
                    display_val = f"{value:.2f}"
            else:
                display_val = str(value)[:20]
            
            feature_names.append(feat_name.replace('_', ' ').title())
            feature_values.append(feat['importance_mean'])
            
            # Color based on importance
            if feat['importance_mean'] > 0.05:
                colors.append('#FF6B6B')  # High importance - Red
            elif feat['importance_mean'] > 0.02:
                colors.append('#FFB84D')  # Medium importance - Orange
            else:
                colors.append('#4ECDC4')  # Lower importance - Teal
    
    if not feature_names:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            y=feature_names,
            x=feature_values,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f"{v:.3f}" for v in feature_values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="🎯 Top Feature Contributions",
            font=dict(size=16, color="#FF6B6B", family="Arial Black")
        ),
        xaxis_title="Importance Score",
        yaxis_title="",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11),
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
    fig.update_yaxes(showgrid=False)
    
    return fig


def create_anime_details_card(features: Dict, anime_row=None) -> str:
    """Create an HTML card with styled anime details."""
    
    if anime_row is not None:
        # For anime selected by name
        name = anime_row.get('name', 'Unknown Anime')
        genres = anime_row.get('genres', 'N/A')
        anime_type = anime_row.get('type', 'N/A')
        episodes = anime_row.get('episodes', 'N/A')
        members = anime_row.get('members', 0)
        favorites = anime_row.get('favorites', 0)
    else:
        # For manual input
        name = "Custom Anime"
        genres = features.get('genres', 'N/A')
        anime_type = features.get('type', 'N/A')
        episodes = int(features.get('episodes', 0))
        members = int(features.get('members', 0))
        favorites = int(features.get('favorites', 0))
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    ">
        <h3 style="margin: 0 0 15px 0; font-size: 1.3rem; color: #fff;">
            🎬 {name}
        </h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9rem;">
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 8px;">
                <div style="opacity: 0.8; font-size: 0.75rem;">Type</div>
                <div style="font-weight: bold; font-size: 1rem;">{anime_type}</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 8px;">
                <div style="opacity: 0.8; font-size: 0.75rem;">Episodes</div>
                <div style="font-weight: bold; font-size: 1rem;">{episodes}</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 8px;">
                <div style="opacity: 0.8; font-size: 0.75rem;">Members</div>
                <div style="font-weight: bold; font-size: 1rem;">{members:,}</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 8px;">
                <div style="opacity: 0.8; font-size: 0.75rem;">Favorites</div>
                <div style="font-weight: bold; font-size: 1rem;">{favorites:,}</div>
            </div>
        </div>
        <div style="
            margin-top: 12px;
            padding: 10px;
            background: rgba(255,255,255,0.15);
            border-radius: 8px;
            font-size: 0.85rem;
        ">
            <div style="opacity: 0.9; margin-bottom: 4px; font-weight: 600;">Genres</div>
            <div style="line-height: 1.4;">{genres}</div>
        </div>
    </div>
    """
    return card_html


def create_confidence_metrics(prediction: float, features: Dict) -> str:
    """Create confidence metrics display."""
    
    # Calculate pseudo-confidence based on feature completeness and values
    members = features.get('members', 0)
    favorites = features.get('favorites', 0)
    scored_by = features.get('scored_by', 0)
    
    # Confidence based on engagement metrics (higher engagement = higher confidence)
    confidence = min(95, 60 + (min(members, 100000) / 100000 * 35))
    
    # Determine confidence level and color
    if confidence >= 85:
        conf_level = "Very High"
        conf_color = "#2ecc71"
        conf_icon = "🟢"
    elif confidence >= 70:
        conf_level = "High"
        conf_color = "#3498db"
        conf_icon = "🔵"
    elif confidence >= 50:
        conf_level = "Moderate"
        conf_color = "#f39c12"
        conf_icon = "🟡"
    else:
        conf_level = "Low"
        conf_color = "#e74c3c"
        conf_icon = "🔴"
    
    metrics_html = f"""
    <div style="
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 18px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    ">
        <h4 style="margin: 0 0 12px 0; font-size: 1.1rem;">
            📊 Prediction Confidence
        </h4>
        <div style="
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        ">
            <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 5px;">
                {conf_icon} {confidence:.1f}%
            </div>
            <div style="font-size: 1rem; opacity: 0.9;">
                {conf_level} Confidence
            </div>
        </div>
        <div style="
            margin-top: 12px;
            padding: 10px;
            background: rgba(255,255,255,0.15);
            border-radius: 8px;
            font-size: 0.85rem;
        ">
            <div style="opacity: 0.9;">Based on engagement metrics and feature completeness</div>
        </div>
    </div>
    """
    return metrics_html


def extract_features_from_anime(anime_row, feature_info: Dict) -> Dict:
    """Extract features from an anime DataFrame row for prediction."""
    
    def safe_get(key, default=0):
        """Safely get value from anime row."""
        val = anime_row.get(key, default)
        if pd.isna(val):
            return default
        # Handle 'UNKNOWN' or other non-numeric strings
        if isinstance(val, str) and val.upper() in ['UNKNOWN', 'N/A', 'NA', '']:
            return default
        return val
    
    def safe_float(key, default=0):
        """Safely convert value to float."""
        val = safe_get(key, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(default)
    
    def count_tokens(text):
        """Count comma-separated tokens."""
        if pd.isna(text) or not text:
            return 0
        return sum(1 for token in str(text).split(',') if token.strip())
    
    # Extract raw values with safe conversion
    members = safe_float('members', 0)
    favorites = safe_float('favorites', 0)
    scored_by = safe_float('scored_by', 0)
    popularity = safe_float('popularity', 1)
    episodes = safe_float('episodes', 1)
    
    genres = str(safe_get('genres', ''))
    producers = str(safe_get('producers', ''))
    studios = str(safe_get('studios', ''))
    licensors = str(safe_get('licensors', ''))
    overview = str(safe_get('overview', ''))
    
    anime_type = str(safe_get('type', 'TV'))
    source = str(safe_get('source', 'Original'))
    anime_rating = str(safe_get('anime_rating', 'PG-13 - Teens 13 or older'))
    
    # Calculate derived features
    genre_count = count_tokens(genres)
    producer_count = count_tokens(producers)
    studio_count = count_tokens(studios)
    licensor_count = count_tokens(licensors)
    
    favorites_per_member = _safe_ratio(favorites, members)
    episodes_missing = int(episodes == 0)
    
    features: Dict[str, object] = {
        "genres": genres,
        "overview": overview,
        "type": anime_type,
        "episodes": episodes,
        "producers": producers,
        "licensors": licensors,
        "studios": studios,
        "source": source,
        "anime_rating": anime_rating,
        "popularity": popularity,
        "favorites": favorites,
        "scored_by": scored_by,
        "members": members,
        "genre_count": float(genre_count),
        "producer_count": float(producer_count),
        "studio_count": float(studio_count),
        "licensor_count": float(licensor_count),
        "overview_char_len": float(len(overview)),
        "overview_word_len": float(len(overview.split())),
        "has_licensor": float(1 if licensor_count > 0 else 0),
        "episodes_missing": float(episodes_missing),
        "log_popularity": _safe_log(popularity),
        "log_favorites": _safe_log(favorites),
        "log_scored_by": _safe_log(scored_by),
        "log_members": _safe_log(members),
        "log_episodes": _safe_log(episodes),
        "members_per_episode": _safe_ratio(members, episodes),
        "favorites_per_episode": _safe_ratio(favorites, episodes),
        "favorites_per_member": favorites_per_member,
        "scored_by_per_member": _safe_ratio(scored_by, members),
        "is_long_series": float(1 if episodes >= 50 else 0),
    }
    
    return features


def main():
    """Entry point for the Streamlit dashboard."""

    st.markdown('<h1 class="main-header">🎌 Anime Rating Predictor</h1>', unsafe_allow_html=True)

    model, feature_info, model_path = load_model_artifacts()
    if model is None:
        st.stop()

    model_perf = feature_info.get("model_performance", {})
    test_r2 = model_perf.get("test_r2", 0.0)
    cv_r2_mean = model_perf.get("cv_r2_mean", 0.0)
    cv_r2_std = model_perf.get("cv_r2_std", 0.0)
    rmse = model_perf.get("rmse", float("nan"))
    mae = model_perf.get("mae", float("nan"))

    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Tuned <b>XGBoost Regressor</b> explaining <b>{test_r2 * 100:.1f}%</b> of the rating variance on the test split.
            </p>
            <p style="font-size: 0.95rem; color: #888;">Artifacts loaded from <code>{model_path.name}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## 📊 Model Snapshot")
        st.metric("Test R²", f"{test_r2:.3f}", delta=f"CV mean {cv_r2_mean:.3f} ± {cv_r2_std:.3f}")
        st.metric("RMSE", f"{rmse:.3f}")
        st.metric("MAE", f"{mae:.3f}")

        st.markdown("### 🎯 Top Features")
        top_df = pd.DataFrame(feature_info.get("top_features", [])).head(5)
        if not top_df.empty:
            for idx, row in top_df.iterrows():
                st.write(f"{idx + 1}. **{row['feature']}** — {row['importance_mean']:.4f}")
        else:
            st.info("Feature importances unavailable.")

        data_info = feature_info.get("data_info", {})
        st.markdown("### 📅 Training Summary")
        st.write(f"**Training Date**: {model_perf.get('training_date', 'N/A')}")
        st.write(f"**Train Samples**: {data_info.get('train_samples', 0):,}")
        st.write(f"**Test Samples**: {data_info.get('test_samples', 0):,}")
        st.write(f"**Input Features**: {data_info.get('n_features', 0)}")

    tab1, tab2, tab3 = st.tabs(["🎯 Predict Rating", "📊 Model Insights", "ℹ️ About"])

    with tab1:
        # Add input method selector
        st.markdown('<p class="sub-header">Choose Input Method</p>', unsafe_allow_html=True)
        input_method = st.radio(
            "How would you like to predict the rating?",
            ["📝 Enter Anime Details Manually", "🔍 Select Anime by Name"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if input_method == "🔍 Select Anime by Name":
            # Load anime dataset
            anime_df = load_anime_dataset()
            
            if anime_df is not None and not anime_df.empty:
                # Single column for search input
                st.markdown('<p class="sub-header">🎬 Select an Anime</p>', unsafe_allow_html=True)
                
                # Create searchable selectbox
                anime_names = anime_df['name'].dropna().unique().tolist() if 'name' in anime_df.columns else []
                
                if anime_names:
                    selected_anime = st.selectbox(
                        "Search and select an anime:",
                        options=sorted(anime_names),
                        help="Type to search for an anime by name"
                    )
                    
                    if selected_anime:
                        # Get anime data
                        anime_row = anime_df[anime_df['name'] == selected_anime].iloc[0]
                        
                        # Display anime details in expandable section
                        with st.expander("📊 View Anime Details", expanded=False):
                            detail_col1, detail_col2 = st.columns(2)
                            with detail_col1:
                                st.write(f"**Type:** {anime_row.get('type', 'N/A')}")
                                st.write(f"**Episodes:** {anime_row.get('episodes', 'N/A')}")
                                st.write(f"**Source:** {anime_row.get('source', 'N/A')}")
                                st.write(f"**Rating:** {anime_row.get('anime_rating', 'N/A')}")
                            
                            with detail_col2:
                                st.write(f"**Members:** {anime_row.get('members', 0):,.0f}")
                                st.write(f"**Favorites:** {anime_row.get('favorites', 0):,.0f}")
                                st.write(f"**Popularity:** #{anime_row.get('popularity', 'N/A')}")
                            
                            if 'genres' in anime_row and pd.notna(anime_row['genres']):
                                st.write(f"**Genres:** {anime_row['genres']}")
                            
                            if 'overview' in anime_row and pd.notna(anime_row['overview']):
                                st.markdown("**Synopsis:**")
                                st.write(anime_row['overview'])
                        
                        # Predict button
                        if st.button("🔮 Predict Rating for This Anime", type="primary", use_container_width=True):
                            # Extract features from anime row
                            user_features = extract_features_from_anime(anime_row, feature_info)
                            prediction = predict_rating(model, feature_info, user_features)
                            
                            if prediction is not None:
                                st.session_state.prediction = prediction
                                st.session_state.input_features = user_features
                                st.session_state.selected_anime = anime_row
                        
                        # Show results after prediction with new layout
                        if "prediction" in st.session_state and "selected_anime" in st.session_state:
                            st.markdown("---")
                            
                            # Top row: Image (left) | Gauge (right)
                            col_image, col_gauge = st.columns([1, 1])
                            
                            with col_image:
                                st.markdown("### �️ Anime")
                                anime_data = st.session_state.selected_anime
                                # Display anime image if available
                                if 'image_url' in anime_data and pd.notna(anime_data['image_url']):
                                    try:
                                        st.image(anime_data['image_url'], use_container_width=True)
                                    except:
                                        st.info("🖼️ Image not available")
                                else:
                                    st.info("🖼️ No image")
                            
                            with col_gauge:
                                st.markdown("### 📊 Predicted Rating")
                                pred_value = st.session_state.prediction
                                
                                # Display prediction gauge
                                gauge_fig = create_prediction_gauge(pred_value)
                                st.plotly_chart(gauge_fig, use_container_width=True)
                                
                                st.markdown(
                                    f'<div class="prediction-result">Predicted Rating: {pred_value:.2f}/10</div>',
                                    unsafe_allow_html=True,
                                )
                                
                                # Rating interpretation
                                if pred_value >= 8.5:
                                    st.success("🌟 Expected to be a standout hit!")
                                elif pred_value >= 7.0:
                                    st.info("� Strong audience reception anticipated.")
                                elif pred_value >= 5.0:
                                    st.warning("🤔 Mixed reception likely.")
                                else:
                                    st.error("👎 Might underperform with viewers.")
                            
                            # Bottom row: Analysis (full width)
                            st.markdown("---")
                            st.markdown("### 🔍 Analysis")
                            
                            analysis_col1, analysis_col2 = st.columns([1, 1])
                            
                            with analysis_col1:
                                # Confidence metrics
                                confidence_html = create_confidence_metrics(
                                    st.session_state.prediction,
                                    st.session_state.input_features
                                )
                                st.markdown(confidence_html, unsafe_allow_html=True)
                            
                            with analysis_col2:
                                # Feature contribution chart
                                contrib_fig = create_feature_contribution_chart(
                                    st.session_state.input_features,
                                    feature_info
                                )
                                if contrib_fig:
                                    st.plotly_chart(contrib_fig, use_container_width=True)
                else:
                    st.warning("No anime names found in the dataset.")
            else:
                st.error("⚠️ Anime dataset not found. Please ensure 'Animes.csv' is in the project directory.")
        
        else:  # Manual input method
            user_features = create_feature_input_form(feature_info)
            if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
                prediction = predict_rating(model, feature_info, user_features)
                if prediction is not None:
                    st.session_state.prediction = prediction
                    st.session_state.input_features = user_features
                    # Clear selected_anime when using manual input
                    if "selected_anime" in st.session_state:
                        del st.session_state.selected_anime
            
            # Display results after prediction with new layout
            if "prediction" in st.session_state and "selected_anime" not in st.session_state:
                st.markdown("---")
                
                # Top row: Placeholder/Info (left) | Gauge (right)
                col_info, col_gauge = st.columns([1, 1])
                
                with col_info:
                    st.markdown("### � Input Summary")
                    # Key input metrics
                    display_cols = [
                        ("Members", "members"),
                        ("Favorites", "favorites"),
                        ("Episodes", "episodes"),
                        ("Genres", "genre_count"),
                    ]
                    metrics_col1, metrics_col2 = st.columns(2)
                    for label, key in display_cols[:2]:
                        metrics_col1.metric(label, f"{st.session_state.input_features.get(key, 0):,.0f}")
                    for label, key in display_cols[2:]:
                        metrics_col2.metric(label, f"{st.session_state.input_features.get(key, 0):,.0f}")
                
                with col_gauge:
                    st.markdown("### 📊 Predicted Rating")
                    pred_value = st.session_state.prediction
                    
                    # Display prediction gauge
                    gauge_fig = create_prediction_gauge(pred_value)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    st.markdown(
                        f'<div class="prediction-result">Predicted Rating: {pred_value:.2f}/10</div>',
                        unsafe_allow_html=True,
                    )

                    if pred_value >= 8.5:
                        st.success("🌟 Expected to be a standout hit!")
                    elif pred_value >= 7.0:
                        st.info("👍 Strong audience reception anticipated.")
                    elif pred_value >= 5.0:
                        st.warning("🤔 Mixed reception likely!")
                    else:
                        st.error("👎 Might underperform with viewers.")
                
                # Bottom row: Analysis (full width)
                st.markdown("---")
                st.markdown("### � Analysis")
                
                analysis_col1, analysis_col2 = st.columns([1, 1])
                
                with analysis_col1:
                    # Confidence metrics
                    confidence_html = create_confidence_metrics(
                        st.session_state.prediction,
                        st.session_state.input_features
                    )
                    st.markdown(confidence_html, unsafe_allow_html=True)
                
                with analysis_col2:
                    # Feature contribution chart
                    contrib_fig = create_feature_contribution_chart(
                        st.session_state.input_features,
                        feature_info
                    )
                    if contrib_fig:
                        st.plotly_chart(contrib_fig, use_container_width=True)

    with tab2:
        st.markdown("## 📊 Model Performance & Insights")

        fig_a, fig_b, fig_c = create_dashboard_figures(feature_info)
        st.plotly_chart(fig_a, use_container_width=True)
        st.plotly_chart(fig_b, use_container_width=True)
        st.plotly_chart(fig_c, use_container_width=True)

        metrics_records = feature_info.get("metrics_summary", [])
        if metrics_records:
            metrics_df = pd.DataFrame(metrics_records)
            st.markdown("### 📐 Metric Snapshot")
            st.dataframe(metrics_df, use_container_width=True)

        data_info = feature_info.get("data_info", {})
        st.markdown("### 📈 Dataset Overview")
        st.write(f"**Average Rating**: {data_info.get('target_mean', float('nan')):.2f}")
        st.write(f"**Rating Std Dev**: {data_info.get('target_std', float('nan')):.2f}")
        st.write(f"**Total Records**: {data_info.get('total_samples', 0):,}")

    with tab3:
        st.markdown("## ℹ️ About This Application")
        st.markdown(
            f"""
            ### 🎌 Anime Rating Predictor

            This interface deploys a tuned **XGBoost Regressor** trained on curated MyAnimeList metadata.

            - **Test R²**: {test_r2:.3f}
            - **RMSE**: {rmse:.3f}
            - **MAE**: {mae:.3f}

            #### 🔬 Highlights
            1. 🧹 Robust preprocessing with TF-IDF + SVD for textual features and scalers for numerics.
            2. 🛡️ Leakage-safe feature engineering (log scaling, ratios, length-based signals).
            3. � Three-fold cross-validation with grid search for reliable hyperparameters.

            #### � What ships with this app
            - Serialized pipeline (`{model_path.name}`) with preprocessing and model weights.
            - Deployment-ready metadata for visualisations and user guidance.
            - Interactive charts showcasing model calibration and driver signals.

            #### ⚠️ Limitations
            - Predictions mirror historical patterns; niche or new titles may deviate.
            - Text quality matters—concise, descriptive overviews offer better signal-to-noise.
            - Engagement metrics (members, favorites) heavily influence predictions.

            Built with ❤️ using Streamlit, scikit-learn, and XGBoost.
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
