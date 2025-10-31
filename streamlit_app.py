import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Tuple

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
        col_form, col_result = st.columns([2, 1])

        with col_form:
            user_features = create_feature_input_form(feature_info)
            if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
                prediction = predict_rating(model, feature_info, user_features)
                if prediction is not None:
                    st.session_state.prediction = prediction
                    st.session_state.input_features = user_features

        with col_result:
            if "prediction" in st.session_state:
                pred_value = st.session_state.prediction
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
                    st.warning("🤔 Mixed reception likely.")
                else:
                    st.error("👎 Might underperform with viewers.")

                if "input_features" in st.session_state:
                    st.markdown("### 🔍 Key Inputs")
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
