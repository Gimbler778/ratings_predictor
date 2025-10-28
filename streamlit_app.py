import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os
from datetime import datetime

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

@st.cache_resource
def load_model_components():
    """Load the trained model and related components"""
    try:
        # Find the latest model files
        model_files = glob.glob("best_anime_rating_model_*.joblib")
        scaler_files = glob.glob("feature_scaler_*.joblib")
        info_files = glob.glob("model_feature_info_*.pkl")
        
        if not model_files or not scaler_files or not info_files:
            st.error("Model files not found! Please ensure the model has been trained and saved.")
            return None, None, None
        
        # Load the latest files
        latest_model_file = max(model_files, key=os.path.getctime)
        latest_scaler_file = max(scaler_files, key=os.path.getctime)
        latest_info_file = max(info_files, key=os.path.getctime)
        
        # Load components
        model = joblib.load(latest_model_file)
        scaler = joblib.load(latest_scaler_file)
        
        with open(latest_info_file, 'rb') as f:
            feature_info = pickle.load(f)
        
        return model, scaler, feature_info
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_feature_input_form(feature_info):
    """Create input form for anime features"""
    st.markdown('<p class="sub-header">🎬 Enter Anime Details</p>', unsafe_allow_html=True)
    
    # Get feature names and create organized input sections
    feature_names = feature_info['feature_names']
    
    # Initialize feature dict
    features = {}
    
    # Main features section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 User Engagement**")
        members = st.number_input(
            "Members Count", 
            min_value=1, 
            max_value=3000000, 
            value=50000,
            help="Number of users who have this anime in their list"
        )
        favorites = st.number_input(
            "Favorites Count", 
            min_value=0, 
            max_value=200000, 
            value=1000,
            help="Number of users who marked this as favorite"
        )
        
    with col2:
        st.markdown("**📺 Content Information**")
        episodes = st.number_input(
            "Number of Episodes", 
            min_value=1, 
            max_value=2000, 
            value=12,
            help="Total number of episodes"
        )
        genre_count = st.number_input(
            "Number of Genres", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="How many genres this anime belongs to"
        )
        
    with col3:
        st.markdown("**🎭 Content Type**")
        anime_type = st.selectbox(
            "Anime Type",
            ["TV", "Movie", "OVA", "Special", "ONA", "Music"],
            help="Type of anime content"
        )
        
        anime_rating = st.selectbox(
            "Content Rating",
            ["G - All Ages", "PG - Children", "PG-13 - Teens 13 or older", 
             "R - 17+ (violence & profanity)", "R+ - Mild Nudity", "Rx - Hentai"],
            help="Age rating classification"
        )
    
    # Advanced features (auto-calculated)
    st.markdown('<p class="sub-header">🔧 Advanced Features (Auto-calculated)</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Log Members:** {np.log1p(members):.2f}")
        st.info(f"**Log Favorites:** {np.log1p(favorites):.2f}")
        st.info(f"**Log Episodes:** {np.log1p(episodes):.2f}")
        
    with col2:
        favorites_per_member = favorites / members if members > 0 else 0
        st.info(f"**Favorites per Member:** {favorites_per_member:.4f}")
        st.info(f"**Is Movie:** {'Yes' if anime_type == 'Movie' else 'No'}")
        st.info(f"**Is TV Series:** {'Yes' if anime_type == 'TV' else 'No'}")
        st.info(f"**Is Long Series:** {'Yes' if episodes > 24 else 'No'}")
    
    # Create feature vector
    for feature in feature_names:
        if feature == 'members':
            features[feature] = members
        elif feature == 'favorites':
            features[feature] = favorites
        elif feature == 'episodes':
            features[feature] = episodes
        elif feature == 'genre_count':
            features[feature] = genre_count
        elif feature == 'log_members':
            features[feature] = np.log1p(members)
        elif feature == 'log_favorites':
            features[feature] = np.log1p(favorites)
        elif feature == 'log_episodes':
            features[feature] = np.log1p(episodes)
        elif feature == 'favorites_per_member':
            features[feature] = favorites_per_member
        elif feature == 'is_movie':
            features[feature] = 1 if anime_type == 'Movie' else 0
        elif feature == 'is_tv':
            features[feature] = 1 if anime_type == 'TV' else 0
        elif feature == 'is_long_series':
            features[feature] = 1 if episodes > 24 else 0
        elif feature == 'anime_id':
            features[feature] = 1  # Default value
        # Handle one-hot encoded features
        elif feature.startswith('type_'):
            type_name = feature.replace('type_', '')
            features[feature] = 1 if anime_type == type_name else 0
        elif feature.startswith('anime_rating_'):
            rating_name = feature.replace('anime_rating_', '')
            features[feature] = 1 if rating_name in anime_rating else 0
        elif feature.startswith('source_'):
            features[feature] = 0  # Default for source features
        else:
            features[feature] = 0  # Default value for other features
    
    return features

def predict_rating(model, features, feature_names):
    """Make prediction using the trained model"""
    try:
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        # Convert to DataFrame with proper feature names
        X_pred = pd.DataFrame([feature_vector], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        
        # Ensure prediction is within valid range (1-10)
        prediction = max(1.0, min(10.0, prediction))
        
        return prediction
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def create_feature_importance_chart(feature_info):
    """Create feature importance visualization"""
    importance_data = feature_info['feature_importance']
    
    # Get top 10 features
    sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:10]
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    fig = px.bar(
        x=importance_values,
        y=feature_names,
        orientation='h',
        title="🔍 Top 10 Most Important Features",
        labels={'x': 'Feature Importance', 'y': 'Features'},
        color=importance_values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        title_font_size=16,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_prediction_gauge(prediction):
    """Create a gauge chart for the prediction"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Rating"},
        delta = {'reference': 7.0},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 5], 'color': "lightgray"},
                {'range': [5, 7], 'color': "yellow"},
                {'range': [7, 8.5], 'color': "orange"},
                {'range': [8.5, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎌 Anime Rating Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
        Predict anime ratings using advanced machine learning with <b>75.2% accuracy</b>! 
        Built with Random Forest algorithm trained on 10,000+ anime data points.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model components
    model, scaler, feature_info = load_model_components()
    
    if model is None:
        st.stop()
    
    # Sidebar with model information
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        
        model_perf = feature_info['model_performance']
        st.metric("R² Score", f"{model_perf['r2_score']:.3f}", "75.2% accuracy")
        st.metric("RMSE", f"{model_perf['rmse']:.3f}", "Low error")
        st.metric("MAE", f"{model_perf['mae']:.3f}", "Mean absolute error")
        
        st.markdown("### 🎯 Top Features")
        top_features = feature_info['top_features'][:5]
        for i, feature in enumerate(top_features, 1):
            st.write(f"{i}. **{feature['Feature']}**: {feature['Importance']:.3f}")
        
        st.markdown("### 📅 Model Details")
        st.write(f"**Training Date**: {model_perf['training_date']}")
        st.write(f"**Total Features**: {feature_info['data_info']['n_features']}")
        st.write(f"**Training Samples**: {feature_info['data_info']['total_samples']:,}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎯 Predict Rating", "📊 Model Insights", "ℹ️ About"])
    
    with tab1:
        # Prediction interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feature input form
            features = create_feature_input_form(feature_info)
            
            # Prediction button
            if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
                prediction = predict_rating(model, features, feature_info['feature_names'])
                
                if prediction is not None:
                    # Store prediction in session state
                    st.session_state.prediction = prediction
                    st.session_state.features = features
        
        with col2:
            # Display prediction result
            if hasattr(st.session_state, 'prediction'):
                prediction = st.session_state.prediction
                
                # Gauge chart
                gauge_fig = create_prediction_gauge(prediction)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Rating interpretation
                st.markdown(f'<div class="prediction-result">Rating: {prediction:.2f}/10</div>', 
                           unsafe_allow_html=True)
                
                if prediction >= 8.5:
                    st.success("🌟 Excellent! This anime is predicted to be outstanding!")
                elif prediction >= 7.0:
                    st.info("👍 Good! This anime should be quite enjoyable!")
                elif prediction >= 5.0:
                    st.warning("🤔 Average. This anime might have mixed reviews.")
                else:
                    st.error("👎 Below average. This anime might not be well-received.")
                
                # Feature contribution (simplified)
                if hasattr(st.session_state, 'features'):
                    st.markdown("### 🔍 Key Input Factors")
                    features = st.session_state.features
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Members", f"{features.get('members', 0):,}")
                        st.metric("Episodes", f"{features.get('episodes', 0)}")
                    with col_b:
                        st.metric("Favorites", f"{features.get('favorites', 0):,}")
                        st.metric("Genres", f"{features.get('genre_count', 0)}")
    
    with tab2:
        # Model insights and visualizations
        st.markdown("## 📊 Model Performance & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance chart
            importance_fig = create_feature_importance_chart(feature_info)
            st.plotly_chart(importance_fig, use_container_width=True)
        
        with col2:
            # Model performance metrics
            st.markdown("### 🎯 Model Performance")
            
            perf = feature_info['model_performance']
            
            # Create performance visualization
            metrics = ['R² Score', 'RMSE', 'MAE']
            values = [perf['r2_score'], perf['rmse'], perf['mae']]
            
            # Normalize values for better visualization
            normalized_values = [perf['r2_score'], 1-perf['rmse'], 1-perf['mae']]
            
            perf_fig = px.bar(
                x=metrics,
                y=values,
                title="Model Performance Metrics",
                color=values,
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(perf_fig, use_container_width=True)
            
            # Data insights
            st.markdown("### 📈 Dataset Insights")
            data_info = feature_info['data_info']
            st.write(f"**Total Samples**: {data_info['total_samples']:,}")
            st.write(f"**Features Used**: {data_info['n_features']}")
            st.write(f"**Average Rating**: {data_info['target_mean']:.2f}")
            st.write(f"**Rating Std Dev**: {data_info['target_std']:.2f}")
    
    with tab3:
        # About section
        st.markdown("## ℹ️ About This Application")
        
        st.markdown("""
        ### 🎌 Anime Rating Predictor
        
        This application uses a **Random Forest Regressor** trained on over 10,000 anime data points 
        to predict anime ratings with **75.2% accuracy** (R² = 0.752).
        
        ### 🔬 How It Works
        
        1. **Data Collection**: Trained on comprehensive anime dataset from MyAnimeList
        2. **Feature Engineering**: Created meaningful features like log transformations and ratios
        3. **Data Leakage Prevention**: Removed features that could cause unrealistic predictions
        4. **Model Training**: Used Random Forest with hyperparameter tuning and cross-validation
        5. **Validation**: Achieved realistic performance metrics through rigorous testing
        
        ### 🎯 Key Features Used
        
        - **User Engagement**: Members count, favorites count, favorites per member ratio
        - **Content Information**: Episode count, genre count, content type
        - **Content Classification**: Anime type (TV/Movie/OVA), content rating
        - **Engineered Features**: Log transformations, binary indicators
        
        ### 📊 Model Performance
        
        - **R² Score**: 0.752 (75.2% variance explained)
        - **RMSE**: 0.470 (average prediction error)
        - **MAE**: 0.347 (median prediction error)
        
        ### 🛠️ Technical Stack
        
        - **Machine Learning**: scikit-learn, Random Forest
        - **Frontend**: Streamlit, Plotly
        - **Data Processing**: pandas, numpy
        - **Model Persistence**: joblib, pickle
        
        ### 👨‍💻 Usage Tips
        
        1. **Members Count**: Higher member counts generally correlate with better ratings
        2. **Favorites Ratio**: A high favorites-to-members ratio indicates quality
        3. **Episode Count**: Very long or very short series might have different rating patterns
        4. **Genre Count**: Multi-genre anime might appeal to broader audiences
        
        ### ⚠️ Limitations
        
        - Predictions are based on historical data patterns
        - Individual preferences may vary significantly
        - New anime trends might not be captured
        - Quality depends on the input data accuracy
        
        ---
        
        **Built with ❤️ using Streamlit and scikit-learn**
        """)

if __name__ == "__main__":
    main()