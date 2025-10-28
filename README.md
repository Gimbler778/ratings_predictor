# Anime Rating Predictor 🎌

A comprehensive machine learning project that predicts anime ratings using advanced regression models with a professional Streamlit web interface for deployment.

## 🎯 Project Overview

This project implements a complete ML pipeline to predict anime average ratings based on various features like user engagement, content characteristics, and metadata. The final model achieves **75.2% accuracy (R² = 0.7517)** using a Random Forest Regressor.

## ✨ Key Features

- **Machine Learning Pipeline**: Complete data preprocessing, feature engineering, and model training
- **Data Leakage Prevention**: Carefully designed to avoid leaky features for realistic performance
- **Multiple Model Comparison**: 7 different regression models tested and compared
- **Interactive Web Interface**: Professional Streamlit GUI for real-time predictions
- **Comprehensive Visualizations**: Feature importance, model performance, and prediction insights
- **Production Ready**: Saved models with deployment configuration

## 🏆 Model Performance

| Model | R² Score | RMSE | MAE | Status |
|-------|----------|------|-----|--------|
| **Random Forest** | **0.7517** | **0.4703** | **0.3467** | ✅ Best |
| Ridge Regression | 0.6833 | 0.5318 | 0.4126 | 🥈 Good |
| Lasso Regression | 0.6829 | 0.5322 | 0.4128 | 🥉 Good |
| Multiple Linear | 0.6826 | 0.5324 | 0.4130 | ✅ Baseline |

## 📊 Dataset Information

- **Source**: Anime dataset with comprehensive metadata
- **Records**: 10,577 anime entries after cleaning
- **Features**: 39 legitimate features (removed data leakage)
- **Target**: Average rating (0-10 scale)

### 🔍 Most Important Features

1. **Log Favorites** (72.97% importance) - User engagement metric
2. **Log Members** (72.84% importance) - Community size indicator  
3. **Genre Count** (44.18% importance) - Content complexity
4. **Content Type** (TV/Movie) - Format classification
5. **Favorites per Member** - Engagement ratio

## 🚀 Quick Start

### Local Deployment

```bash
# Clone the repository
git clone <repository-url>
cd ratings_predictor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

Access the application at: `http://localhost:8501`

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with `streamlit_app.py` as main file

## 📁 Project Structure

```
ratings_predictor/
├── 📓 rating_predictor.ipynb        # Main ML development notebook
├── 🌐 streamlit_app.py             # Streamlit web application
├── 📋 requirements.txt             # Python dependencies
├── ⚙️  .streamlit/config.toml       # Streamlit configuration
├── 🤖 best_anime_rating_model_*.joblib  # Trained model
├── 📏 feature_scaler_*.joblib       # Feature preprocessing
├── 📄 model_feature_info_*.pkl     # Model metadata
├── 📊 Animes.csv                   # Original dataset
├── 📖 DEPLOYMENT.md                # Deployment guide
└── 📝 README.md                    # This file
```

## 🖥️ Web Interface Features

### 🎯 Prediction Interface
- **Smart Input Forms**: Organized by feature categories
- **Real-time Predictions**: Instant rating predictions with confidence
- **Interactive Gauges**: Visual rating display with color coding

### 📈 Model Insights
- **Feature Importance**: Interactive bar charts
- **Performance Metrics**: Model accuracy and error analysis
- **Prediction Confidence**: Visual feedback on prediction quality

### 📱 Professional Design
- **Responsive Layout**: Works on desktop and mobile
- **Custom Styling**: Anime-themed color scheme
- **Error Handling**: User-friendly validation and messages

## 🧠 Technical Implementation

### Data Preprocessing
- ✅ Missing value imputation with domain knowledge
- ✅ Feature engineering (log transformations, ratios)
- ✅ Categorical encoding with one-hot encoding
- ✅ Data leakage prevention (removed rank, popularity)

### Model Training
- ✅ Train/test split with proper validation
- ✅ Cross-validation for robust evaluation
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Feature scaling for linear models

### Deployment Architecture
- ✅ Model serialization with joblib
- ✅ Feature preprocessing pipeline
- ✅ Streamlit caching for performance
- ✅ Error handling and input validation

## 📈 Key Insights

### 🎯 Performance Analysis
- **Realistic Accuracy**: 75.2% variance explained (excellent for rating prediction)
- **Low Prediction Error**: RMSE of 0.47 on 0-10 scale
- **Stable Model**: Consistent cross-validation performance

### 🔍 Feature Insights
- **User Engagement**: Favorites and member count are strongest predictors
- **Content Characteristics**: Genre diversity and format matter
- **Non-leaky Features**: All predictors are available before rating assignment

### 📊 Data Quality
- **Clean Dataset**: Comprehensive preprocessing pipeline
- **No Data Leakage**: Removed features influenced by target variable
- **Balanced Features**: Good mix of numerical and categorical predictors

## 🛠️ Technologies Used

- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **Model Persistence**: joblib, pickle
- **Development**: Jupyter Notebook, Python 3.11+

## 📖 Usage Examples

### Making Predictions
```python
import joblib

# Load the trained model
model = joblib.load('best_anime_rating_model_*.joblib')
scaler = joblib.load('feature_scaler_*.joblib')

# Example prediction
features = [members, favorites, episodes, genre_count, ...]
prediction = model.predict([features])[0]
print(f"Predicted Rating: {prediction:.2f}")
```

### Using the Web Interface
1. **Input anime features** in the organized form sections
2. **Click "Predict Rating"** for instant results
3. **View insights** in the Model Insights tab
4. **Explore visualizations** for deeper understanding

## 🚨 Important Notes

### Data Leakage Prevention
This project initially achieved unrealistic perfect scores (R² = 0.9929) due to data leakage. We systematically identified and removed:
- **Rank** (directly derived from ratings)
- **Popularity scores** (influenced by ratings)  
- **Scored by counts** (correlated with ratings)

The final model uses only legitimate predictors available before rating assignment.

### Model Limitations
- Predictions are based on historical anime data patterns
- Performance may vary for anime with unique characteristics
- Model works best within the training data distribution

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anime dataset providers for comprehensive metadata
- Streamlit team for the excellent web framework
- scikit-learn contributors for robust ML tools
- Open source community for inspiration and tools

---

## 🎮 Try It Live!

**[Launch the Anime Rating Predictor →](your-streamlit-app-url)**

*Predict anime ratings with machine learning precision!*