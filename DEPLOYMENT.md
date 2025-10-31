# 🎌 Anime Rating Predictor - Deployment Guide

**Updated Model**: XGBoost Regressor (R² = 80.6%)  
**Last Updated**: November 1, 2025

---

## 📋 Quick Start (Local Deployment)

### Prerequisites
- Python 3.11+ (recommended) or 3.10+
- pip or conda package manager
- Git (for cloning the repository)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Gimbler778/ratings_predictor.git
   cd ratings_predictor
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the Application**
   - Local URL: `http://localhost:8501`
   - The app will automatically open in your default browser
   - Model artifacts are automatically loaded on startup

---

## ☁️ Streamlit Cloud Deployment

### 🚀 Deployment Steps

1. **Prepare Your Repository**
   ```bash
   git add .
   git commit -m "Update model with XGBoost (R² 0.8058) and new UI"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **"New app"** button
   - **Repository**: Select your GitHub repository
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - Click **"Deploy"**

3. **Monitor Deployment**
   - Streamlit Cloud will install dependencies and deploy automatically
   - First deployment takes 2-5 minutes
   - View logs during deployment for any issues
   - Live app URL format: `https://[username]-ratings-predictor.streamlit.app`

### ✅ Verification After Deployment

After deployment, verify:
- ✅ App loads without errors
- ✅ Prediction form displays correctly
- ✅ Model loads successfully (check for error messages)
- ✅ Sample prediction works (e.g., default anime values)
- ✅ Tabs display: "Predict Rating", "Model Insights", "About"

### 🔧 Version Compatibility

**Python Environment**: Python 3.10+
**Key Dependencies**:
- `streamlit>=1.29.0`
- `pandas>=2.0.0`
- `numpy>=1.26.0`
- `scikit-learn>=1.3.0`
- `joblib>=1.3.0`
- `xgboost>=1.7.0`
- `plotly>=5.15.0`

If you get version conflicts, use the optimized `requirements_streamlit.txt`:
```bash
# For deployment issues, use:
cp requirements_streamlit.txt requirements.txt
git add requirements.txt
git commit -m "Use streamlined requirements for cloud deployment"
git push
```

---

## 📁 File Structure

### Required Files (for deployment)
```
ratings_predictor/
├── streamlit_app.py                          # Main Streamlit application
├── rating_predictor.ipynb                    # ML development notebook
├── requirements.txt                          # Python dependencies
├── Animes.csv                                # Training dataset
├── .streamlit/
│   └── config.toml                          # Streamlit configuration
└── artifacts/
    ├── best_anime_rating_model_20251101_002054.joblib  # 🟢 Latest model
    ├── model_feature_info_20251101_002054.pkl          # 🟢 Metadata
    └── model_metrics_20251101_002054.csv               # 🟢 Performance metrics
```

### Model Artifacts Explained
- **`best_anime_rating_model_*.joblib`**: Serialized XGBoost pipeline with preprocessing
- **`model_feature_info_*.pkl`**: Feature metadata, importance rankings, and training info
- **`model_metrics_*.csv`**: Performance metrics (R², RMSE, MAE, etc.)

**Note**: The app automatically loads the latest timestamped artifacts. Only keep the most recent versions in `artifacts/` folder.

---

## ⚙️ Configuration

### Streamlit Config (`.streamlit/config.toml`)
The app uses custom styling defined in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
```

### Model Caching
- Models are cached using `@st.cache_resource` for performance
- First load: ~2-3 seconds
- Subsequent loads: instant (from cache)
- Cache clears when app is restarted

---

## 🐛 Troubleshooting

### Local Development Issues

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError** | Run `pip install -r requirements.txt` to install dependencies |
| **Model files not found** | Ensure `.joblib` and `.pkl` files are in `artifacts/` folder |
| **Port 8501 already in use** | Use `streamlit run streamlit_app.py --server.port 8502` |
| **Python version error** | Upgrade to Python 3.11+ with `python --version` |
| **NumPy warnings on Windows** | These are non-fatal; can be suppressed in `.streamlit/config.toml` |

### Streamlit Cloud Issues

| Issue | Solution |
|-------|----------|
| **Dependency conflict (Python 3.13)** | Use `requirements_streamlit.txt` with flexible versions |
| **Model artifact loading failed** | Ensure `.joblib` and `.pkl` files are committed to GitHub |
| **"Memory limit exceeded"** | Reduce cache size or optimize data loading |
| **App crashes on prediction** | Check that all input features are properly validated |

### Debug Mode

Enable detailed logging locally:
```bash
streamlit run streamlit_app.py --logger.level=debug
```

Check Streamlit Cloud logs:
1. Go to your app settings on share.streamlit.io
2. Click **"Manage app"** → **"Settings"**
3. View deployment logs for errors

---

## 🎯 Production Checklist

Before deploying to production:

- [ ] All dependencies listed in `requirements.txt`
- [ ] Latest model artifacts in `artifacts/` folder
- [ ] Model files named with consistent timestamps
- [ ] `.pkl` metadata file matches `.joblib` model
- [ ] `streamlit_app.py` has error handling for missing models
- [ ] `README.md` updated with current model performance
- [ ] Tested locally with `streamlit run streamlit_app.py`
- [ ] Committed and pushed to GitHub
- [ ] Deployed to Streamlit Cloud successfully
- [ ] Verified all features work (prediction, insights, about)

---

## 🚀 Live Features

### ✅ Fully Implemented
- 🎯 Interactive prediction form with organized input sections
- 📊 Real-time XGBoost-powered predictions (R² = 0.8058)
- 📈 Model performance dashboards with 3 visualization sets
- 🔍 Feature importance analysis with permutation importance
- 🎨 Anime-themed professional UI with custom styling
- 📱 Responsive design (desktop, tablet, mobile)
- 💾 Model artifact caching for performance
- ⚠️ Comprehensive error handling

### 🔄 Model Features
- **Algorithm**: XGBoost Regressor (Gradient Boosting)
- **Input Features**: 39 engineered features (numeric, categorical, text)
- **Preprocessing**: StandardScaling, OneHotEncoding, TF-IDF + SVD
- **Performance**: CV R² = 78.6% ± 0.66%, Test R² = 80.6%
- **Speed**: <2 seconds prediction on Streamlit Cloud

---

## 📚 Additional Resources

### Documentation
- **[README.md](README.md)**: Project overview, model details, and usage
- **[rating_predictor.ipynb](rating_predictor.ipynb)**: Full ML development process
- **This file**: Deployment and troubleshooting guide

### Links
- **Live App**: [ratingspredictor.streamlit.app](https://ratingspredictor.streamlit.app)
- **GitHub**: [github.com/Gimbler778/ratings_predictor](https://github.com/Gimbler778/ratings_predictor)
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)

### Support
- Report issues: [GitHub Issues](https://github.com/Gimbler778/ratings_predictor/issues)
- Feature requests: [GitHub Discussions](https://github.com/Gimbler778/ratings_predictor/discussions)
