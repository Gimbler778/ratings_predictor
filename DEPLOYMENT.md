# Anime Rating Predictor - Deployment Guide

## Quick Start (Local Deployment)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the Application**
   - Open your browser to: `http://localhost:8501`

## Streamlit Cloud Deployment

### ⚠️ Important: Python 3.13 Compatibility
For Streamlit Cloud deployment, use the compatible requirements:

**Option A: Use streamlined requirements (RECOMMENDED)**
```bash
# Rename requirements_streamlit.txt to requirements.txt for deployment
mv requirements_streamlit.txt requirements.txt
```

**Option B: Update existing requirements.txt**
Use flexible version ranges instead of exact versions:
```
streamlit>=1.29.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.3.0
joblib>=1.3.0
plotly>=5.15.0
```

### Deployment Steps
1. **Fix Requirements**: Ensure compatible package versions (see above)
2. **Push to GitHub**: Commit all changes including fixed requirements
3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository and branch
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

### Alternative: Community Cloud
1. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Repository: Your GitHub repo
5. Branch: main
6. Main file path: streamlit_app.py
7. Click "Deploy!"

## File Structure for Deployment
```
ratings_predictor/
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── best_anime_rating_model_*.joblib  # Trained model
├── feature_scaler_*.joblib   # Feature scaler
├── model_feature_info_*.pkl  # Model metadata
├── Animes.csv               # Original dataset
└── rating_predictor.ipynb   # Development notebook
```

## Environment Variables (Optional)
Create a `.env` file for sensitive configurations:
```
STREAMLIT_THEME_PRIMARY_COLOR="#FF6B6B"
STREAMLIT_SERVER_PORT=8501
```

## Troubleshooting

### Streamlit Cloud Deployment Issues:

**🚨 Python 3.13 + Pandas Compatibility Error**
```
× No solution found when resolving dependencies:
pandas==2.1.3 depends on numpy>=1.26.0,<=1.26.4
numpy==1.24.3 (conflicting versions)
```

**Solution:**
1. Use the `requirements_streamlit.txt` file (rename to `requirements.txt`)
2. Or update requirements.txt with flexible versions:
   ```
   streamlit>=1.29.0
   pandas>=2.0.0  
   numpy>=1.26.0
   scikit-learn>=1.3.0
   joblib>=1.3.0
   plotly>=5.15.0
   ```
3. Commit and push changes
4. Restart deployment on Streamlit Cloud

### Common Local Issues:
1. **ModuleNotFoundError**: Run `pip install -r requirements.txt`
2. **Model files not found**: Ensure model files are in the same directory
3. **Port already in use**: Change port in config.toml or use `--server.port 8502`
4. **NumPy warnings on Windows**: Use Python 3.11 for better stability

### Performance Tips:
- Model loading is cached for better performance
- Large datasets are processed efficiently with pandas
- Plotly charts are optimized for web display

## Production Considerations
- Model files are loaded once and cached
- Input validation prevents errors
- Responsive design works on mobile devices
- Error handling provides user-friendly messages

## Features Available:
✅ Interactive prediction interface  
✅ Real-time rating predictions  
✅ Model performance visualizations  
✅ Feature importance analysis  
✅ Professional gauge charts  
✅ Responsive design  
✅ Model metadata display  

## Support
For issues or questions, refer to the model development notebook: `rating_predictor.ipynb`
