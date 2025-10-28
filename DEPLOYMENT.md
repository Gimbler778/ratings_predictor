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

### Option 1: Direct Deployment
1. Push your code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and branch
5. Set main file path: `streamlit_app.py`
6. Click "Deploy"

### Option 2: Community Cloud
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

### Common Issues:
1. **ModuleNotFoundError**: Run `pip install -r requirements.txt`
2. **Model files not found**: Ensure model files are in the same directory
3. **Port already in use**: Change port in config.toml or use `--server.port 8502`

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