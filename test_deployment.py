# Quick Test Script for Streamlit App
import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# Check if required files exist
files_to_check = [
    'streamlit_app.py',
    'requirements.txt',
    '.streamlit/config.toml'
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"✅ {file} exists")
    else:
        print(f"❌ {file} missing")

# Test imports
try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import joblib
    print("✅ joblib imported successfully")
except ImportError as e:
    print(f"❌ joblib import failed: {e}")

# Check for model files
import glob
model_files = glob.glob("best_anime_rating_model_*.joblib")
scaler_files = glob.glob("feature_scaler_*.joblib")
info_files = glob.glob("model_feature_info_*.pkl")

if model_files:
    print(f"✅ Model file found: {model_files[0]}")
else:
    print("❌ No model files found")

if scaler_files:
    print(f"✅ Scaler file found: {scaler_files[0]}")
else:
    print("❌ No scaler files found")

if info_files:
    print(f"✅ Info file found: {info_files[0]}")
else:
    print("❌ No info files found")

print("\n" + "="*50)
print("DEPLOYMENT STATUS CHECK COMPLETE")
print("="*50)