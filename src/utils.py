import os
import joblib
import pandas as pd
import streamlit as st
from src.config import MODELS_DIR, DATA_PATH

@st.cache_resource
def load_models():
    """Loads all trained models and the scaler."""
    try:
        return {
            'regression': joblib.load(os.path.join(MODELS_DIR, "regression.pkl")),
            'classifier': joblib.load(os.path.join(MODELS_DIR, "classifier.pkl")),
            'kmeans': joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl")),
            'scaler': joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        }
    except FileNotFoundError:
        return None

def load_data():
    """Loads the dataset for visualization."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None
