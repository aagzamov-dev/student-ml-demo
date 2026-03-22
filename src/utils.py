import json
import os

import joblib
import pandas as pd
import streamlit as st

from src.config import CLASSIFIER_PATH, DATA_PATH, METRICS_PATH, SCALER_PATH


@st.cache_resource
def load_model_bundle():
    try:
        return {
            "classifier": joblib.load(CLASSIFIER_PATH),
            "scaler": joblib.load(SCALER_PATH),
        }
    except FileNotFoundError:
        return None


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return None
