import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES

def preprocess_features(df, scaler=None):
    """Handles scaling of numerical features."""
    X = df[FEATURES]
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def get_sample_input(data_dict):
    """Converts UI inputs into a DataFrame for prediction."""
    ordered = {feature: data_dict[feature] for feature in FEATURES}
    return pd.DataFrame([ordered], columns=FEATURES)
