import numpy as np
from src.config import CLUSTER_NAMES

def predict_performance(models, input_df):
    """Runs all three ML tasks for a given student profile."""
    # Scale input
    X_scaled = models['scaler'].transform(input_df)
    
    # Regression
    reg_pred = models['regression'].predict(X_scaled)[0]
    
    # Classification
    clf_prob = models['classifier'].predict_proba(X_scaled)[0][1]
    clf_pred = 1 if clf_prob >= 0.5 else 0
    
    # Clustering
    cluster_id = models['kmeans'].predict(X_scaled)[0]
    cluster_name = CLUSTER_NAMES[cluster_id]
    
    return {
        'score': round(float(reg_pred), 2),
        'pass_fail': "PASS" if clf_pred == 1 else "FAIL",
        'pass_prob': round(float(clf_prob) * 100, 2),
        'cluster': cluster_name
    }
