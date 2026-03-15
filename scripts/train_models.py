import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score
from src.config import DATA_PATH, MODELS_DIR, FEATURES, REGRESSION_TARGET, CLASSIFICATION_TARGET, N_CLUSTERS
from src.features import preprocess_features

def main():
    if not os.path.exists(DATA_PATH):
        print("Data not found. Run generate_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Preprocessing
    X_scaled, scaler = preprocess_features(df)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # 2. Regression
    y_reg = df[REGRESSION_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train)
    joblib.dump(reg_model, os.path.join(MODELS_DIR, "regression.pkl"))
    print(f"Regression MAE: {mean_absolute_error(y_test, reg_model.predict(X_test)):.2f}")

    # 3. Classification
    y_clf = df[CLASSIFICATION_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train, y_train)
    joblib.dump(clf_model, os.path.join(MODELS_DIR, "classifier.pkl"))
    print(f"Classifier Accuracy: {accuracy_score(y_test, clf_model.predict(X_test)):.2f}")

    # 4. Clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.pkl"))
    # Save clustered data back for visualization
    df.to_csv(DATA_PATH, index=False)
    print("Models trained and saved successfully.")

if __name__ == "__main__":
    main()
