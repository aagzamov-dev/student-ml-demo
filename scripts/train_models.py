import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_setup import load_project_env

load_project_env()

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSIFIER_PATH,
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.features import preprocess_features


def validate_dataset(df: pd.DataFrame) -> None:
    required_columns = FEATURES + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing_columns))
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The target column must contain both pass and fail examples.")


def remove_stale_artifacts() -> None:
    obsolete_files = [
        os.path.join(MODELS_DIR, "regression.pkl"),
        os.path.join(MODELS_DIR, "kmeans.pkl"),
        os.path.join(MODELS_DIR, "cluster_profiles.json"),
    ]
    for path in obsolete_files:
        if os.path.exists(path):
            os.remove(path)


def main():
    if not os.path.exists(DATA_PATH):
        print("Data not found. Run generate_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)
    os.makedirs(MODELS_DIR, exist_ok=True)
    remove_stale_artifacts()

    X_scaled, scaler = preprocess_features(df)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "model_name": "RandomForestClassifier",
        "demo_question": "Will this student pass the final exam?",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1_score": round(float(f1_score(y_test, predictions)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "pass_rate_percent": round(float(df[TARGET_COLUMN].mean() * 100), 2),
        "confusion_matrix": {
            "true_fail_pred_fail": int(matrix[0][0]),
            "true_fail_pred_pass": int(matrix[0][1]),
            "true_pass_pred_fail": int(matrix[1][0]),
            "true_pass_pred_pass": int(matrix[1][1]),
        },
    }

    joblib.dump(classifier, CLASSIFIER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Classification model trained successfully.")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1_score']}")


if __name__ == "__main__":
    main()
