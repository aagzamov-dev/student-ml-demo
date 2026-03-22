import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSIFIER_PATH, DATA_PATH, METRICS_PATH, SCALER_PATH
from src.features import get_sample_input
from src.predict import predict_student_outcome
from src.utils import load_data, load_metrics, load_model_bundle


def validate_artifacts():
    required_files = [DATA_PATH, CLASSIFIER_PATH, SCALER_PATH, METRICS_PATH]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))


def main():
    validate_artifacts()

    bundle = load_model_bundle()
    df = load_data()
    metrics = load_metrics()

    if bundle is None or df is None or metrics is None:
        raise RuntimeError("Could not load the trained classifier, dataset, or metrics.")

    sample = get_sample_input(
        {
            "study_hours_per_day": 6.5,
            "attendance_percent": 91.0,
            "assignments_completed": 17,
            "sleep_hours": 7.0,
            "previous_score": 82.0,
            "internet_usage_hours": 2.0,
            "participation_score": 72,
            "extra_tutoring": 1,
            "practice_tests_completed": 6,
            "stress_level": 4,
        }
    )
    prediction = predict_student_outcome(bundle, sample)

    if prediction["prediction"] not in {"Likely Pass", "Likely Fail"}:
        raise AssertionError("Prediction label is invalid.")
    if not 0 <= prediction["pass_probability"] <= 100:
        raise AssertionError("Pass probability is outside the expected range.")

    print("Classification demo test passed.")
    print(f"Rows loaded: {len(df)}")
    print(json.dumps(prediction, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
