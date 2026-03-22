import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(DATA_DIR, "students.csv")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

FEATURES = [
    "study_hours_per_day",
    "attendance_percent",
    "assignments_completed",
    "sleep_hours",
    "previous_score",
    "internet_usage_hours",
    "participation_score",
    "extra_tutoring",
    "practice_tests_completed",
    "stress_level",
]

TARGET_COLUMN = "pass_fail"
SCORE_COLUMN = "final_exam_score"

RANDOM_STATE = 42
TEST_SIZE = 0.2

PASS_LABELS = {
    0: "Likely Fail",
    1: "Likely Pass",
}
