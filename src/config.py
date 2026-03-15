import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "students.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Features
FEATURES = [
    'study_hours_per_day', 'attendance_percent', 'assignments_completed',
    'sleep_hours', 'previous_score', 'internet_usage_hours',
    'participation_score', 'extra_tutoring', 'practice_tests_completed',
    'stress_level'
]

# Targets
REGRESSION_TARGET = 'final_exam_score'
CLASSIFICATION_TARGET = 'pass_fail'

# Clustering
N_CLUSTERS = 3
CLUSTER_NAMES = {
    0: "High Performers",
    1: "Average Students",
    2: "At-Risk Students"
}
