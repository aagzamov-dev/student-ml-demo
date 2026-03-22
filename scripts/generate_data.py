import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH


def generate_student_data(n_samples=1000, seed=42):
    if n_samples < 50:
        raise ValueError("Please generate at least 50 samples so the train/test split remains meaningful.")

    np.random.seed(seed)
    
    # Independent Features
    study_hours = np.random.uniform(1, 12, n_samples)
    attendance = np.random.uniform(60, 100, n_samples)
    prev_score = np.random.uniform(40, 100, n_samples)
    sleep_hours = np.random.uniform(5, 9, n_samples)
    assignments = np.random.randint(5, 21, n_samples)
    participation = np.random.uniform(20, 100, n_samples)
    stress = np.random.uniform(1, 10, n_samples)
    tutoring = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    practice_tests = np.random.randint(0, 11, n_samples)
    internet = np.random.uniform(1, 6, n_samples)

    # Calculate Regression Target: Final Exam Score
    # Weighted formula + noise to simulate reality
    final_score = (
        0.3 * prev_score + 
        0.2 * attendance + 
        0.2 * (study_hours * 100 / 12) + 
        0.1 * (assignments * 100 / 20) + 
        0.1 * participation + 
        0.1 * (practice_tests * 10) - 
        0.05 * (stress * 10) +
        np.random.normal(0, 4, n_samples)
    )
    
    final_score = np.clip(final_score, 0, 100)
    
    # Calculate Classification Target: Pass/Fail (threshold 50)
    pass_fail = (final_score >= 50).astype(int)

    df = pd.DataFrame({
        'study_hours_per_day': study_hours,
        'attendance_percent': attendance,
        'assignments_completed': assignments,
        'sleep_hours': sleep_hours,
        'previous_score': prev_score,
        'internet_usage_hours': internet,
        'participation_score': participation,
        'extra_tutoring': tutoring,
        'practice_tests_completed': practice_tests,
        'stress_level': stress,
        'final_exam_score': final_score,
        'pass_fail': pass_fail
    })

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset generated at {DATA_PATH} with {n_samples} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic student performance dataset.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of student rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible data generation.")
    args = parser.parse_args()
    generate_student_data(n_samples=args.samples, seed=args.seed)
