import pandas as pd
import streamlit as st

from src.charts import (
    plot_feature_importance,
    plot_pass_distribution,
    plot_probability_gauge,
    plot_score_distribution,
    plot_study_vs_previous,
)
from src.config import FEATURES
from src.features import get_sample_input
from src.predict import predict_student_outcome
from src.utils import load_data, load_metrics, load_model_bundle

st.set_page_config(page_title="Student Pass Predictor", page_icon="📘", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --bg: #09111f;
        --bg-soft: #101a2d;
        --panel: #111c31;
        --panel-2: #16243d;
        --border: rgba(148, 163, 184, 0.18);
        --text: #edf3ff;
        --muted: #9fb1cc;
        --accent: #56b6ff;
        --good: #2ecc71;
        --bad: #ff6b6b;
        --warn: #f7c46c;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(86, 182, 255, 0.08), transparent 28%),
            radial-gradient(circle at top right, rgba(46, 204, 113, 0.06), transparent 22%),
            linear-gradient(180deg, #09111f 0%, #0f1728 100%);
        color: var(--text);
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, span, div {
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1322 0%, #11192b 100%);
        border-right: 1px solid var(--border);
    }
    .hero, .card, .callout {
        background: rgba(17, 28, 49, 0.92);
        border: 1px solid var(--border);
        border-radius: 22px;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
    }
    .hero {
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.15rem 1.2rem;
    }
    .callout {
        padding: 1rem 1.1rem;
        background: linear-gradient(135deg, rgba(24, 38, 64, 0.95), rgba(16, 28, 47, 0.95));
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.4rem;
        color: #f8fbff;
    }
    .hero p, .subtle, .stCaption {
        color: var(--muted) !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(17, 28, 49, 0.94);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.7rem;
    }
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stNumberInput > div > div,
    .stTextInput > div > div {
        background: var(--panel) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .stDataEditor, .stDataFrame {
        background: transparent;
    }
    .stSlider label, .stToggle label, .stSelectbox label {
        color: var(--text) !important;
    }
    .stMarkdown a {
        color: #88c8ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SCENARIOS = {
    "Strong Student": {
        "study_hours_per_day": 8.0,
        "attendance_percent": 96.0,
        "assignments_completed": 18,
        "sleep_hours": 7.5,
        "previous_score": 88.0,
        "internet_usage_hours": 1.5,
        "participation_score": 84,
        "extra_tutoring": 1,
        "practice_tests_completed": 7,
        "stress_level": 3,
    },
    "Average Student": {
        "study_hours_per_day": 5.0,
        "attendance_percent": 84.0,
        "assignments_completed": 14,
        "sleep_hours": 6.5,
        "previous_score": 69.0,
        "internet_usage_hours": 2.8,
        "participation_score": 61,
        "extra_tutoring": 0,
        "practice_tests_completed": 4,
        "stress_level": 5,
    },
    "At Risk Student": {
        "study_hours_per_day": 2.5,
        "attendance_percent": 66.0,
        "assignments_completed": 8,
        "sleep_hours": 5.5,
        "previous_score": 48.0,
        "internet_usage_hours": 5.2,
        "participation_score": 35,
        "extra_tutoring": 0,
        "practice_tests_completed": 1,
        "stress_level": 8,
    },
}

FLOAT_FEATURES = {
    "study_hours_per_day",
    "attendance_percent",
    "sleep_hours",
    "previous_score",
    "internet_usage_hours",
}
INT_FEATURES = {
    "assignments_completed",
    "participation_score",
    "extra_tutoring",
    "practice_tests_completed",
    "stress_level",
}
SLIDER_LIMITS = {
    "study_hours_per_day": (0.0, 12.0, 0.5),
    "attendance_percent": (50.0, 100.0, 1.0),
    "assignments_completed": (0, 20, 1),
    "sleep_hours": (4.0, 10.0, 0.5),
    "previous_score": (0.0, 100.0, 1.0),
    "internet_usage_hours": (0.0, 8.0, 0.5),
    "participation_score": (0, 100, 1),
    "extra_tutoring": (0, 1, 1),
    "practice_tests_completed": (0, 10, 1),
    "stress_level": (1, 10, 1),
}

bundle = load_model_bundle()
df = load_data()
metrics = load_metrics()

if bundle is None or df is None:
    st.error("Model or dataset not found. Run the README commands to generate data and train the classifier.")
    st.stop()


def clean_student_inputs(values: dict) -> dict:
    cleaned = {}
    for feature in FEATURES:
        value = values[feature]
        if feature in FLOAT_FEATURES:
            cleaned[feature] = float(value)
        else:
            cleaned[feature] = int(round(float(value)))
    return cleaned


def set_student_inputs(values: dict) -> None:
    st.session_state.student_inputs = clean_student_inputs(values)


def get_student_editor_df(values: dict) -> pd.DataFrame:
    return pd.DataFrame([clean_student_inputs(values)], columns=FEATURES)


def get_dataset_display(dataframe: pd.DataFrame) -> pd.DataFrame:
    display_df = dataframe.copy()
    display_df.insert(0, "student_id", range(1, len(display_df) + 1))
    display_df["pass_label"] = display_df["pass_fail"].map({1: "Pass", 0: "Fail"})
    return display_df


dataset_display = get_dataset_display(df)

if "scenario" not in st.session_state:
    st.session_state.scenario = "Average Student"
if "student_inputs" not in st.session_state:
    set_student_inputs(SCENARIOS[st.session_state.scenario])
if "selected_student_id" not in st.session_state:
    st.session_state.selected_student_id = 1


def apply_scenario(name: str) -> None:
    st.session_state.scenario = name
    set_student_inputs(SCENARIOS[name])


def load_dataset_student(student_id: int) -> None:
    row = dataset_display.loc[dataset_display["student_id"] == student_id, FEATURES].iloc[0].to_dict()
    set_student_inputs(row)
    st.session_state.selected_student_id = int(student_id)


st.sidebar.title("Student Pass Predictor")
page = st.sidebar.radio("Choose a page", ["Overview", "Test Student", "Student Data", "Model Evidence"])

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <h1>Student Pass Predictor</h1>
            <p>
                One machine learning classifier predicts whether a student is likely to pass the final exam.
                Use the student data, edit the inputs, and show the result live.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Students", len(df))
    metric_b.metric("Features", len(FEATURES))
    metric_c.metric("Pass Rate", f"{df['pass_fail'].mean() * 100:.1f}%")
    metric_d.metric("Accuracy", metrics["accuracy"] if metrics else "N/A")

    chart_left, chart_right = st.columns([1.05, 1])
    with chart_left:
        st.plotly_chart(plot_pass_distribution(df), use_container_width=True)
        st.plotly_chart(plot_study_vs_previous(df), use_container_width=True)
    with chart_right:
        st.markdown(
            """
            <div class="callout">
                <h3 style="margin-top:0;">What This Demo Shows</h3>
                <p class="subtle">
                    The model uses study time, attendance, previous score, assignments, tutoring,
                    participation, stress, and similar features to classify a student as likely to pass or fail.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_score_distribution(df), use_container_width=True)

elif page == "Test Student":
    st.markdown(
        """
        <div class="hero">
            <h1>Live Student Test</h1>
            <p>
                Load a ready-made scenario, pick a real student from the dataset, or edit a custom student row.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_col, picker_col = st.columns([0.95, 1.15])
    with source_col:
        input_mode = st.selectbox("Input source", ["Quick scenario", "Dataset student", "Custom editable row"])
    with picker_col:
        if input_mode == "Quick scenario":
            scenario = st.selectbox(
                "Scenario",
                list(SCENARIOS.keys()),
                index=list(SCENARIOS.keys()).index(st.session_state.scenario),
            )
            if scenario != st.session_state.scenario:
                apply_scenario(scenario)
        elif input_mode == "Dataset student":
            student_id = st.selectbox(
                "Student from dataset",
                dataset_display["student_id"].tolist(),
                index=max(st.session_state.selected_student_id - 1, 0),
            )
            if student_id != st.session_state.selected_student_id:
                load_dataset_student(student_id)
        else:
            st.caption("Edit the one-row table below and the prediction updates immediately.")

    if input_mode == "Custom editable row":
        editor_df = st.data_editor(
            get_student_editor_df(st.session_state.student_inputs),
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="custom_student_editor",
        )
        if not editor_df.empty:
            set_student_inputs(editor_df.iloc[0].to_dict())
    elif input_mode == "Dataset student":
        selected_view = dataset_display.loc[
            dataset_display["student_id"] == st.session_state.selected_student_id
        ].copy()
        st.dataframe(selected_view, use_container_width=True, hide_index=True)

    left, right = st.columns([1.02, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Student Inputs")
        defaults = st.session_state.student_inputs
        inputs = {}
        for feature in FEATURES:
            min_value, max_value, step = SLIDER_LIMITS[feature]
            label = feature.replace("_", " ").title()
            if feature == "extra_tutoring":
                inputs[feature] = 1 if st.toggle(label, value=bool(defaults[feature])) else 0
            elif feature in FLOAT_FEATURES:
                inputs[feature] = st.slider(label, float(min_value), float(max_value), float(defaults[feature]), float(step))
            else:
                inputs[feature] = st.slider(label, int(min_value), int(max_value), int(defaults[feature]), int(step))
        set_student_inputs(inputs)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        input_df = get_sample_input(st.session_state.student_inputs)
        result = predict_student_outcome(bundle, input_df)
        status = result["prediction"]
        status_color = "#2ecc71" if result["predicted_class"] == 1 else "#ff6b6b"

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction")
        st.markdown(
            f"<p style='font-size:1.35rem;'>Result: <strong style='color:{status_color};'>{status}</strong></p>",
            unsafe_allow_html=True,
        )
        prob_a, prob_b = st.columns(2)
        prob_a.metric("Pass Probability", f"{result['pass_probability']}%")
        prob_b.metric("Fail Risk", f"{result['risk_probability']}%")
        st.plotly_chart(plot_probability_gauge(result["pass_probability"]), use_container_width=True)

        comparison = pd.DataFrame(
            {
                "Signal": ["Study Hours", "Attendance", "Previous Score", "Stress", "Assignments"],
                "Student": [
                    st.session_state.student_inputs["study_hours_per_day"],
                    st.session_state.student_inputs["attendance_percent"],
                    st.session_state.student_inputs["previous_score"],
                    st.session_state.student_inputs["stress_level"],
                    st.session_state.student_inputs["assignments_completed"],
                ],
                "Dataset Average": [
                    round(float(df["study_hours_per_day"].mean()), 2),
                    round(float(df["attendance_percent"].mean()), 2),
                    round(float(df["previous_score"].mean()), 2),
                    round(float(df["stress_level"].mean()), 2),
                    round(float(df["assignments_completed"].mean()), 2),
                ],
            }
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Student Data":
    st.markdown(
        """
        <div class="hero">
            <h1>Student Data</h1>
            <p>
                Browse the generated dataset, inspect all students, and load any row into the live test page.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_left, preview_right = st.columns([0.85, 1.15])
    with preview_left:
        selected_student_id = st.number_input(
            "Student ID to load into test page",
            min_value=1,
            max_value=int(len(dataset_display)),
            value=int(st.session_state.selected_student_id),
            step=1,
        )
        if st.button("Load This Student", use_container_width=True):
            load_dataset_student(int(selected_student_id))
            st.success(f"Student {selected_student_id} loaded into Test Student.")
    with preview_right:
        selected_row = dataset_display.loc[dataset_display["student_id"] == int(selected_student_id)]
        st.dataframe(selected_row, use_container_width=True, hide_index=True)

    st.subheader("All Students")
    st.dataframe(dataset_display, use_container_width=True, height=520, hide_index=True)

elif page == "Model Evidence":
    st.markdown(
        """
        <div class="hero">
            <h1>Model Evidence</h1>
            <p>
                Use this page to explain the model performance, feature importance, and evaluation results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", metrics["accuracy"])
        m2.metric("Precision", metrics["precision"])
        m3.metric("Recall", metrics["recall"])
        m4.metric("F1 Score", metrics["f1_score"])

        matrix = metrics["confusion_matrix"]
        matrix_df = pd.DataFrame(
            [
                ["Actual Fail", matrix["true_fail_pred_fail"], matrix["true_fail_pred_pass"]],
                ["Actual Pass", matrix["true_pass_pred_fail"], matrix["true_pass_pred_pass"]],
            ],
            columns=["Actual / Predicted", "Predicted Fail", "Predicted Pass"],
        )

        left, right = st.columns([1.18, 1])
        with left:
            st.plotly_chart(plot_feature_importance(bundle["classifier"]), use_container_width=True)
        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Evaluation Summary")
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
            st.write(f"Training rows: {metrics['train_rows']}")
            st.write(f"Testing rows: {metrics['test_rows']}")
            st.write(f"Dataset pass rate: {metrics['pass_rate_percent']}%")
            st.markdown("</div>", unsafe_allow_html=True)
