import streamlit as st
import pandas as pd
from src.utils import load_models, load_data
from src.charts import *
from src.predict import predict_performance
from src.config import FEATURES, N_CLUSTERS, CLUSTER_NAMES

# Page Config
st.set_page_config(page_title="Student ML Analyzer", layout="wide")

# CSS for polish
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_with_html=True)

# Sidebar Navigation
st.sidebar.title("📚 Education AI")
page = st.sidebar.radio("Navigate", ["Overview", "Dataset Insights", "Predictions", "Clustering Analysis"])

# Load Resources
models = load_models()
df = load_data()

if models is None or df is None:
    st.error("⚠️ Models or Data not found! Please run the training scripts as described in the README.")
    st.stop()

if page == "Overview":
    st.title("🎓 Student Performance Analysis Dashboard")
    st.markdown("""
    Welcome to the **Machine Learning Demo**. This project demonstrates three core ML paradigms using a synthetic student dataset.
    
    ### 🔬 Machine Learning Tasks:
    1. **Regression**: Predicting the exact `Final Exam Score` (0-100).
    2. **Classification**: Categorizing students into `Pass` or `Fail`.
    3. **Clustering**: Automatically grouping students into performance tiers.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", f"{len(df)} Students")
    col2.metric("Features", f"{len(FEATURES)} Variables")
    col3.metric("ML Tasks", "3 Paradigms")
    
    st.plotly_chart(plot_distribution(df, 'final_exam_score'), use_container_width=True)

elif page == "Dataset Insights":
    st.title("📊 Exploration & Insights")
    tab1, tab2 = st.tabs(["Distributions", "Correlations"])
    
    with tab1:
        feature = st.selectbox("Select Feature to Examine", FEATURES)
        st.plotly_chart(plot_distribution(df, feature), use_container_width=True)
    
    with tab2:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
        st.info("💡 Note how 'Previous Score' and 'Attendance' correlate strongly with the final grade.")

elif page == "Predictions":
    st.title("🔮 Performance Predictor")
    st.write("Adjust student parameters below to see ML predictions in real-time.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Profile")
        inputs = {}
        inputs['study_hours_per_day'] = st.slider("Study Hours", 0.0, 12.0, 5.0)
        inputs['attendance_percent'] = st.slider("Attendance %", 50.0, 100.0, 85.0)
        inputs['previous_score'] = st.slider("Previous Score", 0.0, 100.0, 70.0)
        inputs['assignments_completed'] = st.number_input("Assignments", 0, 20, 15)
        inputs['sleep_hours'] = st.slider("Sleep Hours", 4.0, 10.0, 7.0)
        inputs['participation_score'] = st.slider("Participation", 0, 100, 60)
        inputs['stress_level'] = st.slider("Stress (1-10)", 1, 10, 5)
        inputs['extra_tutoring'] = 1 if st.checkbox("Extra Tutoring") else 0
        inputs['practice_tests_completed'] = st.number_input("Practice Tests", 0, 10, 3)
        inputs['internet_usage_hours'] = st.slider("Daily Internet Use", 0.0, 6.0, 2.0)

    with col2:
        input_df = pd.DataFrame([inputs])
        results = predict_performance(models, input_df)
        
        st.subheader("Results")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Predicted Score", f"{results['score']}/100")
            status_color = "green" if results['pass_fail'] == "PASS" else "red"
            st.markdown(f"### Status: <span style='color:{status_color}'>{results['pass_fail']}</span>", unsafe_with_html=True)
        
        with res_col2:
            st.metric("Pass Probability", f"{results['pass_prob']}%")
            st.metric("Assigned Cluster", results['cluster'])
        
        st.divider()
        st.subheader("How the Model Thinks")
        st.plotly_chart(plot_feature_importance(models['regression'], FEATURES), use_container_width=True)

elif page == "Clustering Analysis":
    st.title("🧩 Unsupervised Learning: Clustering")
    st.write("The model grouped students into 3 distinct clusters based on their habits and performance.")
    
    st.plotly_chart(plot_clusters(df), use_container_width=True)
    
    st.subheader("Cluster Explanations")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success(f"**{CLUSTER_NAMES[0]}**")
        st.write("High attendance, high previous scores, and consistent study habits.")
    with c2:
        st.warning(f"**{CLUSTER_NAMES[1]}**")
        st.write("Moderate engagement and average scores. Potential for growth.")
    with c3:
        st.error(f"**{CLUSTER_NAMES[2]}**")
        st.write("Low attendance and lower scores. Requires immediate intervention.")
