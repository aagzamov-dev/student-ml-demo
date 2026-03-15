import plotly.express as px
import pandas as pd
from src.config import CLUSTER_NAMES

def plot_correlation_heatmap(df):
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", 
                    color_continuous_scale='RdBu_r', title="Feature Correlation Heatmap")
    return fig

def plot_distribution(df, column):
    fig = px.histogram(df, x=column, nbins=30, marginal="box", 
                       title=f"Distribution of {column.replace('_', ' ').title()}")
    return fig

def plot_clusters(df):
    # We use Attendance and Previous Score for visualization
    df_copy = df.copy()
    df_copy['Cluster Name'] = df_copy['cluster'].map(CLUSTER_NAMES)
    
    fig = px.scatter(df_copy, x='attendance_percent', y='previous_score', 
                     color='Cluster Name', hover_data=['final_exam_score'],
                     title="Student Clusters: Attendance vs Previous Score")
    return fig

def plot_feature_importance(model, features):
    importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False)
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance (Random Forest)")
    return fig
