import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import FEATURES, SCORE_COLUMN, TARGET_COLUMN

PLOTLY_TEMPLATE = "plotly_dark"


def _apply_dark_layout(fig):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#111c31",
        plot_bgcolor="#111c31",
        font_color="#edf3ff",
        title_font_color="#edf3ff",
        legend_font_color="#edf3ff",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_pass_distribution(df):
    counts = (
        df[TARGET_COLUMN]
        .map({0: "Fail", 1: "Pass"})
        .value_counts()
        .rename_axis("Outcome")
        .reset_index(name="Students")
    )
    fig = px.bar(
        counts,
        x="Outcome",
        y="Students",
        color="Outcome",
        color_discrete_map={"Pass": "#2ecc71", "Fail": "#ff6b6b"},
        title="Pass vs Fail Distribution",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False)
    return _apply_dark_layout(fig)


def plot_score_distribution(df):
    fig = px.histogram(
        df,
        x=SCORE_COLUMN,
        nbins=24,
        title="Final Exam Score Distribution",
        color_discrete_sequence=["#56b6ff"],
        template=PLOTLY_TEMPLATE,
    )
    return _apply_dark_layout(fig)


def plot_feature_importance(model):
    importance = pd.DataFrame(
        {"Feature": FEATURES, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=True)
    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Which Inputs Matter Most",
        color="Importance",
        color_continuous_scale="Tealgrn",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False)
    return _apply_dark_layout(fig)


def plot_study_vs_previous(df):
    df_copy = df.copy()
    df_copy["Outcome"] = df_copy[TARGET_COLUMN].map({0: "Fail", 1: "Pass"})
    fig = px.scatter(
        df_copy,
        x="previous_score",
        y="study_hours_per_day",
        color="Outcome",
        color_discrete_map={"Pass": "#2ecc71", "Fail": "#ff6b6b"},
        hover_data=["attendance_percent", SCORE_COLUMN],
        title="Study Hours vs Previous Score",
        template=PLOTLY_TEMPLATE,
    )
    return _apply_dark_layout(fig)


def plot_probability_gauge(pass_probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pass_probability,
            number={"suffix": "%"},
            title={"text": "Chance Of Passing"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71"},
                "steps": [
                    {"range": [0, 50], "color": "#4a1f2b"},
                    {"range": [50, 75], "color": "#56451f"},
                    {"range": [75, 100], "color": "#183d30"},
                ],
            },
        )
    )
    fig.update_layout(height=320)
    return _apply_dark_layout(fig)
