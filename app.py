"""
IPL Win Prediction — Streamlit Dashboard
Loads the model trained in IPL_Win_Prediction.ipynb and serves live
pre-match win-probability estimates.

Run locally with:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

TEAM_RENAME = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}


@st.cache_resource
def load_artifacts():
    bundle = joblib.load("ipl_win_predictor.pkl")
    # Handle both layouts: data/matches.csv (intended) or matches.csv (if uploaded flat)
    matches_path = "data/matches.csv" if os.path.exists("data/matches.csv") else "matches.csv"
    matches = pd.read_csv(matches_path)
    # deliveries not needed at inference time — kept in data/deliveries.csv.gz for reference
    for col in ["team1", "team2", "toss_winner", "winner"]:
        matches[col] = matches[col].replace(TEAM_RENAME)
    return bundle, matches


bundle, matches = load_artifacts()
model = bundle["model"]
le_team = bundle["le_team"]
le_venue = bundle["le_venue"]
le_toss = bundle["le_toss"]
feature_cols = bundle["feature_cols"]

st.title("🏏 IPL Pre-Match Win Predictor")
st.caption(
    "Trained on 1,095 IPL matches (2008–2024) with an ETL + feature-engineering "
    "pipeline (team win ratio, venue advantage, head-to-head, toss impact, recent "
    "form, batting strength) feeding an XGBoost classifier."
)

teams = sorted(le_team.classes_)
venues = sorted(le_venue.classes_)

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", teams, index=teams.index("Chennai Super Kings") if "Chennai Super Kings" in teams else 0)
with col2:
    team2 = st.selectbox("Team 2", [t for t in teams if t != team1])

venue = st.selectbox("Venue", venues)
toss_winner = st.radio("Toss winner", [team1, team2], horizontal=True)
toss_decision = st.radio("Toss decision", ["bat", "field"], horizontal=True)


def team_stat(team, col_win, col_played=None):
    played = ((matches["team1"] == team) | (matches["team2"] == team)).sum()
    wins = (matches["winner"] == team).sum()
    return wins / played if played > 0 else 0.5


def venue_stat(team, venue):
    sub = matches[((matches["team1"] == team) | (matches["team2"] == team)) & (matches["venue"] == venue)]
    if len(sub) == 0:
        return 0.5
    return (sub["winner"] == team).sum() / len(sub)


def h2h_stat(t1, t2):
    sub = matches[((matches["team1"] == t1) & (matches["team2"] == t2)) | ((matches["team1"] == t2) & (matches["team2"] == t1))]
    if len(sub) == 0:
        return 0.5
    return (sub["winner"] == t1).sum() / len(sub)


def recent_form(team, n=10):
    sub = matches[(matches["team1"] == team) | (matches["team2"] == team)].sort_values("date").tail(n)
    if len(sub) == 0:
        return 0.5
    return (sub["winner"] == team).mean()


if st.button("Predict win probability", type="primary"):
    row = {
        "team1_enc": le_team.transform([team1])[0],
        "team2_enc": le_team.transform([team2])[0],
        "venue_enc": le_venue.transform([venue])[0],
        "toss_winner_is_team1": int(toss_winner == team1),
        "toss_decision_enc": le_toss.transform([toss_decision])[0],
        "team1_win_ratio": team_stat(team1, "winner"),
        "team2_win_ratio": team_stat(team2, "winner"),
        "team1_venue_ratio": venue_stat(team1, venue),
        "team2_venue_ratio": venue_stat(team2, venue),
        "team1_h2h_ratio": h2h_stat(team1, team2),
        "toss_convert_rate": (matches["toss_winner"] == matches["winner"]).mean(),
        "team1_recent_form": recent_form(team1),
        "team2_recent_form": recent_form(team2),
    }
    row["form_diff"] = row["team1_recent_form"] - row["team2_recent_form"]
    row["win_ratio_diff"] = row["team1_win_ratio"] - row["team2_win_ratio"]
    row["matches_played_diff"] = 0
    row["team1_avg_runs"] = 150.0
    row["team2_avg_runs"] = 150.0
    row["avg_runs_diff"] = 0.0

    X_input = pd.DataFrame([row])[feature_cols]
    proba_team1 = model.predict_proba(X_input)[0, 1]

    fig = go.Figure(
        go.Bar(
            x=[proba_team1 * 100, (1 - proba_team1) * 100],
            y=[team1, team2],
            orientation="h",
            marker_color=["#1f77b4", "#ff7f0e"],
            text=[f"{proba_team1*100:.1f}%", f"{(1-proba_team1)*100:.1f}%"],
            textposition="auto",
        )
    )
    fig.update_layout(title="Win Probability", xaxis_title="Probability (%)", xaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "⚠️ **Read this honestly:** cross-validated accuracy on held-out matches is "
        "~53%, only slightly above the baseline of simply predicting the toss winner "
        "(~51%). T20 cricket outcomes are close to a coin flip when predicted from "
        "team-level stats alone — this dashboard shows a probabilistic lean, not a "
        "confident forecast. Player-level and pitch-report data would meaningfully "
        "improve this."
    )

st.divider()
st.caption("Data: 2008–2024 IPL seasons · Model: XGBoost · Built with pandas, scikit-learn, XGBoost, Streamlit, Plotly")
