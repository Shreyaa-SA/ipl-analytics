"""
IPL Win Predictor — Stadium Edition
Loads the model trained in IPL_Win_Prediction.ipynb and serves live
pre-match win-probability estimates, an all-time Champions Timeline,
and a Team Stats leaderboard — all computed live from matches.csv.

Run locally with: streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

TEAM_RENAME = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Rough brand colors so the UI feels like the real franchises, not generic Plotly defaults
TEAM_COLORS = {
    "Chennai Super Kings": "#F9CD05",
    "Mumbai Indians": "#045093",
    "Royal Challengers Bengaluru": "#DA1818",
    "Kolkata Knight Riders": "#3A225D",
    "Delhi Capitals": "#17479E",
    "Punjab Kings": "#D71920",
    "Rajasthan Royals": "#EA1C8D",
    "Sunrisers Hyderabad": "#F26522",
    "Gujarat Titans": "#1C1C1C",
    "Lucknow Super Giants": "#00AEEF",
    "Deccan Chargers": "#4B2E83",
    "Pune Warriors": "#5A2E86",
    "Rising Pune Supergiant": "#7B2D8B",
    "Kochi Tuskers Kerala": "#F58220",
    "Gujarat Lions": "#E9762B",
}

TEAM_EMOJI = {
    "Chennai Super Kings": "🦁", "Mumbai Indians": "🔵", "Royal Challengers Bengaluru": "🔴",
    "Kolkata Knight Riders": "🟣", "Delhi Capitals": "🔷", "Punjab Kings": "⚔️",
    "Rajasthan Royals": "💗", "Sunrisers Hyderabad": "🟠", "Gujarat Titans": "⚫",
    "Lucknow Super Giants": "🩵",
}

# ── Theming ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Outfit', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #1a2f4b 0%, #0b1220 45%, #06090f 100%);
    }

    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(255,138,0,0.15), rgba(0,120,255,0.12));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 18px;
    }
    .hero h1 { font-weight: 800; font-size: 2.3rem; margin: 0; color: #fff; }
    .hero p { color: #b9c4d6; margin-top: 6px; font-size: 0.95rem; }

    .stat-card {
        border-radius: 16px;
        padding: 18px 20px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .stat-card .big { font-size: 1.8rem; font-weight: 800; color: #fff; }
    .stat-card .small { color: #9fb0c9; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }

    .champ-row {
        display: flex; align-items: center; gap: 14px;
        padding: 10px 16px; border-radius: 12px;
        background: rgba(255,255,255,0.03);
        border-left: 4px solid var(--accent, #F9CD05);
        margin-bottom: 8px;
    }
    .champ-year { font-weight: 800; color: #fff; width: 60px; }
    .champ-team { font-weight: 600; color: #eaeef5; flex: 1; }
    .champ-venue { color: #8194ad; font-size: 0.82rem; }

    div[data-testid="stMetricValue"] { color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    bundle = joblib.load("ipl_win_predictor.pkl")
    matches_path = "data/matches.csv" if os.path.exists("data/matches.csv") else "matches.csv"
    matches = pd.read_csv(matches_path)
    for col in ["team1", "team2", "toss_winner", "winner"]:
        if col in matches.columns:
            matches[col] = matches[col].replace(TEAM_RENAME)
    return bundle, matches


bundle, matches = load_artifacts()
model = bundle["model"]
le_team = bundle["le_team"]
le_venue = bundle["le_venue"]
le_toss = bundle["le_toss"]
feature_cols = bundle["feature_cols"]

teams = sorted(le_team.classes_)
venues = sorted(le_venue.classes_)


def team_color(t):
    return TEAM_COLORS.get(t, "#4C9AFF")


def team_stat(team, *_):
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


def h2h_matches(t1, t2):
    """Every past meeting between t1 and t2, oldest first — used to build the year-by-year view."""
    sub = matches[((matches["team1"] == t1) & (matches["team2"] == t2)) | ((matches["team1"] == t2) & (matches["team2"] == t1))].copy()
    if sub.empty:
        return sub
    if "season" in sub.columns:
        sub["season_year"] = sub["season"].astype(str).str.extract(r"(\d{4})")
    elif "date" in sub.columns:
        sub["season_year"] = pd.to_datetime(sub["date"], errors="coerce").dt.year.astype("Int64").astype(str)
    else:
        sub["season_year"] = "?"
    sort_col = "date" if "date" in sub.columns else "season_year"
    sub = sub.sort_values(sort_col)
    cols = [c for c in ["season_year", "date", "venue", "winner"] if c in sub.columns]
    return sub[cols]


def recent_form(team, n=10):
    sub = matches[(matches["team1"] == team) | (matches["team2"] == team)].sort_values("date").tail(n)
    if len(sub) == 0:
        return 0.5
    return (sub["winner"] == team).mean()


@st.cache_data
def get_champions():
    """Pull the Final of each season straight from matches.csv — no hardcoded list."""
    if "match_type" not in matches.columns or "season" not in matches.columns:
        return pd.DataFrame()
    finals = matches[matches["match_type"].astype(str).str.strip().str.lower() == "final"].copy()
    if finals.empty:
        return pd.DataFrame()
    finals["season_year"] = finals["season"].astype(str).str.extract(r"(\d{4})").astype(float)
    finals = finals.dropna(subset=["season_year"])
    finals["season_year"] = finals["season_year"].astype(int)
    finals = finals.sort_values("season_year").drop_duplicates("season_year", keep="last")
    cols = [c for c in ["season_year", "winner", "team1", "team2", "venue", "city"] if c in finals.columns]
    return finals[cols].reset_index(drop=True)


# ── Hero ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🏏 IPL Pre-Match Win Predictor</h1>
        <p>XGBoost model trained on 1,095 IPL matches (2008–2024) · ETL + feature-engineering
        pipeline (team win ratio, venue advantage, head-to-head, toss impact, recent form,
        batting strength).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_champs, tab_stats = st.tabs(["🎯 Predict a Match", "🏆 Champions Timeline", "📊 Team Leaderboard"])

# ── Tab 1: Predictor ────────────────────────────────────────────────────
with tab_predict:
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=teams.index("Chennai Super Kings") if "Chennai Super Kings" in teams else 0)
    with col2:
        team2 = st.selectbox("Team 2", [t for t in teams if t != team1])

    venue = st.selectbox("Venue", venues)
    toss_winner = st.radio("Toss winner", [team1, team2], horizontal=True)
    toss_decision = st.radio("Toss decision", ["bat", "field"], horizontal=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="stat-card"><div class="big">{team_stat(team1)*100:.0f}%</div><div class="small">{team1} career win rate</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><div class="big">{h2h_stat(team1, team2)*100:.0f}%</div><div class="small">{team1} h2h vs {team2}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card"><div class="big">{team_stat(team2)*100:.0f}%</div><div class="small">{team2} career win rate</div></div>', unsafe_allow_html=True)

    st.write("")

    # ── Year-by-year head-to-head between the two selected teams ───────
    h2h_df = h2h_matches(team1, team2)
    if h2h_df.empty:
        st.caption(f"📅 {team1} and {team2} haven't played each other yet in this dataset.")
    else:
        with st.expander(f"📅 {team1} vs {team2} — year-by-year history ({len(h2h_df)} matches)", expanded=True):
            display_df = h2h_df.rename(columns={
                "season_year": "Year", "date": "Date", "venue": "Venue", "winner": "Winner",
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            yearly = h2h_df.copy()
            yearly["team1_won"] = (yearly["winner"] == team1).astype(int)
            yearly_pct = yearly.groupby("season_year")["team1_won"].mean().reset_index()
            yearly_pct["win_pct"] = yearly_pct["team1_won"] * 100

            fig_h2h = go.Figure(
                go.Bar(
                    x=yearly_pct["season_year"],
                    y=yearly_pct["win_pct"],
                    marker_color=team_color(team1),
                    text=[f"{v:.0f}%" for v in yearly_pct["win_pct"]],
                    textposition="auto",
                )
            )
            fig_h2h.update_layout(
                title=f"{team1} win % vs {team2}, by year",
                yaxis_title="Win %", yaxis_range=[0, 100], xaxis_title="Year",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaeef5", height=280,
            )
            st.plotly_chart(fig_h2h, use_container_width=True)

    if st.button("🔮 Predict win probability", type="primary", use_container_width=True):
        row = {
            "team1_enc": le_team.transform([team1])[0],
            "team2_enc": le_team.transform([team2])[0],
            "venue_enc": le_venue.transform([venue])[0],
            "toss_winner_is_team1": int(toss_winner == team1),
            "toss_decision_enc": le_toss.transform([toss_decision])[0],
            "team1_win_ratio": team_stat(team1),
            "team2_win_ratio": team_stat(team2),
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
        favored = team1 if proba_team1 >= 0.5 else team2

        fig = go.Figure(
            go.Bar(
                x=[proba_team1 * 100, (1 - proba_team1) * 100],
                y=[team1, team2],
                orientation="h",
                marker_color=[team_color(team1), team_color(team2)],
                text=[f"{proba_team1*100:.1f}%", f"{(1-proba_team1)*100:.1f}%"],
                textposition="auto",
            )
        )
        fig.update_layout(
            title=f"{TEAM_EMOJI.get(favored, '🏆')} Edge: {favored}",
            xaxis_title="Win Probability (%)", xaxis_range=[0, 100],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaeef5", height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "⚠️ **Read this honestly:** cross-validated accuracy on held-out matches is "
            "~53%, only slightly above the baseline of simply predicting the toss winner "
            "(~51%). T20 cricket outcomes are close to a coin flip when predicted from "
            "team-level stats alone — this dashboard shows a probabilistic lean, not a "
            "confident forecast. Player-level and pitch-report data would meaningfully "
            "improve this."
        )

# ── Tab 2: Champions Timeline ───────────────────────────────────────────
with tab_champs:
    champs = get_champions()
    if champs.empty:
        st.warning("No `match_type == 'Final'` rows found in matches.csv, so the timeline can't be built from this dataset.")
    else:
        st.subheader(f"🏆 {champs['season_year'].min()}–{champs['season_year'].max()} · IPL Champions")
        for _, row in champs.sort_values("season_year", ascending=False).iterrows():
            winner = row["winner"]
            venue_txt = f'{row.get("venue", "")}, {row.get("city", "")}'.strip(", ")
            st.markdown(
                f"""
                <div class="champ-row" style="--accent:{team_color(winner)};">
                    <div class="champ-year">{int(row['season_year'])}</div>
                    <div class="champ-team">{TEAM_EMOJI.get(winner, '🏆')} {winner}</div>
                    <div class="champ-venue">{venue_txt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.subheader("Most Titles")
        title_counts = champs["winner"].value_counts()
        fig2 = go.Figure(
            go.Bar(
                x=title_counts.values,
                y=title_counts.index,
                orientation="h",
                marker_color=[team_color(t) for t in title_counts.index],
                text=title_counts.values,
                textposition="auto",
            )
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaeef5", xaxis_title="Titles won", yaxis=dict(autorange="reversed"),
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Team Leaderboard ─────────────────────────────────────────────
with tab_stats:
    st.subheader("📊 All-Time Team Leaderboard")
    rows = []
    for t in teams:
        played = int(((matches["team1"] == t) | (matches["team2"] == t)).sum())
        wins = int((matches["winner"] == t).sum())
        rows.append({
            "Team": f"{TEAM_EMOJI.get(t, '🏏')} {t}",
            "Matches": played,
            "Wins": wins,
            "Win %": round(wins / played * 100, 1) if played else 0.0,
        })
    board = pd.DataFrame(rows).sort_values("Win %", ascending=False).reset_index(drop=True)
    board.index = board.index + 1
    st.dataframe(board, use_container_width=True)

    fig3 = go.Figure(
        go.Bar(
            x=board["Win %"],
            y=board["Team"],
            orientation="h",
            marker_color=[team_color(t.split(" ", 1)[-1]) for t in board["Team"]],
            text=[f"{v}%" for v in board["Win %"]],
            textposition="auto",
        )
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#eaeef5", xaxis_title="Win %", yaxis=dict(autorange="reversed"),
        height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.caption("Data: 2008–2024 IPL seasons · Model: XGBoost · Built with pandas, scikit-learn, XGBoost, Streamlit, Plotly")
