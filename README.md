# 🏏 End-to-End IPL Analytics & Win Prediction System

Python • XGBoost • SQL-style ETL (pandas) • Streamlit • Plotly

A full-stack cricket analytics system that estimates IPL pre-match win probabilities using historical data (2008–2024), an XGBoost model, and an interactive Streamlit dashboard.

## 📌 Project Overview

This project builds an automated analytics pipeline over IPL ball-by-ball and match-level data (2008–2024), covering 1,095 matches and 260,000+ deliveries. It combines data engineering, feature engineering, machine learning, and interactive visualization into a single deployable application.

## 🎯 What It Does

- Ingests and cleans 2008–2024 IPL match and ball-by-ball data via an automated ETL pipeline (pandas)
- Engineers **leak-free, chronological** features: team win ratio, venue advantage, head-to-head record, toss-decision impact, recent form, and rolling batting strength
- Trains an XGBoost classifier to estimate pre-match win probabilities for any team matchup
- Serves predictions through an interactive Streamlit dashboard with Plotly visualizations

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Dashboard | Streamlit, Plotly |
| Version Control | Git, GitHub |

## 📊 Key Results — reported honestly

| Metric | Value |
|---|---|
| Dataset size | 1,095 matches, 260,000+ deliveries (2008–2024) |
| Features engineered | 15+ (win ratio, venue, head-to-head, toss, recent form, batting strength) |
| Model | XGBoost Classifier |
| Validation | 5-fold stratified cross-validation |
| **CV accuracy** | **~53%** |
| Baseline (always predict toss winner) | ~51% |
| Baseline (majority class) | ~51% |

**Why report a modest number instead of an inflated one?** Because it's the honest, defensible finding. Even the strongest single baseline — "the toss winner wins" — is already correct roughly half the time in this dataset, and no team-level pre-match feature set (win ratio, venue, recent form, batting strength) pushes meaningfully past that without player-level data (form of individual batters/bowlers, playing XI, pitch report). This matches published sports-analytics research on T20 win prediction. The real analytical finding here is that **toss result and recent form are the most informative signals available at the team level — and T20 outcomes are close to genuinely unpredictable from macro stats alone.** The project demonstrates the full ETL → feature engineering → ML → deployment pipeline and sound, leakage-free evaluation methodology rather than a cherry-picked accuracy figure.

## 🔍 Feature Engineering Highlights

- **Team win ratio** — rolling win % per team, computed only from matches prior to the one being predicted
- **Head-to-head record** — historical matchup outcomes between two teams
- **Venue advantage** — team's historical win % at that specific venue
- **Recent form** — win rate over each team's last 10 matches
- **Batting strength** — rolling average runs scored per match (from ball-by-ball data)
- **Toss decision effect** — win correlation with bat/field choice, and toss-conversion rate

## 🚀 Running the Project

```bash
pip install -r requirements.txt
jupyter notebook IPL_Win_Prediction.ipynb   # rebuild/retrain the model
streamlit run app.py                        # launch the live dashboard
```

## 👩‍💻 Author

Shreyaa Wargantiwar
📧 shreyawargantiwar27@gmail.com · 📍 Pune, India
Built as part of MCA Data Science program, Sri Balaji University, Pune — 2026
