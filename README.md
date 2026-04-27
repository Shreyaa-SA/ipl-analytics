# 🏏 End-to-End IPL Analytics & Win Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-orange?logo=xgboost)
![SQL](https://img.shields.io/badge/SQL-ETL_Pipeline-lightgrey?logo=postgresql)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Visualizations-blueviolet?logo=plotly)

> **A full-stack cricket analytics system that predicts IPL match win probabilities using 15+ years of historical data, XGBoost ML models, and an interactive Streamlit dashboard.**

---

## 📌 Project Overview

This project builds an automated analytics pipeline over the entire IPL dataset (2008–2024), covering **100,000+ match deliveries**. It combines data engineering, feature engineering, machine learning, and interactive visualization into a single deployable application.

---

## 🎯 What It Does

- Ingests and processes **15+ years of IPL ball-by-ball data** via an automated ETL pipeline
- Engineers features like team win ratios, venue performance, and player statistics
- Trains an **XGBoost classifier** to estimate pre-match win probabilities for any team matchup
- Serves predictions through an **interactive Streamlit dashboard** with real-time Plotly charts

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Database & ETL | SQL, PostgreSQL / SQLite |
| Machine Learning | XGBoost, Scikit-learn |
| Dashboard | Streamlit, Plotly |
| Version Control | Git, GitHub |

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Dataset size | 100,000+ deliveries (2008–2024) |
| Features engineered | 15+ (win ratio, venue, player stats, toss, etc.) |
| Model | XGBoost Classifier |
| Validation approach | Chronological train/test split |

---

## 🔍 Feature Engineering Highlights

- **Team win ratio** — rolling win % per team per season
- **Head-to-head record** — historical matchup outcomes between two teams
- **Venue advantage** — home/neutral ground win rates per team
- **Player impact score** — top batsmen and bowlers weighted contribution
- **Toss decision effect** — win correlation with bat/field choice by venue

---

## 👩‍💻 Author

**Shreya Wargantiwar**
📧 shreyawargantiwar27@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/shreyawargantiwar)
🐙 [GitHub](https://github.com/Shreyaa-SA)
📍 Pune, India

---

*Built as part of MCA Data Science program, Sri Balaji University, Pune — 2026*
