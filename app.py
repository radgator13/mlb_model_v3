﻿import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="MLB Model Dashboard", layout="wide")

st.title("⚾ MLB Win Probability Dashboard")

tab1, tab2 = st.tabs(["📊 Predictions", "✅ Model Results Log"])

# === TAB 1: Daily Predictions
with tab1:
    PREDICTION_DIR = "predictions"

    if not os.path.exists(PREDICTION_DIR):
        st.warning("⚠️ 'predictions/' folder not found.")
        st.stop()

    csv_files = sorted([
        f for f in os.listdir(PREDICTION_DIR)
        if f.startswith("predictions_") and f.endswith(".csv")
    ])

    available_dates = [f.replace("predictions_", "").replace(".csv", "") for f in csv_files]

    if not available_dates:
        st.warning("⚠️ No prediction files found in 'predictions/'")
        st.stop()

    selected_date = st.selectbox("Select a prediction date", available_dates, index=len(available_dates) - 1)
    selected_csv = os.path.join(PREDICTION_DIR, f"predictions_{selected_date}.csv")

    try:
        df = pd.read_csv(selected_csv)
    except FileNotFoundError:
        st.error(f"❌ Could not load {selected_csv}")
        st.stop()

    df['home_win_prob'] = df['home_win_prob'].astype(float).round(3)
    df = df.sort_values(by='home_win_prob', ascending=False)

    teams = sorted(set(df['home_team']) | set(df['away_team']))
    team_filter = st.sidebar.multiselect("Filter by Team", teams)

    if team_filter:
        df = df[df['home_team'].isin(team_filter) | df['away_team'].isin(team_filter)]

    st.dataframe(df.reset_index(drop=True), use_container_width=True)

    avg_conf = df['confidence'].apply(lambda x: len(x)).mean()
    st.sidebar.metric("🔥 Avg Confidence", f"{avg_conf:.2f} / 5")

    st.success(f"✅ Showing {len(df)} predictions from {selected_date}")

# === TAB 2: Prediction Results Log
with tab2:
    RESULTS_LOG = "prediction_results.csv"

    if not os.path.exists(RESULTS_LOG):
        st.warning("⚠️ prediction_results.csv not found.")
        st.stop()

    results_df = pd.read_csv(RESULTS_LOG)
    results_df['home_win_prob'] = results_df['home_win_prob'].astype(float).round(3)

    # Summary stats
    total = len(results_df)
    correct = results_df['is_correct'].sum()
    accuracy = correct / total if total else 0

    st.metric("📈 Overall Accuracy", f"{accuracy:.2%}")
    st.metric("🧠 Total Predictions Logged", total)

    # Filter by confidence?
    min_conf = st.slider("Minimum Confidence 🔥 Level (1-5)", 1, 5, 1)
    results_df['conf_score'] = results_df['confidence'].apply(lambda x: len(str(x)))
    filtered = results_df[results_df['conf_score'] >= min_conf]

    st.write(f"✅ Showing {len(filtered)} predictions with confidence ≥ {min_conf}")
    st.dataframe(filtered.drop(columns=['conf_score']).reset_index(drop=True), use_container_width=True)
