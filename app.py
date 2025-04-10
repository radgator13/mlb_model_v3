import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="MLB Model Predictions", layout="wide")
st.title("⚾ MLB Win Probability Dashboard")

# === Load latest prediction
csv_path = "predictions_latest.csv"

if not os.path.exists(csv_path):
    st.error(f"No prediction file found: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)

# === Format and filter
df['home_win_prob'] = df['home_win_prob'].astype(float).round(3)
df = df.sort_values(by='home_win_prob', ascending=False)

teams = sorted(set(df['home_team']) | set(df['away_team']))
team_filter = st.sidebar.multiselect("Filter by Team", teams)

if team_filter:
    df = df[df['home_team'].isin(team_filter) | df['away_team'].isin(team_filter)]

st.dataframe(df.reset_index(drop=True), use_container_width=True)

# === Sidebar metrics
avg_conf = df['confidence'].apply(lambda x: len(x)).mean()
st.sidebar.metric("Average Confidence 🔥", f"{avg_conf:.2f} / 5")

st.success(f"✅ Showing {len(df)} predictions with confidence scoring.")
