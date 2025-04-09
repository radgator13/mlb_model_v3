import streamlit as st
import pandas as pd
import datetime

# === Load today's prediction file ===
DATE_TODAY = datetime.date.today().isoformat()
CSV_PATH = f"predictions_{DATE_TODAY}.csv"

st.set_page_config(page_title="MLB Model Predictions", layout="wide")
st.title(f"⚾ MLB Win Probabilities — {DATE_TODAY}")

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"No prediction file found: {CSV_PATH}")
    st.stop()

# === Format output ===
df['home_win_prob'] = df['home_win_prob'].apply(lambda x: f"{x:.3f}")
df = df.sort_values(by='home_win_prob', ascending=False)

# === Filter sidebar ===
teams = sorted(set(df['home_team']) | set(df['away_team']))
team_filter = st.sidebar.multiselect("Filter by Team", teams)

if team_filter:
    df = df[df['home_team'].isin(team_filter) | df['away_team'].isin(team_filter)]

# === Show table ===
st.dataframe(df.reset_index(drop=True), use_container_width=True)

# === Style summary ===
avg_confidence = df['confidence'].apply(lambda x: len(x)).mean()
st.sidebar.metric("Average 🔥 Score", f"{avg_confidence:.2f} / 5")

st.success(f"✅ Showing {len(df)} predictions with confidence scoring.")
