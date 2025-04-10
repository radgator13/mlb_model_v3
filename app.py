import streamlit as st
import pandas as pd
import datetime
import os

st.set_page_config(page_title="MLB Model Predictions", layout="wide")
st.title("⚾ MLB Win Probability Dashboard")

# === Get list of prediction files
csv_files = sorted([f for f in os.listdir() if f.startswith("predictions_") and f.endswith(".csv")])
available_dates = [f.replace("predictions_", "").replace(".csv", "") for f in csv_files]

if not available_dates:
    st.warning("⚠️ No prediction files found.")
    st.stop()

# === Date selector (dropdown)
selected_date = st.selectbox("Select a prediction date", available_dates, index=len(available_dates)-1)

# === Load selected file
selected_csv = f"predictions_{selected_date}.csv"

try:
    df = pd.read_csv(selected_csv)
except FileNotFoundError:
    st.error(f"❌ Could not load {selected_csv}")
    st.stop()

# === Format and display
df['home_win_prob'] = df['home_win_prob'].apply(lambda x: f"{float(x):.3f}")
df = df.sort_values(by='home_win_prob', ascending=False)

teams = sorted(set(df['home_team']) | set(df['away_team']))
team_filter = st.sidebar.multiselect("Filter by Team", teams)

if team_filter:
    df = df[df['home_team'].isin(team_filter) | df['away_team'].isin(team_filter)]

st.dataframe(df.reset_index(drop=True), use_container_width=True)

# === Sidebar metrics
avg_conf = df['confidence'].apply(lambda x: len(x)).mean()
st.sidebar.metric("Average Confidence 🔥", f"{avg_conf:.2f} / 5")

st.success(f"✅ Loaded {len(df)} predictions from {selected_date}")
