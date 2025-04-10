import streamlit as st
import pandas as pd
import os

# === App config ===
st.set_page_config(page_title="MLB Model Predictions", layout="wide")
st.title("⚾ MLB Win Probability Dashboard")

# === Folder for archived predictions
PREDICTION_DIR = "predictions"

# === Load prediction files
if not os.path.exists(PREDICTION_DIR):
    st.warning("⚠️ 'predictions/' folder not found. Please generate predictions first.")
    st.stop()

# === Get all dated prediction files
csv_files = sorted([
    f for f in os.listdir(PREDICTION_DIR)
    if f.startswith("predictions_") and f.endswith(".csv")
])

available_dates = [f.replace("predictions_", "").replace(".csv", "") for f in csv_files]

if not available_dates:
    st.warning("⚠️ No prediction files found in 'predictions/'")
    st.stop()

# === Date dropdown
selected_date = st.selectbox("Select a prediction date", available_dates, index=len(available_dates) - 1)
selected_csv = os.path.join(PREDICTION_DIR, f"predictions_{selected_date}.csv")

# === Load selected file
try:
    df = pd.read_csv(selected_csv)
except FileNotFoundError:
    st.error(f"❌ Could not load {selected_csv}")
    st.stop()

# === Format output
df['home_win_prob'] = df['home_win_prob'].astype(float).round(3)
df = df.sort_values(by='home_win_prob', ascending=False)

# === Sidebar team filter
teams = sorted(set(df['home_team']) | set(df['away_team']))
team_filter = st.sidebar.multiselect("Filter by Team", teams)

if team_filter:
    df = df[df['home_team'].isin(team_filter) | df['away_team'].isin(team_filter)]

# === Display results
st.dataframe(df.reset_index(drop=True), use_container_width=True)

# === Sidebar metrics
avg_conf = df['confidence'].apply(lambda x: len(x)).mean()
st.sidebar.metric("Average Confidence 🔥", f"{avg_conf:.2f} / 5")

st.success(f"✅ Showing {len(df)} predictions from {selected_date} with confidence scoring.")
