import streamlit as st
import pandas as pd
import os

# === Page setup
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
    df['predicted_margin'] = df['predicted_margin'].round(4)
    df['predicted_total_runs'] = df['predicted_total_runs'].round(4)

    df = df.sort_values(by='home_win_prob', ascending=False)

    teams = sorted(set(df['home_team']) | set(df['away_team']))
    team_filter = st.sidebar.multiselect("Filter by Team", teams)

    if team_filter:
        df = df[df['home_team'].isin(team_filter) | df['away_team'].isin(team_filter)]

    st.dataframe(df.reset_index(drop=True), use_container_width=True)

    avg_conf = df['confidence'].apply(lambda x: len(str(x))).mean()
    st.sidebar.metric("Avg Confidence", f"{avg_conf:.2f} / 5")

    st.success(f"Showing {len(df)} predictions from {selected_date}")

# === TAB 2: Prediction Results Log
with tab2:
    RESULTS_LOG = "prediction_results.csv"

    if not os.path.exists(RESULTS_LOG):
        st.warning("⚠️ prediction_results.csv not found.")
        st.stop()

    results_df = pd.read_csv(RESULTS_LOG)
    results_df['home_win_prob'] = results_df['home_win_prob'].astype(float).round(3)

    all_dates = sorted(results_df['date'].unique(), reverse=True)
    selected_log_date = st.selectbox("Select a log date", all_dates)

    df_filtered = results_df[results_df['date'] == selected_log_date].copy()

    # Map 'home'/'away' to team names
    df_filtered["predicted_winner"] = df_filtered.apply(
        lambda row: row["home_team"] if row["predicted_winner"] == "home" else row["away_team"], axis=1)
    df_filtered["actual_winner"] = df_filtered.apply(
        lambda row: row["home_team"] if row["actual_winner"] == "home" else row["away_team"], axis=1)

    # Convert boolean to text
    df_filtered["is_correct"] = df_filtered["is_correct"].apply(lambda x: "Correct" if x else "Wrong")

    # 🔥 Confidence filter
    min_conf = st.slider("Minimum Confidence Level (1–5)", 1, 5, 1)
    df_filtered["conf_score"] = df_filtered["confidence"].apply(lambda x: len(str(x)))
    df_filtered = df_filtered[df_filtered["conf_score"] >= min_conf]

    # Clean up edge columns (remove emojis if they exist)
    if "run_line_edge" in df_filtered.columns:
        df_filtered["run_line_edge"] = df_filtered["run_line_edge"].astype(str).str.replace("✅ ", "").str.replace("❌ ", "")
    if "ou_edge" in df_filtered.columns:
        df_filtered["ou_edge"] = df_filtered["ou_edge"].astype(str).str.replace("✅ ", "").str.replace("❌ ", "")

    df_display = df_filtered.drop(columns=["conf_score"])
    st.write(f"Showing {len(df_display)} predictions from {selected_log_date} with confidence ≥ {min_conf}")
    st.dataframe(df_display.reset_index(drop=True), use_container_width=True)

    # === Footer Summary ===
    st.markdown("---")
    st.subheader("📊 Model Summary")

    wins_today = (df_filtered["is_correct"] == "Correct").sum()
    losses_today = (df_filtered["is_correct"] == "Wrong").sum()

    season_df = results_df.copy()
    season_df = season_df[season_df["actual_winner"].isin(["home", "away"])]
    total_season = len(season_df)
    correct_season = season_df["is_correct"].apply(lambda x: x == True or x == "Correct").sum()
    incorrect_season = total_season - correct_season

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {selected_log_date} Summary")
        st.markdown(f"✅ Wins: **{wins_today}**")
        st.markdown(f"❌ Losses: **{losses_today}**")
    with col2:
        st.markdown("### Season Summary")
        st.markdown(f"🏟️ Total Picks: **{total_season}**")
        st.markdown(f"✅ Wins: **{correct_season}**")
        st.markdown(f"❌ Losses: **{incorrect_season}**")
