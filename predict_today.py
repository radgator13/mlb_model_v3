import pandas as pd
import numpy as np
import datetime
import os
import joblib
import subprocess

# === Today's date
DATE_TODAY = datetime.date.today().isoformat()

# === File paths
BOX_PATH = "pending_boxscores.csv"
PITCHING_PATH = "mlb_pitching_stats_2025.csv"
BATTING_PATH = "mlb_batting_stats_2025.csv"
STRENGTH_PATH = "team_strengths.csv"
ROLLING_STATS_PATH = "team_rolling_stats_2025.csv"
MODEL_PATH = "mlb_win_model.pkl"
OUTPUT_CSV_LATEST = "predictions_latest.csv"
OUTPUT_CSV_DATED = f"predictions_{DATE_TODAY}.csv"

# === Load CSVs
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

# === Ensure today's boxscore data exists
def ensure_boxscore_data_for_today():
    if not os.path.exists(BOX_PATH):
        print("⚠️ pending_boxscores.csv not found. Generating it...")
        subprocess.run(["python", "build_pending_boxscores.py"])
        return

    df = pd.read_csv(BOX_PATH)
    if DATE_TODAY not in df['date'].astype(str).values:
        print(f"⚠️ No games for today ({DATE_TODAY}) found. Rebuilding...")
        subprocess.run(["python", "build_pending_boxscores.py"])

ensure_boxscore_data_for_today()

# === Load data
games = load_csv(BOX_PATH)
pitching = load_csv(PITCHING_PATH)
batting = load_csv(BATTING_PATH)
rolling_stats = load_csv(ROLLING_STATS_PATH)
strengths_df = load_csv(STRENGTH_PATH)
team_strengths = dict(zip(strengths_df['team'], strengths_df['strength']))

# === Filter for today's games
games['date'] = pd.to_datetime(games['date']).dt.date.astype(str)
todays_games = games[games['date'] == DATE_TODAY].copy()
if todays_games.empty:
    print(f"No games found for today: {DATE_TODAY}")
    exit()

# === Load model
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    exit()
model = joblib.load(MODEL_PATH)

# === Feature builder
def build_features(row):
    gamePk = row['gamePk']

    def safe_stat(df, side, col):
        df_sub = df[(df['gamePk'] == gamePk) & (df['team_side'] == side)]
        return df_sub[col].mean() if not df_sub.empty and col in df_sub else np.nan

    def rolling_stat(side, col):
        df_sub = rolling_stats[(rolling_stats['gamePk'] == gamePk) & (rolling_stats['team_side'] == side)]
        return df_sub[col].values[0] if not df_sub.empty else np.nan

    return pd.Series({
        'home_pitcher_era': safe_stat(pitching, 'home', 'era'),
        'away_pitcher_era': safe_stat(pitching, 'away', 'era'),
        'home_strikeouts': safe_stat(pitching, 'home', 'strikeouts'),
        'away_strikeouts': safe_stat(pitching, 'away', 'strikeouts'),
        'home_hits': safe_stat(batting, 'home', 'hits'),
        'away_hits': safe_stat(batting, 'away', 'hits'),
        'home_runs': row.get('home_runs', np.nan),
        'away_runs': row.get('away_runs', np.nan),
        'home_strength': team_strengths.get(row['home_team'], 70),
        'away_strength': team_strengths.get(row['away_team'], 70),
        'home_avg_runs_last5': rolling_stat('home', 'runs_scored_last5'),
        'away_avg_runs_last5': rolling_stat('away', 'runs_scored_last5'),
        'home_avg_runs_allowed_last5': rolling_stat('home', 'runs_allowed_last5'),
        'away_avg_runs_allowed_last5': rolling_stat('away', 'runs_allowed_last5'),
        'home_avg_hits_last5': rolling_stat('home', 'hits_last5'),
        'away_avg_hits_last5': rolling_stat('away', 'hits_last5'),
    })

# === Apply features
features_df = todays_games.apply(build_features, axis=1)
features_df['home_team'] = todays_games['home_team']
features_df['away_team'] = todays_games['away_team']
features_df = pd.get_dummies(features_df, columns=['home_team', 'away_team'])

# === Align features with model
missing_cols = [col for col in model.get_booster().feature_names if col not in features_df.columns]
for col in missing_cols:
    features_df[col] = 0
features_df = features_df[model.get_booster().feature_names]

# === Make predictions
probs = model.predict_proba(features_df)[:, 1]
todays_games['home_win_prob'] = probs

# === Add confidence score
def add_confidence(prob):
    delta = abs(prob - 0.5)
    if delta >= 0.4:
        return "🔥🔥🔥🔥🔥"
    elif delta >= 0.3:
        return "🔥🔥🔥🔥"
    elif delta >= 0.2:
        return "🔥🔥🔥"
    elif delta >= 0.1:
        return "🔥🔥"
    else:
        return "🔥"

todays_games['confidence'] = todays_games['home_win_prob'].apply(add_confidence)

# === Output columns
cols = ['date', 'gamePk', 'home_team', 'away_team', 'home_win_prob', 'confidence']

# === Save both latest and dated files
todays_games[cols].to_csv(OUTPUT_CSV_LATEST, index=False)
todays_games[cols].to_csv(OUTPUT_CSV_DATED, index=False)

print(f"✅ Saved predictions to:\n- {OUTPUT_CSV_LATEST}\n- {OUTPUT_CSV_DATED}")
print(todays_games[cols])
# === Output columns
cols = ['date', 'gamePk', 'home_team', 'away_team', 'home_win_prob', 'confidence']

# === Save predictions
latest_path = "predictions_latest.csv"
archive_folder = "predictions"
dated_path = os.path.join(archive_folder, f"predictions_{DATE_TODAY}.csv")

# Create folder if it doesn't exist
os.makedirs(archive_folder, exist_ok=True)

# Save both
todays_games[cols].to_csv(latest_path, index=False)
todays_games[cols].to_csv(dated_path, index=False)

print(f"✅ Saved predictions to:")
print(f"  - {latest_path}")
print(f"  - {dated_path}")
