import pandas as pd
import numpy as np
import datetime
import os
import joblib

DATE_TODAY = datetime.date.today().isoformat()

# === File paths ===
if os.path.exists("game_boxscores.csv"):
    BOX_PATH = "game_boxscores.csv"
elif os.path.exists("pending_boxscores.csv"):
    BOX_PATH = "pending_boxscores.csv"
else:
    print("❌ No boxscore file found.")
    exit()

PITCHING_PATH = "mlb_pitching_stats_2025.csv"
BATTING_PATH = "mlb_batting_stats_2025.csv"
STRENGTH_PATH = "team_strengths.csv"
MODEL_PATH = "mlb_win_model.pkl"
OUTPUT_CSV = f"predictions_{DATE_TODAY}.csv"

# === Load CSVs ===
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

games = load_csv(BOX_PATH)
pitching = load_csv(PITCHING_PATH)
batting = load_csv(BATTING_PATH)
strengths_df = load_csv(STRENGTH_PATH)
team_strengths = dict(zip(strengths_df['team'], strengths_df['strength']))

# === Filter today's games
todays_games = games[games['date'] == DATE_TODAY].copy()
if todays_games.empty:
    print(f"No games found for today: {DATE_TODAY}")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    exit()

model = joblib.load(MODEL_PATH)

# === Feature engineering
def build_features(row):
    gamePk = row['gamePk']

    def safe_stat(df, side, col):
        df_sub = df[(df['gamePk'] == gamePk) & (df['team_side'] == side)]
        return df_sub[col].mean() if not df_sub.empty and col in df_sub else np.nan

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
    })

features_df = todays_games.apply(build_features, axis=1)
features_df['home_team'] = todays_games['home_team']
features_df['away_team'] = todays_games['away_team']

features_df = pd.get_dummies(features_df, columns=['home_team', 'away_team'])

# === Align columns to model
missing_cols = [col for col in model.get_booster().feature_names if col not in features_df.columns]
for col in missing_cols:
    features_df[col] = 0
features_df = features_df[model.get_booster().feature_names]

# === Predict
probs = model.predict_proba(features_df)[:, 1]
todays_games['home_win_prob'] = probs

# === Add confidence scores
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

# === Export
cols = ['date', 'gamePk', 'home_team', 'away_team', 'home_win_prob', 'confidence']
todays_games[cols].to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved predictions to {OUTPUT_CSV}")
print(todays_games[cols])
