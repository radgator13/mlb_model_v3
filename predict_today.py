import pandas as pd
import numpy as np
import datetime
import os
import joblib
import subprocess

# === Config
DATE_TODAY = datetime.date.today().isoformat()

BOX_PATH = "pending_boxscores.csv"
PITCHING_PATH = "mlb_pitching_stats_2025.csv"
BATTING_PATH = "mlb_batting_stats_2025.csv"
STRENGTH_PATH = "team_strengths.csv"
ROLLING_STATS_PATH = "team_rolling_stats_2025.csv"

WIN_MODEL = "mlb_win_model.pkl"
MARGIN_MODEL = "mlb_margin_model.pkl"
TOTAL_MODEL = "mlb_total_model.pkl"

OUTPUT_CSV = "predictions_latest.csv"
ARCHIVE_CSV = f"predictions/predictions_{DATE_TODAY}.csv"
ODDS_PATH = f"odds_{DATE_TODAY}.csv"

# === Load helpers
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def ensure_boxscore_data_for_today():
    if not os.path.exists(BOX_PATH):
        subprocess.run(["python", "build_pending_boxscores.py"])
    else:
        df = pd.read_csv(BOX_PATH)
        if DATE_TODAY not in df['date'].astype(str).values:
            subprocess.run(["python", "build_pending_boxscores.py"])

ensure_boxscore_data_for_today()

# === Load data
games = load_csv(BOX_PATH)
pitching = load_csv(PITCHING_PATH)
batting = load_csv(BATTING_PATH)
rolling = load_csv(ROLLING_STATS_PATH)
strengths_df = load_csv(STRENGTH_PATH)
team_strengths = dict(zip(strengths_df['team'], strengths_df['strength']))

# === Load models
win_model = joblib.load(WIN_MODEL)
margin_model = joblib.load(MARGIN_MODEL)
total_model = joblib.load(TOTAL_MODEL)

# === Filter today's games
games['date'] = pd.to_datetime(games['date']).dt.date.astype(str)
todays_games = games[games['date'] == DATE_TODAY].copy()
if todays_games.empty:
    print(f"No games found for today: {DATE_TODAY}")
    exit()

# === Feature engineering
def build_features(row):
    gamePk = row['gamePk']

    def stat(df, side, col):
        df_sub = df[(df['gamePk'] == gamePk) & (df['team_side'] == side)]
        return df_sub[col].mean() if not df_sub.empty and col in df_sub else np.nan

    def rolling_stat(side, col):
        df_sub = rolling[(rolling['gamePk'] == gamePk) & (rolling['team_side'] == side)]
        return df_sub[col].values[0] if not df_sub.empty else np.nan

    return pd.Series({
        'home_pitcher_era': stat(pitching, 'home', 'era'),
        'away_pitcher_era': stat(pitching, 'away', 'era'),
        'home_strikeouts': stat(pitching, 'home', 'strikeouts'),
        'away_strikeouts': stat(pitching, 'away', 'strikeouts'),
        'home_hits': stat(batting, 'home', 'hits'),
        'away_hits': stat(batting, 'away', 'hits'),
        'home_strength': team_strengths.get(row['home_team'], 70),
        'away_strength': team_strengths.get(row['away_team'], 70),
        'home_avg_runs_last5': rolling_stat('home', 'runs_scored_last5'),
        'away_avg_runs_last5': rolling_stat('away', 'runs_scored_last5'),
        'home_avg_runs_allowed_last5': rolling_stat('home', 'runs_allowed_last5'),
        'away_avg_runs_allowed_last5': rolling_stat('away', 'runs_allowed_last5'),
        'home_avg_hits_last5': rolling_stat('home', 'hits_last5'),
        'away_avg_hits_last5': rolling_stat('away', 'hits_last5'),
    })

features_df = todays_games.apply(build_features, axis=1)
features_df['home_team'] = todays_games['home_team']
features_df['away_team'] = todays_games['away_team']
features_df = pd.get_dummies(features_df, columns=['home_team', 'away_team'])

# === Align to model
def align(df, model):
    missing = [col for col in model.get_booster().feature_names if col not in df.columns]
    for col in missing:
        df[col] = 0
    return df[model.get_booster().feature_names]

X = align(features_df.copy(), win_model)

# === Predict
todays_games['home_win_prob'] = win_model.predict_proba(X)[:, 1]
todays_games['predicted_margin'] = margin_model.predict(X)
todays_games['predicted_total_runs'] = total_model.predict(X)

# === Confidence 🔥
def confidence(prob):
    delta = abs(prob - 0.5)
    if delta >= 0.4: return "🔥🔥🔥🔥🔥"
    elif delta >= 0.3: return "🔥🔥🔥🔥"
    elif delta >= 0.2: return "🔥🔥🔥"
    elif delta >= 0.1: return "🔥🔥"
    else: return "🔥"

todays_games['confidence'] = todays_games['home_win_prob'].apply(confidence)

# === ALWAYS Overwrite odds file from model
print(f"🔁 Overwriting odds file using model predictions: {ODDS_PATH}")
odds_df = todays_games[['gamePk', 'home_team', 'away_team']].copy()
odds_df['run_line'] = -1.5
odds_df['ou_line'] = todays_games['predicted_total_runs'].round(1).apply(lambda x: round(x * 2) / 2)
odds_df.to_csv(ODDS_PATH, index=False)
print(f"✅ Created updated odds file: {ODDS_PATH}")

# === Merge odds into game data
odds_df = odds_df[['gamePk', 'run_line', 'ou_line']]
todays_games = todays_games.merge(odds_df, on='gamePk', how='left')

# === Edge logic
todays_games['run_line_edge'] = todays_games.apply(
    lambda row: "✅ Cover" if row['predicted_margin'] > row['run_line'] else "❌ No cover", axis=1
)

todays_games['ou_edge'] = todays_games.apply(
    lambda row: "✅ Over" if row['predicted_total_runs'] > row['ou_line'] else "❌ Under", axis=1
)

# === Save
cols = [
    'date', 'gamePk', 'home_team', 'away_team',
    'home_win_prob', 'confidence',
    'predicted_margin', 'run_line', 'run_line_edge',
    'predicted_total_runs', 'ou_line', 'ou_edge'
]

todays_games[cols].to_csv(OUTPUT_CSV, index=False)
os.makedirs("predictions", exist_ok=True)
todays_games[cols].to_csv(ARCHIVE_CSV, index=False)

print(f"✅ Saved predictions to:\n- {OUTPUT_CSV}\n- {ARCHIVE_CSV}")
print(todays_games[cols])
