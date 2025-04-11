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
features_df['gamePk'] = todays_games['gamePk']
features_df['home_team'] = todays_games['home_team']
features_df['away_team'] = todays_games['away_team']
features_df = pd.get_dummies(features_df, columns=['home_team', 'away_team'])

# === Debugging missing data
print("🔍 Feature DF shape (pre-dropna):", features_df.shape)
missing_values = features_df.isna().sum()
missing_summary = missing_values[missing_values > 0]
if not missing_summary.empty:
    print("⚠️ Missing values detected:")
    print(missing_summary)

# Save skipped rows
problematic = features_df[features_df.isna().any(axis=1)]
if not problematic.empty:
    problematic.to_csv("skipped_predictions_debug.csv", index=False)
    print(f"⚠️ Saved rows with missing data to: skipped_predictions_debug.csv")

# Drop incomplete rows before prediction
features_df = features_df.dropna()

# === Align to model
def align(df, model):
    missing = [col for col in model.get_booster().feature_names if col not in df.columns]
    for col in missing:
        df[col] = 0
    return df[model.get_booster().feature_names]

X = align(features_df.copy(), win_model)

# === Predict
preds_df = todays_games[todays_games['gamePk'].isin(features_df['gamePk'])].copy()
preds_df['home_win_prob'] = win_model.predict_proba(X)[:, 1]
preds_df['predicted_margin'] = margin_model.predict(X)
preds_df['predicted_total_runs'] = total_model.predict(X)

# === Confidence 🔥
def confidence(prob):
    delta = abs(prob - 0.5)
    if delta >= 0.4: return "🔥🔥🔥🔥🔥"
    elif delta >= 0.3: return "🔥🔥🔥🔥"
    elif delta >= 0.2: return "🔥🔥🔥"
    elif delta >= 0.1: return "🔥🔥"
    else: return "🔥"

preds_df['confidence'] = preds_df['home_win_prob'].apply(confidence)

# === Odds merging
if not os.path.exists(ODDS_PATH):
    print(f"❌ Odds file not found: {ODDS_PATH}. Please run fetch_odds.py first.")
    exit()

odds_df = load_csv(ODDS_PATH)[['gamePk', 'run_line', 'ou_line']]
preds_df = preds_df.merge(odds_df, on='gamePk', how='left')

# === Run line edge (clean format)
def run_line_edge(row):
    if pd.isna(row['run_line']):
        return "N/A"
    team = row['home_team'] if row['run_line'] < 0 else row['away_team']
    side = "-1.5" if row['run_line'] < 0 else "+1.5"
    return f"{team} {side} Cover"

preds_df['run_line_edge'] = preds_df.apply(run_line_edge, axis=1)

# === O/U edge (clean format)
def ou_edge(row):
    if pd.isna(row['ou_line']):
        return "N/A"
    return f"Over {row['ou_line']}" if row['predicted_total_runs'] > row['ou_line'] else f"Under {row['ou_line']}"

preds_df['ou_edge'] = preds_df.apply(ou_edge, axis=1)

# === Save predictions
cols = [
    'date', 'gamePk', 'home_team', 'away_team',
    'home_win_prob', 'confidence',
    'predicted_margin', 'run_line', 'run_line_edge',
    'predicted_total_runs', 'ou_line', 'ou_edge'
]

preds_df[cols].to_csv(OUTPUT_CSV, index=False)
os.makedirs("predictions", exist_ok=True)
preds_df[cols].to_csv(ARCHIVE_CSV, index=False)

print(f"✅ Saved predictions to:\n- {OUTPUT_CSV}\n- {ARCHIVE_CSV}")
print(preds_df[cols])

# === Final validation
expected_gamePks = set(todays_games['gamePk'])
predicted_gamePks = set(preds_df['gamePk'])
missed = expected_gamePks - predicted_gamePks
if missed:
    print(f"⚠️ {len(missed)} game(s) skipped due to incomplete features: {missed}")
