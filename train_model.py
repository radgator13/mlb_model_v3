import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

# === File paths ===
BOX_PATH = "mlb_boxscores_2025.csv"
PITCHING_PATH = "mlb_pitching_stats_2025.csv"
BATTING_PATH = "mlb_batting_stats_2025.csv"
STRENGTH_PATH = "team_strengths.csv"
ROLLING_STATS_PATH = "team_rolling_stats_2025.csv"
OUTPUT_MODEL = "mlb_win_model.pkl"

# === Load CSVs ===
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

boxscores = load_csv(BOX_PATH)
pitching = load_csv(PITCHING_PATH)
batting = load_csv(BATTING_PATH)
strengths_df = load_csv(STRENGTH_PATH)
rolling_stats = load_csv(ROLLING_STATS_PATH)

# === Validate team_strengths.csv
strengths_df.columns = [col.strip().lower() for col in strengths_df.columns]
team_strengths = dict(zip(strengths_df['team'], strengths_df['strength']))

# === Validate boxscore columns
required_cols = {'home_runs', 'away_runs', 'gamePk', 'home_team', 'away_team'}
if not required_cols.issubset(set(boxscores.columns)):
    raise ValueError(f"❌ Missing required boxscore columns: {required_cols - set(boxscores.columns)}")

box = boxscores.copy()
box['home_win'] = (box['home_runs'] > box['away_runs']).astype(int)

# === Feature Engineering ===
def build_features(row):
    gamePk = row['gamePk']

    def stat(df, side, col):
        df_sub = df[(df['gamePk'] == gamePk) & (df['team_side'] == side)]
        return df_sub[col].mean() if not df_sub.empty and col in df_sub else np.nan

    def rolling_stat(side, col):
        df_sub = rolling_stats[(rolling_stats['gamePk'] == gamePk) & (rolling_stats['team_side'] == side)]
        return df_sub[col].values[0] if not df_sub.empty else np.nan

    return pd.Series({
        'home_pitcher_era': stat(pitching, 'home', 'era'),
        'away_pitcher_era': stat(pitching, 'away', 'era'),
        'home_strikeouts': stat(pitching, 'home', 'strikeouts'),
        'away_strikeouts': stat(pitching, 'away', 'strikeouts'),
        'home_hits': stat(batting, 'home', 'hits'),
        'away_hits': stat(batting, 'away', 'hits'),
        'home_runs': row['home_runs'],
        'away_runs': row['away_runs'],
        'home_strength': team_strengths.get(row['home_team'], 70),
        'away_strength': team_strengths.get(row['away_team'], 70),
        'home_avg_runs_last5': rolling_stat('home', 'runs_scored_last5'),
        'away_avg_runs_last5': rolling_stat('away', 'runs_scored_last5'),
        'home_avg_runs_allowed_last5': rolling_stat('home', 'runs_allowed_last5'),
        'away_avg_runs_allowed_last5': rolling_stat('away', 'runs_allowed_last5'),
        'home_avg_hits_last5': rolling_stat('home', 'hits_last5'),
        'away_avg_hits_last5': rolling_stat('away', 'hits_last5'),
    })

features = box.apply(build_features, axis=1)
target = box['home_win']

# === Clean data
X = features.dropna(thresh=int(features.shape[1] * 0.75)).copy()
y = target.loc[X.index]

# === Add team names using .loc to avoid SettingWithCopyWarning
X.loc[:, 'home_team'] = box.loc[X.index, 'home_team']
X.loc[:, 'away_team'] = box.loc[X.index, 'away_team']

# === One-hot encode teams
X = pd.get_dummies(X, columns=['home_team', 'away_team'])

# === Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=4,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# === Evaluate
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
print(f"✅ Accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"✅ AUC: {roc_auc_score(y_test, probs):.3f}")

# === Save model
joblib.dump(model, OUTPUT_MODEL)
print(f"🎯 Model saved to {OUTPUT_MODEL}")
