import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
import joblib
import os

# === File paths ===
BOX_PATH = "mlb_boxscores_2025.csv"
PITCHING_PATH = "mlb_pitching_stats_2025.csv"
BATTING_PATH = "mlb_batting_stats_2025.csv"
STRENGTH_PATH = "team_strengths.csv"
ROLLING_STATS_PATH = "team_rolling_stats_2025.csv"
WIN_MODEL = "mlb_win_model.pkl"
MARGIN_MODEL = "mlb_margin_model.pkl"
TOTAL_MODEL = "mlb_total_model.pkl"

# === Load CSVs
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

boxscores = load_csv(BOX_PATH)
pitching = load_csv(PITCHING_PATH)
batting = load_csv(BATTING_PATH)
strengths_df = load_csv(STRENGTH_PATH)
rolling_stats = load_csv(ROLLING_STATS_PATH)

# === Setup team strength lookup
strengths_df.columns = [col.strip().lower() for col in strengths_df.columns]
team_strengths = dict(zip(strengths_df['team'], strengths_df['strength']))

# === Basic target variables
required = {'home_runs', 'away_runs', 'gamePk', 'home_team', 'away_team'}
if not required.issubset(set(boxscores.columns)):
    raise ValueError("❌ Missing columns in boxscores")

box = boxscores.copy()
box['home_win'] = (box['home_runs'] > box['away_runs']).astype(int)
box['total_runs'] = box['home_runs'] + box['away_runs']
box['run_margin'] = box['home_runs'] - box['away_runs']

# === Feature builder
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
features = features.dropna(thresh=int(features.shape[1] * 0.75)).copy()

# === Targets
win_target = box.loc[features.index, 'home_win']
margin_target = box.loc[features.index, 'run_margin']
total_target = box.loc[features.index, 'total_runs']

# === Team encoding
features['home_team'] = box.loc[features.index, 'home_team']
features['away_team'] = box.loc[features.index, 'away_team']
features = pd.get_dummies(features, columns=['home_team', 'away_team'])

# === Train/Test Split
X_train, X_test, y_win_train, y_win_test = train_test_split(features, win_target, test_size=0.2, random_state=42, stratify=win_target)
_, _, y_margin_train, y_margin_test = train_test_split(features, margin_target, test_size=0.2, random_state=42)
_, _, y_total_train, y_total_test = train_test_split(features, total_target, test_size=0.2, random_state=42)

# === Train Win Model (classifier)
win_model = xgb.XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=4, eval_metric='logloss')
win_model.fit(X_train, y_win_train)
preds_win = win_model.predict(X_test)
probs_win = win_model.predict_proba(X_test)[:, 1]
print(f"✅ Accuracy (Win Model): {accuracy_score(y_win_test, preds_win):.3f}")
print(f"✅ AUC (Win Model): {roc_auc_score(y_win_test, probs_win):.3f}")

# === Train Run Margin Model (regression)
margin_model = xgb.XGBRegressor(n_estimators=250, learning_rate=0.1, max_depth=4)
margin_model.fit(X_train, y_margin_train)
preds_margin = margin_model.predict(X_test)
print(f"📏 MAE (Run Margin): {mean_absolute_error(y_margin_test, preds_margin):.2f}")

# === Train Total Runs Model (regression)
total_model = xgb.XGBRegressor(n_estimators=250, learning_rate=0.1, max_depth=4)
total_model.fit(X_train, y_total_train)
preds_total = total_model.predict(X_test)
print(f"📏 MAE (Total Runs): {mean_absolute_error(y_total_test, preds_total):.2f}")

# === Save all models
joblib.dump(win_model, WIN_MODEL)
joblib.dump(margin_model, MARGIN_MODEL)
joblib.dump(total_model, TOTAL_MODEL)
print(f"🎯 Models saved: {WIN_MODEL}, {MARGIN_MODEL}, {TOTAL_MODEL}")
