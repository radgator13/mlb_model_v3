import subprocess
import datetime
import os
import shutil
import pandas as pd

# === Constants
TODAY = datetime.date.today()
YESTERDAY = TODAY - datetime.timedelta(days=1)

DATE_TODAY = TODAY.isoformat()
DATE_YESTERDAY = YESTERDAY.isoformat()

ARCHIVE_FOLDER = "predictions"
LATEST_FILE = "predictions_latest.csv"
TODAY_FILE = f"{ARCHIVE_FOLDER}/predictions_{DATE_TODAY}.csv"
YESTERDAY_FILE = f"{ARCHIVE_FOLDER}/predictions_{DATE_YESTERDAY}.csv"
RESULTS_LOG = "prediction_results.csv"
BOX_FILE = "mlb_boxscores_2025.csv"
ODDS_FILE = f"odds_{DATE_TODAY}.csv"

# === Step 1: Build pending boxscores
print("📅 Step 1: Building pending boxscores...")
subprocess.run(["python", "build_pending_boxscores.py"], check=True)

# === Step 2: Fetch live odds
print("📡 Step 2: Fetching live MLB odds...")
subprocess.run(["python", "fetch_odds.py"], check=True)

# === Step 3: Run predictions
print("🧠 Step 3: Running model predictions...")
subprocess.run(["python", "predict_today.py"], check=True)

# === Step 4: Archive latest prediction to dated file
os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
if os.path.exists(LATEST_FILE):
    shutil.copy(LATEST_FILE, TODAY_FILE)
    print(f"📁 Archived today's prediction to {TODAY_FILE}")
else:
    print("❌ predictions_latest.csv not found!")

# === Step 5: Log results for today + yesterday
def log_results(prediction_file, date_str):
    if not os.path.exists(prediction_file):
        print(f"⚠️ No prediction file for {date_str} found.")
        return

    if not os.path.exists(BOX_FILE):
        print(f"⚠️ Cannot log {date_str}: boxscore file not found.")
        return

    pred_df = pd.read_csv(prediction_file)
    box_df = pd.read_csv(BOX_FILE)
    box_df = box_df[['gamePk', 'home_runs', 'away_runs']]

    merged = pred_df.merge(box_df, on='gamePk', how='left')

    # Basic result tracking
    merged['actual_winner'] = merged.apply(
        lambda row: "home" if row["home_runs"] > row["away_runs"] else
                    "away" if row["home_runs"] < row["away_runs"] else "tie",
        axis=1
    )
    merged["predicted_winner"] = merged["home_win_prob"].apply(lambda p: "home" if p >= 0.5 else "away")
    merged["is_correct"] = merged["predicted_winner"] == merged["actual_winner"]
    merged["date"] = date_str

    log_cols = ['date', 'gamePk', 'home_team', 'away_team', 'home_win_prob',
                'predicted_winner', 'actual_winner', 'is_correct', 'confidence']

    # Include edge data if present
    if 'run_line' in pred_df.columns and 'run_line_edge' in pred_df.columns:
        log_cols += ['run_line', 'run_line_edge']
    if 'ou_line' in pred_df.columns and 'ou_edge' in pred_df.columns:
        log_cols += ['ou_line', 'ou_edge']

    merged = merged[log_cols]

    # Merge into results log
    if os.path.exists(RESULTS_LOG):
        existing = pd.read_csv(RESULTS_LOG)
        existing = existing[~((existing['date'] == date_str) & (existing['gamePk'].isin(merged['gamePk'])))]
        merged = pd.concat([existing, merged], ignore_index=True)

    merged.to_csv(RESULTS_LOG, index=False)
    print(f"📊 Logged prediction results for {date_str} to {RESULTS_LOG}")

log_results(TODAY_FILE, DATE_TODAY)
log_results(YESTERDAY_FILE, DATE_YESTERDAY)

# === Step 6: Git push
print("🚀 Step 6: Committing and pushing to GitHub...")
subprocess.run(["git", "add", "-f", LATEST_FILE, TODAY_FILE, RESULTS_LOG, ODDS_FILE], check=True)
subprocess.run(["git", "commit", "-m", f'📈 Auto-pipeline update for {DATE_TODAY} + backfill {DATE_YESTERDAY}'], check=False)
subprocess.run(["git", "push", "origin", "master"], check=True)

# === Step 7: Clean up duplicate logs
if os.path.exists(RESULTS_LOG):
    df = pd.read_csv(RESULTS_LOG)
    df = df[df['actual_winner'].isin(['home', 'away'])]
    df = df.sort_values(by="actual_winner", ascending=False)
    df = df.drop_duplicates(subset=["date", "gamePk"], keep="first")
    df = df.sort_values(by=["date", "gamePk"]).reset_index(drop=True)
    df.to_csv(RESULTS_LOG, index=False)
    print(f"🧹 Cleaned duplicates in {RESULTS_LOG}. Rows now: {len(df)}")

print("✅ All steps completed. Dashboard is up to date.")
