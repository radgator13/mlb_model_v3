import subprocess
import datetime
import os
import shutil
import pandas as pd

# === Config
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

# === Step 1: Build today's schedule
print("📅 Step 1: Building pending boxscores...")
subprocess.run(["python", "build_pending_boxscores.py"], check=True)

# === Step 2: Fetch live sportsbook odds
print("📡 Step 2: Fetching live MLB odds...")
subprocess.run(["python", "fetch_odds.py"], check=True)

# === Step 3: Run model predictions
print("🧠 Step 3: Running model predictions...")
subprocess.run(["python", "predict_today.py"], check=True)

# === Step 4: Archive today's prediction file
os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
if os.path.exists(LATEST_FILE):
    shutil.copy(LATEST_FILE, TODAY_FILE)
    print(f"📁 Archived to {TODAY_FILE}")

# === Step 5: Log Results for Today & Yesterday
def log_results(prediction_file, date_str):
    if not os.path.exists(prediction_file) or not os.path.exists(BOX_FILE):
        print(f"❌ Skipping {date_str}: Missing prediction or boxscore file.")
        return

    pred_df = pd.read_csv(prediction_file)
    box_df = pd.read_csv(BOX_FILE)[['gamePk', 'home_runs', 'away_runs']]

    # Check for missing gamePk matches before merging
    missing_gamepks = set(pred_df['gamePk']) - set(box_df['gamePk'])
    if missing_gamepks:
        print(f"⚠️ {len(missing_gamepks)} game(s) in predictions not found in boxscore:")
        for g in missing_gamepks:
            print(f" - Missing gamePk: {g}")
        missed_rows = pred_df[pred_df['gamePk'].isin(missing_gamepks)]
        missed_rows.to_csv("missed_results_debug.csv", index=False)
        print(f"⚠️ Saved missing predictions to missed_results_debug.csv")

    # Merge predictions with actual results
    merged = pred_df.merge(box_df, on="gamePk", how="left")
    merged = merged.dropna(subset=["home_runs", "away_runs"])
    if merged.empty:
        print(f"⏳ No completed games to log for {date_str}")
        return

    merged["actual_winner"] = merged.apply(
        lambda row: "home" if row["home_runs"] > row["away_runs"]
        else "away" if row["home_runs"] < row["away_runs"]
        else "tie", axis=1
    )
    merged["predicted_winner"] = merged["home_win_prob"].apply(lambda p: "home" if p >= 0.5 else "away")
    merged["is_correct"] = merged["predicted_winner"] == merged["actual_winner"]
    merged["date"] = date_str

    log_cols = [
        'date', 'gamePk', 'home_team', 'away_team',
        'home_win_prob', 'predicted_winner', 'actual_winner',
        'is_correct', 'confidence'
    ]
    if 'run_line' in pred_df.columns: log_cols += ['run_line']
    if 'run_line_edge' in pred_df.columns: log_cols += ['run_line_edge']
    if 'ou_line' in pred_df.columns: log_cols += ['ou_line']
    if 'ou_edge' in pred_df.columns: log_cols += ['ou_edge']

    merged = merged[log_cols]

    if os.path.exists(RESULTS_LOG):
        existing = pd.read_csv(RESULTS_LOG)
        existing = existing[~((existing['date'] == date_str) & (existing['gamePk'].isin(merged['gamePk'])))]
        merged = pd.concat([existing, merged], ignore_index=True)

    merged.to_csv(RESULTS_LOG, index=False)
    print(f"✅ Logged {len(merged)} results to {RESULTS_LOG}")

# Score today and yesterday
log_results(TODAY_FILE, DATE_TODAY)
log_results(YESTERDAY_FILE, DATE_YESTERDAY)

# === Step 6: Git add + push
print("🚀 Step 6: Committing and pushing files...")
subprocess.run([
    "git", "add", "-f",
    LATEST_FILE, TODAY_FILE, RESULTS_LOG, ODDS_FILE,
    "predict_today.py", "fetch_odds.py", "run_pipeline.py", "train_model.py"
], check=True)
subprocess.run(["git", "commit", "-m", f"📈 Full pipeline update for {DATE_TODAY}"], check=False)
subprocess.run(["git", "push", "origin", "master"], check=True)

# === Step 7: Clean duplicates in results log
if os.path.exists(RESULTS_LOG):
    df = pd.read_csv(RESULTS_LOG)
    df = df[df['actual_winner'].isin(['home', 'away'])]  # remove tie/incomplete
    df = df.drop_duplicates(subset=["date", "gamePk"], keep="first")
    df = df.sort_values(by=["date", "gamePk"]).reset_index(drop=True)
    df.to_csv(RESULTS_LOG, index=False)
    print(f"🧹 Cleaned and finalized {len(df)} results.")
