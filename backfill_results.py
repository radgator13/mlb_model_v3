import pandas as pd
import os
import glob

# === Files
RESULTS_LOG = "prediction_results.csv"
BOX_FILE = "mlb_boxscores_2025.csv"
PRED_DIR = "predictions"

# === Load boxscores
if not os.path.exists(BOX_FILE):
    print("❌ Missing boxscore file:", BOX_FILE)
    exit()

boxscores = pd.read_csv(BOX_FILE)[['gamePk', 'home_runs', 'away_runs']]

# === Load existing results
if os.path.exists(RESULTS_LOG):
    existing = pd.read_csv(RESULTS_LOG)
else:
    existing = pd.DataFrame()

# === Find all prediction files
pred_files = glob.glob(os.path.join(PRED_DIR, "predictions_*.csv"))

total_logged = 0
for file_path in pred_files:
    filename = os.path.basename(file_path)
    date_str = filename.replace("predictions_", "").replace(".csv", "")

    pred_df = pd.read_csv(file_path)
    if 'date' in existing.columns:
        already_logged = existing[
            (existing['date'] == date_str) &
            (existing['gamePk'].isin(pred_df['gamePk']))
        ]
        if not already_logged.empty:
            continue  # Skip already-logged date

    # Merge predictions with boxscores
    merged = pred_df.merge(boxscores, on='gamePk', how='left')
    merged = merged.dropna(subset=["home_runs", "away_runs"])
    if merged.empty:
        print(f"⚠️ No completed games for {date_str}")
        continue

    # Score predictions
    merged["actual_winner"] = merged.apply(
        lambda row: "home" if row["home_runs"] > row["away_runs"]
        else "away" if row["home_runs"] < row["away_runs"]
        else "tie", axis=1
    )
    merged["predicted_winner"] = merged["home_win_prob"].apply(lambda p: "home" if p >= 0.5 else "away")
    merged["is_correct"] = merged["predicted_winner"] == merged["actual_winner"]
    merged["date"] = date_str

    # Select columns
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

    existing = pd.concat([existing, merged], ignore_index=True)
    total_logged += len(merged)
    print(f"✅ Logged {len(merged)} results for {date_str}")

# Save results
existing = existing.sort_values(by=["date", "gamePk"]).reset_index(drop=True)
existing.to_csv(RESULTS_LOG, index=False)
print(f"📦 Finalized {len(existing)} total rows in {RESULTS_LOG}")
