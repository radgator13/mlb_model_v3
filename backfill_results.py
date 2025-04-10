import pandas as pd
import os

DATE = "2025-04-09"
PRED_FILE = f"predictions/predictions_{DATE}.csv"
BOX_FILE = "mlb_boxscores_2025.csv"
RESULTS_LOG = "prediction_results.csv"

# === Load predictions
pred_df = pd.read_csv(PRED_FILE)

# === Load actuals
boxscores = pd.read_csv(BOX_FILE)
box = boxscores[['gamePk', 'home_runs', 'away_runs']]

# === Merge
merged = pred_df.merge(box, on="gamePk", how="left")

# === Evaluate
merged["actual_winner"] = merged.apply(
    lambda row: "home" if row["home_runs"] > row["away_runs"] else
                "away" if row["home_runs"] < row["away_runs"] else "tie", axis=1
)
merged["predicted_winner"] = merged["home_win_prob"].apply(lambda p: "home" if p >= 0.5 else "away")
merged["is_correct"] = merged["predicted_winner"] == merged["actual_winner"]
merged["date"] = DATE

# === Select columns
merged = merged[['date', 'gamePk', 'home_team', 'away_team', 'home_win_prob',
                 'predicted_winner', 'actual_winner', 'is_correct', 'confidence']]

# === Append to results log
if os.path.exists(RESULTS_LOG):
    existing = pd.read_csv(RESULTS_LOG)
    full = pd.concat([existing, merged], ignore_index=True)
else:
    full = merged

# === Save updated results log
full.to_csv(RESULTS_LOG, index=False)
print(f"✅ Backfilled results for {DATE} into {RESULTS_LOG}")
