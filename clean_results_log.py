import pandas as pd

# Load your prediction results
RESULTS_LOG = "prediction_results.csv"
df = pd.read_csv(RESULTS_LOG)

# Step 1: Remove rows with no actual results
df = df[df['actual_winner'].isin(['home', 'away'])]

# Step 2: Keep only the latest (correct) version of each game/date
# This removes earlier duplicate rows like "tie" outcomes
df = df.sort_values(by="actual_winner", ascending=False)  # Keep home/away over tie
df = df.drop_duplicates(subset=["date", "gamePk"], keep="first")

# Optional: Reset index and sort for consistency
df = df.sort_values(by=["date", "gamePk"]).reset_index(drop=True)

# Step 3: Save the cleaned file
df.to_csv(RESULTS_LOG, index=False)
print(f"✅ Cleaned {RESULTS_LOG}. Rows now: {len(df)}")
