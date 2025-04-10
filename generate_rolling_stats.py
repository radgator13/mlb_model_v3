import pandas as pd

BOX_PATH = "mlb_boxscores_2025.csv"
OUTPUT_PATH = "team_rolling_stats_2025.csv"
WINDOW = 5

df = pd.read_csv(BOX_PATH)
df = df[['date', 'gamePk', 'home_team', 'away_team', 'home_runs', 'away_runs', 'home_hits', 'away_hits']]
df['date'] = pd.to_datetime(df['date'])

home_df = df[['date', 'gamePk', 'home_team', 'home_runs', 'away_runs', 'home_hits']].copy()
home_df.columns = ['date', 'gamePk', 'team', 'runs_scored', 'runs_allowed', 'hits']
home_df['team_side'] = 'home'

away_df = df[['date', 'gamePk', 'away_team', 'away_runs', 'home_runs', 'away_hits']].copy()
away_df.columns = ['date', 'gamePk', 'team', 'runs_scored', 'runs_allowed', 'hits']
away_df['team_side'] = 'away'

long_df = pd.concat([home_df, away_df], ignore_index=True)
long_df.sort_values(by=['team', 'date'], inplace=True)

rolling = long_df.groupby('team').rolling(WINDOW, on='date')[['runs_scored', 'runs_allowed', 'hits']].mean().reset_index()
long_df = long_df.merge(rolling, on=['team', 'date'], suffixes=('', '_last5'))

output = long_df[['gamePk', 'team_side', 'team', 'runs_scored_last5', 'runs_allowed_last5', 'hits_last5']].dropna()
output.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved to {OUTPUT_PATH}")
