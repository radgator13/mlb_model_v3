import pandas as pd
import numpy as np

# === Config ===
BOX_PATH = "pending_boxscores.csv"
PITCHING_PATH = "mlb_pitching_stats_2025.csv"
BATTING_PATH = "mlb_batting_stats_2025.csv"

# === Defaults (fallback league/team average)
DEFAULT_PITCHING = {
    'era': 4.25,
    'strikeouts': 8.5,
}

DEFAULT_BATTING = {
    'hits': 8.2,
    'runs': 4.6,
}

# === Load data
boxscores = pd.read_csv(BOX_PATH)
pitching = pd.read_csv(PITCHING_PATH)
batting = pd.read_csv(BATTING_PATH)

# === Helpers
def ensure_team_side_rows(df, boxscores, stats_dict, stat_type):
    rows = []

    for _, game in boxscores.iterrows():
        gamePk = game['gamePk']
        for side, team in [('home', game['home_team']), ('away', game['away_team'])]:
            match = df[(df['gamePk'] == gamePk) & (df['team_side'] == side)]
            if match.empty:
                row = {
                    'gamePk': gamePk,
                    'team': team,
                    'team_side': side
                }
                row.update(stats_dict)
                rows.append(row)

    return pd.DataFrame(rows)

# === Fill missing pitching rows
new_pitching_rows = ensure_team_side_rows(
    df=pitching,
    boxscores=boxscores,
    stats_dict=DEFAULT_PITCHING,
    stat_type='pitching'
)

# === Fill missing batting rows
new_batting_rows = ensure_team_side_rows(
    df=batting,
    boxscores=boxscores,
    stats_dict=DEFAULT_BATTING,
    stat_type='batting'
)

# === Append and save updated files
if not new_pitching_rows.empty:
    pitching = pd.concat([pitching, new_pitching_rows], ignore_index=True)
    pitching.to_csv(PITCHING_PATH, index=False)
    print(f"✅ Added {len(new_pitching_rows)} fallback rows to {PITCHING_PATH}")

if not new_batting_rows.empty:
    batting = pd.concat([batting, new_batting_rows], ignore_index=True)
    batting.to_csv(BATTING_PATH, index=False)
    print(f"✅ Added {len(new_batting_rows)} fallback rows to {BATTING_PATH}")

if new_pitching_rows.empty and new_batting_rows.empty:
    print("✅ No missing stats found — all matchups already covered.")
