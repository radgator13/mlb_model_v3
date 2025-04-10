import requests
import pandas as pd
import datetime
import os

# === Config
API_KEY = "0a302e68b19b0d18932ef87357bd957e"
SPORT = "baseball_mlb"
REGION = "us"
MARKETS = "spreads,totals"
DATE_TODAY = datetime.date.today().isoformat()
BOX_PATH = "pending_boxscores.csv"
ODDS_FILE = f"odds_{DATE_TODAY}.csv"

# === Load today's scheduled games
if not os.path.exists(BOX_PATH):
    print("❌ pending_boxscores.csv not found. Run build_pending_boxscores.py first.")
    exit()

scheduled = pd.read_csv(BOX_PATH)
scheduled = scheduled[scheduled['date'] == DATE_TODAY]
scheduled['home_team'] = scheduled['home_team'].str.strip()
scheduled['away_team'] = scheduled['away_team'].str.strip()

# === Fetch odds from OddsAPI
print("📡 Fetching MLB odds from OddsAPI...")
url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
params = {
    "apiKey": API_KEY,
    "regions": REGION,
    "markets": MARKETS,
    "oddsFormat": "american"
}

res = requests.get(url, params=params)
if res.status_code != 200:
    print(f"❌ API Error: {res.status_code} - {res.text}")
    exit()

odds_data = res.json()

rows = []
for game in odds_data:
    odds_home = game['home_team'].strip()
    odds_away = game['away_team'].strip()
    
    # Look for match in pending_boxscores.csv (either way)
    match = scheduled[
        ((scheduled['home_team'] == odds_home) & (scheduled['away_team'] == odds_away)) |
        ((scheduled['home_team'] == odds_away) & (scheduled['away_team'] == odds_home))
    ]

    if match.empty:
        continue

    gamePk = match.iloc[0]['gamePk']
    box_home = match.iloc[0]['home_team']
    box_away = match.iloc[0]['away_team']

    # Grab first bookmaker
    if not game['bookmakers']:
        continue

    bookmaker = game['bookmakers'][0]
    run_line = None
    ou_line = None

    for market in bookmaker.get("markets", []):
        if market["key"] == "spreads":
            for outcome in market["outcomes"]:
                if outcome['name'] == odds_home:
                    run_line = outcome.get("point")
        elif market["key"] == "totals":
            for outcome in market["outcomes"]:
                ou_line = outcome.get("point")

    if run_line is None or ou_line is None:
        continue

    rows.append({
        "gamePk": gamePk,
        "home_team": box_home,
        "away_team": box_away,
        "run_line": run_line,
        "ou_line": ou_line
    })

# === Save to CSV
if not rows:
    print("⚠️ No matching odds found.")
    exit()

df = pd.DataFrame(rows)
df.to_csv(ODDS_FILE, index=False)
print(f"✅ Saved matched odds to: {ODDS_FILE}")
print(df)
