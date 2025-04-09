import requests
import pandas as pd
import datetime

# === Config ===
DATE_TODAY = datetime.date.today().isoformat()  # or '2025-04-09' if testing
OUTPUT_CSV = "pending_boxscores.csv"

def get_schedule(date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print(f"❌ Failed to fetch schedule: {res.status_code}")
        return []
    data = res.json()
    return data.get("dates", [])[0].get("games", []) if data.get("dates") else []

def build_pending_boxscores(date):
    games = get_schedule(date)
    rows = []

    for game in games:
        gamePk = game.get("gamePk")
        home = game['teams']['home']['team']['name']
        away = game['teams']['away']['team']['name']
        rows.append({
            'gamePk': gamePk,
            'date': date,
            'home_team': home,
            'away_team': away,
            'home_runs': '',  # leave blank for pending
            'away_runs': '',
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Created {OUTPUT_CSV} with {len(df)} games.")

if __name__ == "__main__":
    build_pending_boxscores("2025-04-09")  # or DATE_TODAY if you prefer
