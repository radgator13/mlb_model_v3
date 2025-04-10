import requests
import pandas as pd
import datetime

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

def build_pending_boxscores(date=None):
    if date is None:
        date = datetime.date.today().isoformat()

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
            'home_runs': '',
            'away_runs': '',
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Created {OUTPUT_CSV} with {len(df)} games for {date}")

if __name__ == "__main__":
    build_pending_boxscores()
