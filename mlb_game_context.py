import requests
import pandas as pd
import datetime
import time
import os

OUTPUT_CSV = 'mlb_game_context_2025.csv'

def get_schedule(date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
    res = requests.get(url)
    if res.status_code != 200:
        print(f"⚠️ Failed to fetch schedule for {date}")
        return []
    games = res.json().get('dates', [])
    return [g for g in (games[0]['games'] if games else []) if g.get('gameType') == 'R']

def get_game_info(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/feed/live"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"⚠️ Error {res.status_code} fetching gamePk {game_pk}")
    except Exception as e:
        print(f"⚠️ Exception fetching gamePk {game_pk}: {e}")
    return None

def parse_game_context(game, feed):
    game_pk = game['gamePk']
    date = game['officialDate']
    venue = game.get('venue', {}).get('name', '')
    home = game['teams']['home']['team']['name']
    away = game['teams']['away']['team']['name']

    weather = feed.get('gameData', {}).get('weather', {})
    temp = weather.get('temp') if 'temp' in weather else ''
    wind = weather.get('wind') if 'wind' in weather else ''
    humidity = weather.get('humidity') if 'humidity' in weather else ''
    condition = weather.get('condition') if 'condition' in weather else ''
    scorer = feed.get('gameData', {}).get('officialScorer', {}).get('fullName', '')

    umps = {}
    for ump in feed.get('gameData', {}).get('umpires', []):
        if 'position' in ump:
            umps[ump['position']] = ump['official']['fullName']

    return {
        'gamePk': game_pk,
        'date': date,
        'venue': venue,
        'home_team': home,
        'away_team': away,
        'weather': condition,
        'temperature': temp,
        'wind': wind,
        'humidity': humidity,
        'official_scorer': scorer,
        'umpire_HP': umps.get('Home Plate', ''),
        'umpire_1B': umps.get('First Base', ''),
        'umpire_2B': umps.get('Second Base', ''),
        'umpire_3B': umps.get('Third Base', '')
    }

def run_full_game_context_build(start_date='2025-03-27'):
    today = datetime.date.today()
    all_data = []

    for n in range((today - datetime.date.fromisoformat(start_date)).days + 1):
        date_str = (datetime.date.fromisoformat(start_date) + datetime.timedelta(days=n)).isoformat()
        games = get_schedule(date_str)
        print(f"📅 {date_str} – {len(games)} games found")

        for game in games:
            game_pk = game['gamePk']
            status = game['status']['detailedState']

            if status != 'Final':
                print(f"  ⏳ Skipping game {game_pk} – not final")
                continue

            print(f"  🔍 Processing game {game_pk}...")

            feed = get_game_info(game_pk)
            if feed:
                parsed = parse_game_context(game, feed)
                all_data.append(parsed)
            else:
                print(f"  ⚠️ No data for gamePk {game_pk}")
            time.sleep(0.5)

    if all_data:
        df = pd.DataFrame(all_data)
        df.drop_duplicates(subset='gamePk', inplace=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Wrote {len(df)} rows to {OUTPUT_CSV}")
    else:
        print("❌ No data written – nothing retrieved.")

if __name__ == "__main__":
    run_full_game_context_build()
