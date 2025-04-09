import requests
import pandas as pd
import datetime
import time
import os

OUTPUT_CSV = 'mlb_game_results_2025.csv'

def get_schedule(date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    games = res.json().get('dates', [])
    return [g for g in (games[0]['games'] if games else []) if g.get('gameType') == 'R']

def get_game_feed(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/feed/live"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"⚠️ Failed to fetch game feed for {game_pk}: {e}")
    return None

def parse_game_result(game, feed):
    game_pk = game['gamePk']
    date = game['officialDate']
    home = game['teams']['home']['team']['name']
    away = game['teams']['away']['team']['name']
    linescore = feed.get('liveData', {}).get('linescore', {})

    home_score = linescore.get('teams', {}).get('home', {}).get('runs', None)
    away_score = linescore.get('teams', {}).get('away', {}).get('runs', None)

    inning = linescore.get('currentInning', '')
    outs = linescore.get('outs', '')
    is_walkoff = feed.get('gameData', {}).get('status', {}).get('isWalkOff', False)
    status = game['status']['detailedState']

    if home_score is None or away_score is None:
        return None

    if home_score > away_score:
        winner = home
        loser = away
    else:
        winner = away
        loser = home

    return {
        'gamePk': game_pk,
        'date': date,
        'home_team': home,
        'home_score': home_score,
        'away_team': away,
        'away_score': away_score,
        'winner': winner,
        'loser': loser,
        'is_walkoff': is_walkoff,
        'inning': inning,
        'outs': outs,
        'status': status
    }

def load_existing_game_ids():
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        return set(df['gamePk'].astype(int))
    return set()

def append_to_csv(data_rows):
    df_new = pd.DataFrame(data_rows)
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.drop_duplicates(subset='gamePk', inplace=True)
    df_combined.to_csv(OUTPUT_CSV, index=False)

def run_game_results_pipeline(start_date='2025-03-27'):
    today = datetime.date.today()
    existing_ids = load_existing_game_ids()
    new_data = []

    for n in range((today - datetime.date.fromisoformat(start_date)).days + 1):
        date_str = (datetime.date.fromisoformat(start_date) + datetime.timedelta(days=n)).isoformat()
        games = get_schedule(date_str)
        print(f"{date_str} – Found {len(games)} games")

        for game in games:
            if game['status']['detailedState'] == 'Final':
                game_pk = game['gamePk']
                if game_pk in existing_ids:
                    continue
                feed = get_game_feed(game_pk)
                if feed:
                    parsed = parse_game_result(game, feed)
                    if parsed:
                        new_data.append(parsed)
                    time.sleep(0.5)

    if new_data:
        append_to_csv(new_data)
        print(f"✅ Added {len(new_data)} game results.")
    else:
        print("No new final results to add.")

if __name__ == "__main__":
    run_game_results_pipeline()
