import requests
import pandas as pd
import datetime
import time
import os

OUTPUT_CSV = 'mlb_boxscores_2025.csv'

def get_schedule(date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    games = res.json().get('dates', [])
    return games[0]['games'] if games else []

def get_boxscore(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else None

def parse_boxscore(game, boxscore):
    game_pk = game['gamePk']
    date = game['officialDate']
    teams = game['teams']
    
    def extract_team_stats(team_key):
        team_info = boxscore['teams'][team_key]
        team_name = team_info['team']['name']
        runs = team_info['teamStats']['batting']['runs']
        hits = team_info['teamStats']['batting']['hits']
        errors = team_info['teamStats']['fielding']['errors']
        return team_name, runs, hits, errors
    
    away_name, away_runs, away_hits, away_errors = extract_team_stats('away')
    home_name, home_runs, home_hits, home_errors = extract_team_stats('home')
    
    return {
        'gamePk': game_pk,
        'date': date,
        'home_team': home_name,
        'away_team': away_name,
        'home_runs': home_runs,
        'away_runs': away_runs,
        'home_hits': home_hits,
        'away_hits': away_hits,
        'home_errors': home_errors,
        'away_errors': away_errors
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

def run_pipeline(start_date='2025-03-27'):
    today = datetime.date.today()
    existing_ids = load_existing_game_ids()
    new_data = []

    for n in range((today - datetime.date.fromisoformat(start_date)).days + 1):
        date_str = (datetime.date.fromisoformat(start_date) + datetime.timedelta(days=n)).isoformat()
        print(f"Checking games for {date_str}...")
        games = get_schedule(date_str)
        for game in games:
            if game['status']['detailedState'] == 'Final':
                game_pk = game['gamePk']
                if game_pk in existing_ids:
                    continue
                boxscore = get_boxscore(game_pk)
                if boxscore:
                    parsed = parse_boxscore(game, boxscore)
                    new_data.append(parsed)
                    time.sleep(0.5)  # polite delay
    if new_data:
        append_to_csv(new_data)
        print(f"✅ Added {len(new_data)} new games.")
    else:
        print("No new completed games to add.")

if __name__ == "__main__":
    run_pipeline()

