import requests
import pandas as pd
import datetime
import time
import os

OUTPUT_CSV = 'mlb_lineups_2025.csv'

def get_schedule(date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    games = res.json().get('dates', [])
    return [g for g in (games[0]['games'] if games else []) if g.get('gameType') == 'R']

def get_boxscore(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"⚠️ Error fetching boxscore for game {game_pk}: {e}")
    return None

def parse_lineups(game, boxscore):
    game_pk = game['gamePk']
    date = game['officialDate']
    rows = []

    for side in ['home', 'away']:
        team_info = boxscore['teams'][side]
        team_name = team_info['team']['name']
        players = team_info.get('players', {})

        for player_id_str, player_data in players.items():
            stats = player_data.get('stats', {})
            pos = player_data.get('position', {})
            batting_order = player_data.get('battingOrder')
            if batting_order and pos:
                batting_slot = int(batting_order[:-1])  # Remove trailing '0'
                rows.append({
                    'gamePk': game_pk,
                    'date': date,
                    'team': team_name,
                    'team_side': side,
                    'player_id': player_data['person']['id'],
                    'player_name': player_data['person']['fullName'],
                    'batting_order': batting_slot,
                    'position': pos.get('abbreviation', '')
                })
    return rows

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
    df_combined.drop_duplicates(subset=['gamePk', 'player_id'], inplace=True)
    df_combined.to_csv(OUTPUT_CSV, index=False)

def run_lineup_pipeline(start_date='2025-03-27'):
    today = datetime.date.today()
    existing_ids = load_existing_game_ids()
    new_data = []

    for n in range((today - datetime.date.fromisoformat(start_date)).days + 1):
        date_str = (datetime.date.fromisoformat(start_date) + datetime.timedelta(days=n)).isoformat()
        games = get_schedule(date_str)
        print(f"{date_str} – Found {len(games)} regular season games")

        for game in games:
            if game['status']['detailedState'] == 'Final':
                game_pk = game['gamePk']
                if game_pk in existing_ids:
                    continue
                boxscore = get_boxscore(game_pk)
                if boxscore:
                    parsed = parse_lineups(game, boxscore)
                    new_data.extend(parsed)
                    time.sleep(0.5)

    if new_data:
        append_to_csv(new_data)
        print(f"✅ Added {len(new_data)} lineup entries.")
    else:
        print("No new lineups to add.")

if __name__ == "__main__":
    run_lineup_pipeline()
