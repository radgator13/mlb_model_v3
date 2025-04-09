import requests
import pandas as pd
import datetime
import time
import os

OUTPUT_CSV = 'mlb_batting_stats_2025.csv'
PENDING_LOG = 'pending_batting_games.csv'

def get_schedule(date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    games = res.json().get('dates', [])
    all_games = games[0]['games'] if games else []
    return [g for g in all_games if g.get('gameType') == 'R']

def get_boxscore(game_pk, max_retries=3):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            print(f"⚠️ Error fetching boxscore for game {game_pk}: {e}")
        time.sleep(1)
    print(f"🚫 Failed to get boxscore for game {game_pk}")
    return None

def parse_batting_stats(game, boxscore):
    game_pk = game['gamePk']
    date = game['officialDate']
    rows = []

    for side in ['home', 'away']:
        team_info = boxscore['teams'][side]
        team_name = team_info['team']['name']
        players = team_info.get('players', {})

        for player_key, player_data in players.items():
            stats = player_data.get('stats', {}).get('batting')
            if not stats:
                continue
            name = player_data['person']['fullName']
            rows.append({
                'gamePk': game_pk,
                'date': date,
                'team': team_name,
                'team_side': side,
                'player_id': player_data['person']['id'],
                'player_name': name,
                'at_bats': stats.get('atBats'),
                'hits': stats.get('hits'),
                'runs': stats.get('runs'),
                'rbi': stats.get('rbi'),
                'home_runs': stats.get('homeRuns'),
                'base_on_balls': stats.get('baseOnBalls'),
                'strike_outs': stats.get('strikeOuts'),
                'stolen_bases': stats.get('stolenBases'),
                'left_on_base': stats.get('leftOnBase'),
                'avg': stats.get('avg'),
                'obp': stats.get('obp'),
                'slg': stats.get('slg'),
                'ops': stats.get('ops')
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

def save_pending_game(game_pk, date):
    if not os.path.exists(PENDING_LOG):
        df = pd.DataFrame(columns=['gamePk', 'date'])
    else:
        df = pd.read_csv(PENDING_LOG)
    if game_pk not in df['gamePk'].values:
        df = pd.concat([df, pd.DataFrame([{'gamePk': game_pk, 'date': date}])])
        df.to_csv(PENDING_LOG, index=False)

def run_batting_pipeline(start_date='2025-03-27'):
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
                    parsed_batting = parse_batting_stats(game, boxscore)
                    new_data.extend(parsed_batting)
                    time.sleep(0.5)
                else:
                    save_pending_game(game_pk, date_str)

    if new_data:
        append_to_csv(new_data)
        print(f"✅ Added {len(new_data)} new batting lines.")
    else:
        print("No new completed games to add.")

if __name__ == "__main__":
    run_batting_pipeline()
