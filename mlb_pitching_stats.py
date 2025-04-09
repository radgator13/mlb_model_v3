import requests
import pandas as pd
import datetime
import time
import os

OUTPUT_CSV = 'mlb_pitching_stats_2025.csv'
PENDING_LOG = 'pending_boxscores.csv'

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
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
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

def parse_pitching_stats(game, boxscore):
    game_pk = game['gamePk']
    date = game['officialDate']
    rows = []
    for side in ['home', 'away']:
        team_info = boxscore['teams'][side]
        team_name = team_info['team']['name']
        pitchers = team_info.get('pitchers', [])
        players = team_info.get('players', {})
        for pitcher_id in pitchers:
            player_key = f'ID{pitcher_id}'
            player_data = players.get(player_key)
            if not player_data or 'stats' not in player_data or 'pitching' not in player_data['stats']:
                continue
            stats = player_data['stats']['pitching']
            name = player_data['person']['fullName']
            rows.append({
                'gamePk': game_pk,
                'date': date,
                'team': team_name,
                'team_side': side,
                'pitcher_id': pitcher_id,
                'pitcher_name': name,
                'innings_pitched': stats.get('inningsPitched'),
                'hits': stats.get('hits'),
                'runs': stats.get('runs'),
                'earned_runs': stats.get('earnedRuns'),
                'walks': stats.get('baseOnBalls'),
                'strikeouts': stats.get('strikeOuts'),
                'home_runs': stats.get('homeRuns'),
                'pitches': stats.get('numberOfPitches'),
                'strikes': stats.get('strikes'),
                'era': stats.get('era'),
                'whip': stats.get('whip')
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
        combined_df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        combined_df = df_new
    combined_df.drop_duplicates(subset=['gamePk', 'pitcher_id'], inplace=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)

def save_pending_game(game_pk, date):
    if not os.path.exists(PENDING_LOG):
        df = pd.DataFrame(columns=['gamePk', 'date'])
    else:
        df = pd.read_csv(PENDING_LOG)
    if game_pk not in df['gamePk'].values:
        df = pd.concat([df, pd.DataFrame([{'gamePk': game_pk, 'date': date}])])
        df.to_csv(PENDING_LOG, index=False)

def run_pitching_pipeline(start_date='2025-03-27'):
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
                    parsed_pitchers = parse_pitching_stats(game, boxscore)
                    new_data.extend(parsed_pitchers)
                    time.sleep(0.5)
                else:
                    save_pending_game(game_pk, date_str)

    if new_data:
        append_to_csv(new_data)
        print(f"✅ Added {len(new_data)} new pitching lines.")
    else:
        print("No new completed games to add.")

if __name__ == "__main__":
    run_pitching_pipeline()
