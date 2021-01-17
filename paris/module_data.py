import pandas as pd
import numpy as np
import sqlite3
import datetime
import pytz
from . import module_api

DATA_DIR = './var/'


def add_to_history(matches, verbose=0):
    n = matches.shape[0]
    for i in range(n):
        match = matches.iloc[i]
        if match['status'] != 'finished':
            if verbose:
                print("Match not finished yet, can't add to history")
        else:
            row = [0]*5
            row[0], found_1 = parse_team_name(match['opponents'][0]['opponent']['name'],
                                              return_found=True)
            row[1], found_2 = parse_team_name(match['opponents'][1]['opponent']['name'],
                                              return_found=True)
            if not found_1:
                with open('var/teams_map.csv', 'a', encoding = 'utf-8') as f:
                    f.write(row[0]+';\n')
                print(f'{row[0]} added to map')
                f.close()
            if not found_2:
                with open('var/teams_map.csv', 'a', encoding = 'utf-8') as f:
                    f.write(row[1]+';\n')
                f.close()
                print(f'{row[1]} added to map')

            row[2] = match['scheduled_at'].to_pydatetime()
            score = str(match['results'][0]['score'])
            score += " - " + str(match['results'][1]['score'])
            row[3] = (score)

            try:
                row[4] = parse_team_name(match['winner']['name'])

            except Exception:
                #print(match['winner'])
                row[4] = None
            finally:
                with sqlite3.connect(DATA_DIR + 'history.db') as conn:
                    c = conn.cursor()
                    infos = tuple(row)
                    query = """INSERT or IGNORE INTO
                               history VALUES (?,?,?,?,?)"""
                    c.execute(query, infos)
                    conn.commit()
                conn.close()
    


def parse_team_name(el, return_found=False):
    rem = ['team', 'e-sports', 'esports', 'esport',
           'gaming', 'club', 'clan', 'international']
    my_map = pd.read_csv(DATA_DIR + 'teams_map.csv', sep=';').values.tolist()

    def team_map(x, map_, return_found=False):
        x = x.strip()
        found = False
        for val in map_:
            if x in val and not found:
                found = True
                x = val[0]
        if return_found:
            return x, found
        else:
            return x

    l = el.strip().lower().split(' ')
    for word in rem:
        if word in l:
            l.remove(word)
            el = " ".join(l)
    return team_map(el.lower(), my_map, return_found)


def init_db(reboot=False):
    if reboot:
        query = """DROP TABLE history"""
        with sqlite3.connect(DATA_DIR + 'history.db') as conn:
            c = conn.cursor()
            c.execute(query)
            conn.commit()
        conn.close()

    query = """
                CREATE TABLE history(
                team1 TEXT,
                team2 TEXT,
                date TEXT,
                score TEXT,
                winner TEXT,
                UNIQUE(team1, team2, date, score, winner)
            );"""
    with sqlite3.connect(DATA_DIR+'history.db') as conn:
        c = conn.cursor()
        c.execute(query)
        conn.commit()
    conn.close()


def get_history():
    with sqlite3.connect(DATA_DIR+'history.db') as conn:
        history = pd.read_sql_query("SELECT * from history", conn)
    conn.close()
    return history


def parse_date(date):
    if type(date) == str:
        date = datetime.datetime.strptime(date,'%Y-%m-%d %H:%M:%S%z')
    return date




class FeaturesBuilder:

    def __init__(self, update_history=False, token=None,
                 recompute_ranking=False, recompute_elo=False):
        if update_history:
            if not token:
                print("Pls specify a token if you want the history to be updated")
                return None
            module_api.update_history(token)
        self.history = get_history().drop_duplicates()
        self.history = self.history[~self.history['winner'].isnull()]
        self.history.date = pd.to_datetime(self.history.date, utc=True)
        self.history = self.history.reset_index(drop=True)
        if recompute_ranking:
            self.ranking = self.compute_ranking(self.history)
            self.ranking.to_csv(DATA_DIR+'ranking.csv')
        else:
            self.ranking = pd.read_csv(DATA_DIR + 'ranking.csv', index_col=0)
            self.ranking.index = pd.to_datetime(self.ranking.index)

        if recompute_elo:
            self.history_elo = self.get_matches_elo(self.history)
            self.history_elo.to_csv(DATA_DIR + 'history_elo.csv')
        else:
            self.history_elo = pd.read_csv(DATA_DIR + 'history_elo.csv',
                                           index_col=0).drop_duplicates()
            self.history_elo.date = pd.to_datetime(self.history_elo.date,
                                                   utc=True)

    # Module elo computing
    def find_date(self, date):
        dates = self.ranking.index
        before = np.where(dates <= date)
        if len(before) > 0:
            t = before[0][-1]
        return dates[t]

    def compute_ranking(self, history):
        t0 = history.date.min()
        now = datetime.datetime.now(tz=pytz.utc)
        weeks = (now-t0).days // 7
        dates = ([pd.to_datetime(t0) +
                 datetime.timedelta(days=i*7) for i in range(weeks)])
        teams = pd.Series(pd.concat((history.team1,
                                     history.team2), axis=0).unique())
        elo = pd.Series(np.ones(len(teams))*1400)
        elo_ranking = pd.DataFrame(columns=['team', 'elo', 'date'])
        elo_ranking.team = teams
        elo_ranking.elo = elo
        elo_ranking.date = t0
        elo_ranking = elo_ranking.set_index(['date', 'team'])
        elo_ranking = elo_ranking.unstack('team').elo
        for i in range(weeks-1):
            matches = (history[(dates[i] < history['date'])
                       & (history['date'] < dates[i+1])])
            if dates[i+1] not in elo_ranking.index:
                elo_vect = elo_ranking.loc[[dates[i]]].copy()
                elo_vect.index = [dates[i+1]]
                for j in range(len(matches)):
                    match = matches.iloc[j]
                    team1 = match.team1
                    team2 = match.team2
                    elo1 = elo_vect[team1]
                    elo2 = elo_vect[team2]
                    p1 = 1/(1+10**((elo2-elo1)/400))
                    p2 = 1/(1+10**((elo1-elo2)/400))
                    S1 = 40*((match['winner'] == team1) - p1)
                    S2 = 40*((match['winner'] == team2) - p2)
                    elo1 += S1
                    elo2 += S2
                    elo_vect[team1] = elo1
                    elo_vect[team2] = elo2
                elo_ranking = elo_ranking.append(elo_vect)
        return elo_ranking

    def get_teams_elo(self, teams, date):
        t = self.find_date(date)
        try:
            teams_elo = self.ranking.loc[t, teams]
        except KeyError:
            print(f'One or both teams not in the ranking : {teams}')
            teams_elo = [-99, -99]
        return teams_elo

    def get_match_elo(self, match):
        elo1, elo2 = self.get_teams_elo([match.team1, match.team2], match.date)
        return elo1, elo2

    def get_matches_elo(self, matches):
        elo1 = np.zeros(len(matches)) - 99
        elo2 = np.zeros(len(matches)) - 99
        for i in range(len(matches)):
            elo1[i], elo2[i] = self.get_match_elo(matches.iloc[i])
        mask = (elo1 != -99) & (elo2 != -99)
        matches = matches[mask].copy()
        elo1 = elo1[mask]
        elo2 = elo2[mask]
        matches['elo1'] = elo1
        matches['elo2'] = elo2
        return matches

    # Module fitness computing
    def get_team_history(self,team,date,rows=5):
        date = datetime.datetime(date.year,date.month,date.day,tzinfo=datetime.timezone.utc)
        team_history = self.history_elo[(self.history_elo['date']<date)&((self.history_elo['team2']==team)|(self.history_elo['team1']==team))]
        return team_history.sort_values('date',ascending=False).head(rows)

    def compute_matches_teams_fitness(self,matches):
        n = len(matches)
        fitness1 = np.zeros(n) - 99
        fitness2 = np.zeros(n) - 99
        for i in range(n):
            fitness1[i], fitness2[i] = (self.
                                        compute_single_match_teams_fitness
                                        (matches.iloc[i]))

        matches['F_S_team1'] = fitness1
        matches['F_S_team2'] = fitness2
        return matches

    def compute_single_match_teams_fitness(self, match):
        rows = 5
        team1_history = self.get_team_history(match.team1, match['date'], rows)
        team2_history = self.get_team_history(match.team2, match['date'], rows)
        #print(team1_history,team2_history)
        team1_FS = 0
        team2_FS = 0
        team1 = match.team1
        team2 = match.team2
        elo1 = match.elo1
        elo2 = match.elo2
        for i in range(len(team1_history)):
            team1_history_match = team1_history.iloc[i]
            team1_wins = int(team1_history_match['winner']==team1)
            if team1_history_match.team1 == match.team1:
                p1 = 1/(1+10**(np.abs(team1_history_match.elo2-elo1)/200))
                team1_FS += 10*(team1_wins - p1) #/(t**(1/2)+1e-8)
            elif team1_history_match.team2 == match.team1:
                p1 = 1/(1+10**(np.abs(team1_history_match.elo1-elo1)/200))
                team1_FS += 10*(team1_wins - p1) #/(t**(1/2)+1e-8)
        for i in range(len(team2_history)):
            team2_history_match = team2_history.iloc[i]
            if team2_history_match.team1 == match.team2:
                p2 = 1/(1+10**(np.abs(team2_history_match.elo2-elo2)/200))
                team2_FS += 10*((team2_history_match['winner']==team2) - p2) #/(t**(1/2)+1e-8)
            elif team2_history_match.team2 == match.team2:
                p2 = 1/(1+10**(np.abs(team2_history_match.elo1-elo2)/200))
                team2_FS += 10*((team2_history_match['winner']==team2) - p2) #/(t**(1/2)+1e-8)
            
        return team1_FS, team2_FS
    # Module antecedent computing

    # Module features

    def get_features(self, matches,
                     features=['odd1', 'odd2', 'date', 'elo1',
                               'elo2', 'F_S_team1', 'F_S_team2'],
                     target=False):

        matches_elo = self.get_matches_elo(matches)
        matches_features = self.compute_matches_teams_fitness(matches_elo)
        print(matches_features.loc[:, ['team1', 'team2', 'odd1',
                                       'odd2', 'elo1', 'elo2',
                                       'F_S_team1', 'F_S_team2']])
        if target:
            if 'status' in matches_features.columns:
                matches_features['y'] = (matches_features['winner']
                                         == 'Team 1').astype(int)
            #else:
            #    matches_features['y'] = ((matches_features['winner']
            #                             == matches_features['team1'])
            #                             .astype(int))

        return matches_features.loc[:, features + target * ['y']]
