from django.db.models import Q
from django.utils import timezone

from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from .models import Match, Prediction, Bet
from .module_data import parse_team_name, FeaturesBuilder

import pytz
import time
import pandas as pd

from datetime import datetime


# clf = load('xgb.joblib')

DATA_DIR = 'var/'


# rankings= get_rankings()


def decision(prediction, strategy):

    if strategy == 'EV':
        ev1 = prediction.delta_ev_1
        ev2 = prediction.delta_ev_2
        if ev1 > ev2 and ev1 > 0:
            decision = 1
        elif ev2 > ev1  and ev2 > 0:
            decision = 0
        else:
            decision = -1
    elif strategy == 'Naive':
        odd1 = prediction.odd1
        odd2 = prediction.odd2
        decision = odd1 < odd2
    else:
        decision = None
    return decision


def compute_time_to_go(matches):
    """
    Compute the time to go before the match starts
    """
    for match in matches:
        delta_t = match.date - timezone.now()
        match.time_to_go = ""
        if delta_t.days > 0:
            match.time_to_go += str(int(delta_t.days)) +'d '
        if delta_t.seconds//3600 > 0:
            match.time_to_go += str((delta_t.seconds//3600)) +'h '
        match.time_to_go += str((delta_t.seconds//60)%60) +'m '


class AutoBettor():
    def __init__(self, clf):
        options = webdriver.ChromeOptions()
        options.add_argument('--ignore-certificate-errors')
        options.add_argument("--test-type")
        options.add_argument("--user-data-dir=driver_data") 
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--remote-debugging-port=9222")
        self.driver = webdriver.Chrome(options = options)
        self.driver.get('https://csgolounge.com/fr/')
        self.FB = FeaturesBuilder()
        self.clf = clf

    def auto_predict(self,fast=False):
        """
        Place bets using decision rules based on different strategies
        """
        if fast:
            five_min = timezone.now() + timezone.timedelta(seconds=5*60)
            predictions = Prediction.objects.filter(match__status='Scheduled',
                                                    match__date__lte=five_min)
        else:
            query = Q(match__status="Scheduled")
            predictions = Prediction.objects.filter(query)

        for pred in predictions:
            decision_ev = decision(pred, 'EV')

            if decision_ev == 1:
                bet, created = Bet.objects.get_or_create(match=pred.match,strategy='EV')
                bet.winner = 'Team 1'
                if not bet.real:
                    bet.amount = 1
                bet.save()
                pred.save()
                print(bet,
                      "new"*created + "not new" * (1-created),
                      f"with strategy : {bet.strategy}")
            elif decision_ev == 0:
                bet, created = Bet.objects.get_or_create(match=pred.match,strategy='EV')
                bet.winner = 'Team 2'
                if not bet.real:
                    bet.amount = 1
                bet.save()
                pred.save()
                print(bet,
                      "new"*created + "not new" * (1-created),
                      f"with strategy : {bet.strategy}")
            elif decision_ev == -1:
                print(f'Not a good bet {pred.match}')

            decision_naive = decision(pred, 'Naive')
            if decision_naive == 1:
                bet, created = Bet.objects.get_or_create(match=pred.match, strategy='Naive')
                bet.winner = 'Team 1'
                if not bet.real:
                    bet.amount = 1
                bet.save()
                print(bet,
                      "new"*created + "not new" * (1-created),
                      f"with strategy : {bet.strategy}")

            elif decision_naive == 0:
                bet, created = Bet.objects.get_or_create(match=pred.match,strategy='Naive')
                bet.winner = 'Team 2'
                if not bet.real:
                    bet.amount = 1
                bet.save()
                print(bet,
                      "new"*created + "not new" * (1-created),
                      f"with strategy : {bet.strategy}")

        bets = Bet.objects.filter(status='Pending')
        for bet in bets:
            if bet.match.status == 'Canceled' or bet.match.status =='Rescheduled':
                bet.delete()
        self.update_bet(fast=fast)

    def update_bet(self,fast=False):
        """
        Compute the results of bets : the winning team and the gain associated
        """
        if fast:
            five_min = timezone.now() + timezone.timedelta(seconds=5*60)
            bets = Bet.objects.filter(match__status='Results', match__date__lte=five_min)

        else:
            query1 = (Q(match__status='Results') | Q(match__status='Expired'))
            query2 = Q(status='Finished') | Q(status='Canceled')
            bets = Bet.objects.filter(query1).exclude(query2)
        for bet in bets:
            match = bet.match
            if match.status == 'Results':
                if len(bet.winner):
                    if bet.winner == 'Team 1': 
                        win = (match.winner == 'Team 1')
                        lose = 1-win
                        bet.gain = bet.amount *((match.odd1-1)* win - lose)
                    elif bet.winner == 'Team 2':
                        win = (match.winner == 'Team 2')
                        lose = 1-win
                        bet.gain = bet.amount *((match.odd2-1)* win - lose)
                    bet.status = 'Finished'
            elif match.status == 'Expired':
                bet.status = 'Canceled'

            bet.save(force_update=True)
        self.update_matches()

    def update_matches(self):
        """
        Check if matches list is outdated and assign the status 'expired' to outdated matches
        """
        history_button = self.driver.find_elements_by_class_name("sys-history-tab")[0]
        history_button.click()
        matches = self.driver.find_elements_by_class_name("lounge-bets-items__item")
        for match in matches:
            date = match.find_elements_by_class_name('lounge-match-date__date')[0].get_attribute("innerHTML")
            date = datetime.strptime(date.replace(" ","")[:-3],"%d.%m.%Y,%H:%M")
            date = pytz.timezone("Europe/Paris").localize(date, is_dst=None)
            bo = match.find_elements_by_class_name('sys-bo')[0].get_attribute('innerHTML')
            team1 = match.find_elements_by_class_name('lounge-team_left')[0]
            team1_name = parse_team_name(team1.find_elements_by_class_name('lounge-team__title')[0].get_attribute('innerHTML'))
            team1_logo = team1.find_elements_by_class_name('sys-t1logo')[0].get_attribute('src')
            team2 = match.find_elements_by_class_name('lounge-team_right')[0]
            team2_name = parse_team_name(team2.find_elements_by_class_name('lounge-team__title')[0].get_attribute('innerHTML'))
            team2_logo = team2.find_elements_by_class_name('sys-t2logo')[0].get_attribute('src')
            team1_odd = team1.find_elements_by_class_name('sys-stat-koef-1')[0].get_attribute('innerHTML')
            team2_odd = team2.find_elements_by_class_name('sys-stat-koef-2')[0].get_attribute('innerHTML')
            is_rescheduled = 'bet-rescheduled' in match.get_attribute('class').split()
            is_canceled = 'bet-canceled' in match.get_attribute('class').split()
            has_results = 'bet-result' in match.get_attribute('class').split()
            is_live = 'bet-now' in match.get_attribute('class').split()
            status = "Rescheduled" * is_rescheduled + "Canceled" * is_canceled + 'Results' * has_results + "Live" * is_live 
            if not len(status):
                status = 'Scheduled'
            right_win = 'bet-winner-right' in match.get_attribute('class').split()
            left_win = 'bet-winner-left' in match.get_attribute('class').split()
            winner = left_win* 'Team 1' + right_win* 'Team 2'

            if len(team1_odd) > 2 and len(team2_odd) > 2:
                team1_odd = float(team1_odd[1:])
                team2_odd = float(team2_odd[1:])
                match_id = int(match.get_attribute('data-id'))

                Match(team1=team1_name,team2=team2_name,odd1=team1_odd,odd2=team2_odd,
                      logo1=team1_logo,logo2=team2_logo,
                      status=status, winner=winner,
                      bo=bo,date=date,match_id=match_id).save()

        twelve_hours_ago = timezone.now() - timezone.timedelta(hours=12)
        Match.objects.filter(Q(date__lte = twelve_hours_ago) & (Q(status = "Scheduled")|Q(status="Live")) ).update(status='Expired')


    def scrap_upcoming_matches(self):
        wait = WebDriverWait(self.driver, 10)
        wait.until(EC.presence_of_element_located((By.XPATH, "//a[@href='/fr/checkin/']")))
        matches_data = pd.DataFrame(columns= ["team1",'team2','odd1','odd2','logo1','logo2','bo','status','winner','date','match_id'])
        total_button = self.driver.find_elements_by_class_name("sys-total-tab")[0]
        total_button.click()
        matches = self.driver.find_elements_by_class_name("lounge-bets-items__item")
        for match in matches:
            date = match.find_elements_by_class_name('lounge-match-date__date')[0].get_attribute("innerHTML")
            date = datetime.strptime(date.replace(" ","")[:-3],"%d.%m.%Y,%H:%M")
            date = pytz.timezone("Europe/Paris").localize(date, is_dst=None)
            bo = match.find_elements_by_class_name('sys-bo')[0].get_attribute('innerHTML')
            team1 = match.find_elements_by_class_name('lounge-team_left')[0]
            team1_name = parse_team_name(team1.find_elements_by_class_name('lounge-team__title')[0].get_attribute('innerHTML'))
            team1_logo = team1.find_elements_by_class_name('sys-t1logo')[0].get_attribute('src')
            team2 = match.find_elements_by_class_name('lounge-team_right')[0]
            team2_name = parse_team_name(team2.find_elements_by_class_name('lounge-team__title')[0].get_attribute('innerHTML'))
            team2_logo = team2.find_elements_by_class_name('sys-t2logo')[0].get_attribute('src')
            team1_odd = team1.find_elements_by_class_name('sys-stat-koef-1')[0].get_attribute('innerHTML')
            team2_odd = team2.find_elements_by_class_name('sys-stat-koef-2')[0].get_attribute('innerHTML')
            is_rescheduled = 'bet-rescheduled' in match.get_attribute('class').split()
            is_canceled = 'bet-canceled' in match.get_attribute('class').split()
            has_results = 'bet-result' in match.get_attribute('class').split()
            is_live = 'bet-now' in match.get_attribute('class').split()
            status = "Rescheduled" * is_rescheduled + "Canceled" * is_canceled + 'Results' * has_results + "Live" * is_live 
            if not len(status):
                status = 'Scheduled'
            right_win = 'bet-winner-right' in match.get_attribute('class').split()
            left_win = 'bet-winner-left' in match.get_attribute('class').split()
            winner = left_win* 'Team 1' + right_win* 'Team 2'

            if len(team1_odd) > 2  and len(team2_odd) > 2:
                team1_odd = float(team1_odd[1:])
                team2_odd = float(team2_odd[1:])
                match_id = int(match.get_attribute('data-id'))
                row = pd.DataFrame([[team1_name,team2_name,team1_odd,team2_odd,team1_logo,team2_logo,bo,status,winner,date,match_id]],
                                   columns= ["team1",'team2','odd1','odd2','logo1','logo2','bo','status','winner','date','match_id'])
                matches_data = matches_data.append(row,ignore_index=True)

        return matches_data

    def save_predictions(self, predictions, model='SVM'):
        for i in range(len(predictions)):
            pred = predictions.iloc[i]
            match_id = pred['match_id']
            odd_p_1 = float(pred['odd_p_1'])
            odd_p_2 = float(pred['odd_p_2'])

            match = Match.objects.filter(match_id=match_id)[0]
            delta_ev_1 = float(match.odd1)/odd_p_1 - 1
            delta_ev_2 = float(match.odd2)/odd_p_2 - 1

            Prediction(match=match, odd1=odd_p_1, odd2=odd_p_2,
                       delta_ev_1=delta_ev_1,
                       delta_ev_2=delta_ev_2, model=model).save()

    def predict(self, features):
        probas = self.clf.predict_proba(features.drop('match_id', axis=1))
        eps = 1e-8
        features['odd_p_1'] = 1/(probas[:, 0] + eps)
        features['odd_p_2'] = 1/(probas[:, 1] + eps)
        return features

    def auto_bet(self,amount='0.5'):
        i = 1
        while 1:
            now = datetime.now()
            print(f'Starting iteration {i} at {now.hour}h{now.minute}')
            try:
                matches_data = self.scrap_upcoming_matches()
                for i in range(matches_data.shape[0]):
                    match = matches_data.iloc[i]
                    team1,team2,odd1,odd2,logo1,logo2,bo,status,winner,date,match_id=match
                    Match(team1=team1,team2=team2,odd1=odd1,odd2=odd2,
                          logo1=logo1,logo2=logo2,
                          status=status, winner=winner,
                          bo=bo,date=date,match_id=match_id).save()
                print('Upcoming matches : ')
                print(matches_data.iloc[:10])
            except IndexError:
                print("Can't scrap matches")
            # m = 5
            # to_m_min = timezone.now() + timezone.timedelta(seconds=m*60)
            # for_m_min = timezone.now() - timezone.timedelta(seconds=m*60)

            features = self.FB.get_features(matches_data,
                                            features=['odd1', 'odd2',
                                                      'date', 'elo1',
                                                      'elo2', 'F_S_team1',
                                                      'F_S_team2', 'match_id'])
            predictions = self.predict(features)
            self.save_predictions(predictions)
            self.auto_predict()
            print('Sleeping for 30s')
            for j in range(30):
                print('#'*j + ' ' + str((j*100)//30) +'%')
                time.sleep(1)
