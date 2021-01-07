import pandas as pd
from datetime import datetime
import pytz
from .module_data import parse_team_name

from selenium import webdriver

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from .models import Match


class MatchPageScrapper():
    def __init__(self):
        options = webdriver.ChromeOptions()
        #options.add_argument('--ignore-certificate-errors')
        #options.add_argument("--test-type")
        options.add_argument("--incognito")
        #.add_argument('--headless')
        #options.add_argument("--user-data-dir=selenium") 
        options.add_argument("start-maximized")
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        #options.add_argument("--remote-debugging-port=9222")
        #options.add_argument("--window-size=1920,1080")
        #options.add_argument("--kiosk")
        self.already_visited = list(map(lambda x:x.match_id,Match.objects.all().only("match_id")))
        self.driver =webdriver.Chrome(options = options)
        self.driver.get('https://csgolounge.com/fr/')
        wait = WebDriverWait(self.driver, 10)
        wait.until(EC.presence_of_element_located((By.XPATH, "//a[@href='https://csgolounge.com/fr/login/']")))

    def is_already_scrapped(self,match_id):
        return match_id in self.already_visited

    def is_page_empty(self):
        return not len(self.driver.find_elements_by_class_name('sys-stat-koef-1'))>0

    def scrap_match_page(self,url):
        self.driver.get(url)
        self.driver.implicitly_wait(1)
        if self.is_already_scrapped(url.split('/')[-2]):
            return None, []
        elif not self.is_page_empty():
            #matches_data = pd.DataFrame(columns= ["team1",'team2','odd1','odd2','logo1','logo2','bo','date','match_id'])
            date = self.driver.find_elements_by_class_name('lounge-match-date__date')[0].get_attribute("innerHTML")
            date = datetime.strptime(date.replace(" ","")[:-3],"%d.%m.%Y,%H:%M")  
            date = pytz.timezone("Europe/Paris").localize(date, is_dst=None) 
            bo = self.driver.find_elements_by_class_name('sys-bo')[0].get_attribute('innerHTML')
            team1 = self.driver.find_elements_by_class_name('lounge-team_left')[0]
            team1_name = parse_team_name(team1.find_elements_by_class_name('lounge-team__title')[0].get_attribute('innerHTML').lower())
            team1_odd = team1.find_elements_by_class_name('sys-stat-koef-1')[0].get_attribute('innerHTML')
            team2 = self.driver.find_elements_by_class_name('lounge-team_right')[0]
            team2_name = parse_team_name(team2.find_elements_by_class_name('lounge-team__title')[0].get_attribute('innerHTML').lower())
            team2_odd = team2.find_elements_by_class_name('sys-stat-koef-2')[0].get_attribute('innerHTML')
            team1_logo = team1.find_elements_by_class_name('sys-t1logo')[0].get_attribute('src') 
            team2_logo = team2.find_elements_by_class_name('sys-t2logo')[0].get_attribute('src')
            match_id = self.driver.current_url.split('/')[-2]
            is_canceled = len(self.driver.find_elements_by_class_name('lounge-match-date_canceled'))>0
            is_rescheduled = len(self.driver.find_elements_by_class_name('lounge-match-date_rescheduled'))>0
            is_live = len(self.driver.find_elements_by_class_name('lounge-match-date_live'))>0
            winner_element = self.driver.find_elements_by_class_name('lounge-team_win')
            urls = list(map(lambda x:x.get_attribute('href'),self.driver.find_elements_by_class_name('match-history-item')))
            if url in urls:
                urls.remove(url)
            if len(winner_element)>0:
                team1_winner = 'lounge-team_left' in self.driver.find_elements_by_class_name('lounge-team_win')[0].get_attribute('class').split()
                team2_winner = 'lounge-team_right' in self.driver.find_elements_by_class_name('lounge-team_win')[0].get_attribute('class').split()
            else:
                team1_winner = False
                team2_winner = False
            winner = team1_winner * 'Team 1' + team2_winner * 'Team 2'
            has_results = len(winner)>0
            status = "Rescheduled" * is_rescheduled + "Canceled" * is_canceled + 'Results' * has_results + "Live" * is_live             

            if len(team1_odd) > 2  and len(team2_odd) > 2:
                team1_odd = float(team1_odd[1:])
                team2_odd = float(team2_odd[1:])
                row = pd.DataFrame([[team1_name,team2_name,team1_odd,team2_odd,
                                     team1_logo,team2_logo,bo,status,winner,date,match_id]],
                columns= ["team1",'team2','odd1','odd2',
                          'logo1','logo2','bo','status','winner','date','match_id'])
                team1,team2,odd1,odd2,logo1,logo2,bo,status,winner,date,match_id=row.iloc[0]
                Match(team1=team1,team2=team2,odd1=odd1,odd2=odd2,logo1=logo1,logo2=logo2,status=status,winner=winner,bo=bo,date=date,match_id=match_id).save()
                self.already_visited.append(match_id)
                return row,urls
            else:
                return None,[]

        else: 
            date = datetime.datetime.now(pytz.utc)
            status = 'Empty'
            match_id = self.driver.current_url.split('/')[-2]
            row = pd.DataFrame([['','',0,0,
                                 '','','',status,'',date,match_id]],
            columns= ["team1",'team2','odd1','odd2',
                      'logo1','logo2','bo','status','winner','date','match_id'])
            team1,team2,odd1,odd2,logo1,logo2,bo,status,winner,date,match_id=row.iloc[0]
            Match(team1=team1,team2=team2,odd1=odd1,odd2=odd2,logo1=logo1,logo2=logo2,status=status,winner=winner,bo=bo,date=date,match_id=match_id).save()
            self.already_visited.append(match_id)
            return row,[]

    def recursive_scrap(self,urls):
        if len(urls) == 0:
            return 'Done'
        else:
            for url in urls:
                data,new_urls = self.scrap_match_page(url)
                print(data)
                self.recursive_scrap(new_urls)

