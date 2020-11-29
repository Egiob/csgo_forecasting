import pandas as pd
from urllib.request import urlopen, Request
import json
import numpy as np
import sqlite3
import datetime
from bs4 import BeautifulSoup
import re
from django.templatetags.static import static
from .module_data import add_to_history


def get_teams(token):
    """
    Calls PandasScore API following endpoints : /csgo/teams
    return the response as a jsons
    """
    per_page = 100
    response=[]
    for i in range(50):  
        page=i
        request = urlopen(f"https://api.pandascore.co/csgo/teams?token={token}&per_page={per_page}&page={page}")
        response += json.load(request)
    return response

def get_matches_history(token,page_number=100):
    """
    Calls PandasScore API following endpoints : /csgo/matches
    return the response as a jsons
    """
    per_page = 100
    response=[]
    for i in range(page_number):
        page = i
        request = urlopen(f"https://api.pandascore.co/csgo/matches?token={token}&page={page}&per_page={per_page}")
        response += json.load(request)
    return response



def update_history(token):
    matches_json = get_matches_history(token)
    matches = pd.read_json(StringIO(json.dumps(matches_json)))
    add_to_history(matches)
