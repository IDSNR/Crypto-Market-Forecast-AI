import requests
import pytrends
import pandas as pd
from pytrends.request import TrendReq
import time
from requests.exceptions import Timeout, ConnectionError
import json



def get_google_trends_data(keyword, beginig_date, finishing_date):
    timeframe_var = str(beginig_date)+" "+str(finishing_date)

    keywords = [keyword]

    pytrends = TrendReq()
    pytrends.build_payload(keywords,timeframe=timeframe_var)
    trend_data = pytrends.interest_over_time()
    data = trend_data[keyword]

    return data.to_json("database_SOL_2.json")

print(get_google_trends_data("Crypto", "2024-01-25", "2024-01-28"))