import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from datetime import timezone

df = pd.read_csv(f"dataset/Earthquake_Predictions.csv")

df = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

timestamp = []
date_stamp = []


def unix_timestamp(given_date):
    unix_date = datetime.datetime(1970, 1, 1,0,0,0)
    r_days = (given_date - unix_date).days
    r_sec = (given_date - unix_date).seconds
    unix_time = (r_days * 86400) + r_sec
    return unix_time


i = 1
for d, t in zip(df['Date'], df['Time']):
    '''
    try:
        date_list = d.split('/')
        date_list = list(map(int, date_list))

        time_list = t.split(':')
        time_list = list(map(int, time_list))

        ts = datetime.datetime(date_list[2], date_list[0], date_list[1], time_list[0], time_list[1], time_list[2])
        
        if date_list[2] >= 1970:
            timestamp.append(time.mktime(ts.timetuple()))
        else:
            timestamp.append(unix_timestamp(given_date=ts))
    except ValueError:
        print({ts})
        timestamp.append('ValueError')
    '''
    date_list = d.split('/')
    date_list = list(map(int, date_list))

    time_list = t.split(':')
    time_list = list(map(int, time_list))

    ts = datetime.datetime(date_list[2], date_list[0], date_list[1], time_list[0], time_list[1], time_list[2])

    if date_list[2] >= 1970:
        timestamp.append(time.mktime(ts.timetuple()))
    else:
        timestamp.append(unix_timestamp(given_date=ts))

print(timestamp)
