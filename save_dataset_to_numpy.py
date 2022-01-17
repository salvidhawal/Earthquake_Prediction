import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from datetime import timezone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

df = pd.read_csv(f"dataset/Earthquake_Predictions.csv")

df = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

timestamp = []
date_stamp = []


def unix_timestamp(given_date):
    unix_date = datetime.datetime(1970, 1, 1, 0, 0, 0)
    r_days = (given_date - unix_date).days
    r_sec = (given_date - unix_date).seconds
    unix_time = (r_days * 86400) + r_sec
    return unix_time


i = 1
for d, t in zip(df['Date'], df['Time']):
    if "T" in d:
        date_time = d.split("T")
        d = date_time[0]
        t1 = date_time[1].split(".")
        t = t1[0]
        datetimeobject = datetime.datetime.strptime(d, '%Y-%m-%d')
        d = datetimeobject.strftime('%m/%d/%Y')

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

timeStamp = pd.Series(timestamp)
df['Timestamp'] = timeStamp.values
final_df = df.drop(['Date', 'Time'], axis=1)

print(final_df.head())

X = final_df[['Timestamp', 'Latitude', 'Longitude']]
y = final_df[['Magnitude', 'Depth']]

#scaler = MaxAbsScaler()
#scaler.fit(X)
#X = scaler.transform(X)

#X["Timestamp"] = MaxAbsScaler().fit_transform(np.array(X["Timestamp"]).reshape(-1,1))
#X["Timestamp"] = X["Timestamp"] / 180.0

column = X["Timestamp"]
max_value = column.max()
print(max_value)

X["Timestamp"] = X["Timestamp"] / max_value
X["Latitude"] = X["Latitude"] / 180
X["Longitude"] = X["Longitude"] / 180

#scaler = MinMaxScaler()
#scaler.fit(y)
#y = scaler.transform(y)

column = y["Depth"]
max_value = column.max()
print(max_value)

y["Magnitude"] = y["Magnitude"] / 10
y["Depth"] = y["Depth"] / max_value

print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
print(X_train.shape, X_test.shape, y_train.shape, X_test.shape)



np.save("dataset/X_train", X_train)
np.save("dataset/X_test", X_test)
np.save("dataset/y_train", y_train)
np.save("dataset/y_test", y_test)