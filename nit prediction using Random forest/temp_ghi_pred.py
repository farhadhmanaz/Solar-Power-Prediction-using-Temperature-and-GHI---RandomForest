import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
#import seaborn as sns
import pprint
import time
from time import sleep
import random

df = pd.read_csv("niteditedfinal.csv")
df = df.fillna(0)
nonzero_mean = df[ df!= 0 ].mean()
#df.head(5)

cols = [0,1,2,3,4]
X = df[df.columns[cols]].values

cols = [5]
Y_temp = df[df.columns[cols]].values

cols = [6]
Y_ghi = df[df.columns[cols]].values

from sklearn.model_selection import  train_test_split

x_train,x_test,y_temp_train,y_temp_test = train_test_split(X,Y_temp)
x_train,x_test,y_ghi_train,y_ghi_test = train_test_split(X,Y_ghi)

#print(x_train.shape,x_test.shape,y_temp_train.shape,y_temp_test.shape)
#print(x_train.shape,x_test.shape,y_ghi_train.shape,y_ghi_test.shape)


from sklearn.ensemble import RandomForestRegressor

rfc1 = RandomForestRegressor()
rfc2 = RandomForestRegressor()

rfc1.fit(x_train,y_temp_train)
rfc2.fit(x_train,y_ghi_train)

y_temp_pred_rfc = rfc1.predict(x_test)
y_ghi_pred_rfc = rfc2.predict(x_test)


data = pd.date_range('2021-04-10 11:00:00', '2021-04-11 11:00:00',freq ='15min' )
#print("date : ", data)

df3 = pd.DataFrame({'date':data})
#print(df3)

dataset = df3
dataset["Month"] = pd.to_datetime(dataset["date"]).dt.month
dataset["Year"] = pd.to_datetime(dataset["date"]).dt.year
dataset["Date"] = pd.to_datetime(dataset["date"]).dt.date
dataset["day"] = pd.to_datetime(dataset["date"]).dt.day

dataset["Hour"] = pd.to_datetime(dataset["date"]).dt.time
dataset["ho"] = pd.to_datetime(dataset["date"]).dt.hour
dataset["mi"] = pd.to_datetime(dataset["date"]).dt.minute
#print(dataset.head())

cols = [2,1,4,6,7]
cr_x = dataset[dataset.columns[cols]]
#print(cr_x.head())

cr_x = cr_x.values

d = []
for x in cr_x:
    c = rfc1.predict([x])
    #time.sleep(10)
    d.append(c)

twenty_temp = d

import pickle 

pickle_out = open("twenty_temp.pickle","wb")
pickle.dump(twenty_temp,pickle_out)
pickle_out.close()
pickle_in = open("twenty_temp.pickle","rb")
twenty_temp = pickle.load(pickle_in)
twenty_temp = np.array(twenty_temp)
print(twenty_temp)

temp = pd.DataFrame(twenty_temp, 
            columns=['Temperature'])
print(temp)

#prediction of ghi

p = []
for x in cr_x:
    c = rfc2.predict([x])
    #time.sleep(10)
    p.append(c)

twenty_ghi = p

pickle_out = open("twenty_ghi.pickle","wb")
pickle.dump(twenty_ghi,pickle_out)
pickle_out.close()
pickle_in = open("twenty_ghi.pickle","rb")
twenty_ghi = pickle.load(pickle_in)
twenty_ghi = np.array(twenty_ghi)
ghi = pd.DataFrame(twenty_ghi, 
            columns=['GHI'])
print(ghi)

#P = ηSI [1 − 0.05(T− 25)]
#η = Panel efficiency(0.18) S = Panel Area(7.4322) I = Irradiance T = Temperature
#P= 0.187.4322I(1-0.05(T-25))
#f = 0.18*7.4322*twenty_ghi*(1-0.05*(twenty_temp-25))
twenty_ghi = twenty_ghi.flatten()
twenty_temp = twenty_temp.flatten()


f = 0.18*7.4322*twenty_ghi
insi = twenty_temp - 25
midd = 0.95*insi

power = f* midd
power = pd.DataFrame(power, 
             columns=['power'])

print("Power: ", power)

result = pd.concat([temp,ghi, power], axis=1, ignore_index=False)
result = pd.concat([result,dataset],axis = 1, ignore_index = False)
print(result)


    

#result.to_csv('results_2020.csv')







