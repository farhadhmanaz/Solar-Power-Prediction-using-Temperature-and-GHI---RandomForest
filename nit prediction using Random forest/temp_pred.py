import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
#import seaborn as sns
import pprint
import random
#%matplotlib inline
import MySQLdb
import glob

df = pd.read_csv("niteditedfinal.csv") 
#print(df.head(5))

df = df.fillna(0)
nonzero_mean = df[ df!= 0 ].mean()
#df.head(5)

cols = [0,1,2,3,4,5,6]
df1 = df[df.columns[cols]]
#print(df1.head())

TestData = df1.tail(8760)
print(TestData.head())

Training_Set = df1[:-8760]
print(Training_Set.head())


Temperature = df1["Temperature"]
print[Temperature]

db = MySQLdb.connect(host="localhost", user="root",passwd="password", db="temp_database")
cur = db.cursor()
temperature = $_GET["Temperature"];
$ghi = $_GET["GHI"];
$minute = $_GET["Minute"];
$year = $_GET["hour"]
$ip = $_GET["127.0.0.7"];
$active = 1;

sql = "INSERT INTO tbl_temperature (temperature, ghi, minute, Year, ip, active) VALUES ('$temperature', '$ghi', '$minute', '$year', '$ip', '$active')"; 



cols = [0,1,2,3,4]
X = Training_Set[Training_Set.columns[cols]].values

cols = [5,6]
Y = Training_Set[Training_Set.columns[cols]].values

from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

from sklearn.svm import NuSVR
from sklearn.ensemble import RandomForestRegressor
svc = NuSVR(kernel='poly')
rfc = RandomForestRegressor()
rfc.fit(x_train,y_train)


y_pred1 = rfc.predict(x_test)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,y_pred1)
#print("Predicted Temp and Irradiation of")
val = [[2019,1,1,10,15]]
rfcp = rfc.predict(val)
print("The predicted value of temp and irradiance for", val ,"is", rfcp)
