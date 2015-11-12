"""
Beating the Benchmark
West Nile Virus Prediction @ Kaggle
__author__ : Abhihsek
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import sys

# Load dataset 
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
sample = pd.read_csv('./sampleSubmission.csv')
weather = pd.read_csv('./weather.csv')

# Get labels
labels = train.WnvPresent.values


# Not using codesum for this benchmark
weather = weather.drop('CodeSum', axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

# replace some missing values and T with -1
weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', -1)
weather = weather.replace(' T', -1)
weather = weather.replace('  T', -1)

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return int(x.split('-')[1])

def create_day(x):
    return int(x.split('-')[2])

def create_year(x):
    return int(x.split('-')[0])




train['dayofyear'] = [i.dayofyear for i in train.Date]
train['month']=[i.month for i in train.Date]
train['year']=[i.year for i in train.Date]

test['dayofyear'] = [i.dayofyear for i in test.Date]
test['month']=[i.month for i in test.Date]
test['year']=[i.year for i in test.Date]



# Add integer latitude/longitude columns
train['Lat_int'] = train.Latitude.apply(int)
train['Long_int'] = train.Longitude.apply(int)
test['Lat_int'] = test.Latitude.apply(int)
test['Long_int'] = test.Longitude.apply(int)

# drop address columns
train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos'], axis = 1)
test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)





# Merge with weather data
train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')

train = train.drop(['Date'], axis = 1)
test = test.drop(['Date'], axis = 1)



# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)

lbl.fit(list(train['Street'].values) + list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)
test['Street'] = lbl.transform(test['Street'].values)

lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
test['Trap'] = lbl.transform(test['Trap'].values)


train=train.drop(['Street','Trap','Block','AddressAccuracy'],axis=1)
test=test.drop(['Street','Trap','Block','AddressAccuracy'],axis=1)
train=train.drop(['StnPressure_x','StnPressure_y','SnowFall_x'],axis=1)
test=test.drop(['StnPressure_x','StnPressure_y','SnowFall_x'],axis=1)
train=train.drop(['DewPoint_x','DewPoint_y'],axis=1)    #可有可无
test=test.drop(['DewPoint_x','DewPoint_y'],axis=1)    #可有可无
train=train.drop(['SeaLevel_x','SeaLevel_y'],axis=1)
test=test.drop(['SeaLevel_x','SeaLevel_y'],axis=1)
#train=train.drop(['Tavg_x','Tavg_y'],axis=1)    #有用参数
train=train.drop(['Tmin_x','Tmin_y'],axis=1) #没大影响
test=test.drop(['Tmin_x','Tmin_y'],axis=1) #没大影响
train=train.drop(['Tmax_x','Tmax_y'],axis=1)    #这个参数反而不重要，取消了好。
test=test.drop(['Tmax_x','Tmax_y'],axis=1)    #这个参数反而不重要，取消了好。
# train=train.drop(['Heat_x','Heat_y'],axis=1)    #不太确定，影响不大。
#train=train.drop(['Cool_x','Cool_y'],axis=1)        #不太确定，影响不大，好像留着好些。
train=train.drop(['ResultSpeed_x','ResultSpeed_y'],axis=1)#去掉的好
test=test.drop(['ResultSpeed_x','ResultSpeed_y'],axis=1)#去掉的好
train=train.drop(['AvgSpeed_x','AvgSpeed_y'],axis=1)#去掉的好
test=test.drop(['AvgSpeed_x','AvgSpeed_y'],axis=1)#去掉的好
train=train.drop(['ResultDir_x','ResultDir_y'],axis=1)#去掉后，对2007,2009改善很大，对后两年没大影响，还有一点点下降。和喷药有关？
test=test.drop(['ResultDir_x','ResultDir_y'],axis=1)#去掉后，对2007,2009改善很大，对后两年没大影响，还有一点点下降。和喷药有关？




# drop columns with -1s
train = train.ix[:,(train != -1).any(axis=0)]
test = test.ix[:,(test != -1).any(axis=0)]

# Random Forest Classifier 
clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=300, min_samples_split=1)

clf.fit(train, labels)

# create predictions and submission file
predictions = clf.predict_proba(test)[:,1]


yearScale={ 2008:1.0, 
            2010:0.5,
            2012:2.0,
            2014:1.0
            }

yearFactor=[yearScale[int(y)] for y in test.year]



#apply the adjustments
predictions=predictions*yearFactor

sample['WnvPresent'] = predictions
sample.to_csv('beat_the_benchmark.csv', index=False)
