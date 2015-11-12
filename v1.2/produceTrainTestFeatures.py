import pandas as pd
import numpy as np
from sklearn import preprocessing
import math, sys



def computeNumMosq(year):
    train=pd.read_csv('./parsedTrain.csv',parse_dates=['Date'])
    train['year']=[i.year for i in train.Date]
    train['dayofyear'] = [i.dayofyear for i in train.Date]

    yearTrain=train[train.year==year].values
    otherYearTrain=train[train.year!=year].values

    years=train.year.unique()
    years=years[years!=year]
    for rec in yearTrain:
        samePlace=otherYearTrain[otherYearTrain[1]==yearTrain[1]]




# Load dataset 
train = pd.read_csv('./parsedTrain.csv',parse_dates=['Date'])
test = pd.read_csv('./test.csv',parse_dates=['Date'])
weather = pd.read_csv('./weather.csv',parse_dates=['Date'])


# Get labels

# Not using codesum for this benchmark

#the CodeSum is not useful...
# CodeSumSet=set()
# for cs in weather.CodeSum:
#     CodeSumSet=CodeSumSet|set(cs.split())

# CodeSumSet-={'FU','FG+','VCFG','BCFG','VCTS','FG','GR','MIFG','SQ','SN'}
# CodeSumSet-={'DZ','BR','TSRA','TS','HZ','RA'}

# for cs in CodeSumSet:
#     cs_values=pd.Series(np.array([0 for i in range(len(weather.CodeSum))]))
#     for i in range(len(weather.CodeSum)):
#         if cs in weather['CodeSum'][i].split():
#             cs_values[i]=1
#     weather[cs]=cs_values
#     print cs,': ', sum(cs_values)


# use LabelEncoder and OneHotEncoder to transform string features to binary numerical tuples.
lbl = preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()
# lbl.fit(list(weather['CodeSum'].values))
# weather['CodeSum'] = lbl.transform(weather['CodeSum'].values)

# allFeatures=list(weather['CodeSum'].values)
# allFeatures=[[i] for i in allFeatures]
# ohe.fit(allFeatures)
# weather=pd.concat([weather,pd.DataFrame(ohe.transform([[i] for i in (list(weather['CodeSum'].values))]).toarray())],axis=1)

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




for c in weather.columns:
    if type(weather[c][0])==type(''):
        weather[c]=weather[c].apply(float)


# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.

# def create_month(x):
#     return x.split('-')[1]

# def create_day(x):
#     return x.split('-')[2]

train['dayofyear'] = [i.dayofyear for i in train.Date]
train['month']=[i.month for i in train.Date]
train['year']=[i.year for i in train.Date]

# train=train[train.year!=2011]




# test['year'] = [i.year for i in test.Date]

# Add integer latitude/longitude columns
# train['Lat_int'] = train.Latitude.apply(int)
# train['Long_int'] = train.Longitude.apply(int)



# drop address columns
NumMosquitos=list(train.NumMosquitos)
train = train.drop(['AddressNumberAndStreet'], axis = 1)
test = test.drop(['Id', 'AddressNumberAndStreet'], axis = 1)

# Merge with weather data


train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')

train = train.drop(['Date'], axis = 1)
test = test.drop(['Date'], axis = 1)


# train=train.drop(['Latitude','Longitude'],axis=1)
# test=test.drop(['Latitude','Longitude'],axis=1)




# Convert categorical data to numbers





# lbl.fit(list(train['Species'].values) + list(test['Species'].values))
# train['Species'] = lbl.transform(train['Species'].values)
# test['Species'] = lbl.transform(test['Species'].values)

# allFeatures=list(train['Species'].values) + list(test['Species'].values)
# allFeatures=[[i] for i in allFeatures]
# ohe.fit(allFeatures)
# train=pd.concat([train,pd.DataFrame(ohe.transform([[i] for i in (list(train['Species'].values))]).toarray())],axis=1)
# test=pd.concat([test,pd.DataFrame(ohe.transform([[i] for i in (list(test['Species'].values))]).toarray())],axis=1)

# train=train.drop(['Species'],axis=1)




# lbl.fit(list(train['Street'].values) + list(test['Street'].values))
# train['Street'] = lbl.transform(train['Street'].values)
# test['Street'] = lbl.transform(test['Street'].values)

# allFeatures=list(train['Street'].values) + list(test['Street'].values)
# allFeatures=[[i] for i in allFeatures]
# ohe.fit(allFeatures)
# train=pd.concat([train,pd.DataFrame(ohe.transform([[i] for i in (list(train['Street'].values))]).toarray())],axis=1)
# test=pd.concat([test,pd.DataFrame(ohe.transform([[i] for i in (list(test['Street'].values))]).toarray())],axis=1)


# lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
# train['Trap'] = lbl.transform(train['Trap'].values)
# test['Trap'] = lbl.transform(test['Trap'].values)

# allFeatures=list(train['Trap'].values) + list(test['Trap'].values)
# allFeatures=[[i] for i in allFeatures]
# ohe.fit(allFeatures)
# train=pd.concat([train,pd.DataFrame(ohe.transform([[i] for i in (list(train['Trap'].values))]).toarray())],axis=1)
# test=pd.concat([test,pd.DataFrame(ohe.transform([[i] for i in (list(test['Trap'].values))]).toarray())],axis=1)


# lbl.fit(list(train['Address'].values) + list(test['Address'].values))
# train['Address'] = lbl.transform(train['Address'].values)
# test['Address'] = lbl.transform(test['Address'].values)

# allFeatures=list(train['Address'].values) + list(test['Address'].values)
# allFeatures=[[i] for i in allFeatures]
# ohe.fit(allFeatures)
# train=pd.concat([train,pd.DataFrame(ohe.transform([[i] for i in (list(train['Address'].values))]).toarray())],axis=1)
# test=pd.concat([test,pd.DataFrame(ohe.transform([[i] for i in (list(test['Address'].values))]).toarray())],axis=1)




train=train.drop(['Street','Trap','Address','Block','AddressAccuracy'],axis=1)
test=test.drop(['Street','Trap','Address','Block'],axis=1)


train = train.ix[:,(train != -1).any(axis=0)]
test = test.ix[:,(test != -1).any(axis=0)]




lat1,long1=41.995,-87.933
lat2,long2=41.786, -87.752

stat1Dist=((train.Latitude-lat1)**2+(train.Longitude-long1)**2)**0.5
stat2Dist=((train.Latitude-lat2)**2+(train.Longitude-long2)**2)**0.5

# weatherInfo=['ResultDir']
# for info in weatherInfo:
#     train[info]=(train[info+'_x']*stat2Dist+train[info+'_y']*stat1Dist)/(stat1Dist+stat2Dist)
    # train=train.drop([info+'_x',info+'_y'],axis=1)

# train['Tdiff']=train['Tmax']-train['Tmin']






train=train.drop(['StnPressure_x','StnPressure_y','SnowFall_x'],axis=1)
# train=train.drop(['Depth_x'],axis=1)    #好像不弄掉好些
# train=train.drop(['DewPoint_x','DewPoint_y'],axis=1)    #差不多
# train=train.drop(['WetBulb_x','WetBulb_y'],axis=1)    #差不多

train['Sunrise_x']=(train.Sunrise_x/100).apply(int)*60+train.Sunrise_x%100
train['Sunset_x']=(train.Sunset_x/100).apply(int)*60+train.Sunset_x%100

train['DayTime']=train['Sunset_x']-train['Sunrise_x']
# train=train.drop(['Sunrise_x'],axis=1)        #重要参数，不可以取消a= sunset和sunrise互相可以部分代替，可能日照时长是真正有意义的参数？主要影响2011 和2013
# train=train.drop(['Sunset_x'],axis=1)        #重要参数，不可以取消

# train=train.drop(['PrecipTotal_x','PrecipTotal_y'],axis=1)    #这个参数不出意外地很重要

train=train.drop(['DewPoint_x','DewPoint_y'],axis=1)    #可有可无
train=train.drop(['SeaLevel_x','SeaLevel_y'],axis=1)
#train=train.drop(['Tavg_x','Tavg_y'],axis=1)    #有用参数
train=train.drop(['Tmin_x','Tmin_y'],axis=1) #没大影响
train=train.drop(['Tmax_x','Tmax_y'],axis=1)    #这个参数反而不重要，取消了好。
# train=train.drop(['Heat_x','Heat_y'],axis=1)    #不太确定，影响不大。
#train=train.drop(['Cool_x','Cool_y'],axis=1)        #不太确定，影响不大，好像留着好些。
train=train.drop(['ResultSpeed_x','ResultSpeed_y'],axis=1)#去掉的好
train=train.drop(['AvgSpeed_x','AvgSpeed_y'],axis=1)#去掉的好
train=train.drop(['ResultDir_x','ResultDir_y'],axis=1)#去掉后，对2007,2009改善很大，对后两年没大影响，还有一点点下降。和喷药有关？

# train=train.drop(['dayofyear','month'],axis=1)




train=train.sort(['Latitude','Longitude','year','dayofyear','WnvPresent'])
train.index=range(len(train.Latitude))





lastDayTime=np.array(train['DayTime'])
lastSunrise_x=np.array(train['Sunrise_x'])
lastSunset_x=np.array(train['Sunset_x'])
lastTavg_x=np.array(train['Tavg_x'])
lastTavg_y=np.array(train['Tavg_y'])
lastPrecipTotal_x=np.array(train['PrecipTotal_x'])
lastPrecipTotal_y=np.array(train['PrecipTotal_y'])





for i in xrange(1,len(train.Latitude)):
    if train.Latitude[i]==train.Latitude[i-1] and train.Longitude[i]==train.Longitude[i-1] and train.year[i]==train.year[i-1]:
        if train.dayofyear[i]!=train.dayofyear[i-1]:
            lastDayTime[i]=train.DayTime[i-1]
            lastSunrise_x[i]=train.Sunrise_x[i-1]
            lastSunset_x[i]=train.Sunset_x[i-1]
            lastTavg_x[i]=train.Tavg_x[i-1]
            lastTavg_y[i]=train.Tavg_y[i-1]
            lastPrecipTotal_x[i]=train.PrecipTotal_x[i-1]
            lastPrecipTotal_y[i]=train.PrecipTotal_y[i-1]
        else:
            lastDayTime[i]=lastDayTime[i-1]
            lastSunrise_x[i]=lastSunrise_x[i-1]
            lastSunset_x[i]=lastSunset_x[i-1]
            lastTavg_x[i]=lastTavg_x[i-1]
            lastTavg_y[i]=lastTavg_y[i-1]
            lastPrecipTotal_x[i]=lastPrecipTotal_x[i-1]
            lastPrecipTotal_y[i]=lastPrecipTotal_y[i-1]


        

train['lastDayTime']=lastDayTime

# train['lastSunrise_x']=lastSunrise_x
# train['lastSunset_x']=lastSunset_x
# train['lastTavg_x']=lastTavg_x
# train['lastTavg_y']=lastTavg_y
# train['lastPrecipTotal_x']=lastPrecipTotal_x
# train['lastPrecipTotal_y']=lastPrecipTotal_y


        

# # make secondLastSurvey features:
secondLastDayTime=np.array(train['lastDayTime'])
# secondLastSunrise_x=np.array(train['lastSunrise_x'])
# secondLastSunset_x=np.array(train['lastSunset_x'])
# secondLastPrecipTotal_x=np.array(train['PrecipTotal_x'])
# secondLastPrecipTotal_y=np.array(train['PrecipTotal_y'])



for i in xrange(2,len(train.Latitude)):
    if train.Latitude[i]==train.Latitude[i-2] and train.Longitude[i]==train.Longitude[i-2] and train.year[i]==train.year[i-2]:
        if train.dayofyear[i]!=train.dayofyear[i-1]:
            secondLastDayTime[i]=train.DayTime[i-2]
    #         secondLastSunrise_x[i]=train.Sunrise_x[i-2]
    #         secondLastSunset_x[i]=train.Sunset_x[i-2]
    #         secondLastPrecipTotal_x[i]=train.PrecipTotal_x[i-2]
    #         secondLastPrecipTotal_y[i]=train.PrecipTotal_y[i-2]
        else:
            secondLastDayTime[i]=secondLastDayTime[i-1]



# 
train['secondLastDayTime']=secondLastDayTime
# train['secondLastSunrise_x']=secondLastSunrise_x
# train['secondLastSunset_x']=secondLastSunset_x
# train['secondLastPrecipTotal_x']=secondLastPrecipTotal_x
# train['secondLastPrecipTotal_y']=secondLastPrecipTotal_y


# train['lastTavg_x']=lastTavg_x
# train['lastTavg_y']=lastTavg_y
# train['lastPrecipTotal_x']=lastPrecipTotal_x
# train['lastPrecipTotal_y']=lastPrecipTotal_y

# train=train.sort(['year','dayofyear'])

# train=train.drop(['DayTime','Sunrise_x','Sunset_x'],axis=1)



labels = pd.DataFrame(train.WnvPresent)
train=train.drop(['WnvPresent'],axis=1)


years=train.year.unique()

# for year in years:




for year in years:

    yearlyTrain=train[train.year==year]
    yearlyLabels=labels[train.year==year]
    columns=yearlyLabels.columns
    yearlyTrain['WnvPresent']=yearlyLabels
    yearlyTrain=yearlyTrain.drop_duplicates()

    yearlyLabels=pd.DataFrame(yearlyTrain['WnvPresent'].values)
    yearlyLabels.columns=columns
    print yearlyLabels.columns

    yearlyTrain=yearlyTrain.drop(['WnvPresent'],axis=1)

    print len(yearlyLabels),len(yearlyTrain)
    



    print (yearlyLabels==1).sum()

    yearlyTrain.to_csv('trainFeatures'+str(year)+'.csv',index=False)
    yearlyLabels.to_csv('labels'+str(year)+'.csv',index=False)


n=len(train)
columns=train.columns
train=train.as_matrix()
labels=labels.WnvPresent.values

newTrain=[]
newLabels=[]
i=0
while i < n:
    record_num=i
    newTrain.append(train[i])
    newLabels.append(labels[i])
    
    i+=1
    while i<n and train[i][1]==train[i-1][1] and train[i][2]==train[i-1][2] and train[i][4]==train[i-1][4] and train[i][6]==train[i-1][6]:
        if labels[i]>=1:
            # for j in xrange(11):
            newTrain.append(train[i])
            newLabels.append(labels[i])
        else:
            if record_num==0:
                newTrain.append(train[i])
                newLabels.append(labels[i])
        i+=1

train=pd.DataFrame(newTrain)
train.columns=columns
labels=pd.DataFrame(newLabels)
labels.columns=['WnvPresent']
print train.Species.unique()
print len(train)




train.to_csv('trainFeatures.csv',index=False)
labels.to_csv('labels.csv',index=False)
test.to_csv('testFeatures.csv',index=False)



