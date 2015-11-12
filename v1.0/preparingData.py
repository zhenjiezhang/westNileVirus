import pandas as pd
import numpy as np

train=pd.read_csv('train.csv',parse_dates=['Date'])
columns=train.columns
train=train.as_matrix()
n=len(train)

parsedTrain=[]

i=0
while i<n:
    record_num=i
    wnv=train[record_num][-1]
    numMosq=train[record_num][-2]
    # print numMosq, wnv
    i+=1

    while i<n and train[i][0]==train[record_num][0] and train[i][2]==train[record_num][2] and train[i][7]==train[record_num][7] and \
    train[i][8]==train[record_num][8]:
        numMosq+=train[i][-2]
        wnv+=train[i][-1]
        i+=1

    parsedTrain+=[train[record_num]]
    # print numMosq, len(parsedTrain)-1
    # print parsedTrain.NumMosquitos[len(parsedTrain)-1]
    # print "writing",numMosq, parsedTrain.NumMosquitos[len(parsedTrain)-1]

    parsedTrain[-1][-2]=numMosq
    parsedTrain[-1][-1]=(0 if wnv==0 else 1)

parsedTrain=pd.DataFrame(parsedTrain)
parsedTrain.columns=columns




parsedTrain=parsedTrain.sort(['Date','Address','WnvPresent'],ascending=[1, 1, 0]).as_matrix()

finalTrain=[]
i=0
n=len(parsedTrain)
while i<n:
    record=parsedTrain[i]
    date=record[0]
    address=record[1]

    Wnv=record[-1]

    finalTrain.append(parsedTrain[i])

    i+=1
    while i<n and parsedTrain[i][0]==date and parsedTrain[i][1]==address:
        if parsedTrain[i][-1]==Wnv:
            finalTrain.append(parsedTrain[i])
        i+=1

finalTrain=pd.DataFrame(finalTrain)
finalTrain.columns=columns





finalTrain.to_csv('parsedTrain.csv',index=False)

