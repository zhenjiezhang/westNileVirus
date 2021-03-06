#To combine multiple rows on the same count into single rows, and if possible, try to replicate rows with wnv to reflect the guessed number of carrier mosquitos


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
        wnv+=train[i][-1]*50/train[i][-2]

        i+=1

    if wnv==0:
        parsedTrain.append(train[record_num])
        parsedTrain[-1][-2]=numMosq

    # print numMosq, len(parsedTrain)-1
    # print parsedTrain.NumMosquitos[len(parsedTrain)-1]
    # print "writing",numMosq, parsedTrain.NumMosquitos[len(parsedTrain)-1]

    else:
        for w in xrange(wnv):
            parsedTrain.append(train[record_num])
            parsedTrain[-1][-2]=numMosq
            parsedTrain[-1][-1]=1


        # parsedTrain[-1][-1]=(0 if wnv==0 else 1)
    # parsedTrain[-1][-1]=wnv

parsedTrain=pd.DataFrame(parsedTrain)
parsedTrain.columns=columns



parsedTrain.to_csv('parsedTrain.csv',index=False)

