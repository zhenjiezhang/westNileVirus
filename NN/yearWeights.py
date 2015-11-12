import pandas as pd


ans=pd.read_csv("west_nile_v55044.csv")
test=pd.read_csv("test.csv",parse_dates=["Date"])

test['year']=[d.year for d in test.Date]

weights={
    2008: 1.0,
    2010: 1.0,
    2012: 16.0,
    2014: 2.0
}

weightedPrediction=[]
for y, p in zip(test.year.values,ans.WnvPresent.values):
    weightedPrediction.append(weights[y]*p)

top=max(weightedPrediction)
weightedPrediction=[i/top for  i in weightedPrediction]

ans.WnvPresent=weightedPrediction

print max(ans.WnvPresent)


ans.to_csv("weightedPrediction.csv",index=False)




