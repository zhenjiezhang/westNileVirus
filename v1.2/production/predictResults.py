"""
Beating the Benchmark
West Nile Virus Prediction @ Kaggle
__author__ : Abhihsek
"""

import pandas as pd
import numpy as np
import math
import sys
from sklearn import ensemble, preprocessing, svm, linear_model
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeRegressor
import datetime


def computeROC(predictions, labels):
    predictions=np.array(predictions)
    labels=np.array(labels)

    threshold=[0.0001*i for i in xrange(10001)]
    TPR=[0.0 for i in xrange(len(threshold))]
    FPR=[0.0 for i in xrange(len(threshold))]
    trueNum=float(len(labels[labels==1]))
    falseNum=len(labels)-trueNum

    for i in xrange(len(threshold)):
        t=threshold[i]
        TPR[i]=len(labels[np.logical_and(predictions>=t, labels==1)])/trueNum
        FPR[i]=len(labels[np.logical_and(predictions>=t, labels==0)])/falseNum
    return TPR, FPR


def evalROC(predictions, testLabels):

    TPR,FPR=computeROC(predictions,testLabels)
    area=sum([(TPR[i]+TPR[i+1])/2*(FPR[i]-FPR[i+1]) for i in xrange(len(TPR)-1)])

    PRs=pd.DataFrame()
    PRs['TPR']=TPR
    PRs['FPR']=FPR
    PRs.to_csv('ROC_Curve.csv',index=False)
    return area

    # sample['WnvPresent'] = predictions
    # sample.to_csv('randomForestResults2007.csv', index=False)





# predictions=[0.01*i for i in xrange(101)]
# labels=[0 for i in xrange(50)]+[1 for i in xrange(51)]
# TPR,FPR=computeROC(predictions,labels)
# PRs=pd.DataFrame()
# PRs['TPR']=TPR
# PRs['FPR']=FPR
# area=sum([(TPR[i]+TPR[i+1])/2*(FPR[i]-FPR[i+1]) for i in xrange(len(TPR)-1)])
# print 'area under ROC= ', area
# PRs.to_csv('ROC_Curve.csv',index=False)
# sys.exit(0)

# leave out the year when in training, and use that year for cross-validation.
def yearDist(year1, year2):
    return 1.0/math.exp((abs(year1-year2)))
def numOfMosWeight(numOfMos, label,max):
    return 1
    # if label==0:
    #     return numOfMos
    # else:
    #     return  max-numOfMos




def makeTrainAndTest():
    train=pd.read_csv('trainFeatures.csv')  

    labels=pd.read_csv('labels.csv')

    numOfMosquitos=train.NumMosquitos.values

    test=pd.read_csv('testFeatures.csv')


    train=train.drop(['NumMosquitos'],axis=1)

    train=train.drop(['Species'],axis=1)
    test=test.drop(['Species'],axis=1)

    # test['NumMosquitos']=predictNumMos(year,40)
    # print test.NumMosquitos[:15]
    # sys.exit(0)

    


    
    # weights=np.array([yearDist(year,train.year.values[i])*numOfMosWeight(numOfMosquitos[i],labels.WnvPresent.values[i],max(numOfMosquitos)) \
        # if labels.WnvPresent.values[i]==0 else yearDist(year,train.year.values[i])*numOfMosWeight(numOfMosquitos[i],labels.WnvPresent.values[i],max(numOfMosquitos))  \
        # for i in xrange(len(labels.WnvPresent))]) 

    weights=np.array([1 for i in xrange(len(labels.WnvPresent))])




    return train.values,labels.WnvPresent.values,test.values, weights


def randomForestTraining(n_estimators=100):


    train,labels,test,weights=makeTrainAndTest()
    # sample = pd.read_csv('./sampleSubmission.csv')

    clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, min_samples_split=1,oob_score=True)



    clf.fit(train, labels, sample_weight=weights)


    prediction = clf.predict_proba(test)

    predictions=prediction[:,1]

    # month=test[:,3]
    # print month[np.logical_and(month<7, predictions>0.2)]

    # predictions[month<7]=0
    print predictions[:20]



    return predictions

def adaBoostTraining(n_estimators=100):
    train,labels,test,weights=makeTrainAndTest()

    # clf.fit(train,labels)
    # print "oob_score=", clf.oob_score_
    # score=cross_val_score(adaclf,train,labels,n_jobs=1,sample_weight=np.array([0.01 if i==0 else 0.95  for i in labels])).mean()
    # create predictions and submission file

    adaclf=ensemble.AdaBoostClassifier(n_estimators=n_estimators)
    adaclf.fit(train, labels, sample_weight=weights)
    prediction = adaclf.predict_proba(test)
    predictions=prediction[:,1]

    # month=train[:][3]
    # predictions[month<7]=0





    return predictions

def svmTraining(c=1,g=0, n_estimators=100):# n_estimators just holds the position. It does not do anything in svm.
    train,labels,test,weights=makeTrainAndTest()
    # print train[0]
    # print len(train[0])

    scaler=preprocessing.MinMaxScaler()
    train=scaler.fit_transform(train)
    test=scaler.transform(test)

    # print train[0]
    # print test[0]

    # sys.exit(0)




    c=1
    g=10**(-7)
    clf=svm.SVC(kernel='rbf',probability=True) #C=c,gamma=g)




    clf.fit(train, labels,sample_weight=weights)
    prediction=clf.predict_proba(test)
    predictions=prediction[:,1]

    month=train[:][3]
    predictions[month<7]=0

    return predictions





def runTraining(methods=['rdf'], n_estimators=100):
    
    models=[]
    try:
        for method in methods:
            models.append({
            'rdf':randomForestTraining,
            'ada':adaBoostTraining,
            'svm':svmTraining
            }[method])
    except KeyError:
        print 'Wrong method'
        sys.exit(0)

    numOfModels=len(models)

    for model in models:
        #predict according to training (without species information)
        predictions=model(n_estimators=n_estimators)

        print predictions[:100]

        #adjust predictions with species frequency of contracting Wnv
        train=pd.read_csv('trainFeatures.csv')
        labels=pd.read_csv('labels.csv').WnvPresent

        test=pd.read_csv('testFeatures.csv')

        sp=pd.concat([train.Species,test.Species],axis=0).unique()


        spWnvFraction=dict()

        normalizer=0
        for i in sp:
            spWnvFraction[i]=1.0*(np.logical_and(train.Species==i,labels==1).sum())
            normalizer+=spWnvFraction[i]
        for i in sp:
            spWnvFraction[i]=spWnvFraction[i]/(normalizer)
            print i,spWnvFraction[i], normalizer

        spFactor=[spWnvFraction[i] for i in test.Species]



        #Adjust Wnv probability with respect to individual years to account for the year to year variances
        yearScale={ 
                2008:1.0,   #1 is better than 2 and 0.5
                2010:1.0,
                2012:16.0,  #16 is beter than 8, 4, 32.  
                2014:2.0,
                }

        yearFactor=[yearScale[y] for y in test.year]



        #apply the adjustments
        predictions=predictions*spFactor*yearFactor
        predictions[predictions>1]=1




        # area=evalROC(predictions,pd.read_csv('testLabels.csv').WnvPresent.values)
        # print area
        # sys.exit(0)

        print 'output'

        sample = pd.read_csv('./sampleSubmission.csv')
        sample['WnvPresent'] = predictions
        dt=datetime.datetime.now()
        sample.to_csv(''.join(methods)+str(dt.year)+str(dt.month)+str(dt.day)+str(dt.hour)+str(dt.minute)+str(dt.second)+'.csv', index=False)






runTraining(methods=['rdf'],n_estimators=100)
# predictNumMos(2009)


