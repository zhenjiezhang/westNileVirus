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

def makeTrainAndTest(year):
    train=pd.read_csv('trainFeatures.csv')
    


    labels=pd.read_csv('labels.csv')


    




    labels=labels[train['year']!=year]
    train=train[train['year']!=year]
    numOfMosquitos=train.NumMosquitos[train['year']!=year].values



    testLabels=pd.read_csv('labels'+str(year)+'.csv')
    test=pd.read_csv('trainFeatures'+str(year)+'.csv')


    train=train.drop(['NumMosquitos'],axis=1)
    test=test.drop(['NumMosquitos'],axis=1)

    train=train.drop(['Species'],axis=1)
    test=test.drop(['Species'],axis=1)

    # test['NumMosquitos']=predictNumMos(year,40)
    # print test.NumMosquitos[:15]
    # sys.exit(0)

    


    
    weights=np.array([yearDist(year,train.year.values[i])*numOfMosWeight(numOfMosquitos[i],labels.WnvPresent.values[i],max(numOfMosquitos)) \
        if labels.WnvPresent.values[i]==0 else yearDist(year,train.year.values[i])*numOfMosWeight(numOfMosquitos[i],labels.WnvPresent.values[i],max(numOfMosquitos))  \
        for i in xrange(len(labels.WnvPresent))]) 

    weights=np.array([1 for i in xrange(len(labels.WnvPresent))])




    return train.values,labels.WnvPresent.values,test.values,testLabels.WnvPresent.values, weights

def yearDist(year1, year2):
    return 1.0/math.exp((abs(year1-year2)))
def numOfMosWeight(numOfMos, label,max):
    return 1
    # if label==0:
    #     return numOfMos
    # else:
    #     return  max-numOfMos

def predictNumMos(year, max_depth):
    train=pd.read_csv('trainFeatures.csv')
    train=train[train['year']!=year]

    test=pd.read_csv('trainFeatures'+str(year)+'.csv')

    trainY=train.NumMosquitos
    trainX=train.drop(['NumMosquitos'],axis=1)

    testX=test.drop(['NumMosquitos'],axis=1)



    lr=linear_model.LinearRegression()
    lr.fit(trainX.values,trainY.values)
    testY=lr.predict(testX.values)


    svr = svm.SVR(kernel='linear')
    # print list(svr.fit(trainX.values,trainY.values).predict(testX.values))

    dt=DecisionTreeRegressor(max_depth=max_depth)
    # return dt.fit(trainX,trainY).predict(testX)
    # print (dt.predict(testX))[238:245]
    # print [float(i) for i in (test.NumMosquitos.values)[238:245]]










def randomForestTraining(year, n_estimators=100):


    train,labels,test,testLabels,weights=makeTrainAndTest(year)
    # sample = pd.read_csv('./sampleSubmission.csv')

    clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, min_samples_split=1,oob_score=True)
    clf.fit(train, labels, sample_weight=weights)

    prediction = clf.predict_proba(test)

    predictions=prediction[:,1]

    # month=test[:,3]
    # print month[np.logical_and(month<7, predictions>0.2)]

    # predictions[month<7]=0


    return predictions, testLabels

def adaBoostTraining(year, n_estimators=100):
    train,labels,test,testLabels,weights=makeTrainAndTest(year)

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





    return predictions,testLabels

def svmTraining(year,c=1,g=0, n_estimators=100):# n_estimators just holds the position. It does not do anything in svm.
    train,labels,test,testLabels,weights=makeTrainAndTest(year)
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

    return predictions,testLabels



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



def runTraining(year, methods=['rdf'], n_estimators=100):
    
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

    if year>=0:

        predictions=np.array([0.0 for i in xrange(pd.read_csv('trainFeatures.csv').year.value_counts()[year])])
        for model in models:
            prediction,testLabels=model(year,n_estimators=n_estimators)
            predictions=predictions+prediction
            area=evalROC(prediction,testLabels)
            print year, ' '+model.__name__+' ', n_estimators, ': ',area
        predictions/=numOfModels

        area=evalROC(predictions,testLabels)
        print year, ' ', methods, ' ', n_estimators, ': ',area



    else:

        for model in models:
            tf=pd.read_csv('trainFeatures.csv')
            labels=pd.read_csv('labels.csv').WnvPresent
            years=tf.year.unique()
            sp=tf.Species.unique()
            places=tf.Latitude.unique()



            numOfYears=len(years)
            totalArea=0
            allPredictions=np.array([])
            allTestLabels=np.array([])
            for y in years:


                
                tf_y=tf[tf.year!=y]
                labels_y=labels[tf.year!=y]

                spWnvFraction=dict()

                for p in places:
                    spWnvFraction[p]=dict()
                    # tf_y_p=tf_y[tf_y.Latitude==p]
                    # labels_y_p=labels_y[tf_y.Latitude==p]
                    normalizer=0


                    # for i in sp:
                    #     spWnvFraction[p][i]=1.0*(np.logical_and(tf_y_p.Species==i,labels_y_p==1).sum())
                    #     normalizer+=spWnvFraction[p][i]
                    # for i in sp:
                    #     spWnvFraction[p][i]=1.0*spWnvFraction[p][i]/(normalizer) if normalizer>0 else 0
                    #     # print y, i,spWnvFraction[i], normalizer


                spWnvFraction=dict()

                normalizer=0
                for i in sp:
                    spWnvFraction[i]=1.0*(np.logical_and(tf_y.Species==i,labels_y==1).sum())
                    normalizer+=spWnvFraction[i]
                for i in sp:
                    spWnvFraction[i]=1.0*spWnvFraction[i]/(normalizer)
                    print y, i,spWnvFraction[i], normalizer








                predictions,testLabels=model(y,n_estimators=n_estimators)

                test=pd.read_csv('trainFeatures'+str(y)+'.csv')

                # lat=test.Latitude.values
                # spe=test.Species.values
                spFactor=[spWnvFraction[i] for i in test.Species]

                predictions=predictions*spFactor



                area=evalROC(predictions,testLabels)
                totalArea+=area

                scale={2007:1.89,2009:0.1,2011:0.25,2013:1.99}
                dt=pd.DataFrame()
                dt['label']=testLabels
                dt['prediction']=predictions
                dt.to_csv('predictions'+str(y)+'.csv')
                allPredictions=np.concatenate((allPredictions,predictions*scale[y]))
                allTestLabels=np.concatenate((allTestLabels,testLabels))

                print y, ' '+model.__name__+' ', n_estimators, ': ',area
            print 'Average: '+model.__name__+' ', n_estimators, ': ', totalArea/numOfYears
            print 'Assembly: '+model.__name__+' ', n_estimators, ': ', evalROC(allPredictions,allTestLabels)


scores=[]
def searchCG_SVM():
    cs=range(-5,15)
    gs=range(-15,3)
    for c in cs:
        for g in gs:
            years=pd.read_csv('trainFeatures.csv').year.unique()
            numOfYears=len(years)
            totalArea=0
            for y in years:
                predictions,testLabels=svmTraining(y,c=c,g=g)
                area=evalROC(predictions,testLabels)
                totalArea+=area
            scores.append(totalArea/numOfYears)
            print 'c= ',c, '; gamma= ',g            
            print 'Average: ', totalArea/numOfYears

# searchCG_SVM()
runTraining(-1,methods=['rdf'],n_estimators=300)
# predictNumMos(2009)


