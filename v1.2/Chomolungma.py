"""
Beating the Benchmark
West Nile Virus Prediction @ Kaggle
__author__ : Chomolungma
"""

import pandas as pd
import numpy as np
import sys
from sklearn import ensemble, preprocessing
from sklearn.cross_validation import train_test_split,cross_val_score


def computeROC(predictions, labels):
	predictions=np.array(predictions)
	labels=np.array(labels)

	threshold=[0.01*i for i in xrange(101)]
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



train=pd.read_csv('trainFeatures.csv').values
labels=pd.read_csv('labels.csv').WnvPresent.values

testLabels=pd.read_csv('labels2007.csv').WnvPresent.values
test=pd.read_csv('trainFeatures2007.csv').values
sample = pd.read_csv('./sampleSubmission.csv')


clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=10, min_samples_split=1,oob_score=True)
# clf.fit(train,labels)
# print "oob_score=", clf.oob_score_
adaclf=ensemble.AdaBoostClassifier(n_estimators=100)


# score=cross_val_score(adaclf,train,labels,n_jobs=1,sample_weight=np.array([0.01 if i==0 else 0.95  for i in labels])).mean()

# print score



# create predictions and submission file

clf.fit(train, labels, sample_weight=np.array([0.05 if labels[i]==0 else 0.95  for i in xrange(len(labels))]))

# adaclf.fit(train, labels, sample_weight=np.array([0.05 if labels[i]==0 else 0.95  for i in xrange(len(labels))]))
prediction = clf.predict_proba(test)
# prediction = adaclf.predict_proba(test)
predictions=prediction[:,1]
# print len(predictions[predictions>0.5])



TPR,FPR=computeROC(predictions,testLabels)
area=sum([(TPR[i]+TPR[i+1])/2*(FPR[i]-FPR[i+1]) for i in xrange(len(TPR)-1)])
print 'area under ROC= ', area

PRs=pd.DataFrame()
PRs['TPR']=TPR
PRs['FPR']=FPR
PRs.to_csv('ROC_Curve.csv',index=False)

# sample['WnvPresent'] = predictions
# sample.to_csv('randomForestResults2007.csv', index=False)
