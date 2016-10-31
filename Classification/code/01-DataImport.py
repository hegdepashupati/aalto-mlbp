
from mlclassifiers import LogisticRegression,ConfusionMatrix

import pandas as pd
import random


'''
Import data and declare features
'''
rawdata = pd.read_csv("/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/classification_dataset_training.csv")
target = ["rating"]
features = ['but','good','place','food','great','very','service','back','really','nice',
            'love','little','ordered','first','much','came','went','try','staff','people',
            'restaurant','order','never','friendly','pretty','come','chicken','again','vegas',
            'definitely','menu','better','delicious','experience','amazing','wait','fresh','bad',
            'price','recommend','worth','enough','customer','quality','taste','atmosphere','however',
            'probably','far','disappointed']

# Test and training split
random.seed(1234)
trainvec = random.sample(range(0,rawdata.shape[0]),round(rawdata.shape[0]*0.7))
traindf = rawdata.loc[trainvec,]
testdf = rawdata.loc[set(range(0,rawdata.shape[0]))-set(trainvec),]



'''
model 1 - logistic regression
'''
logreg = LogisticRegression()
logreg.train(traindf,target,features)

traindf['pred1'] = logreg.predict(newdata=traindf,type="class")
testdf['pred1'] = logreg.predict(newdata=testdf,type="class")

ConfusionMatrix(traindf['rating'],traindf['pred1'])
ConfusionMatrix(testdf['rating'],testdf['pred1'])

