from mlclassifiers import LogisticRegression, ConfusionMatrix, KfoldCV, DecisionTree
import pandas as pd
import random
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

'''
Import data and declare features
'''
rawdata = pd.read_csv(
    "/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/classification_dataset_training.csv")
target = ['rating']
features = ['but', 'good', 'place', 'food', 'great', 'very', 'service', 'back', 'really', 'nice',
            'love', 'little', 'ordered', 'first', 'much', 'came', 'went', 'try', 'staff', 'people',
            'restaurant', 'order', 'never', 'friendly', 'pretty', 'come', 'chicken', 'again', 'vegas',
            'definitely', 'menu', 'better', 'delicious', 'experience', 'amazing', 'wait', 'fresh', 'bad',
            'price', 'recommend', 'worth', 'enough', 'customer', 'quality', 'taste', 'atmosphere', 'however',
            'probably', 'far', 'disappointed']

# Test and training split
random.seed(1234)
trainvec = random.sample(range(0, rawdata.shape[0]), round(rawdata.shape[0] * 0.7))
traindf = rawdata.loc[trainvec,]
validationdf = rawdata.loc[set(range(0, rawdata.shape[0])) - set(trainvec),]

'''
model 1 - logistic regression
'''
# train model
logreg = LogisticRegression(alpha=0.01)
logreg.train(traindf, target, features)

# prediction on train and test data
traindf['pred_LR'] = logreg.predict(newdata=traindf, type="class")
validationdf['pred_LR'] = logreg.predict(newdata=validationdf, type="class")

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LR'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LR'])

# crossvalidation summary
KfoldCV(model=logreg, data=rawdata, target=target, features=features, k=5)

'''
model 2 - logitic regression with L2 regularization
'''
# train model
logreg = LogisticRegression(alpha=0.01, L2regular=True, lambreg=0.1)
logreg.train(traindf, target, features)

# prediction on train and test data
traindf['pred_LRL2'] = logreg.predict(newdata=traindf, type="class")
validationdf['pred_LRL2'] = logreg.predict(newdata=validationdf, type="class")

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL2'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL2'])

# crossvalidation summary
KfoldCV(model=logreg, data=rawdata, target=target, features=features, k=5)

'''
model 3 - logistic regression with L1 regularization
'''
from sklearn import linear_model

y = traindf[target]
x = traindf[features]

logreg = linear_model.LogisticRegression(C=0.1, penalty='l1', tol=0.01)

# train model
logreg.fit(x, y)

# prediction on train and test data
traindf['pred_LRL1'] = logreg.predict(traindf[features])
validationdf['pred_LRL1'] = logreg.predict(validationdf[features])

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL1'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL1'])

fpr, tpr, _ = roc_curve(validationdf['rating'], validationdf['pred_LRL1'])
plt.plot(fpr, tpr, label='Logistic Regression with L1', color="red")

'''
model 3 - decision trees
'''
# train model
dt = DecisionTree(maxdepth=5, minnodesize=10, minnodeprop=0.01)
dt.train(traindf, target, features)

# prediction on train and test data
traindf['pred_LR'] = dt.predict(newdata=traindf, type="class")
validationdf['pred_LR'] = dt.predict(newdata=validationdf, type="class")

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LR'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LR'])

fpr, tpr, _ = roc_curve(traindf['rating'], traindf['pred_LR'])
plt.plot(fpr, tpr, label='RF')

# crossvalidation summary
KfoldCV(model=dt, data=rawdata, target=target, features=features, k=5)

'''
scikitlearn decision tree
'''
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

y = traindf[target]
x = traindf[features]
dt = DecisionTreeClassifier(random_state=99, criterion='entropy', max_depth=4, min_samples_split=20,
                            min_samples_leaf=10)
dt.fit(x, y)

dot_data = export_graphviz(dt, feature_names=features, out_file="dt.dot")
graph = pydotplus.graph_from_dot_file("dt.dot")
graph.write_pdf("dt.pdf")

traindf['pred'] = dt.predict_proba(traindf[features])
validationdf['pred'] = dt.predict(validationdf[features])

ConfusionMatrix(traindf['rating'], traindf['pred'])
ConfusionMatrix(validationdf['rating'], validationdf['pred'])

'''
ROC curve
'''
fpr, tpr, _ = roc_curve(traindf['rating'], traindf['pred_LR'])
plt.plot(fpr, tpr, label='logistic reg')

fpr, tpr, _ = roc_curve(traindf['rating'], traindf['pred_LRL2'])
plt.plot(fpr, tpr, label='logistic reg with L2 regularisation')

fpr, tpr, _ = roc_curve(traindf['rating'], traindf['pred_LRL1'])
plt.plot(fpr, tpr, label='logistic reg with L1 regularisation')
