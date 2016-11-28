import pandas as pd
import random

'''
Import data and declare features
'''
rawdf = pd.read_csv(
    "/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/train.csv")
testdf = pd.read_csv(
    "/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/test.csv")
target = ['rating']
orgfeatures = ['but', 'good', 'place', 'food', 'great', 'very', 'service', 'back', 'really', 'nice',
               'love', 'little', 'ordered', 'first', 'much', 'came', 'went', 'try', 'staff', 'people',
               'restaurant', 'order', 'never', 'friendly', 'pretty', 'come', 'chicken', 'again', 'vegas',
               'definitely', 'menu', 'better', 'delicious', 'experience', 'amazing', 'wait', 'fresh', 'bad',
               'price', 'recommend', 'worth', 'enough', 'customer', 'quality', 'taste', 'atmosphere', 'however',
               'probably', 'far', 'disappointed']
allfeatures = list(set(rawdf.columns.values) - set(['ID', 'rating']))
newfeatures = [feature for feature in allfeatures if feature.endswith("_m")]

# Test and training split
random.seed(1234)
trainvec = random.sample(range(0, rawdf.shape[0]), round(rawdf.shape[0] * 0.7))
traindf = rawdf.loc[trainvec,]
validationdf = rawdf.loc[set(range(0, rawdf.shape[0])) - set(trainvec),]

from patsy import dmatrices, ModelDesc, Term, LookupFactor
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

'''
define plotting functions
'''


def PlotCVROC(df, target, features, clf):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle

    # generate scikilt learn structures
    formula = ModelDesc([Term([LookupFactor(target[0])])], [Term([LookupFactor(c)]) for c in features])
    y, x = dmatrices(formula, df, return_type="dataframe")
    y = np.array(y.values.flatten())

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=3)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['#1ca099', '#ffa500', '#ffff00'])
    lw = 2

    i = 0
    for (train, test), color in zip(cv.split(x, y), colors):
        probas_ = clf.fit(x.loc[train], y[train]).predict_proba(x.loc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='Fold %d (area = %0.3f)' % (i, roc_auc),
                 alpha=0.4)

        i += 1
    plt.plot([0, 1], [0, 1], linestyle=':', lw=lw, color='k',
             label='Random guess', alpha=0.4)

    mean_tpr /= cv.get_n_splits(x, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='#e60000', linestyle='--',
             label='Mean (area = %0.3f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    plt.legend(loc="lower right")
    fig = plt.gcf()

    return fig


# plot weights for different features for different values of C
def FindOptimalLambda(df, target, features, clf, fname="test.png", crange=np.logspace(2, -6, 20)):
    # generate scikilt learn structures
    formula = ModelDesc([Term([LookupFactor(target[0])])], [Term([LookupFactor(c)]) for c in features])
    y, x = dmatrices(formula, df, return_type="dataframe")
    y = np.array(y.values.flatten())

    # create a dataframe to store features weights for different c values
    summdf = pd.DataFrame(columns=features, dtype='float', index=(1 / crange))

    for c in crange:
        clf.set_params(C=c)
        clf.fit(x, y)
        coeffdf = pd.DataFrame({'coeff': np.transpose(clf.coef_).flatten()}, index=x.columns).transpose()
        summdf.loc[1 / c] = coeffdf.loc['coeff']

    summdf.plot(legend=False, logx=True)
    fig = plt.gcf()
    fig.axes[0].set_xlabel(r"$\lambda$")
    fig.axes[0].set_ylabel("weights")
    fig.savefig(fname)

    return "Done"


def ModelSummary(rawdf, target, features, clf, fname="test.png"):
    formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in features])
    y, x = dmatrices(formula, rawdf, return_type="dataframe")
    y = y.values.flatten()

    clf.fit(x, y)

    plt.figure()
    plot = PlotCVROC(rawdf, target, features, logreg)
    plt.savefig(fname)

    scores = cross_val_score(clf, x, y, cv=5)
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(clf, x, y, cv=5, scoring='roc_auc')
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return "done"


'''
model 0 - original features, no regularisation
'''
logreg = linear_model.LogisticRegression(penalty='l2', C=10e10)
ModelSummary(rawdf, target, orgfeatures, logreg, fname="00_noregularisation_of.png")

'''
model 1 - original features, L1 regularization
'''
logreg = linear_model.LogisticRegression(penalty='l1')
ModelSummary(rawdf, target, orgfeatures, logreg, fname="01_L1regularisation_of.png")
FindOptimalLambda(rawdf, target, orgfeatures, logreg, fname="01_L1regularisation_of_wm.png")

'''
model2 - original features, L2 regularization
'''
logreg = linear_model.LogisticRegression(penalty='l2')
ModelSummary(rawdf, target, orgfeatures, logreg, fname="02_L2regularisation_of.png")
FindOptimalLambda(rawdf, target, orgfeatures, logreg, fname="02_L2regularisation_of_wm.png")

'''
model 3 - original L1 selected features, L2 regularisation
'''
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in orgfeatures])
y, x = dmatrices(formula, rawdf, return_type="dataframe")
y = y.values.flatten()

logreg = linear_model.LogisticRegressionCV(penalty='l1', solver='liblinear')
logreg.fit(x, y)
print(logreg.C_)

coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
nflist = coeffdf[coeffdf.coeff != 0].feature.values.tolist()
print(len(nflist))

logreg = linear_model.LogisticRegression(penalty='l2')
ModelSummary(rawdf, target, nflist, logreg, fname="03_L2regularisation_L1selectedf.png")

'''
model 4 - higher features, no regularisation
'''
logreg = linear_model.LogisticRegression(penalty='l2', C=10e10)
ModelSummary(rawdf, target, allfeatures, logreg, fname="04_noregularisation_hf.png")

'''
model 5 - higher features, L1 regularization
'''
logreg = linear_model.LogisticRegression(penalty='l1')
ModelSummary(rawdf, target, allfeatures, logreg, fname="05_L1regularisation_hf.png")
FindOptimalLambda(rawdf, target, allfeatures, logreg, fname="05_L1regularisation_hf_wm.png")

'''
model 6 - higher features, L2 regularization
'''
logreg = linear_model.LogisticRegression(penalty='l2')
ModelSummary(rawdf, target, allfeatures, logreg, fname="06_L2regularisation_hf.png")
FindOptimalLambda(rawdf, target, allfeatures, logreg, fname="06_L2regularisation_hf_wm.png")

'''
model 7 - higher L1 selected features, L2 regularisation
'''
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in allfeatures])
y, x = dmatrices(formula, rawdf, return_type="dataframe")
y = y.values.flatten()

logreg = linear_model.LogisticRegression(C=0.1, penalty='l1', tol=0.01)
logreg.fit(x, y)

coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
nflist = coeffdf[coeffdf.coeff != 0].feature.values.tolist()
print(len(nflist))

logreg = linear_model.LogisticRegression(penalty='l2')
ModelSummary(rawdf, target, nflist, logreg, fname="07_L2regularisation_L1selectedf.png")

# classification summary
testdf['pred'] = (np.array([pred[0] for pred in logreg.predict_proba(testdf[nflist])]) < 0.5) * 1

from mlclassifiers import ConfusionMatrix
from sklearn.metrics import roc_auc_score

ConfusionMatrix(testdf['rating'], testdf['pred'])

testdf['pred'] = 1 - np.array([pred[0] for pred in logreg.predict_proba(testdf[nflist])])
roc_auc_score(testdf['rating'], testdf['pred'])
