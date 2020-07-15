# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:23:46 2020

@author: misak
"""

#packages
import pandas as pd 
import numpy as np
import missingno as msno
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt


from missingpy import MissForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
from sklearn.metrics import confusion_matrix
from sklearn import svm

#funkce
def rmse(predictions, actual):
    return np.sqrt(((predictions - actual) ** 2).mean())

#load DF
dataset = pd.read_csv("C:/Users/misak/Desktop/konici_projekt/sedmileti/dataset_7let.csv", error_bad_lines=False, sep=';') 
dataset.head()

dataset['sqSire'] = dataset['sireLevel']**2
dataset['sqDam'] = dataset['damLevel']**2
dataset['sqSireOfdam'] = dataset['sireOfdamLevel']**2


#summary statistics
len(dataset)

dataset.info()
dataset[["horseLevel", "damLevel", "sireLevel",  "sireOfdamLevel", "sqSire", "sqDam", "sqSireOfdam"]].describe()

#correlations
df = pd.DataFrame(dataset,columns=["horseLevel", "damLevel", "sireLevel",  "sireOfdamLevel", "sqSire", "sqDam", "sqSireOfdam"])
df.corr()

#NAs display
df = pd.DataFrame(dataset,columns=["horseLevel", "damLevel", "sireLevel",  "sireOfdamLevel"])
msno.matrix(df) 

#histotams and density plots
dataset['horseLevel'].plot.hist(bins=10, alpha=0.5)
dataset['sireLevel'].plot.hist(bins=10, alpha=0.5)
dataset['damLevel'].plot.hist(bins=10, alpha=0.5)
dataset['sireOfdamLevel'].plot.hist(bins=10, alpha=0.5)

sns.distplot(dataset['horseLevel'], hist=False, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


#random forrest imputation
imputer = MissForest()
imputedData = imputer.fit_transform(df)
imputedData = pd.DataFrame(imputedData, columns = df.columns)


#create train/test df
msk = np.random.rand(len(imputedData)) < 0.8
train = imputedData[msk]
test = imputedData[~msk]

#OLS
train['const'] = 1
reg1 = sm.OLS(endog=train['horseLevel'], exog=train[['damLevel', 'sireLevel', 'sireOfdamLevel']], 
    missing='drop')

results1 = reg1.fit()

print(results1.summary())

#predicting with OLS in sample
ypred = results1.predict(exog=train[['damLevel', 'sireLevel', 'sireOfdamLevel']])
print(ypred)

print(rmse(ypred, train['horseLevel']))

fig, ax = plt.subplots()
ax.plot(train['horseLevel'], ypred, 'ro')




#predicting with OLS out-of sample
ynewpred =  results1.predict(exog=test[['damLevel', 'sireLevel', 'sireOfdamLevel']])
print(ynewpred)

print(rmse(ynewpred, test['horseLevel']))

fig, ax = plt.subplots()
ax.plot(test['horseLevel'], ynewpred, 'ro')

#random forrests
#in sample
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train[['damLevel', 'sireLevel', 'sireOfdamLevel']], train['horseLevel'])

predictions1 = rf.predict(train[['damLevel', 'sireLevel', 'sireOfdamLevel']])
print(rmse(predictions1, train['horseLevel']))

#out-of sample
rf.fit(train[['damLevel', 'sireLevel', 'sireOfdamLevel']], train['horseLevel'])

predictions2 = rf.predict(test[['damLevel', 'sireLevel', 'sireOfdamLevel']])
print(rmse(predictions2, test['horseLevel']))


#########PREMIUM HORSES DETECTION
train['Premium'] = np.where(train['horseLevel']>=125, 1, 0)  
test['Premium'] = np.where(test['horseLevel']>=125, 1, 0)

#logit
y, X = dmatrices('Premium ~ damLevel + sireLevel + sireOfdamLevel', train, return_type = 'dataframe')


logit_model = sm.Logit(y, X)
result=logit_model.fit()
print(result.summary2())

logreg = LogisticRegression()
logreg.fit(X, y)
#predikce in sample

y_pred_prob = logreg.predict_proba(X)[:, 1]
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X, y)))

y_pred = np.where(y_pred_prob >= 0.4, 1, 0)
y_predA = pd.DataFrame()
y_predA["Premium"] = y_pred 

confusion_matrix = confusion_matrix(y, y_predA)
print(confusion_matrix)


#predikce out of sample
X1 = test.iloc[:, 1:4]
X1.insert(0, "Intercept", 1)

y1 = test.iloc[:, 4:5]

y_pred_prob1 = logreg.predict_proba(X1)[:, 1]
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X1, y1)))

y_pred1 = np.where(y_pred_prob1 >= 0.4, 1, 0)
y_predA1 = pd.DataFrame()
y_predA1["Premium"] = y_pred1 

confusion_matrix = confusion_matrix(y1, y_predA1)
print(confusion_matrix)


#SUPPORT VECTOR MACHINES
y, X = dmatrices('Premium ~ damLevel + sireLevel + sireOfdamLevel', train, return_type = 'dataframe')

# linear kernel computation
clf = svm.SVC(kernel='linear' , C = 10)
clf.fit(X,y)

trainSVM = train.iloc[:, 1:4]
trainSVM.insert(0, "Intercept", 1)

testSVM = test.iloc[:, 1:4]
testSVM.insert(0, "Intercept", 1)

# predict on training examples
SVM1 = clf.predict(trainSVM)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y, SVM1)
print(confusion_matrix)


# predict on testing examples
SVM2 = clf.predict(testSVM)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y1, SVM2)
print(confusion_matrix)


# Radial basis kernel computation
clf = svm.SVC(kernel='rbf' , C = 10)
clf.fit(X,y)


# predict on training examples
SVM1 = clf.predict(trainSVM)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y, SVM1)
print(confusion_matrix)


# predict on testing examples
SVM2 = clf.predict(testSVM)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y1, SVM2)
print(confusion_matrix)


#Polynomial kernel computation
clf = svm.SVC(kernel='poly' , C = 10)
clf.fit(X,y)


# predict on training examples
SVM1 = clf.predict(trainSVM)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y, SVM1)
print(confusion_matrix)


# predict on testing examples
SVM2 = clf.predict(testSVM)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y1, SVM2)
print(confusion_matrix)















