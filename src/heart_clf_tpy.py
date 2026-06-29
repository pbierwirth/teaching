#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 13:21:48 2026

@author: philipp
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.compose import ColumnTransformer

import os
os.environ['MKL_VERBOSE'] = '0'

data = pd.read_csv('/Users/philipp/Downloads/heart.csv')
data = data.drop_duplicates()

y = data.target
y = y*-1+1
x = data.drop('target', axis=1)

#%%
x.info()

unique_values = {i: len(x[i].value_counts()) for i in x.columns}
cat_idx = [i for i,j in enumerate(unique_values) if unique_values[j] < 6]
con_idx = [i for i,j in enumerate(unique_values) if unique_values[j] > 6]

# get continious values
conX = x.iloc[:, con_idx]

#%% calculate effect sizes for continous 
def get_d(a,b, bootstrap = False, permu = 10000):
    if bootstrap==True:
        d_b = []
        for _ in range(permu):
            idx_a = np.random.randint(0,len(a),size=len(a))
            idx_b = np.random.randint(0,len(b),size=len(b)) 
            d_b.append(
                (a[idx_a].mean()-b[idx_b].mean()) /
                np.concatenate((a[idx_a],b[idx_b])).std()
                )   
        d = {'mean': np.array(d_b).mean(), 
             'std': np.array(d_b).std()}
    else:
        d = (a.mean()-b.mean())/pd.concat((a,b)).std()
    return d


effect_sizes = {i: get_d(np.array(conX[i].loc[y==1]),
       np.array(conX[i].loc[y==0]), bootstrap=True)  for i in conX.columns}

fig = plt.figure();
ax = fig.add_subplot()
for i,n in enumerate(effect_sizes):
    ax.bar(i+1, effect_sizes[n]['mean'], 
           edgecolor='black', color='#2c8cbe', width=0.67)
    ax.errorbar(i+1, effect_sizes[n]['mean'],
                 effect_sizes[n]['std'], color='black')

ax.set_xticks(np.linspace(1,len(effect_sizes),len(effect_sizes)))
ax.set_xticklabels(effect_sizes.keys())
ax.set_ylabel('Cohens d')
ax.axhline(0,color='black', linewidth=1)

# %% calculate AUC and accuracy for single predictor
def roc_auc(x,y):
    idx = np.argsort(x)
    sorted_y = y[idx][::-1]
    
    tp = np.cumsum(sorted_y)/np.sum(y==1)
    fp = np.cumsum(sorted_y==0)/np.sum(y==0)
    
    auc = np.sum(np.diff(fp) * 0.5*(tp[:-1] + tp[1:]))
    
    if auc < 0.5:
        auc = 1-auc
    
    return auc,fp, tp

auc = {cond: roc_auc(conX[cond] ,np.array(y)) for i,cond in enumerate(conX)}

# %% multivariavte approach
# clf = XGBClassifier()
# clf = GradientBoostingClassifier()
# clf = GaussianNB()

x = np.array(x)
y = np.array(y)

clf = TabPFNClassifier() 
folds = KFold(n_splits=5)

auc = []
for train_idx, test_idx in folds.split(x):
   
    clf.fit(x[train_idx,:], y[train_idx]);
    y_hat = clf.predict(x[test_idx])
    
    auci,_,_=roc_auc(x[test_idx,:], y_hat)
    auc.append(auci)

print(f'mean AUC is: {np.array(auc).mean()}')

#%%
clf = SVC(kernel='rbf', C = 0.01)
clf = GaussianNB()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('fit', clf)
    ])

cv = cross_validate(pipeline, conX, y, cv=10, scoring='roc_auc')
cv['test_score'].mean()




#%% 
pipeline_numeric = Pipeline([
    ('scaler', MinMaxScaler()),
     ('pca', PCA(n_components=1))
    ])
pipeline_categorical = Pipeline([
    ('ohe', OneHotEncoder(drop='first'))
    ])

ct = ColumnTransformer([
    ('numeric_transformer', pipeline_numeric, con_idx),
    ('cat_transformer', pipeline_categorical, cat_idx)
    ])

clf = SVC(kernel='linear', C=1)
pip = Pipeline([
    ('colums', ct),
    ('clf', clf)
    ])


cv = cross_validate(pip, x,y,return_estimator=True, cv = 10, scoring='roc_auc')
cv['test_score'].mean()


svc = pip.fit(x,y)
score = svc.decision_function(x)

(score[y==1].mean() - score[y==0].mean())/score.std()


plt.hist(score[y==1],10,alpha=0.2)
plt.hist(score[y==0],10,alpha=0.2)

plt.plot(np.sort(score[y==1]), marker='.', linewidth=0)

from scipy.stats import norm


x = np.linspace(1/1000,1,len(score[y==1]))
tq = norm.ppf(x, loc = score[y==1].mean(), scale=score[y==1].std())

plt.plot(tq, np.sort(score[y==1]), marker='.', linewidth=0)
plt.plot(tq, tq, linewidth=1)


plt.plot(tq,tq)
