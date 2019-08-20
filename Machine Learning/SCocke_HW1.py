# -*- coding: utf-8 -*-
"""


@author: Steven Cocke
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold  
from sklearn import datasets
from itertools import product
import matplotlib.pyplot as plt


# Part 1: Bring in breast cancer data
# 3. find some simple data

cancer = datasets.load_breast_cancer()

M = cancer.data

L = cancer.target

n_folds = 5

data = (M,L,n_folds)


# Part 2: Define functions


# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parameters 
#for each
# 2. expand to include larger number of classifiers and hyperparameter settings

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data 
  kf = KFold(n_splits=n_folds)  
  ret = {} 
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): 
    clf = a_clf(**clf_hyper) 
            
    clf.fit(M[train_index], L[train_index])   
    pred = clf.predict(M[test_index])         
    ret[ids]= {'clf': clf,                   
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret



def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1) 
                         
                        
        k1Test = k1Test.replace('            ',' ') 
        k1Test = k1Test.replace('          ',' ')
        
        
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) 
        else:
            clfsAccuracyDict[k1Test] = [v1]            
        
            

def myHyperParamSearch(clfsList,clfDict):
    for clf in clfsList:
    
  
        clfString = str(clf)
        
        for k1, v1 in clfDict.items(): 
            if k1 in clfString:
                k2,v2 = zip(*v1.items()) 
                for values in product(*v2): 
                    hyperParams = dict(zip(k2, values)) 
                    results = run(clf, data, hyperParams) 
                    populateClfAccuracyDict(results) 
 


clfsList = [RandomForestClassifier, LogisticRegression, KNeighborsClassifier] 

clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], 
                                      "n_jobs": [1,2,3]},                                      
           'LogisticRegression': {"tol": [0.001,0.01,0.1]},           
           'KNeighborsClassifier': {'n_neighbors': np.arange(3, 15),
                                     'weights': ['uniform', 'distance'],
                                     'algorithm': ['ball_tree', 'kd_tree', 'brute']}}

                   
clfsAccuracyDict = {}

myHyperParamSearch(clfsList,clfDict)    

print(clfsAccuracyDict)


n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())

filename_prefix = 'clf_Histograms_'

plot_num = 1 


               
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parameters settings

#create the histograms
for k1, v1 in clfsAccuracyDict.items():
    fig = plt.figure(figsize =(10,10)) 
    ax  = fig.add_subplot(1, 1, 1) 
    plt.hist(v1, facecolor='green', alpha=0.75) 
    ax.set_title(k1, fontsize=10) 
    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=15) 
    ax.set_ylabel('Frequency', fontsize=15) 
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
    ax.yaxis.set_ticks(np.arange(0, n+1, 1)) 
    ax.xaxis.set_tick_params(labelsize=20) 
    ax.yaxis.set_tick_params(labelsize=20) 
    

    
    plot_num_str = str(plot_num) 
    filename = filename_prefix + plot_num_str 
    plt.savefig(filename, bbox_inches = 'tight') 
    plot_num = plot_num+1 
    
plt.show()

#create box plots
filename_prefix = 'clf_Boxplots_'

plot_num = 1 

for k1, v1 in clfsAccuracyDict.items():
    fig = plt.figure(figsize =(10,10)) 
    ax  = fig.add_subplot(1, 1, 1) 
    plt.boxplot(v1) 
    ax.set_title(k1, fontsize=10) 
    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=15) 
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
    ax.xaxis.set_tick_params(labelsize=20) 
    

    
    plot_num_str = str(plot_num) 
    filename = filename_prefix + plot_num_str 
    plt.savefig(filename, bbox_inches = 'tight') 
    plot_num = plot_num+1 
    
plt.show()

# 6. Investigate grid search function
##We used grid search funciton in data mining and is attached to my assignment submission.

