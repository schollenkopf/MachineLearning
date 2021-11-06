#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 21:52:37 2021

@author: paulnelsonbecker
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 19:26:24 2021

@author: paulnelsonbecker
"""
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from project1 import *

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

##dataset:
y = np.array(y, copy=False, subok=True, ndmin=2).T
X = np.append(X,y,axis=1)
X = Y2.astype(float64)
attributeNames = pca_names.tolist()
y = X[:,1]
firsthalf = X[:,0]
firsthalf = np.array(firsthalf, copy=False, subok=True, ndmin=2).T
secondhalf = X[:,2:-1]
X = np.concatenate((firsthalf,secondhalf),axis=1)

firsthalf = attributeNames[0]
firsthalf = np.array(firsthalf, copy=False, subok=True, ndmin=2).T
secondhalf = attributeNames[2:-1]
secondhalf = np.array(secondhalf, copy=False, subok=True, ndmin=2).T
attributeNames = np.concatenate((firsthalf,secondhalf))




N, M = X.shape

# Add offset attribute
#X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
#M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Maximum number of neighbors
L=40

# Initialize variables
#T = len(lambdas)

w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))


lambdas = np.power(10.,(0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))
Array_hidden_units = np.array([1,2,3,4,5,7])


k=0

outerloop_nr = 0
K2=5
Error_LR = np.zeros((K,2))
Error_KNN = np.zeros((K,2))
Error_base = np.zeros((K,1))

w2 = np.empty((M+1,K))
for train_index, test_index in CV.split(X,y):

    
    
    w = np.empty((M+1,K2,len(lambdas)))
    
    
    
    CV2 = model_selection.KFold(K2, shuffle=True)
    
    
    sizes_test2 = np.empty(K2)
    
    errors_vs_innerloop_LR = np.zeros((K2,len(lambdas)))
    errors_vs_innerloop_KNN = np.zeros((K2,L))
    innerloop_nr = 0
    for train_index2, test_index2 in CV2.split(X[train_index,:],y[train_index]):
        sizes_test2[innerloop_nr] = len(test_index2)
        
        
        #KNN
   
        X_train = X[train_index2,:]
        y_train = y[train_index2]
        X_test = X[test_index2,:]
        y_test = y[test_index2]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train, y_train);
            y_est = knclassifier.predict(X_test);
            errors_vs_innerloop_KNN[innerloop_nr,l-1] = (np.sum(y_est!=y_test))/len(y_test)
        #print(errors_vs_innerloop_KNN)
            
        ####regularized linear regression####
        X_train_LR = X[train_index2]
        y_train_LR = y[train_index2]
        X_test_LR = X[test_index2]
        y_test_LR = y[test_index2]
        
        X_train_LR = np.concatenate((np.ones((X_train_LR.shape[0],1)),X_train_LR),1)
        X_test_LR = np.concatenate((np.ones((X_test_LR.shape[0],1)),X_test_LR),1)
        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train_LR[:, 1:], 0)
        sigma = np.std(X_train_LR[:, 1:], 0)
        
        X_train_LR[:, 1:] = (X_train_LR[:, 1:] - mu) / sigma
        X_test_LR[:, 1:] = (X_test_LR[:, 1:] - mu) / sigma
        
        # precompute terms
        Xty = X_train_LR.T @ y_train_LR
        XtX = X_train_LR.T @ X_train_LR
        
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M+1)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,innerloop_nr,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
            errors_vs_innerloop_LR[innerloop_nr,l] = np.power(y_test_LR-X_test_LR @ w[:,innerloop_nr,l].T,2).mean(axis=0)
    

        innerloop_nr = innerloop_nr + 1
        
        
    
    ###Chose best KNN and train it on full x_train####
    
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    
    
    for l in range(1,L+1):
        ms_knn = 0
        for k in range(K2):
            ms_knn = ms_knn + (sizes_test2[k]/len(train_index))*errors_vs_innerloop_KNN[k,l-1]

        if (l==1):
            best = l
            lowest_error = ms_knn
        elif (ms_knn<lowest_error):
            best = l
            lowest_error = ms_knn
    print("Best Model KNN: {}".format(best) )
    knclassifier = KNeighborsClassifier(n_neighbors=best);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    err = (np.sum(y_est!=y_test))/len(y_test)

    
    Error_KNN[outerloop_nr,0] = np.round(best)
    Error_KNN[outerloop_nr,1] = err
    print(Error_KNN)
    ###Chose best LR and train it on full x_train###
    for la in range(len(lambdas)):
        ms_lr = 0
        for k in range(K2):
            ms_lr = ms_lr + (sizes_test2[k]/len(train_index))*errors_vs_innerloop_LR[k,la]
            
        if (la==0):
            lambda_ = lambdas[la]
            lowest_error_LR = ms_lr
        elif (ms_lr<lowest_error_LR):
            lambda_ = lambdas[la]
            lowest_error_LR = ms_lr
    
    
    X_train_LR = X[train_index]
    y_train_LR = y[train_index]
    X_test_LR = X[test_index]
    y_test_LR = y[test_index]
        
    X_train_LR = np.concatenate((np.ones((X_train_LR.shape[0],1)),X_train_LR),1)
    
    X_test_LR = np.concatenate((np.ones((X_test_LR.shape[0],1)),X_test_LR),1)
    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train_LR[:, 1:], 0)
    sigma = np.std(X_train_LR[:, 1:], 0)
        
    X_train_LR[:, 1:] = (X_train_LR[:, 1:] - mu) / sigma
    X_test_LR[:, 1:] = (X_test_LR[:, 1:] - mu) / sigma
        
    # precompute terms
    Xty = X_train_LR.T @ y_train_LR
    XtX = X_train_LR.T @ X_train_LR
    
    lambdaI = lambda_ * np.eye(M+1)
    lambdaI[0,0] = 0 # remove bias regularization
    w2[:,outerloop_nr] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    Error_LR[outerloop_nr,0] = lambda_
    Error_LR[outerloop_nr,1] = np.power(y_test_LR-X_test_LR @ w2[:,outerloop_nr].T,2).mean(axis=0)
    
        
    ###BaseModel####
    male = 0
    female = 0
    for i in range(len(y_test)):
        if (y_test[i]==0):
            male = male +1
        else:
            female = female +1
    if (male>female):
        err = (np.sum(0!=y_test))/len(y_test)
    else:
        err = (np.sum(1!=y_test))/len(y_test)
    
    Error_base[outerloop_nr] = err
    
    
    
    
    
    
    outerloop_nr = outerloop_nr + 1
print(Error_KNN)
print(Error_LR)
print(Error_base)
    
    
  