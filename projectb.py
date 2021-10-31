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


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

##dataset:
X = Y2.astype(float64)
attributeNames = pca_names.tolist()
y = y.astype(float64)
##grades are in y




N, M = X.shape

# Add offset attribute
#X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
#M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 4
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)


# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))


lambdas = np.power(10.,(0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))
Array_hidden_units = np.array([1,5,10])


k=0

outerloop_nr = 0
K2=4
Error_LR = np.zeros((K2,2))
Error_ANN = np.zeros((K2,2))
Error_base = np.zeros((K2,1))

w2 = np.empty((M+1,K,len(lambdas)))
for train_index, test_index in CV.split(X,y):

    
    
    w = np.empty((M+1,K2,len(lambdas)))
    
    
    
    CV2 = model_selection.KFold(K2, shuffle=True)
    
    
    sizes_test2 = np.empty(K2)
    
    errors_vs_innerloop_LR = np.zeros((K2,len(lambdas)))
    errors_vs_innerloop = np.zeros((K2,len(Array_hidden_units)))
    innerloop_nr = 0
    for train_index2, test_index2 in CV2.split(X[train_index,:],y[train_index]):
        sizes_test2[innerloop_nr] = len(test_index2)
        
        
        ####NEURAL NETWORK#####
        X_train_ANN = torch.Tensor(X[train_index2,:])
        y_train_ANN = torch.Tensor(y[train_index2])
        X_test_ANN = torch.Tensor(X[test_index2,:])
        y_test_ANN = torch.Tensor(y[test_index2])
        
        
        for i in range(len(Array_hidden_units)):
            
            n_hidden_units = Array_hidden_units[i]
            
            print("Training ANN Model with:")
            print(outerloop_nr)
            print(innerloop_nr)
            print(n_hidden_units)
           
            model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
            loss_fn = torch.nn.MSELoss()
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_ANN,
                                                       y=y_train_ANN,
                                                       n_replicates=1,
                                                       max_iter=10000)
            print('\n\tBest loss: {}\n'.format(final_loss))
            y_test_est = net(X_test_ANN)
            se = (y_test_est.float()-y_test_ANN.float())**2 # squared error
            mse = (torch.sum(se).type(torch.float)/len(y_test_ANN)).data.numpy()
            errors_vs_innerloop[innerloop_nr,i] = mse
            print(errors_vs_innerloop)
            
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
            print("Training LR Model with:")
            print(outerloop_nr)
            print(innerloop_nr)
            print(lambdas[l])
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M+1)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,innerloop_nr,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
            errors_vs_innerloop_LR[innerloop_nr,l] = np.power(y_test_LR-X_test_LR @ w[:,innerloop_nr,l].T,2).mean(axis=0)
    
        

        
        
        
        
        
        
        
        
        
        innerloop_nr = innerloop_nr + 1
        
        
    
    ###Chose best ANN and train it on full x_train####
    for model in range(len(Array_hidden_units)):
        ms_ann = 0
        for k in range(K2):
            ms_ann = ms_ann + (sizes_test2[k]/len(train_index))*errors_vs_innerloop[k,model]
            
        print("Chosing best model ANN")
        print(model)
        print(ms_ann)
        if (model==0):
            n = Array_hidden_units[model]
            lowest_error = ms_ann
        elif (ms_ann<lowest_error):
            n = Array_hidden_units[model]
            lowest_error = ms_ann
            
    X_train_ANN = torch.Tensor(X[train_index,:])
    y_train_ANN = torch.Tensor(y[train_index])
    X_test_ANN = torch.Tensor(X[test_index,:])
    y_test_ANN = torch.Tensor(y[test_index])
    print("Best Model: {}".format(n))
    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n), #M features to n_hidden_units
                    torch.nn.Tanh(), # 1st transfer function,
                    torch.nn.Linear(n,n),
                    torch.nn.Tanh(),
                    torch.nn.Linear(n, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
    loss_fn = torch.nn.MSELoss()
    net, final_loss, learning_curve = train_neural_net(model,
                                                   loss_fn,
                                                   X=X_train_ANN,
                                                   y=y_train_ANN,
                                                   n_replicates=1,
                                                   max_iter=10000)
    y_test_est = net(X_test_ANN)
    se = (y_test_est.float()-y_test_ANN.float())**2 # squared error
    mse = (torch.sum(se).type(torch.float)/len(y_test_ANN)).data.numpy()
    Error_ANN[outerloop_nr,0] = n
    Error_ANN[outerloop_nr,1] = mse

    ###Chose best LR and train it on full x_train###
    for la in range(len(lambdas)):
        ms_lr = 0
        for k in range(K2):
            ms_lr = ms_lr + (sizes_test2[k]/len(train_index))*errors_vs_innerloop_LR[k,la]
            
        print("Chosing best model LR")
        print(la)
        print(ms_lr)
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
    w2[:,outerloop_nr,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    Error_LR[outerloop_nr,0] = lambda_
    Error_LR[outerloop_nr,1] = np.power(y_test_LR-X_test_LR @ w[:,outerloop_nr,l].T,2).mean(axis=0)
    
        
    ###BaseModel####
    mean = np.ones(len(y[test_index]))*np.mean(y[train_index])
    se = (mean-y[test_index])**2 # squared error
    mse = (sum(se)/len(y[test_index]))
    
    Error_base[outerloop_nr] = mse
    
    
    
    
    
    
    outerloop_nr = outerloop_nr + 1
print(Error_ANN)
print(Error_LR)
print(Error_base)
    
    
  