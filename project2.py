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

##dataset:
X = Y2.astype(float64)
attributeNames = pca_names.tolist()
y = y.astype(float64)
##grades are in y




N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,(0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))


w_rlr = np.empty(M)
mu = np.empty(M-1)
sigma = np.empty(M-1)




    

    
cross_validation = 10  
    
    
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
mu[:] = np.mean(X[:, 1:], 0)
sigma[:] = np.std(X[:, 1:], 0)
    
X[:, 1:] = (X[:, 1:] - mu[:] ) / sigma[:] 

    
Xty = X.T @ y
XtX = X.T @ X
    


    # Estimate weights for the optimal value of lambda, on entire training set
lambdaI = opt_lambda * np.eye(M)
lambdaI[0,0] = 0 # Do no regularize the bias term
w_rlr[:] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    
figure(1, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()


show()

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m],2)))


#predict(18)
def predictvalue(index):
    predict = numpy.dot(w_rlr[:],X[index,:])
    print("Prediction: {}".format(int(np.round(predict))))
    print("Actual: {}".format(int(y[index])))

print('Ran Exercise 8.1.1')



