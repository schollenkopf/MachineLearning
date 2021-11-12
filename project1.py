#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:13:21 2021

@author: paulnelsonbecker
"""


import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import pandas as pd
from similarity import similarity,binarize
from categoric2numeric import  categoric2numeric

filename = 'student-mat2.csv'
df = pd.read_csv(filename, ";")


# =============================================================================
# print(round(df.describe(),1))
# # correlation matrix 
# corr = df.corr()
# print(round(corr,1))
# =============================================================================
###########
import matplotlib.pyplot as plt 
import seaborn as sns


# =============================================================================
# ##############
# df.hist(column=['school','sex','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','higher','internet','goout','Dalc','Walc','health','absences','G3'],figsize=(10,10))
# plt.savefig('distribution.png',dpi=300,bbox_inches='tight')
# ################
# sns.lineplot(df['G3'], df['absences'])
# plt.savefig('lineplot.png',dpi=300,bbox_inches='tight')
# ###############
# fig, ax = plt.subplots(figsize=(10,10)) 
# sns.heatmap(round(df.corr(),1), annot=True)
# plt.savefig('heatmap.png',dpi=300,bbox_inches='tight')
# plt.show
# =============================================================================

raw_data = df.values  


cols = range(0, 26) 
X = raw_data[:, cols]


attributeNames = np.asarray(df.columns[cols])
#print(attributeNames)

classLabels = raw_data[:,-1] # -1 takes the last column

classNames = np.unique(classLabels)
classGroups = np.array([5,10,15,20])

classDict2 = dict(zip(classNames,np.array([0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])))
classDict = dict(zip(classNames,range(len(classNames))))


y = np.array([classDict[cl] for cl in classLabels])
#####groups of grades##########
y2 = np.array([classDict2[cl] for cl in classLabels])


N, M = X.shape

C = len(classNames)


pca_data = X
pca_names = attributeNames

##Transform categoric variables to one-out-of-K coding:
##MJOb
X_num, attribute_names = categoric2numeric(pca_data[:,5])
firsthalfnames = np.concatenate((pca_names[0:5],attribute_names))
pca_names = np.concatenate((firsthalfnames,pca_names[6:26]))
firsthalf = np.concatenate((pca_data[:,0:5], X_num),axis=1)
pca_data = np.concatenate((firsthalf, pca_data[:,6:26]), axis=1)

#print(pca_names)
#print(pca_data.shape)

##FJOB
X_num, attribute_names = categoric2numeric(pca_data[:,10])
for i in range(len(attribute_names)):
    attribute_names[i] = 'F ' + attribute_names[i] 
firsthalfnames = np.concatenate((pca_names[0:10],attribute_names))
pca_names = np.concatenate((firsthalfnames,pca_names[11:30]))
firsthalf = np.concatenate((pca_data[:,0:10], X_num),axis=1)
pca_data = np.concatenate((firsthalf, pca_data[:,11:30]), axis=1)


pca_names = np.concatenate((pca_names[0:15],pca_names[16:34]))
pca_data = np.concatenate((pca_data[:,0:15], pca_data[:,16:34]), axis=1)

count = 0;

# Bin School
for i in range(N):
    if (pca_data[i,0]=="GP"):
        pca_data[i,0] = 1
        
    else:
        pca_data[i,0] = 0
        

# Bin Female
for i in range(N):
    if (pca_data[i,1]=="F"):
        pca_data[i,1] = 1
        
    else:
        pca_data[i,1] = 0
        




# Bin schoolsup
for i in range(N):
    if (pca_data[i,19]=="yes"):
        pca_data[i,19] = 1
       
    else:
        pca_data[i,19] = 0
       

# Bin famsup
for i in range(N):
    if (pca_data[i,20]=="yes"):
        pca_data[i,20] = 1
        
    else:
        pca_data[i,20] = 0
        

#BIn paid
for i in range(N):
    if (pca_data[i,21]=="yes"):
        pca_data[i,21] = 1
        
    else:
        pca_data[i,21] = 0
        
        
for i in range(N):
    if (pca_data[i,22]=="yes"):
        pca_data[i,22] = 1
        
    else:
        pca_data[i,22] = 0
        
        
for i in range(N):
    
    if (pca_data[i,23]=="yes"):
        pca_data[i,23] = 1
        
    else:
        pca_data[i,23] = 0
        
        
for i in range(N):
    if (pca_data[i,24]=="yes"):
        pca_data[i,24] = 1
        
    else:
        pca_data[i,24] = 0
      
X_num, attribute_names = categoric2numeric(pca_data[:,15])
firsthalfnames = np.concatenate((pca_names[0:15],attribute_names))
pca_names = np.concatenate((firsthalfnames,pca_names[16:36]))
firsthalf = np.concatenate((pca_data[:,0:15], X_num),axis=1)
pca_data = np.concatenate((firsthalf, pca_data[:,16:36]), axis=1)
pca_data_noGrades = pca_data[:,0:32]
pca_names_noGrades = pca_names[0:32]
y = pca_data[:,34]

df = pd.DataFrame(pca_data, columns=['school', 'sex','age', 'Medu', 'Fedu', 'at_home', 'health', 'other', 'services',
 'teacher', 'F at_home', 'F health', 'F other', 'F services', 'F teacher',
 'father','mother','other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
 'paid', 'activities', 'higher', 'internet', 'goout', 'Dalc', 'Walc', 'health',
 'absences', 'G1', 'G2', 'G3'])

    
    
#WITHOUT GRADES:
pca_data = pca_data_noGrades
pca_names = pca_names_noGrades
df = pd.DataFrame(pca_data, columns=['school', 'sex','age', 'Medu', 'Fedu', 'at_home', 'health', 'other', 'services',
 'teacher', 'F at_home', 'F health', 'F other', 'F services', 'F teacher',
 'father','mother','other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
 'paid', 'activities', 'higher', 'internet', 'goout', 'Dalc', 'Walc', 'health',
 'absences'])





df.to_csv('k-encoding.csv',index=True)

N, M = pca_data.shape

pca_data = pca_data.astype(np.int)

#Y = pca_data - np.ones((N,1))*pca_data.mean(axis=0)
#Y2 = Y*(1/np.std(Y,0))
Y2 = pca_data


# =============================================================================
# #####PLOT 1-STANDARD DEVIATION########
# print("______________________________________")
# r = np.arange(1,pca_data.shape[1]+1)
# plt.bar(r, np.std(pca_data,0))
# plt.xticks(r, pca_names, rotation='vertical')
# plt.ylabel('Standard deviation')
# plt.xlabel('Attributes')
# plt.title('Students: attribute standard deviations')
# plt.savefig('std.png',dpi=300,bbox_inches='tight')
# plt.show()
# #####################
# 
# =============================================================================

U,S,V = svd(Y2,full_matrices=False)


# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Compute the projection onto the principal components
Z = U*S;
######PLOT 1.5-Projected data#####
# =============================================================================
# C = len(classGroups)
# for c in range(C):
#     plt.plot(Z[y2==c,0], Z[y2==c,1], '.', alpha=.5)
# plt.xlabel('PC'+str(0+1))
# plt.ylabel('PC'+str(1+1))
# plt.title('Projection' )
# plt.legend(classGroups)
# plt.axis('equal')
# plt.savefig('projection.png',dpi=300,bbox_inches='tight')
# =============================================================================



threshold = 0.9

#####PLOT 2-VARIANCE EXPLAINED########
# =============================================================================
# plt.figure()
# plt.plot(range(1,len(rho)+1),rho,'x-')
# plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
# plt.plot([1,len(rho)],[threshold, threshold],'k--')
# plt.title('Variance explained by principal components');
# plt.xlabel('Principal component');
# plt.ylabel('Variance explained');
# plt.legend(['Individual','Cumulative','Threshold'])
# plt.grid()
# plt.savefig('variance.png',dpi=300,bbox_inches='tight')
# plt.show()
# 
# =============================================================================

#####################PLOT 3-PCA Coefficients#####


# =============================================================================
vt = V.T
# pcs = [0,1]
# legendStrs = ['PC'+str(e+1) for e in pcs]
# bw = .2
# r = np.arange(1,M+1)
# for i in pcs:    
#     plt.bar(r+i*bw, vt[:,i], width=bw)
# plt.xticks(r+bw, pca_names, rotation='vertical')
# plt.xlabel('Attributes')
# plt.ylabel('Component coefficients')
# plt.legend(legendStrs)
# plt.grid()
# plt.title('PCA Component Coefficients')
# plt.savefig('pcomponents.png',dpi=300,bbox_inches='tight')
# plt.show()
# =============================================================================

# =============================================================================
pcs = [1]
legendStrs = ['PC'+str(e+1) for e in pcs]
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, vt[:,i], width=bw)
plt.xticks(r+bw, pca_names, rotation='vertical')
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()
# 
# =============================================================================



# =============================================================================
# #########Similarity school####
# i=0
# noti = list(range(0,2))+list(range(21,27))
# print(pca_names[noti])
# # Compute similarity between image i and all others
# sim1 = similarity(pca_data[:,i], pca_data[:,noti].T, 'smc')
# print(sim1)
# 
# i=1
# sim1 = similarity(pca_data[:,i], pca_data[:,noti].T, 'smc')
# print(sim1)
# 
# i=21
# sim1 = similarity(pca_data[:,i], pca_data[:,noti].T, 'smc')
# print(sim1)
# 
# i=22
# sim1 = similarity(pca_data[:,i], pca_data[:,noti].T, 'smc')
# print(sim1)
# 
# i=23
# sim1 = similarity(pca_data[:,i], pca_data[:,noti].T, 'smc')
# print(sim1)
# 
# i=24
# sim1 = similarity(pca_data[:,i], pca_data[:,noti].T, 'smc')
# print(sim1)
# 
# i=25
# sim1 = similarity(pca_data[:,i], pca_data[:,noti].T, 'smc')
# print(sim1)
# 
# i=26
# sim1 = similarity(pca_data[:,i], pca_data[:,noti].T, 'smc')
# print(sim1)
# 
# =============================================================================

