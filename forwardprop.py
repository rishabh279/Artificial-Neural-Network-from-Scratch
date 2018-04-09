# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 00:35:32 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt

Nclass = 500

#np.random.randn(Nclass,2) creates gaussian  distribution and np.array([0,-2]) centers the data at (0,-2)
X1=np.random.randn(Nclass,2)+np.array([0,-2])

X2=np.random.randn(Nclass,2)+np.array([2,2])

X3=np.random.randn(Nclass,2)+np.array([-2,2])

X=np.vstack([X1,X2,X3])

Y = np.array([0]*Nclass+[1]*Nclass+[2]*Nclass)

plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=0.5)

#randomly intialize weight
D=2#Dimensions
M=3#hidden Layer size
K=3#classes

W1=np.random.randn(D,M)
b1=np.random.rand(M)
W2=np.random.randn(M,K)
b2=np.random.randn(K)

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
def forward(X,W1,b1,W2,b2):
  Z=sigmoid(X.dot(W1)+b1)
  
  A=Z.dot(W2)+b2
  
  expA=np.exp(A)
  
  Y=expA/expA.sum(axis=1,keepdims=True)
  
  return Y
  
def classification(Y,P):
  return np.mean(Y==P)

predicted_Y=forward(X,W1,b1,W2,b2)
P=np.argmax(predicted_Y,axis=1)

assert(len(P)==len(Y))

print(classification(Y,P))  


  