# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 18:03:23 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import getData 

Xtrain,Ytrain,Xtest,Ytest=getData()

N,D=Xtrain.shape
M=5
K=len(set(Ytrain)|set(Ytest))
W1=np.random.randn(D,M)
b1=np.zeros(M)
W2=np.random.randn(M,K)
b2=np.zeros(K)

def indicator(Y,K):
  N = len(Y)
  ind=np.zeros((N,K))
  for i in range(N):
    ind[i,Y[i]]=1
  return ind

def softmax(A):
  expA=np.exp(A)
  return expA/expA.sum(axis=1,keepdims=True)
  
def forward(X,W1,b1,W2,b2):
  A=np.tanh(X.dot(W1)+b1)
  return A,softmax(A.dot(W2)+b2)

def classification(T,Yhat):
  return np.mean(Yhat==T)
  
def crossEntropy(T,Yhat):
  return -np.mean(T*np.log(Yhat))

train_costs=[]
test_costs=[]  
#gradient Descent
Ytrain_ind=indicator(Ytrain,K)
Ytest_ind=indicator(Ytest,K)
learning_rate=0.001

for i in range(10000):

  Atrain,Yhat_train=forward(Xtrain,W1,b1,W2,b2)
  Atest,Yhat_test=forward(Xtest,W1,b1,W2,b2)

  train_costs.append(crossEntropy(Ytrain_ind,Yhat_train))
  test_costs.append(crossEntropy(Ytest_ind,Yhat_test))
    
  W2 -= learning_rate*Atrain.T.dot(Yhat_train - Ytrain_ind)  
  b2-=learning_rate*(Yhat_train-Ytrain_ind).sum(axis=0)
  dz=(Yhat_train-Ytrain_ind).dot(W2.T)*(1-Atrain*Atrain)
  W1-=learning_rate*Xtrain.T.dot(dz)
  b1-=learning_rate*dz.sum(axis=0)
  
print('Train Classification Rate',classification(Ytrain,np.argmax(Yhat_train,axis=1)))
print('Test Classification Rate',classification(Ytest,np.argmax(Yhat_test,axis=1)))

plt.plot(train_costs,label='Train')
plt.plot(test_costs,label='Test')
plt.legend()


  
  