# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 01:28:24 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def forward(X,W1,b1,W2,b2):
  Z=1/(1+np.exp(-X.dot(W1)-b1))
  A=Z.dot(W2)+b2
  expA = np.exp(A)
  Y=expA/expA.sum(axis=1,keepdims=True)
  return Y,Z
  
def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()
 
def classification(T,Y):
  return (T==Y).mean()  
 
def derivative_w2(Z,T,Y):  
  N,K=T.shape
  #M=Z.shape[1]
  
  #slow(do after doing derivations)
  '''
  ret1=np.zeros((M,K))
  for n in range(N):
    for m in range(M):
      for k in range(K):
          ret1[m,k]=ret1[m,k]+(T[n,k]-Y[n,k])*Z[n,m]
  '''
  ret1 = Z.T.dot(T-Y)
  return ret1
  
def derivative_b2(T,Y):
  return (T-Y).sum(axis=0)
  
def derivative_w1(X,Z,T,Y,W2):
  N,D=X.shape
  M,K=W2.shape

  #slow(do after doing derivations) 
  dz=(T-Y).dot(W2.T)*Z*(1-Z)
  ret1=X.T.dot(dz)
  return ret1

def derivative_b1(T,Y,W2,Z):
  return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)
  
def main():
  Nclass=500
  D=2#dimensionality  of input 
  M=3#hidden layer size
  K=3#number of classes
  
  X1=np.random.randn(Nclass,D)+np.array([0,-2])
  X2=np.random.randn(Nclass,D)+np.array([2,2])
  X3=np.random.randn(Nclass,D)+np.array([-2,2])
  
  X=np.vstack([X1,X2,X3])
  
  Y=np.array([0]*Nclass+[1]*Nclass+[2]*Nclass)
  N=len(Y)
  
  T=np.zeros((N,K))
  for i in range(N):
    T[i,Y[i]]=1
  
  plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=0.5)
  plt.show()
  
  W1=np.random.randn(D,M)
  b1=np.random.randn(M)
  W2=np.random.randn(M,K)
  b2=np.random.randn(K)
  
  learning_rate=1e-3
  costs=[]
  for epoch in range(1000):
    output,hidden=forward(X,W1,b1,W2,b2)
    if epoch%100 == 0:
      c = cost(T, output)
      P=np.argmax(output,axis=1)
      r = classification(Y, P)
      print(r)
      costs.append(c)
        
    W2+=learning_rate*derivative_w2(hidden,T,output)
    b2+=learning_rate*derivative_b2(T,output)   
    W1+=learning_rate*derivative_w1(X,hidden,T,output,W2)
    b1+=learning_rate*derivative_b1(T,output,W2,hidden) 
    
  plt.plot(costs) 

if __name__ == '__main__':   
  main()  


  