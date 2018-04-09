# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 19:40:21 2018
Illustrating softmax function
@author: rishabh
"""

import numpy as np

#This is for one example
a=np.random.randn(5)

expa=np.exp(a)

answer=expa/expa.sum()

answer.sum()

#This is for multiple examples
A=np.random.randn(100,5)

expA=np.exp(A)

Answer=expA/expA.sum(axis=1,keepdims=True)

check=Answer.sum(axis=1)

