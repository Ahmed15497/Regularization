# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:22:39 2019

@author: ahmed
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from functions import *




file_data = 'ex2data2.txt'
dataset = pd.read_csv(file_data,sep = ',' , header = None)
del file_data

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = np.reshape(y,(np.size(y),1))

m = np.size(y)

X_mapped = featureMappping(X[:,0],X[:,1], 6)

theta = np.zeros((np.size(X_mapped,axis=1), 1))



reg_param = 1;

res = minimize(fun = CostFunm,
	       x0 = theta,
	       method='Newton-CG',
	       args=(X_mapped,y,reg_param),
	       jac=Gradient)



