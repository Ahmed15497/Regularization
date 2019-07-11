# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:22:58 2019

@author: ahmed
"""

import numpy as np

lamda = 1;

def featureMappping(X1,X2,degree):
    output_feature_vec = np.ones(len(X1))[:,None]
    for i in range(1,7):
        for j in range(i+1):
            new_feature = np.array(X1**(i-j)*X2**j)[:,None]
            output_feature_vec = np.hstack((output_feature_vec,new_feature))
    return output_feature_vec


def sigmoid(X):
    X = 1/(1+np.exp(-X))
    return X


def CostFunm(theta,x,y,reg_param):
    m,n = x.shape
    theta = np.reshape(theta,(n,1))
    y = np.reshape(y,(m,1))
    h = sigmoid(np.dot(x,theta))
    term1 = np.log(h)
    term2 = np.log(1-h)
    term1 = np.reshape(term1,(m,1))
    term2 = np.reshape(term2,(m,1))
    term = y * term1 + (1-y) * term2
    J = -(1/m) * np.sum(term)
    J += (reg_param/(2*m))*np.sum(theta**2)
    return J


def Gradient(theta,x,y,reg_param):
    m,n = x.shape
    theta = np.reshape(theta,(n,1))
    y = np.reshape(y,(m,1))
    h = sigmoid(np.dot(x,theta))
    grad = (1/m)*np.sum(((h-y)*x),axis=0)
    grad = grad[:,None] + (reg_param/m)*theta
    
    return grad.flatten()  # flatten is very important !!!!! 