#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:25:29 2019

@author: krunaaltavkar
"""

from linear_regression import linear_regression_noreg,linear_regression_invertible, regularized_linear_regression, tune_lambda, mean_absolute_error
#from data_loader import data_processing_linear_regression, data_processing_linear_regression_with_mapping
#import data_loader
import numpy as np
import pandas as pd

filename = 'winequality-white.csv'

def data_processing_linear_regression_with_mapping(filename,mapping_power):
  

    white = pd.read_csv(filename, low_memory=False, sep=';').values
    
    [N, d] = white.shape
    print("Mapping and Power:",mapping_power)
    maped_X = mapping_data(white[:,:-1],mapping_power)
    print("MX DiM:", maped_X.shape)
    white = np.insert(maped_X, maped_X.shape[1], white[:,-1], axis=1)
    print("White Dim:", white.shape)
    
    
    
    np.random.seed(3)
    # prepare data
    ridx = np.random.permutation(N)
    ntr = int(np.round(N * 0.8))
    nval = int(np.round(N * 0.1))
    ntest = N - ntr - nval
    
    # spliting training, validation, and test
    
    Xtrain = np.hstack([np.ones([ntr, 1]), white[ridx[0:ntr], 0:-1]])
    
    ytrain = white[ridx[0:ntr], -1]
    
    Xval = np.hstack([np.ones([nval, 1]), white[ridx[ntr:ntr + nval], 0:-1]])
    yval = white[ridx[ntr:ntr + nval], -1]
    
    Xtest = np.hstack([np.ones([ntest, 1]), white[ridx[ntr + nval:], 0:-1]])
    ytest = white[ridx[ntr + nval:], -1]
    
    
    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    if power == 2:
        new_X = X**power
        mapped_X = np.concatenate((X,new_X), axis = 1)
    else:
        for i in range(2, power+1):
            if i == 2:
                new_X = X**i
                mapped_X = np.concatenate((X,new_X), axis = 1)
            else:
                new_X = X**i
                mapped_X = np.concatenate((mapped_X,new_X), axis = 1)     
    return mapped_X

print("\n======== Question 1.6.1 (power = 2) ========")
print("if your maaping function is correct, simplely change the 'power' value to see how MAE change when 'power' changes")
power = 2
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression_with_mapping(filename, power)
print("AM:", Xtrain.shape)
bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
print("Best Lambda =  ", bestlambd, sep="")
w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
print("dimensionality of the model parameter is ", len(w), ".", sep="")
print("model parameter is ", np.array_str(w))
mae = mean_absolute_error(w, Xtrain, ytrain)
print("MAE on train is %.5f" % mae)
mae = mean_absolute_error(w, Xval, yval)
print("MAE on val is %.5f" % mae)
mae = mean_absolute_error(w, Xtest, ytest)
print("MAE on test is %.5f" % mae)

print("\n======== Question 1.6.2 (power = higher)========")
print("if your maaping function is correct, simplely change the 'power' value to see how MSE change when 'power' changes")
power = 20
for i in range(2, power):
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression_with_mapping(filename, i)
    print(Xtrain.shape)
    bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
    print('best lambd is ' + str(bestlambd))
    w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
    mae = mean_absolute_error(w, Xtrain, ytrain)
    print('when power = ' + str(i))
    print("MAE on train is %.5f" % mae)
    mae = mean_absolute_error(w, Xval, yval)
    print("MAE on val is %.5f" % mae)
    mae = mean_absolute_error(w, Xtest, ytest)
    print("MAE on test is %.5f" % mae)