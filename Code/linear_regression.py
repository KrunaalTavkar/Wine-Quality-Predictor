"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.
The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression, 
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    modified_features = np.dot(X, w)
    err = sum([abs(a-b) for a,b in zip(modified_features, y)])/len(y)
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here #
    #####################################################	
    transpose_dot = np.dot(X.T, X)
    transpose_inv = np.linalg.inv(transpose_dot)
    second_transpose_dot = np.dot(X.T, y)
    w = np.dot(transpose_inv, second_transpose_dot)
    return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    transpose_dot = np.dot(X.T, X)
    identity_matrix = np.identity(X.shape[1])
    eigenvalues = np.abs(np.linalg.eigvals(transpose_dot))
    min_eigenvalues = np.amin(eigenvalues)
    if min_eigenvalues < 0.00001:
        invertible = False
    while invertible == False:
        transpose_dot += 0.1*identity_matrix
        eigenvalues = np.abs(np.linalg.eigvals(transpose_dot))
        min_eigenvalues = np.amin(eigenvalues)
        if min_eigenvalues < 0.00001:
            invertible = False
        else:
            invertible = True
    transpose_inv = np.linalg.inv(transpose_dot)
    second_transpose_dot = np.dot(X.T, y)
    w = np.dot(transpose_inv, second_transpose_dot)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    transpose_dot = np.dot(X.T, X)
    identity_matrix = np.identity(X.shape[1])
    transpose_dot += lambd*identity_matrix
    transpose_inv = np.linalg.inv(transpose_dot)
    second_transpose_dot = np.dot(X.T, y)
    w = np.dot(transpose_inv, second_transpose_dot)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    min_err = None
    all_possible_lambds = [10**i for i in range(-19,20)]
    for lambd in all_possible_lambds:
        temp_w = regularized_linear_regression(Xtrain, ytrain, lambd)
        new_err = mean_absolute_error(temp_w, Xval, yval)
        if min_err is None:
            min_err = new_err
            bestlambda = lambd
        elif new_err < min_err:
            min_err = new_err
            bestlambda = lambd
    
    return bestlambda
    

###### Q1.6 ######
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


