#!/usr/bin/env python3
from numpy.core.shape_base import block
import data_loader as DL
import numpy as np
import matplotlib.pyplot as plt
import linear_basis_function_regression as regression
import basis_function_computation as basisFunc
import debug_operations as debugOp

"""
    Plot training and test RMS values
"""
def plotRms(title, train_err, test_err):
    plt.figure
    plt.plot(train_err.keys(), train_err.values(), '-bo')
    plt.plot(test_err.keys(), test_err.values(), '-ro')
    plt.ylabel(title)
    plt.legend(['Training error','Test error'])
    plt.title('Fit with polynomials, no regularization')
    plt.xlabel('Polynomial degree')
    plt.show()

"""
    1. Load the provided data
    2. Extract target values and input
    3. Call linear basis function regression on raw data
    4. Call linear basis function regression on normalised data
"""
def main(f_start, t_index, min_degree, max_degree, debugFlag):
    #load data
    (countries, features, values) = DL.load_unicef_data()
    
    #extract target and input values
    targets = values[:,t_index]
    x = values[:,f_start:]

    #set indices for training and test data respectively
    N_points, N_features = np.shape(values)
    N_train = 100
    training_indices = np.arange(N_train)
    testing_indices = np.arange(N_train, N_points)
    h_param = 0
    h_set = np.zeros(1)

    #compute for non-normalised data
    basis_value_x = basisFunc.getBasisFunctionValueMatrix(x, max_degree)
    (nn_train_err, nn_test_err, learned_polynomials) = regression.linearPolynomialRegression(min_degree, max_degree, x, basis_value_x, targets, training_indices, testing_indices, h_set)
    plotRms("RMS for non-normalised input", nn_train_err[h_param], nn_test_err[h_param])

    #debug printing for results
    if debugFlag:
        debugOp.debugPrintDict(nn_train_err[h_param], "Training error for non-normalised data")
        debugOp.debugPrintDict(nn_test_err[h_param], "Testing error for non-normalised data")

    #compute for normalised data
    x = DL.normalize_data(x)
    basis_value_x = basisFunc.getBasisFunctionValueMatrix(x, max_degree)
    (norm_train_err, norm_test_err, learned_polynomials) = regression.linearPolynomialRegression(min_degree, max_degree, x, basis_value_x, targets, training_indices, testing_indices, h_set)
    plotRms("RMS for normalised input", norm_train_err[h_param], norm_test_err[h_param])

    #debug printing for results
    if debugFlag:
        debugOp.debugPrintDict(norm_train_err[h_param], "Training error for normalised data")
        debugOp.debugPrintDict(norm_test_err[h_param], "Testing error for normalised data")    

"""
    Define required range of degrees for the plots here
"""
if __name__ == "__main__":
    f_start = 7
    t_index = 1
    min_degree = 1
    max_degree = 8
    debugFlag = False
    main(f_start, t_index, min_degree, max_degree, debugFlag)