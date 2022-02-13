import numpy as np
import math

"""
    Compute transpose of optimal parameters and multiply by tranpose of basis value matrix
"""
def computePredictionVector(w_opt, basis_value_x):
    #compute the prediction vector
    y_pred = basis_value_x.dot(w_opt)
    #return value
    return y_pred

"""
    1. Compute predictions
    2. Evaluate difference with ground truth
"""
def computeRms(basis_value_x, w_opt, y):
    #compute prediction
    y_pred = computePredictionVector(w_opt, basis_value_x)
    #check lengths
    assert y_pred.shape == y.shape
    #compute rms value
    rms = np.sqrt(np.mean(np.square(y_pred - y)))
    #return value
    return rms    

"""
    1. If hyper-param is 0, directly use pinv. Otherwise follow normal computation method
    3. Multiply this with ground truth to get optimal parameter vector
"""
def evaluateOptimalParameters(x, y, hyperParam):

    if hyperParam < 0:
        print("Negative hyperparameter not allowed")
        exit
    elif hyperParam == 0:
        x_pinv = np.linalg.pinv(x)
        w_opt = x_pinv.dot(y)
        return w_opt
    else:
        xTx = (x.T).dot(x)
        nr, nc = np.shape(x)
        id_matrix = np.identity(nc)
        sum1 = np.add(xTx, hyperParam*id_matrix)
        product1 = np.linalg.inv(sum1)
        product2 = (x.T).dot(y)
        w_opt = product1.dot(product2)
        return w_opt
