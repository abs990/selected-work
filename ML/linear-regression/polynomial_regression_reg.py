#!/usr/bin/env python3
import data_loader as DL
import numpy as np
import matplotlib.pyplot as plt
import linear_basis_function_regression as regression
import math as math
import basis_function_computation as basisFunc

"""
    Plot average validation set error versus hyperparameter
"""
def plotAvgValError(v_error):
    #debug print for results
    debugFlag = False
    if debugFlag:
        print("Average set validation error by hyperparam")
        for h_param in v_error.keys():
            print(str(h_param)+" "+str(v_error[h_param]))

    #save unregularized error separately
    unreg_err = v_error[0]
    #delete unregularized error from dictionary
    v_error.pop(0, 'None')
    #prepare to plot results
    plt.figure
    plt.semilogx(v_error.keys(),v_error.values(),'-bo')
    plt.ylabel('Average validation set error')
    plt.title('Average validation set error versus hyperparameter')
    plt.xlabel('Hyperparameter')

    #show each hyperParam and corresponding ang validation error
    for h_param, error in zip(v_error.keys(), v_error.values()):
        label = "({:.2f},{:.3f})".format(h_param, error)
        plt.annotate(label, (h_param, error), textcoords="offset points", xytext=(0,5), ha='center')

    #show value for hyperParam=0 as a horizontal line
    plt.axhline(y=unreg_err, color='r', linestyle='-')
    plt.legend(['Regularized model error','Unregularized model error (Error = '+str("{:.3f}".format(unreg_err))+' )'])
    
    #show plot
    plt.show()

"""
    Function to get training and validation indices based on validation set window
"""
def getDataSplit(v_start, v_size, N_initial):
    #construct validation set index list
    assert v_start >= 0 and v_start < N_initial
    v_end = v_start + v_size - 1
    assert v_end >= 0 and v_end < N_initial
    assert v_start<=v_end
    v_end_lim = v_end + 1        
    v_indices = np.arange(v_start, v_end_lim)

    #construct training set index list
    N_train = N_initial - v_size
    train_indices = np.zeros(N_train)
    if v_start == 0:
        train_indices = np.arange(v_end_lim,N_initial)
    elif v_end_lim == N_initial:
        train_indices = np.arange(0,v_start)
    else:
        train_list1 = np.arange(0, v_start)
        train_list2 = np.arange(v_end_lim, N_initial)
        train_indices = np.concatenate((train_list1, train_list2), axis=None)

    #return index lists
    return (train_indices, v_indices)

"""
    1. Load the provided data
    2. Extract target values and input
    4. Call linear basis function regression on normalised data for each hyperparameter
"""
def main(f_start, t_index, degree, h_set, k):
    #compute validation set size
    N_train = 100
    assert k>0 and N_train%k==0
    v_size = int(N_train/k)

    #load data
    (countries, features, values) = DL.load_unicef_data()
    
    #extract first N_train target values
    targets = values[0:N_train, t_index]
    #normalise x. Then extract first N_train values for k cross validation
    x = values[:, f_start:]
    x = DL.normalize_data(x)
    x = x[0:N_train,:]

    #dictionary to store average validation set error
    v_error = {}
    for h_param in h_set:
        v_error[h_param] = 0

    #get basis function value matrix
    degree = 2
    basis_value_x = basisFunc.getBasisFunctionValueMatrix(x,degree) 

    #Perform k-fold cross validation by changing validation set window
    #Initially store sum of validation errors in v_error
    #After all hyperparameters are evaluated, compute average
    v_start_lim = N_train - v_size + 1  
    v_start_list = np.arange(0, v_start_lim, k)
    for vStart in v_start_list:
        (train_ids, val_ids) = getDataSplit(vStart, v_size, N_train)
        (train_err, test_err, learned_poly) = regression.linearPolynomialRegression(degree,degree,x,basis_value_x,targets,train_ids,val_ids,h_set)
        for h_param in h_set:
            v_error[h_param] = v_error[h_param] + test_err[h_param][degree]
    #average out the set validation errors
    for h_param in h_set:
        v_error[h_param] = v_error[h_param]/k      

    #plot the result
    plotAvgValError(v_error)

"""
    Define required range of degrees, param for cross validation and hyperparameters
"""
if __name__ == "__main__":
    #range of degrees and indices
    degree = 2
    f_start = 7
    t_index= 1 
    k = 10
    #hyperparameter range. First value in the set will be 0 always
    N_hyperparams = 9
    min_pow = -2
    max_pow = 6
    base = 10
    #construct hyperparameter set
    h_set = np.zeros(N_hyperparams)
    h_set[0] = 0
    pow_set = np.arange(min_pow, max_pow)
    for index in range(1,N_hyperparams):
        h_set[index] = math.pow(base,pow_set[index-1])   
    #start program
    main(f_start, t_index, degree, h_set, k)