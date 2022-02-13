import regression_operations as regOp
import numpy as np

"""
    Linear basis function regression over a range of degrees, for a particular set of hyperparameters
"""
def linearPolynomialRegression(min_degree, max_degree, x, basis_value_x, targets, trainIndices, testIndices, hyperParamSet):

    #validate degree range and hyperparameter set length
    assert max_degree > 0
    assert min_degree > 0
    assert min_degree <= max_degree
    assert len(hyperParamSet) > 0   

    #get values required to proceed with regression computation
    t_train = targets[trainIndices]
    t_test = targets[testIndices]
    n_data, n_features = x.shape

    #Dictionaries to store training errors, test errors and learned polynomials
    train_err = {}
    test_err = {}
    learned_poly = {}

    #iterate and evaluate for each hyperparameter
    for hyperParam in hyperParamSet:
        curr_train_err = {}
        curr_test_err = {}
        curr_learned_poly = {}
        #iterate and evaluate training and test error for each degree
        for degree in np.arange(min_degree,max_degree+1):
            #compute number of columns to extract as per current degree
            if min_degree < max_degree:
                feature_lim = degree * n_features + 1
            else:
                nr, feature_lim = basis_value_x.shape      
            #extract training set values for current degree
            train_basis_value_x = basis_value_x[trainIndices, 0:feature_lim]
            #evaluate optimal parameters at current degree
            w_opt = regOp.evaluateOptimalParameters(train_basis_value_x, t_train, hyperParam)
            curr_learned_poly[degree] = w_opt
            #evaluate rms for training data
            train_rms = regOp.computeRms(train_basis_value_x, w_opt, t_train)
            curr_train_err[degree] = train_rms
            #extract test set values for current degree
            test_basis_value_x = basis_value_x[testIndices, 0:feature_lim]
            #evaluate rms for test data
            test_rms = regOp.computeRms(test_basis_value_x, w_opt, t_test)
            curr_test_err[degree] = test_rms
        
        #store results for current hyperparameter
        train_err[hyperParam] = curr_train_err
        test_err[hyperParam] = curr_test_err
        learned_poly[hyperParam] = curr_learned_poly

    return (train_err, test_err, learned_poly)