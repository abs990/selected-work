#!/usr/bin/env python3
import data_loader as DL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import linear_basis_function_regression as regression
import textwrap
import regression_operations as regOp
import basis_function_computation as basisFunc
import visualize_1d as visual1d

"""
    Plot results for each input feature as a bar chart
"""
def plotBarChart(train_err, test_err):
    #Extract info for display
    f_labels = train_err.keys()
    train_err_vals = train_err.values()
    test_err_vals = test_err.values()

    #debug statement for results
    debugPrint = False
    if debugPrint:
        for label in f_labels:
            print(label)
            print("TR_err="+str(train_err[label])+","+"TST_err="+str(test_err[label]))

    #set parameters for plot
    label_loc = np.arange(len(f_labels))
    width = 0.45
    rel_pos = 0.5 * width
    train_label = 'Train error'
    test_label = 'Test error'

    #construct the plot
    fig, ax = plt.subplots()
    train_bars = ax.bar(label_loc - rel_pos, train_err_vals, width, label = train_label)
    test_bars = ax.bar(label_loc + rel_pos, test_err_vals, width, label = test_label)
    #set title
    ax.set_title('RMS error for individual features')
    #set label for y axis
    ax.set_ylabel('RMS error')
    #set labels for x axis. Used fixed locator to avoid user warning
    ax.set_xlabel('Features')
    ax.xaxis.set_major_locator(ticker.FixedLocator(label_loc))
    text_wrap_lim = 12
    f = lambda x: textwrap.fill(x, text_wrap_lim)
    ax.set_xticklabels(map(f, f_labels))
    ax.set_xticks(label_loc)
    #give a legend
    ax.legend()
    #pad labels from bars
    padding_mag = 3
    ax.bar_label(train_bars, padding = padding_mag)
    ax.bar_label(test_bars, padding = padding_mag)
    #show the chart
    plt.show()

"""
    1. Define target and inputs
    2. Iteratively perform regression for each input feature.
    3. Produce bar chart of results of (2)
    4. Perform detailed analysis of feature 11,12,13 (indices are 10,11,12)
"""
def main(f_start, f_end, t_index, degree):
    #load data
    (countries, features, values) = DL.load_unicef_data()
    
    #extract target values
    targets = values[:,t_index]
    
    #set indices for training and test data respectively
    N_data, N_features = np.shape(values)
    N_train = 100
    training_indices = np.arange(N_train)
    testing_indices = np.arange(N_train, N_data)
    h_param = 0
    h_set = np.zeros(1)

    #define dictionaries to store result for each input feature
    train_err = {}
    test_err = {}
    learned_poly = {}

    #perform regression for each input feature
    f_lim = f_end+1
    for f_index in np.arange(f_start, f_lim):
        #get feature values
        x = values[:, f_index]
        #get basis function value matrix
        basis_value_x = basisFunc.getBasisFunctionValueMatrix(x, degree)
        #perform regression
        (f_train_err, f_test_err, f_poly) = regression.linearPolynomialRegression(degree, degree, x, basis_value_x, targets, training_indices, testing_indices, h_set)
        train_err[features[f_index]] = f_train_err[h_param][degree]
        test_err[features[f_index]] = f_test_err[h_param][degree]
        learned_poly[f_index] = f_poly[h_param][degree]

    #plot bar chart
    plotBarChart(train_err, test_err)

    #analyse GNI, Life expectancy, literacy
    fa_start = 10
    fa_lim = 13
    for f_index in np.arange(fa_start, fa_lim):
        analysisTitle = 'Analysis of '+features[f_index]
        visual1d.featurePostRegressionAnalysis(values[:, f_index], targets, training_indices, testing_indices, learned_poly[f_index], analysisTitle)

"""
    Define required parameters here
"""
if __name__ == "__main__":
    f_start = 7
    f_end = 14
    t_index = 1
    degree = 3
    main(f_start,f_end, t_index, degree)