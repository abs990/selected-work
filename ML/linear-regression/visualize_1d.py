import data_loader as DL
import numpy as np
import matplotlib.pyplot as plt
import basis_function_computation as basisFunc
import regression_operations as regOp

def featurePostRegressionAnalysis(x, targets, trainingIndices, testingIndices, learnedPolynomial, plotTitle):
    #get train and test values
    x_train = x[trainingIndices]
    x_test = x[testingIndices]

    #prep to display polynomial
    numSamplePoints = 300
    degree = 3
    x_poly = np.zeros((numSamplePoints, 1))
    x_poly[:,0] = np.linspace(min(x_train).item(), max(x_train).item(), num=numSamplePoints)
    x_poly_basis_vals = basisFunc.getBasisFunctionValueMatrix(x_poly, degree)
    y_poly = regOp.computePredictionVector(learnedPolynomial, x_poly_basis_vals)
    y_train = targets[trainingIndices]
    y_test = targets[testingIndices]

    #display values
    plt.figure()
    plt.title(plotTitle)
    plt.plot(x_poly,y_poly,'g-')
    plt.plot(x_train,y_train,'b.')
    plt.plot(x_test,y_test,'r.')
    plt.xlabel('Feature value')
    plt.ylabel('U5MR rate')
    plt.legend(['Learned Polynomial', 'Training points', 'Testing points'])
    plt.show()
